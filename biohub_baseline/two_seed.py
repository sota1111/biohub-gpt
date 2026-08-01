"""Deterministic, detector-only two-seed 3-D logit blending.

Torch is imported lazily so the classical production path remains CPU/offline compatible.
Models passed to :func:`infer_two_seed_nodes` must map ``N,C,Z,Y,X`` patches to a
single-channel logit tensor on the same grid.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class TwoSeedConfig:
    patch_shape: tuple[int, int, int] = (32, 128, 128)
    overlap: tuple[int, int, int] = (8, 32, 32)
    blend_weight: float = 0.475
    threshold: float = 0.96875
    min_distance_um: float = 2.0
    min_instance_voxels: int = 2
    voxel_spacing_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

    def validate(self) -> None:
        if any(size < 1 for size in self.patch_shape):
            raise ValueError("patch_shape must be positive")
        if any(value < 0 or value >= size for value, size in zip(self.overlap, self.patch_shape)):
            raise ValueError("overlap must be non-negative and smaller than patch_shape")
        if not 0.0 <= self.blend_weight <= 1.0:
            raise ValueError("blend_weight must be in [0, 1]")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if self.min_distance_um <= 0 or self.min_instance_voxels < 1:
            raise ValueError("instance constraints must be positive")
        if len(self.voxel_spacing_um) != 3 or any(value <= 0 for value in self.voxel_spacing_um):
            raise ValueError("voxel_spacing_um must have three positive values")


def _starts(length: int, patch: int, overlap: int) -> list[int]:
    if length <= patch:
        return [0]
    stride = patch - overlap
    values = list(range(0, length - patch + 1, stride))
    if values[-1] != length - patch:
        values.append(length - patch)
    return values


def patch_slices(shape: Iterable[int], config: TwoSeedConfig) -> list[tuple[slice, slice, slice]]:
    """Return a stable full-coverage patch grid in z/y/x order."""
    config.validate()
    shape = tuple(int(value) for value in shape)
    if len(shape) != 3 or any(value < 1 for value in shape):
        raise ValueError("shape must contain three positive values")
    axes = [_starts(n, p, o) for n, p, o in zip(shape, config.patch_shape, config.overlap)]
    return [
        tuple(slice(start, min(start + size, limit)) for start, size, limit in zip(origin, config.patch_shape, shape))
        for origin in product(*axes)
    ]


def _window(shape: tuple[int, int, int], torch: Any, device: Any) -> Any:
    axes = []
    for size in shape:
        if size == 1:
            axes.append(torch.ones(1, device=device))
        else:
            axes.append(torch.hann_window(size, periodic=False, device=device).clamp_min(1e-3))
    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]


def _aligned_blend(primary: Any, secondary: Any, weight: float) -> Any:
    """Blend seeds after per-patch affine logit alignment."""
    p_mean, s_mean = primary.mean(), secondary.mean()
    p_std = primary.float().std(unbiased=False).clamp_min(1e-4)
    s_std = secondary.float().std(unbiased=False).clamp_min(1e-4)
    aligned = (secondary - s_mean) * (p_std / s_std).clamp(0.5, 2.0) + p_mean
    return (1.0 - weight) * primary + weight * aligned


def infer_blended_logits(volume: np.ndarray, primary: Any, secondary: Any, config: TwoSeedConfig, device: str = "cuda") -> np.ndarray:
    """Run overlap-weighted two-model inference and return one z/y/x logit volume."""
    config.validate()
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured GPU inference path")
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected 3-D z/y/x volume, got {array.shape}")
    target = torch.device(device)
    output = torch.zeros(array.shape, dtype=torch.float32, device=target)
    weights = torch.zeros_like(output)
    primary.eval()
    secondary.eval()
    with torch.inference_mode():
        for region in patch_slices(array.shape, config):
            patch = torch.from_numpy(array[region]).to(target)
            mean, std = patch.mean(), patch.std(unbiased=False).clamp_min(1e-6)
            batch = ((patch - mean) / std)[None, None]
            first = primary(batch)
            second = secondary(batch)
            if isinstance(first, (tuple, list)):
                first = first[-1]
            if isinstance(second, (tuple, list)):
                second = second[-1]
            first, second = first.squeeze(), second.squeeze()
            if first.shape != patch.shape or second.shape != patch.shape:
                raise ValueError("both models must return logits on the input patch grid")
            blended = _aligned_blend(first, second, config.blend_weight)
            window = _window(tuple(patch.shape), torch, target)
            output[region] += blended.float() * window
            weights[region] += window
    return (output / weights.clamp_min(1e-8)).cpu().numpy()


def logits_to_nodes(logits: np.ndarray, config: TwoSeedConfig) -> list[tuple[float, float, float]]:
    """Convert logits to deterministic anisotropy-aware node candidates."""
    config.validate()
    logits = np.asarray(logits, dtype=np.float32)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    foreground = probability >= config.threshold
    labels, count = ndimage.label(foreground)
    if count == 0:
        return []
    maxima_window = tuple(
        max(1, 2 * int(np.floor(config.min_distance_um / spacing)) + 1)
        for spacing in config.voxel_spacing_um
    )
    maxima = probability == ndimage.maximum_filter(probability, size=maxima_window, mode="nearest")
    candidates: list[tuple[float, tuple[float, float, float]]] = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        if int(component.sum()) < config.min_instance_voxels:
            continue
        peaks = np.argwhere(component & maxima)
        if not len(peaks):
            peaks = np.asarray([np.unravel_index(np.argmax(probability * component), logits.shape)])
        for peak in peaks:
            point = tuple(float(value) for value in peak)
            candidates.append((float(probability[tuple(peak)]), point))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    kept: list[tuple[float, float, float]] = []
    spacing = np.asarray(config.voxel_spacing_um)
    for _, point in candidates:
        if all(np.linalg.norm((np.asarray(point) - np.asarray(other)) * spacing) >= config.min_distance_um for other in kept):
            kept.append(point)
    return sorted(kept)


def infer_two_seed_nodes(volume: np.ndarray, primary: Any, secondary: Any, config: TwoSeedConfig, device: str = "cuda") -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    logits = infer_blended_logits(volume, primary, secondary, config, device)
    return logits, logits_to_nodes(logits, config)
