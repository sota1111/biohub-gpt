from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_spacing(spacing: tuple[float, float, float]) -> np.ndarray:
    values = np.asarray(spacing, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("voxel_spacing must contain three finite positive z/y/x values")
    return values


def estimate_phase_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift: tuple[int, int, int] | None = None,
) -> tuple[float, float, float]:
    """Return the integer z/y/x shift that aligns ``moving`` to ``reference``.

    The implementation uses phase-only correlation and deterministic peak
    selection. Empty/constant frames deliberately return zero rather than
    introducing an arbitrary drift.
    """
    reference = np.asarray(reference, dtype=float)
    moving = np.asarray(moving, dtype=float)
    if reference.ndim != 3 or moving.shape != reference.shape:
        raise ValueError("phase correlation requires equal-shaped 3-D frames")
    reference = reference - float(reference.mean())
    moving = moving - float(moving.mean())
    if not np.any(reference) or not np.any(moving):
        return (0.0, 0.0, 0.0)
    cross_power = np.fft.fftn(reference) * np.conj(np.fft.fftn(moving))
    magnitude = np.abs(cross_power)
    cross_power = np.divide(
        cross_power,
        magnitude,
        out=np.zeros_like(cross_power),
        where=magnitude > np.finfo(float).eps,
    )
    correlation = np.abs(np.fft.ifftn(cross_power))
    peak = np.asarray(np.unravel_index(int(np.argmax(correlation)), correlation.shape))
    shape = np.asarray(reference.shape)
    shift = np.where(peak > shape // 2, peak - shape, peak).astype(float)
    if max_shift is not None:
        limits = np.asarray(max_shift, dtype=float)
        if limits.shape != (3,) or np.any(limits < 0):
            raise ValueError("max_shift must contain three non-negative z/y/x values")
        shift = np.clip(shift, -limits, limits)
    return tuple(float(value) for value in shift)


@dataclass(frozen=True)
class SpatialTransform:
    """Map raw voxel coordinates to a drift-corrected physical reference."""

    voxel_spacing: tuple[float, float, float]
    alignment_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def forward(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        spacing = _validate_spacing(self.voxel_spacing)
        corrected_voxel = np.asarray(point, dtype=float) + np.asarray(
            self.alignment_shift, dtype=float
        )
        return tuple(float(value) for value in corrected_voxel * spacing)

    def inverse(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        spacing = _validate_spacing(self.voxel_spacing)
        raw = np.asarray(point, dtype=float) / spacing - np.asarray(
            self.alignment_shift, dtype=float
        )
        return tuple(float(value) for value in raw)


def estimate_reference_transforms(
    frames: list[np.ndarray],
    voxel_spacing: tuple[float, float, float],
    max_shift: tuple[int, int, int] | None = None,
) -> list[SpatialTransform]:
    """Estimate every frame directly against the first non-empty reference."""
    _validate_spacing(voxel_spacing)
    if not frames:
        return []
    reference = next((frame for frame in frames if np.any(frame)), frames[0])
    transforms = []
    for frame in frames:
        shift = estimate_phase_shift(reference, frame, max_shift)
        transforms.append(SpatialTransform(voxel_spacing, shift))
    return transforms
