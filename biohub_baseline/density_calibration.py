"""Deterministic density/stage-aware calibration for classical nucleus candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class DensityCalibration:
    confidence_sigma: float
    dense_confidence_step: float
    min_component_voxels: int
    dense_component_step: int
    nms_radius_um: float
    dense_nms_step_um: float
    candidate_density_target: float = 2e-5
    late_stage_fraction: float = 0.6
    late_stage_relaxation: float = 0.15
    voxel_spacing_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

    def validate(self) -> None:
        if self.confidence_sigma <= 0 or self.dense_confidence_step < 0:
            raise ValueError("confidence constraints must be non-negative")
        if self.min_component_voxels < 1 or self.dense_component_step < 0:
            raise ValueError("component constraints must be non-negative")
        if self.nms_radius_um <= 0 or self.dense_nms_step_um < 0:
            raise ValueError("NMS constraints must be non-negative")
        if self.candidate_density_target <= 0:
            raise ValueError("candidate_density_target must be positive")
        if not 0 <= self.late_stage_fraction <= 1 or self.late_stage_relaxation < 0:
            raise ValueError("invalid development-stage constraints")
        if len(self.voxel_spacing_um) != 3 or any(value <= 0 for value in self.voxel_spacing_um):
            raise ValueError("voxel_spacing_um must contain three positive values")


def _robust_location_scale(volume: np.ndarray) -> tuple[float, float]:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, 1.0
    location = float(np.median(finite))
    scale = float(np.median(np.abs(finite - location)) * 1.4826)
    return location, max(scale, float(np.std(finite)) * 1e-3, np.finfo(float).eps)


def detect_density_calibrated(
    volume: np.ndarray,
    config: DensityCalibration,
    frame_index: int,
    frame_count: int,
) -> list[tuple[float, float, float]]:
    """Filter candidates using robust confidence, component mass and physical NMS."""
    config.validate()
    array = np.asarray(volume, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"expected a 3-D z/y/x frame, got {array.shape}")
    if frame_count < 1 or not 0 <= frame_index < frame_count:
        raise ValueError("frame index must be within frame_count")
    location, scale = _robust_location_scale(array)
    confidence = (array - location) / scale
    provisional = confidence >= config.confidence_sigma
    density_ratio = float(provisional.sum()) / (array.size * config.candidate_density_target)
    density_level = max(0, int(np.ceil(np.log2(max(1.0, density_ratio)))))
    late = frame_index / max(1, frame_count - 1) >= config.late_stage_fraction
    threshold = (
        config.confidence_sigma
        + density_level * config.dense_confidence_step
        - (config.late_stage_relaxation if late else 0.0)
    )
    foreground = ndimage.binary_fill_holes(confidence >= threshold)
    labels, count = ndimage.label(foreground)
    minimum = config.min_component_voxels + density_level * config.dense_component_step
    candidates: list[tuple[float, tuple[float, float, float]]] = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        if int(component.sum()) < minimum:
            continue
        peak = np.unravel_index(np.argmax(np.where(component, confidence, -np.inf)), array.shape)
        weights = np.maximum(confidence[component] - threshold, 0.0) + np.finfo(float).eps
        coordinates = np.argwhere(component)
        center = tuple(float(value) for value in np.average(coordinates, axis=0, weights=weights))
        candidates.append((float(confidence[peak]), center))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    radius = config.nms_radius_um + density_level * config.dense_nms_step_um
    spacing = np.asarray(config.voxel_spacing_um)
    kept: list[tuple[float, float, float]] = []
    for _, point in candidates:
        if all(np.linalg.norm((np.asarray(point) - np.asarray(other)) * spacing) >= radius for other in kept):
            kept.append(point)
    return sorted(kept)
