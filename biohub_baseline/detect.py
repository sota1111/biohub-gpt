from __future__ import annotations

import numpy as np
from scipy import ndimage


def _refine_peak(
    volume: np.ndarray, peak: tuple[int, int, int], radius: int
) -> tuple[float, float, float]:
    slices = tuple(
        slice(max(0, coordinate - radius), min(size, coordinate + radius + 1))
        for coordinate, size in zip(peak, volume.shape)
    )
    patch = volume[slices].astype(float, copy=False)
    background = float(np.percentile(patch, 25))
    weights = np.maximum(patch - background, 0.0)
    if float(weights.sum()) == 0:
        return tuple(float(value) for value in peak)
    local = ndimage.center_of_mass(weights)
    return tuple(float(part.start + offset) for part, offset in zip(slices, local))


def detect_adaptive_centroids(
    volume: np.ndarray,
    threshold_percentile: float = 99.0,
    local_sigma: float = 2.0,
    local_offset: float = 0.5,
    peak_distance: int = 2,
    refine_radius: int = 2,
    nms_distance: float = 2.0,
) -> list[tuple[float, float, float]]:
    """Detect locally bright peaks and refine them to sub-voxel z/y/x coordinates."""
    volume = np.asarray(volume, dtype=float)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D z/y/x frame, got shape {volume.shape}")
    if peak_distance < 1 or refine_radius < 1 or nms_distance <= 0:
        raise ValueError("peak/refinement distances must be positive")

    local_mean = ndimage.gaussian_filter(volume, sigma=local_sigma)
    residual = volume - local_mean
    positive = residual[residual > 0]
    if positive.size == 0:
        return []
    residual_floor = float(np.percentile(positive, threshold_percentile))
    noise = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
    adaptive_floor = np.maximum(residual_floor, local_offset * noise)
    window = 2 * peak_distance + 1
    maxima = volume == ndimage.maximum_filter(volume, size=window, mode="nearest")
    peak_mask = maxima & (residual >= adaptive_floor)
    peaks = [tuple(int(value) for value in point) for point in np.argwhere(peak_mask)]
    peaks.sort(key=lambda point: (-float(residual[point]), point))

    kept: list[tuple[float, float, float]] = []
    for peak in peaks:
        refined = _refine_peak(volume, peak, refine_radius)
        if all(np.linalg.norm(np.subtract(refined, existing)) >= nms_distance for existing in kept):
            kept.append(refined)
    return sorted(kept)


def detect_centroids(
    volume: np.ndarray,
    threshold_percentile: float = 99.5,
    min_voxels: int = 4,
    detection_model: dict[str, object] | None = None,
) -> list[tuple[float, float, float]]:
    """Return z/y/x centroids of bright connected components in one 3-D frame."""
    if detection_model is not None:
        return detect_adaptive_centroids(
            volume,
            threshold_percentile=float(
                detection_model.get("threshold_percentile", threshold_percentile)
            ),
            local_sigma=float(detection_model.get("local_sigma", 2.0)),
            local_offset=float(detection_model.get("local_offset", 0.5)),
            peak_distance=int(detection_model.get("peak_distance", 2)),
            refine_radius=int(detection_model.get("refine_radius", 2)),
            nms_distance=float(detection_model.get("nms_distance", 2.0)),
        )
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D z/y/x frame, got shape {volume.shape}")
    threshold = float(np.percentile(volume, threshold_percentile))
    mask = volume > threshold
    labels, count = ndimage.label(mask)
    if count == 0:
        return []
    sizes = np.bincount(labels.ravel())
    kept = [label for label in range(1, count + 1) if sizes[label] >= min_voxels]
    if not kept:
        return []
    centers = ndimage.center_of_mass(volume, labels, kept)
    return [tuple(float(value) for value in center) for center in centers]
