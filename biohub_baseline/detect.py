from __future__ import annotations

import numpy as np
from scipy import ndimage


def _validate_spacing(values: object) -> tuple[float, float, float]:
    spacing = tuple(float(value) for value in values)  # type: ignore[arg-type]
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("voxel_spacing must contain three positive values")
    return spacing


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


def detect_touching_centroids(
    volume: np.ndarray,
    threshold_percentile: float = 99.0,
    local_sigma: float = 2.0,
    local_offset: float = 0.5,
    marker_distance: float = 2.0,
    min_component_voxels: int = 4,
    min_instance_voxels: int = 2,
    max_component_voxels: int = 4096,
    min_peak_distance: float = 0.75,
    min_component_intensity_ratio: float = 0.1,
    separation_confidence: float = 0.2,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[tuple[float, float, float]]:
    """Split touching foreground instances with anisotropic marker-controlled watershed."""
    volume = np.asarray(volume, dtype=float)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D z/y/x frame, got shape {volume.shape}")
    spacing = _validate_spacing(voxel_spacing)
    if (
        marker_distance <= 0
        or min_component_voxels < 1
        or min_instance_voxels < 1
        or max_component_voxels < min_component_voxels
        or min_peak_distance <= 0
        or not 0 <= min_component_intensity_ratio <= 1
        or not 0 <= separation_confidence <= 1
    ):
        raise ValueError("invalid touching-instance detector constraints")

    local_mean = ndimage.gaussian_filter(volume, sigma=local_sigma)
    residual = volume - local_mean
    positive = residual[residual > 0]
    if positive.size == 0:
        return []
    residual_floor = float(np.percentile(positive, threshold_percentile))
    noise = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
    foreground = residual >= max(residual_floor, local_offset * noise)
    foreground = ndimage.binary_fill_holes(foreground)
    foreground_labels, component_count = ndimage.label(foreground)
    global_max = max(float(np.nanmax(volume)), np.finfo(float).eps)

    centers: list[tuple[float, float, float]] = []
    window = tuple(
        max(3, 2 * int(np.floor(marker_distance / axis_spacing)) + 1)
        for axis_spacing in spacing
    )
    for component_id in range(1, component_count + 1):
        component = foreground_labels == component_id
        component_size = int(component.sum())
        if component_size < min_component_voxels or component_size > max_component_voxels:
            continue
        if float(np.nanmax(volume[component])) / global_max < min_component_intensity_ratio:
            continue
        distance = ndimage.distance_transform_edt(component, sampling=spacing)
        peak_mask = (
            component
            & (distance == ndimage.maximum_filter(distance, size=window, mode="constant"))
            & (distance >= min_peak_distance)
        )
        marker_labels, marker_count = ndimage.label(peak_mask)
        if marker_count == 0:
            centers.append(
                tuple(float(value) for value in ndimage.center_of_mass(component))
            )
            continue

        peak_strengths = [
            float(distance[marker_labels == marker_id].max())
            for marker_id in range(1, marker_count + 1)
        ]
        strongest = max(peak_strengths)
        accepted = [
            marker_id
            for marker_id, strength in enumerate(peak_strengths, start=1)
            if strength / strongest >= separation_confidence
        ]
        markers = np.zeros(volume.shape, dtype=np.int32)
        for output_id, marker_id in enumerate(accepted, start=1):
            markers[marker_labels == marker_id] = output_id
        if len(accepted) == 1:
            centers.append(
                tuple(float(value) for value in ndimage.center_of_mass(component))
            )
            continue

        normalized = np.clip(
            255 * (1.0 - distance / max(float(distance.max()), np.finfo(float).eps)),
            0,
            255,
        ).astype(np.uint8)
        watershed = ndimage.watershed_ift(normalized, markers)
        for instance_id in range(1, len(accepted) + 1):
            instance = component & (watershed == instance_id)
            if int(instance.sum()) < min_instance_voxels:
                continue
            centers.append(
                tuple(float(value) for value in ndimage.center_of_mass(instance))
            )
    return sorted(centers)


def detect_centroids(
    volume: np.ndarray,
    threshold_percentile: float = 99.5,
    min_voxels: int = 4,
    detection_model: dict[str, object] | None = None,
) -> list[tuple[float, float, float]]:
    """Return z/y/x centroids of bright connected components in one 3-D frame."""
    if detection_model is not None:
        if detection_model.get("name") == "touching-watershed-v1":
            return detect_touching_centroids(
                volume,
                threshold_percentile=float(
                    detection_model.get("threshold_percentile", threshold_percentile)
                ),
                local_sigma=float(detection_model.get("local_sigma", 2.0)),
                local_offset=float(detection_model.get("local_offset", 0.5)),
                marker_distance=float(detection_model.get("marker_distance", 2.0)),
                min_component_voxels=int(
                    detection_model.get("min_component_voxels", min_voxels)
                ),
                min_instance_voxels=int(detection_model.get("min_instance_voxels", 2)),
                max_component_voxels=int(
                    detection_model.get("max_component_voxels", 4096)
                ),
                min_peak_distance=float(detection_model.get("min_peak_distance", 0.75)),
                min_component_intensity_ratio=float(
                    detection_model.get("min_component_intensity_ratio", 0.1)
                ),
                separation_confidence=float(
                    detection_model.get("separation_confidence", 0.2)
                ),
                voxel_spacing=_validate_spacing(
                    detection_model.get("voxel_spacing", (1.0, 1.0, 1.0))
                ),
            )
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
