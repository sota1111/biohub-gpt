from __future__ import annotations

import numpy as np
from scipy import ndimage


def detect_centroids(
    volume: np.ndarray, threshold_percentile: float = 99.5, min_voxels: int = 4
) -> list[tuple[float, float, float]]:
    """Return z/y/x centroids of bright connected components in one 3-D frame."""
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
