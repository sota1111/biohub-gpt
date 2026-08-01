import numpy as np
import pytest

from biohub_baseline.track import LinkConfig, link_constrained
from biohub_baseline.two_seed import (
    TwoSeedConfig,
    infer_blended_logits,
    logits_to_nodes,
    patch_slices,
)


def test_patch_grid_is_deterministic_and_covers_volume():
    config = TwoSeedConfig(patch_shape=(4, 6, 6), overlap=(1, 2, 2))
    regions = patch_slices((7, 11, 12), config)
    covered = np.zeros((7, 11, 12), dtype=bool)
    for region in regions:
        covered[region] = True
    assert covered.all()
    assert regions == patch_slices((7, 11, 12), config)


def test_anisotropic_node_conversion_is_stable_and_filters_small_instances():
    logits = np.full((5, 9, 9), -20.0, dtype=np.float32)
    logits[2, 3:5, 3:5] = 20.0
    logits[4, 8, 8] = 20.0
    config = TwoSeedConfig(threshold=0.9, min_instance_voxels=2, min_distance_um=1.0)
    first = logits_to_nodes(logits, config)
    assert first == logits_to_nodes(logits, config)
    assert len(first) == 1


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        patch_slices((4, 4, 4), TwoSeedConfig(patch_shape=(4, 4, 4), overlap=(4, 0, 0)))


def test_frozen_tracker_accepts_empty_neural_detection_frames():
    assert link_constrained([[], [], []], LinkConfig(max_distance=12.0)) == []


def test_gpu_blend_is_deterministic_when_cuda_is_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    class Scale(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, value):
            return value * self.scale

    rng = np.random.default_rng(1988)
    volume = rng.normal(size=(7, 11, 12)).astype(np.float32)
    config = TwoSeedConfig(patch_shape=(4, 6, 6), overlap=(1, 2, 2), blend_weight=0.475)
    first = infer_blended_logits(volume, Scale(1.0).cuda(), Scale(1.5).cuda(), config)
    second = infer_blended_logits(volume, Scale(1.0).cuda(), Scale(1.5).cuda(), config)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
