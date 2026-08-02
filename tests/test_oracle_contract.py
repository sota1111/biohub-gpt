from copy import deepcopy

import pytest

from biohub_baseline.oracle_contract import digest, fixture_rows, perturbations


def test_fixture_contract_shapes_are_known():
    assert len(fixture_rows("empty")) == 0
    assert len(fixture_rows("one_node")) == 1
    assert len(fixture_rows("one_edge")) == 3
    assert len(fixture_rows("one_division")) == 7


def test_perturbations_cover_each_contract_layer_without_mutating_oracle():
    oracle = fixture_rows("one_division")
    before = deepcopy(oracle)
    variants = perturbations(oracle, (1.625, 0.40625, 0.40625))
    assert set(variants) == {
        "coordinate_axis_swap_zx", "voxel_spacing_isotropic", "frame_index_plus_one",
        "node_id_remap", "edge_direction_reverse", "division_flatten",
        "matching_tolerance_zero",
    }
    assert oracle == before
    assert digest(variants["node_id_remap"][0]) != digest(oracle)


def test_unknown_fixture_is_rejected():
    with pytest.raises(KeyError):
        fixture_rows("unknown")
