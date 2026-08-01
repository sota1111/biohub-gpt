import pytest

from biohub_baseline.real_cv import embryo_id, split_digest, validate_fixed_split


def test_embryo_held_out_split_is_fixed_and_disjoint():
    result = validate_fixed_split(
        ["6bba_b", "44b6_b", "44b6_a", "6bba_a"], ["44b6"], ["6bba"]
    )
    assert result == {
        "screen": ["44b6_a", "44b6_b"],
        "confirm": ["6bba_a", "6bba_b"],
    }
    assert split_digest(result) == split_digest(result)
    assert not {embryo_id(item) for item in result["screen"]} & {
        embryo_id(item) for item in result["confirm"]
    }


def test_split_rejects_series_leakage_and_unassigned_embryos():
    with pytest.raises(ValueError, match="embryo leakage"):
        validate_fixed_split(["44b6_a"], ["44b6"], ["44b6"])
    with pytest.raises(ValueError, match="unassigned embryo"):
        validate_fixed_split(["other_a", "44b6_a", "6bba_a"], ["44b6"], ["6bba"])


def test_dataset_id_requires_embryo_prefix():
    with pytest.raises(ValueError, match="no embryo prefix"):
        embryo_id("invalid")


def test_fixed_split_digest_is_immutable():
    split = {"screen": ["44b6_0113de3b"], "confirm": ["6bba_05b6850b"]}
    assert split_digest(split) == "f26b52e2a6174a21984e9d3fe089a2a2ab7291d5740217bfd2270d90e9efd320"
