"""
Tests for the subject IID alignment logic used in load_raw()
(scripts/analysis/10_phase1_snp_attribution_within_blocks.py).

The alignment logic is tested inline to avoid importing the full script, which
has a torch dependency at module level.  The logic under test is:
  - identical order  → accept as-is
  - same set, different order  → reorder to match expected
  - different set  → raise ValueError
  - duplicate IIDs in either list  → raise ValueError
"""
import pytest


def _align_iids(raw_iids, expected_iids):
    """
    Mirrors the IID alignment block in load_raw().
    Returns the (possibly reordered) raw_iids list.
    Raises ValueError for genuine mismatches.
    """
    raw_iids = list(raw_iids)
    expected_iids = list(expected_iids)

    n_raw = len(raw_iids)
    n_exp = len(expected_iids)
    if n_raw != n_exp:
        raise ValueError(f"IID count mismatch: raw={n_raw}, expected={n_exp}")

    if len(set(raw_iids)) != n_raw:
        raise ValueError("Duplicate IIDs in raw file")
    if len(set(expected_iids)) != n_exp:
        raise ValueError("Duplicate IIDs in expected list")

    raw_set = set(raw_iids)
    exp_set = set(expected_iids)
    if raw_set != exp_set:
        extra = raw_set - exp_set
        missing = exp_set - raw_set
        raise ValueError(f"IID set mismatch: extra={extra}, missing={missing}")

    if raw_iids == expected_iids:
        return raw_iids, False

    iid_to_pos = {iid: i for i, iid in enumerate(raw_iids)}
    new_order = [iid_to_pos[iid] for iid in expected_iids]
    reordered = [raw_iids[i] for i in new_order]
    assert reordered == expected_iids
    return reordered, True


def test_identical_order_passes():
    iids = ["S01", "S02", "S03"]
    result, was_reordered = _align_iids(iids, iids)
    assert result == iids
    assert not was_reordered


def test_reordered_set_is_fixed():
    raw = ["S03", "S01", "S02"]
    expected = ["S01", "S02", "S03"]
    result, was_reordered = _align_iids(raw, expected)
    assert result == expected
    assert was_reordered


def test_different_set_raises():
    raw = ["S01", "S02", "S03"]
    expected = ["S01", "S02", "S99"]  # S99 not in raw
    with pytest.raises(ValueError, match="IID set mismatch"):
        _align_iids(raw, expected)


def test_duplicate_iids_in_raw_raises():
    raw = ["S01", "S01", "S02"]  # duplicate S01
    expected = ["S01", "S02", "S03"]
    with pytest.raises(ValueError, match="Duplicate"):
        _align_iids(raw, expected)


def test_count_mismatch_raises():
    raw = ["S01", "S02"]
    expected = ["S01", "S02", "S03"]
    with pytest.raises(ValueError, match="count mismatch"):
        _align_iids(raw, expected)
