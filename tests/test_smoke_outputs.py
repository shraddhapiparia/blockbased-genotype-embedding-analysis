"""
Shape checks for Phase 1 and Phase 2 synthetic pipeline outputs.
These tests are skipped when the outputs don't exist — they are populated by
running `bash test_run.sh` before pytest, as happens in CI.

Expected shapes (from synthetic configs):
  Phase 1 MSE all_blocks.npy : (30, 4, 4)  — 30 subjects, 4 blocks, d=4
  Phase 2 ORD individual_embeddings.npy : (30, 16) — d_model=16
"""
from pathlib import Path

import pytest

_ROOT   = Path(__file__).resolve().parents[1]
_P1_OUT = _ROOT / "results" / "synthetic_test"
_P2_OUT = _ROOT / "results" / "synthetic_test2"

_ALL_BLOCKS   = _P1_OUT / "MSE" / "embeddings" / "all_blocks.npy"
_INDIV_EMB    = _P2_OUT / "ORD" / "embeddings" / "individual_embeddings.npy"
_POOL_ATTN    = _P2_OUT / "ORD" / "embeddings" / "pooling_attention_weights.csv"
_P2_SUMMARY   = _P2_OUT / "phase2_summary.csv"


@pytest.mark.skipif(not _ALL_BLOCKS.exists(), reason="Phase 1 synthetic outputs not generated")
def test_all_blocks_shape():
    """all_blocks.npy must be (N=30, B=4, d_max=4) for the synthetic config."""
    import numpy as np
    arr = np.load(_ALL_BLOCKS)
    assert arr.shape == (30, 4, 4), (
        f"Expected all_blocks shape (30, 4, 4), got {arr.shape}"
    )


@pytest.mark.skipif(not _INDIV_EMB.exists(), reason="Phase 2 synthetic outputs not generated")
def test_individual_embeddings_shape():
    """individual_embeddings.npy must be (N=30, d_model=16) for the synthetic config."""
    import numpy as np
    arr = np.load(_INDIV_EMB)
    assert arr.shape == (30, 16), (
        f"Expected individual_embeddings shape (30, 16), got {arr.shape}"
    )


@pytest.mark.skipif(not _POOL_ATTN.exists(), reason="Phase 2 synthetic outputs not generated")
def test_pooling_attention_weights_has_block_columns():
    """pooling_attention_weights.csv must have one column per block (4 blocks)."""
    import pandas as pd
    df = pd.read_csv(_POOL_ATTN)
    block_cols = [c for c in df.columns if c not in ("IID", "subject_id", "FID")]
    assert len(block_cols) == 4, (
        f"Expected 4 block columns in pooling_attention_weights.csv, got {len(block_cols)}: "
        f"{block_cols}"
    )


@pytest.mark.skipif(not _P2_SUMMARY.exists(), reason="Phase 2 synthetic outputs not generated")
def test_phase2_summary_exists_and_nonempty():
    """phase2_summary.csv must be a non-empty file."""
    import pandas as pd
    df = pd.read_csv(_P2_SUMMARY)
    assert len(df) > 0, "phase2_summary.csv is empty"
