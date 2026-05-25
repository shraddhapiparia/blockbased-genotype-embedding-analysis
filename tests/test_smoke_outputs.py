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

_ALL_BLOCKS        = _P1_OUT / "MSE" / "embeddings" / "all_blocks.npy"
_INDIV_EMB         = _P2_OUT / "ORD" / "embeddings" / "individual_embeddings.npy"
_POOL_ATTN         = _P2_OUT / "ORD" / "embeddings" / "pooling_attention_weights.csv"
_P2_SUMMARY        = _P2_OUT / "phase2_summary.csv"
_INITIAL_REPR      = _P2_OUT / "ORD" / "embeddings" / "block_initial_repr.npy"
_CTX_CHANGE_BLOCK  = _P2_OUT / "ORD" / "embeddings" / "per_block_contextualization_change.csv"
_P2_BLOCK_DIAG     = _P2_OUT / "ORD" / "diagnostics" / "phase2_block_diagnostics.csv"
_PCA_SUMMARY       = _P2_OUT / "ORD" / "baselines" / "pca_baseline_summary.csv"
_MP_SUMMARY        = _P2_OUT / "ORD" / "baselines" / "mean_pool_baseline_summary.csv"


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


@pytest.mark.skipif(
    not (_P2_SUMMARY.exists() and _INITIAL_REPR.exists()),
    reason="Phase 2 synthetic outputs not generated (or pre-date diagnostics)"
)
def test_phase2_summary_diagnostic_columns():
    """phase2_summary.csv must contain the new diagnostic columns."""
    import pandas as pd
    df = pd.read_csv(_P2_SUMMARY)
    required = {
        "mean_pool_attn_entropy",
        "mean_context_delta_l2",
        "pca_val_recon_loss",
        "raw_mean_pool_val_recon_loss",
        "embedhead_mean_pool_val_recon_loss",
    }
    missing = required - set(df.columns)
    assert not missing, f"phase2_summary.csv missing columns: {missing}"


@pytest.mark.skipif(not _INITIAL_REPR.exists(), reason="Phase 2 synthetic outputs not generated")
def test_block_initial_repr_shape():
    """block_initial_repr.npy must be (N, B, d_model) = (30, 4, 16) for synthetic config."""
    import numpy as np
    arr = np.load(_INITIAL_REPR)
    assert arr.ndim == 3, f"Expected 3-D array, got shape {arr.shape}"
    assert arr.shape[0] == 30, f"Expected 30 subjects, got {arr.shape[0]}"


@pytest.mark.skipif(not _CTX_CHANGE_BLOCK.exists(), reason="Phase 2 synthetic outputs not generated")
def test_per_block_ctx_change_has_block_rows():
    """per_block_contextualization_change.csv must have one row per block."""
    import pandas as pd
    df = pd.read_csv(_CTX_CHANGE_BLOCK)
    assert len(df) == 4, f"Expected 4 block rows, got {len(df)}"
    assert "mean_context_delta_l2" in df.columns


@pytest.mark.skipif(not _P2_BLOCK_DIAG.exists(), reason="Phase 2 synthetic outputs not generated")
def test_phase2_block_diagnostics_exists():
    """phase2_block_diagnostics.csv must be non-empty."""
    import pandas as pd
    df = pd.read_csv(_P2_BLOCK_DIAG)
    assert len(df) > 0, "phase2_block_diagnostics.csv is empty"


@pytest.mark.skipif(not _PCA_SUMMARY.exists(), reason="Phase 2 synthetic outputs not generated")
def test_pca_baseline_summary_exists():
    """pca_baseline_summary.csv must exist and contain key columns."""
    import pandas as pd
    df = pd.read_csv(_PCA_SUMMARY)
    assert "pca_val_recon_loss" in df.columns


@pytest.mark.skipif(not _MP_SUMMARY.exists(), reason="Phase 2 synthetic outputs not generated")
def test_mean_pool_baseline_summary_exists():
    """mean_pool_baseline_summary.csv must exist and contain key columns."""
    import pandas as pd
    df = pd.read_csv(_MP_SUMMARY)
    assert "raw_mean_pool_val_recon_loss" in df.columns
