"""
Tests for Phase 2 diagnostics and baseline utilities.
All tests use tiny synthetic tensors; no real data or checkpoints required.
"""
import math
import numpy as np
import pytest
from pathlib import Path

torch = pytest.importorskip("torch", reason="torch not installed")
pd    = pytest.importorskip("pandas", reason="pandas not installed")

from scripts.core.attention_phase2 import (
    AttentionAggregator,
    compute_contextualization_change,
    _extract_initial_tokens,
    _extract_pool_attn_by_token,
    _pool_attn_entropy,
    _pool_attn_topk_mass,
    run_pca_baseline,
    run_mean_pool_baseline,
    run_phase2_block_diagnostics,
)


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    m = AttentionAggregator(
        n_blocks=4, d_in=6, d_model=8, n_heads=2,
        n_layers=1, d_ff=16, dropout=0.0,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.RandomState(1)
    N, B, d_in = 20, 4, 6
    emb  = rng.randn(N, B, d_in).astype(np.float32)
    tr_ix = np.arange(14)
    va_ix = np.arange(14, 17)
    te_ix = np.arange(17, 20)
    return emb, tr_ix, va_ix, te_ix, B, d_in


# ── Part A: get_initial_tokens / _extract_initial_tokens ──────────────────────

def test_get_initial_tokens_shape(tiny_model):
    """get_initial_tokens must return (batch, B, d_model)."""
    x = torch.randn(5, 4, 6)
    with torch.no_grad():
        tokens = tiny_model.get_initial_tokens(x)
    assert tokens.shape == (5, 4, 8), f"Expected (5, 4, 8), got {tokens.shape}"


def test_extract_initial_tokens_shape(tiny_model, synthetic_data):
    """_extract_initial_tokens must return (N, B, d_model) for the full dataset."""
    emb, *_ = synthetic_data
    arr = _extract_initial_tokens(tiny_model, emb, batch_size=8)
    assert arr.shape == (20, 4, 8), f"Expected (20, 4, 8), got {arr.shape}"


def test_initial_tokens_differ_from_contextual(tiny_model, synthetic_data):
    """Initial tokens before self-attention should differ from contextual tokens after."""
    emb, *_ = synthetic_data
    initial = _extract_initial_tokens(tiny_model, emb, batch_size=32)
    x_t = torch.tensor(emb)
    with torch.no_grad():
        _, _, contextual = tiny_model.encode(x_t, return_self_attn=False)
    contextual_np = contextual.numpy()
    # They must not be identical (transformer layers change the representation)
    assert not np.allclose(initial, contextual_np), (
        "initial and contextual tokens are identical — Transformer may not be modifying them"
    )


# ── Part B: compute_contextualization_change ───────────────────────────────────

def test_ctx_change_required_columns():
    """per_block_df must contain all required columns."""
    N, B, d = 10, 4, 8
    rng = np.random.RandomState(0)
    init = rng.randn(N, B, d).astype(np.float32)
    ctx  = init + rng.randn(N, B, d).astype(np.float32) * 0.1
    bnames = [f"block_{i}" for i in range(B)]
    per_block, per_subj = compute_contextualization_change(init, ctx, bnames)

    required_pb = {
        "block_id", "mean_context_delta_l2", "std_context_delta_l2",
        "mean_context_delta_cosine", "std_context_delta_cosine",
        "context_change_rank",
    }
    missing = required_pb - set(per_block.columns)
    assert not missing, f"per_block missing columns: {missing}"

    required_ps = {
        "mean_context_delta_l2_per_subject",
        "mean_context_delta_cosine_per_subject",
    }
    missing2 = required_ps - set(per_subj.columns)
    assert not missing2, f"per_subj missing columns: {missing2}"


def test_ctx_change_row_counts():
    """per_block must have B rows; per_subj must have N rows."""
    N, B, d = 10, 5, 8
    init = np.zeros((N, B, d), dtype=np.float32)
    ctx  = np.ones((N, B, d),  dtype=np.float32)
    bnames = [f"block_{i}" for i in range(B)]
    per_block, per_subj = compute_contextualization_change(init, ctx, bnames)
    assert len(per_block) == B, f"Expected {B} rows, got {len(per_block)}"
    assert len(per_subj)  == N, f"Expected {N} rows, got {len(per_subj)}"


def test_ctx_change_l2_nonneg():
    """Mean L2 context-change must be >= 0 for all blocks."""
    N, B, d = 8, 3, 6
    rng = np.random.RandomState(2)
    init = rng.randn(N, B, d).astype(np.float32)
    ctx  = rng.randn(N, B, d).astype(np.float32)
    bnames = [f"b{i}" for i in range(B)]
    per_block, _ = compute_contextualization_change(init, ctx, bnames)
    assert (per_block["mean_context_delta_l2"] >= 0).all()


def test_ctx_change_identical_inputs_zero_delta():
    """When initial == contextual, L2 delta must be ~0 for all blocks."""
    N, B, d = 6, 3, 4
    rng = np.random.RandomState(3)
    x = rng.randn(N, B, d).astype(np.float32)
    bnames = [f"b{i}" for i in range(B)]
    per_block, per_subj = compute_contextualization_change(x, x, bnames)
    assert np.allclose(per_block["mean_context_delta_l2"].values, 0.0, atol=1e-5)
    assert np.allclose(per_subj["mean_context_delta_l2_per_subject"].values, 0.0, atol=1e-5)


def test_ctx_change_rank_is_permutation():
    """context_change_rank must be a permutation of 1..B."""
    N, B, d = 12, 5, 8
    rng = np.random.RandomState(4)
    init = rng.randn(N, B, d).astype(np.float32)
    ctx  = rng.randn(N, B, d).astype(np.float32)
    bnames = [f"b{i}" for i in range(B)]
    per_block, _ = compute_contextualization_change(init, ctx, bnames)
    ranks = sorted(per_block["context_change_rank"].tolist())
    assert ranks == list(range(1, B + 1)), f"Ranks not a permutation of 1..{B}: {ranks}"


# ── Pool attention entropy / top-k mass ────────────────────────────────────────

def test_pool_attn_entropy_positive():
    """Entropy must be > 0 for non-degenerate attention distributions."""
    pa = np.array([[0.25, 0.25, 0.25, 0.25],
                   [0.70, 0.10, 0.10, 0.10]])
    assert _pool_attn_entropy(pa) > 0


def test_pool_attn_entropy_uniform_max():
    """Uniform distribution has higher entropy than peaked distribution."""
    B = 8
    uniform = np.full((20, B), 1.0 / B)
    peaked  = np.zeros((20, B)); peaked[:, 0] = 1.0
    assert _pool_attn_entropy(uniform) > _pool_attn_entropy(peaked)


def test_pool_attn_topk_mass_bounds():
    """top-1 mass in [0,1]; top-all mass == 1.0."""
    pa = np.array([[0.1, 0.5, 0.3, 0.1],
                   [0.25, 0.25, 0.25, 0.25]])
    m1 = _pool_attn_topk_mass(pa, 1)
    m4 = _pool_attn_topk_mass(pa, 4)
    assert 0.0 <= m1 <= 1.0
    assert math.isclose(m4, 1.0, abs_tol=1e-5)


# ── Part D: PCA baseline ───────────────────────────────────────────────────────

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")


def test_pca_baseline_n_components_bounded_by_n_train(synthetic_data, tmp_path):
    """n_components must be <= n_train - 1 (PCA cannot exceed training samples)."""
    emb, tr_ix, va_ix, te_ix, B, d_in = synthetic_data
    n_comp, va_mse, te_mse = run_pca_baseline(
        emb, tr_ix, va_ix, te_ix,
        block_dims=None, latent_mask=None,
        n_components_req=8, out_dir=tmp_path / "pca",
    )
    assert n_comp <= len(tr_ix) - 1, (
        f"n_comp={n_comp} > n_train-1={len(tr_ix)-1}"
    )
    assert np.isfinite(va_mse), "val MSE is not finite"


def test_pca_baseline_creates_required_files(synthetic_data, tmp_path):
    """PCA baseline must create embeddings, CSV, and summary file."""
    emb, tr_ix, va_ix, te_ix, *_ = synthetic_data
    run_pca_baseline(
        emb, tr_ix, va_ix, te_ix,
        block_dims=None, latent_mask=None,
        n_components_req=8, out_dir=tmp_path / "pca2",
    )
    assert (tmp_path / "pca2" / "pca_subject_embeddings.npy").exists()
    assert (tmp_path / "pca2" / "pca_subject_embeddings.csv").exists()
    assert (tmp_path / "pca2" / "pca_baseline_summary.csv").exists()


def test_pca_baseline_with_heterogeneous_block_dims(synthetic_data, tmp_path):
    """PCA baseline must work when block_dims varies per block."""
    emb, tr_ix, va_ix, te_ix, B, d_in = synthetic_data
    # block_dims must be <= d_in for each block
    block_dims = [3, 4, 5, 6]
    n_comp, va_mse, _ = run_pca_baseline(
        emb, tr_ix, va_ix, te_ix,
        block_dims=block_dims, latent_mask=None,
        n_components_req=8, out_dir=tmp_path / "pca3",
    )
    assert np.isfinite(va_mse)
    # flat_dim = sum(block_dims) = 18; n_comp <= min(13, 18, 8) = 8
    assert n_comp <= 8


def test_pca_embedding_shape(synthetic_data, tmp_path):
    """PCA subject embeddings must have n_components columns."""
    emb, tr_ix, va_ix, te_ix, *_ = synthetic_data
    n_comp, _, _ = run_pca_baseline(
        emb, tr_ix, va_ix, te_ix,
        block_dims=None, latent_mask=None,
        n_components_req=6, out_dir=tmp_path / "pca4",
    )
    arr = np.load(tmp_path / "pca4" / "pca_subject_embeddings.npy")
    assert arr.shape == (20, n_comp), f"Expected (20, {n_comp}), got {arr.shape}"


# ── Part E: mean-pool baseline ─────────────────────────────────────────────────

def test_mean_pool_creates_required_files(tiny_model, synthetic_data, tmp_path):
    """Mean-pool baseline must create embeddings and summary files."""
    emb, tr_ix, va_ix, te_ix, *_ = synthetic_data
    raw_va, raw_te, eh_va, eh_te = run_mean_pool_baseline(
        tiny_model, emb, tr_ix, va_ix, te_ix,
        latent_mask=None, latent_mask_t=None,
        out_dir=tmp_path / "mp",
    )
    assert (tmp_path / "mp" / "mean_pool_subject_embeddings.npy").exists()
    assert (tmp_path / "mp" / "mean_pool_subject_embeddings.csv").exists()
    assert (tmp_path / "mp" / "mean_pool_baseline_summary.csv").exists()
    assert np.isfinite(raw_va), "raw val MSE is not finite"
    assert np.isfinite(eh_va),  "embedhead val MSE is not finite"


def test_mean_pool_embedding_shape(tiny_model, synthetic_data, tmp_path):
    """Mean-pool embeddings must have shape (N, d_model)."""
    emb, tr_ix, va_ix, te_ix, *_ = synthetic_data
    run_mean_pool_baseline(
        tiny_model, emb, tr_ix, va_ix, te_ix,
        latent_mask=None, latent_mask_t=None,
        out_dir=tmp_path / "mp_shape",
    )
    arr = np.load(tmp_path / "mp_shape" / "mean_pool_subject_embeddings.npy")
    assert arr.shape == (20, 8), f"Expected (20, 8), got {arr.shape}"  # d_model=8


def test_mean_pool_val_recon_finite(tiny_model, synthetic_data, tmp_path):
    """Both mean-pool val MSEs (raw and embedhead) must be finite."""
    emb, tr_ix, va_ix, te_ix, *_ = synthetic_data
    raw_va, _, eh_va, _ = run_mean_pool_baseline(
        tiny_model, emb, tr_ix, va_ix, te_ix,
        latent_mask=None, latent_mask_t=None,
        out_dir=tmp_path / "mp_fin",
    )
    assert np.isfinite(raw_va), "raw val MSE is not finite"
    assert np.isfinite(eh_va),  "embedhead val MSE is not finite"


def test_mean_pool_summary_columns(tiny_model, synthetic_data, tmp_path):
    """mean_pool_baseline_summary.csv must have both variant column names."""
    emb, tr_ix, va_ix, te_ix, *_ = synthetic_data
    run_mean_pool_baseline(
        tiny_model, emb, tr_ix, va_ix, te_ix,
        latent_mask=None, latent_mask_t=None,
        out_dir=tmp_path / "mp_cols",
    )
    df = pd.read_csv(tmp_path / "mp_cols" / "mean_pool_baseline_summary.csv")
    for col in ("raw_mean_pool_val_recon_loss", "embedhead_mean_pool_val_recon_loss"):
        assert col in df.columns, f"Missing column: {col}"


# ── Part C: Phase 2 block diagnostics (graceful with missing Phase 1) ──────────

def test_block_diagnostics_no_crash_missing_phase1(tmp_path):
    """run_phase2_block_diagnostics must not crash when vae_summary.csv is absent."""
    block_names = [f"block_{i}" for i in range(4)]
    rng = np.random.RandomState(5)
    pool_attn = rng.dirichlet(np.ones(4), size=20)
    blk_mse   = rng.rand(4)
    run_phase2_block_diagnostics(
        block_names, pool_attn, blk_mse,
        ctx_per_block_df=None,
        p1_dir=str(tmp_path / "nonexistent"),
        lt="ORD",
        lt_dir=tmp_path / "p2_diag",
    )
    assert (tmp_path / "p2_diag" / "diagnostics" / "phase2_block_diagnostics.csv").exists()


def test_block_diagnostics_creates_correlation_summary(tmp_path):
    """run_phase2_block_diagnostics must create phase2_attention_diagnostic_summary.csv."""
    block_names = [f"block_{i}" for i in range(5)]
    rng = np.random.RandomState(6)
    pool_attn = rng.dirichlet(np.ones(5), size=30)
    blk_mse   = rng.rand(5)
    run_phase2_block_diagnostics(
        block_names, pool_attn, blk_mse,
        ctx_per_block_df=None,
        p1_dir=str(tmp_path / "nonexistent"),
        lt="ORD",
        lt_dir=tmp_path / "p2_corr",
    )
    assert (
        tmp_path / "p2_corr" / "diagnostics" / "phase2_attention_diagnostic_summary.csv"
    ).exists()


def test_block_diagnostics_correct_n_rows(tmp_path):
    """phase2_block_diagnostics.csv must have one row per block."""
    B = 6
    block_names = [f"block_{i}" for i in range(B)]
    rng = np.random.RandomState(7)
    pool_attn = rng.dirichlet(np.ones(B), size=25)
    blk_mse   = rng.rand(B)
    run_phase2_block_diagnostics(
        block_names, pool_attn, blk_mse,
        ctx_per_block_df=None,
        p1_dir=str(tmp_path / "nonexistent"),
        lt="ORD",
        lt_dir=tmp_path / "p2_rows",
    )
    df = pd.read_csv(tmp_path / "p2_rows" / "diagnostics" / "phase2_block_diagnostics.csv")
    assert len(df) == B, f"Expected {B} rows, got {len(df)}"


# ── Multi-token pooling ────────────────────────────────────────────────────────

@pytest.mark.parametrize("K", [1, 4])
def test_multi_pool_embedding_dim(K):
    """Embedding dim must be K * d_model."""
    torch.manual_seed(K)
    d_model = 8
    m = AttentionAggregator(
        n_blocks=4, d_in=6, d_model=d_model, n_heads=2,
        n_layers=1, d_ff=16, dropout=0.0, n_pool_tokens=K,
    )
    m.eval()
    x = torch.randn(5, 4, 6)
    with torch.no_grad():
        emb, pool_attn, _ = m.encode(x)
    assert emb.shape == (5, K * d_model), (
        f"K={K}: expected emb shape (5, {K * d_model}), got {tuple(emb.shape)}"
    )
    assert m.emb_dim == K * d_model


@pytest.mark.parametrize("K", [1, 4])
def test_multi_pool_attn_shape_and_sums(K):
    """pool_attn returned by encode() must be (batch, B) and sum to ~1 per subject."""
    torch.manual_seed(K + 10)
    m = AttentionAggregator(
        n_blocks=4, d_in=6, d_model=8, n_heads=2,
        n_layers=1, d_ff=16, dropout=0.0, n_pool_tokens=K,
    )
    m.eval()
    x = torch.randn(7, 4, 6)
    with torch.no_grad():
        _, pool_attn, _ = m.encode(x)
    assert pool_attn.shape == (7, 4), (
        f"K={K}: expected pool_attn (7, 4), got {tuple(pool_attn.shape)}"
    )
    sums = pool_attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(7), atol=1e-5), (
        f"K={K}: pool_attn row sums not ~1: {sums.tolist()}"
    )


@pytest.mark.parametrize("K", [1, 4])
def test_multi_pool_forward_recon_shape(K):
    """forward() must return recon of shape (batch, B, d_in) regardless of K."""
    torch.manual_seed(K + 20)
    B, d_in = 4, 6
    m = AttentionAggregator(
        n_blocks=B, d_in=d_in, d_model=8, n_heads=2,
        n_layers=1, d_ff=16, dropout=0.0, n_pool_tokens=K,
    )
    m.eval()
    x = torch.randn(5, B, d_in)
    with torch.no_grad():
        recon, emb, pool_attn = m(x, return_self_attn=False)
    assert recon.shape == (5, B, d_in), (
        f"K={K}: expected recon (5, {B}, {d_in}), got {tuple(recon.shape)}"
    )


def test_multi_pool_by_token_shape():
    """_extract_pool_attn_by_token must return (N, K, B)."""
    K, B, d_in = 4, 4, 6
    torch.manual_seed(99)
    m = AttentionAggregator(
        n_blocks=B, d_in=d_in, d_model=8, n_heads=2,
        n_layers=1, d_ff=16, dropout=0.0, n_pool_tokens=K,
    )
    m.eval()
    rng = np.random.RandomState(9)
    data = rng.randn(20, B, d_in).astype(np.float32)
    out = _extract_pool_attn_by_token(m, data, batch_size=8)
    assert out.shape == (20, K, B), f"Expected (20, {K}, {B}), got {out.shape}"


def test_k1_by_token_equals_mean_pool_attn():
    """For K=1, _extract_pool_attn_by_token squeezed must match encode() pool_attn."""
    torch.manual_seed(77)
    B, d_in = 4, 6
    m = AttentionAggregator(
        n_blocks=B, d_in=d_in, d_model=8, n_heads=2,
        n_layers=1, d_ff=16, dropout=0.0, n_pool_tokens=1,
    )
    m.eval()
    rng = np.random.RandomState(8)
    data = rng.randn(16, B, d_in).astype(np.float32)
    by_token = _extract_pool_attn_by_token(m, data, batch_size=8)  # (N, 1, B)
    with torch.no_grad():
        _, pool_attn, _ = m.encode(torch.tensor(data))              # (N, B)
    pool_attn_np = pool_attn.numpy()
    assert np.allclose(by_token[:, 0, :], pool_attn_np, atol=1e-5), (
        "K=1 by_token[:, 0, :] must match encode() pool_attn"
    )
