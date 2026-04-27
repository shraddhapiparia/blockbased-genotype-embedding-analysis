"""
Tests for mathematical properties of BlockVAE and AttentionAggregator.
All tests use tiny synthetic tensors; no real data or checkpoints required.
"""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from scripts.core.VAE_phase1 import BlockVAE
from scripts.core.attention_phase2 import AttentionAggregator


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_vae():
    model = BlockVAE(p=10, d=4, drop=0.0, loss_type="MSE")
    model.eval()
    return model


@pytest.fixture(scope="module")
def small_aggregator():
    model = AttentionAggregator(
        n_blocks=4, d_in=8, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0
    )
    model.eval()
    return model


# ── attention aggregator tests ────────────────────────────────────────────────

def test_pool_attn_sums_to_one(small_aggregator):
    """Pooling attention weights must sum to 1.0 across blocks for every subject."""
    x = torch.randn(5, 4, 8)
    with torch.no_grad():
        _, pool_attn, _ = small_aggregator.encode(x)
    sums = pool_attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(5), atol=1e-5), (
        f"pool_attn row sums: {sums.tolist()} — expected all ~1.0"
    )


def test_pool_attn_shape(small_aggregator):
    """Pooling attention output shape must be (batch, n_blocks)."""
    x = torch.randn(3, 4, 8)
    with torch.no_grad():
        _, pool_attn, _ = small_aggregator.encode(x)
    assert pool_attn.shape == (3, 4), (
        f"Expected pool_attn shape (3, 4), got {tuple(pool_attn.shape)}"
    )


# ── VAE KL divergence tests ───────────────────────────────────────────────────

def test_kl_divergence_nonneg():
    """
    Element-wise KL divergence from N(mu, exp(lv)) to N(0,1) must be >= 0.
    Formula: kl_i = -0.5 * (1 + lv_i - mu_i² - exp(lv_i)) >= 0
    """
    torch.manual_seed(0)
    mu = torch.randn(50, 4)
    lv = torch.randn(50, 4).clamp(-20, 2)
    kl_per_dim = -0.5 * (1.0 + lv - mu.pow(2) - lv.exp())
    assert (kl_per_dim >= 0).all(), (
        f"KL divergence has negative elements: min={kl_per_dim.min().item():.6f}"
    )


# ── VAE encoder NaN tests ─────────────────────────────────────────────────────

def test_valid_input_no_nan_output(small_vae):
    """A finite, well-formed input must not produce NaN in mu or lv."""
    torch.manual_seed(1)
    x = torch.randn(8, 10)
    with torch.no_grad():
        mu, lv = small_vae.encode(x)
    assert not torch.isnan(mu).any(), "NaN in mu for valid input"
    assert not torch.isnan(lv).any(), "NaN in lv for valid input"


def test_nan_input_propagates_through_encoder(small_vae):
    """
    NaN input to the encoder must produce NaN output — not silently pass
    finite values that would mask upstream data corruption.
    """
    x_nan = torch.full((8, 10), float("nan"))
    with torch.no_grad():
        mu_nan, _ = small_vae.encode(x_nan)
    assert torch.isnan(mu_nan).any(), (
        "Expected NaN output for NaN input; encoder may be silently discarding NaNs"
    )
