"""
Tests for VAE latent-space diagnostics and free-bits support.
All tests use tiny synthetic tensors; no real data or checkpoints required.
"""
import math
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from scripts.core.VAE_phase1 import BlockVAE, compute_latent_diagnostics


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_vae():
    """Tiny MSE VAE with known weights (eval mode, no dropout)."""
    torch.manual_seed(0)
    model = BlockVAE(p=20, d=4, drop=0.0, loss_type="MSE")
    model.eval()
    return model


@pytest.fixture(scope="module")
def va_x(trained_vae):
    torch.manual_seed(1)
    return torch.rand(30, 20) * 2.0  # simulate genotype [0, 2]


# ── compute_latent_diagnostics: key presence and ranges ───────────────────────

def test_diag_required_keys(trained_vae, va_x):
    diag = compute_latent_diagnostics(trained_vae, va_x)
    required = {
        "n_active_latents", "frac_dims_collapsed", "latent_underused",
        "kl_per_dim_min", "kl_per_dim_median", "kl_per_dim_max",
        "mu_var_min", "mu_var_median", "mu_var_max",
        "sigma_min", "sigma_median", "sigma_max",
        "_kl_per_dim", "_mu_var_per_dim", "_sigma_per_dim",
    }
    missing = required - diag.keys()
    assert not missing, f"compute_latent_diagnostics missing keys: {missing}"


def test_n_active_latents_in_range(trained_vae, va_x):
    diag = compute_latent_diagnostics(trained_vae, va_x)
    d = 4
    assert 0 <= diag["n_active_latents"] <= d, (
        f"n_active_latents={diag['n_active_latents']} out of [0, {d}]"
    )


def test_frac_dims_collapsed_in_range(trained_vae, va_x):
    diag = compute_latent_diagnostics(trained_vae, va_x)
    fdc = diag["frac_dims_collapsed"]
    assert 0.0 <= fdc <= 1.0, f"frac_dims_collapsed={fdc} out of [0,1]"


def test_frac_dims_collapsed_consistent_with_n_active(trained_vae, va_x):
    diag = compute_latent_diagnostics(trained_vae, va_x)
    d = 4
    expected_fdc = round(1.0 - diag["n_active_latents"] / d, 4)
    assert math.isclose(diag["frac_dims_collapsed"], expected_fdc, abs_tol=1e-4), (
        f"frac_dims_collapsed={diag['frac_dims_collapsed']} inconsistent with "
        f"n_active={diag['n_active_latents']} / d={d}"
    )


def test_latent_underused_is_bool(trained_vae, va_x):
    diag = compute_latent_diagnostics(trained_vae, va_x)
    assert isinstance(diag["latent_underused"], bool)


def test_diag_arrays_have_correct_length(trained_vae, va_x):
    diag = compute_latent_diagnostics(trained_vae, va_x)
    d = 4
    assert diag["_kl_per_dim"].shape == (d,),    f"_kl_per_dim shape {diag['_kl_per_dim'].shape}"
    assert diag["_mu_var_per_dim"].shape == (d,), f"_mu_var_per_dim shape {diag['_mu_var_per_dim'].shape}"
    assert diag["_sigma_per_dim"].shape == (d,),  f"_sigma_per_dim shape {diag['_sigma_per_dim'].shape}"


def test_kl_per_dim_nonneg(trained_vae, va_x):
    """Per-dimension KL from N(mu,sigma) to N(0,1) must be >= 0 for a proper posterior."""
    diag = compute_latent_diagnostics(trained_vae, va_x)
    kl = diag["_kl_per_dim"]
    assert np.all(kl >= -1e-6), f"Negative per-dim KL: {kl}"


def test_sigma_per_dim_positive(trained_vae, va_x):
    diag = compute_latent_diagnostics(trained_vae, va_x)
    assert diag["sigma_min"] > 0, f"sigma_min={diag['sigma_min']} must be > 0"


# ── free-bits: zero disables, positive clamps ─────────────────────────────────

def test_free_bits_zero_matches_standard_kl():
    """free_bits=0.0 must produce the same KL as the original formula."""
    torch.manual_seed(42)
    model = BlockVAE(p=10, d=4, drop=0.0, loss_type="MSE")
    model.eval()
    x = torch.rand(8, 10)
    recon, mu, lv = model(x)

    _, _, kl_std = model.compute_loss(x, recon, mu, lv, beta=0.5, free_bits=0.0)
    # Reference: pure per-element mean over (B, d) — matches compute_loss exactly
    kl_ref = (-0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())).item()
    assert math.isclose(kl_std.item(), kl_ref, rel_tol=1e-5), (
        f"free_bits=0.0 KL={kl_std.item():.6f} != reference {kl_ref:.6f}"
    )


def test_free_bits_positive_no_crash():
    torch.manual_seed(7)
    model = BlockVAE(p=10, d=4, drop=0.0, loss_type="MSE")
    x = torch.rand(8, 10)
    recon, mu, lv = model(x)
    loss, rl, kl = model.compute_loss(x, recon, mu, lv, beta=0.5, free_bits=0.5)
    assert torch.isfinite(loss), f"loss is not finite with free_bits=0.5"
    assert kl.item() >= 0.0, f"KL with free_bits should be >= 0"


def test_free_bits_positive_raises_kl():
    """With free_bits > 0, KL should be >= free_bits * d (all dims clamped at least)."""
    torch.manual_seed(3)
    model = BlockVAE(p=10, d=4, drop=0.0, loss_type="MSE")
    # Force near-zero posterior: small weights → mu≈0, lv≈0 → KL per dim ≈ 0
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    x = torch.rand(8, 10)
    recon, mu, lv = model(x)
    free_bits = 0.3
    _, _, kl_fb = model.compute_loss(x, recon, mu, lv, beta=1.0, free_bits=free_bits)
    assert kl_fb.item() >= free_bits * 4 - 1e-5, (
        f"KL={kl_fb.item():.4f} should be >= free_bits*d={free_bits*4:.4f}"
    )


# ── beta_eff_at_best is derivable from config (sanity check) ──────────────────

def test_beta_eff_at_best_formula():
    beta_max = 0.5
    beta_warmup = 50
    best_epoch = 25
    expected = beta_max * min(1.0, best_epoch / beta_warmup)
    assert math.isclose(expected, 0.25, rel_tol=1e-9)

    best_epoch_after = 100
    expected_full = beta_max * min(1.0, best_epoch_after / beta_warmup)
    assert math.isclose(expected_full, beta_max, rel_tol=1e-9)
