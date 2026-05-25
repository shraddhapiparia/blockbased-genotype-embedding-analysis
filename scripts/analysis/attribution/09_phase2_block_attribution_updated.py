#!/usr/bin/env python3
"""
09_phase2_block_attribution.py

Post-hoc Phase 2 block-level attribution via leave-one-block-out (LOBO) masking.

Requires torch (install via the 'vae' conda environment or any env with PyTorch).
Run as:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/analysis/09_phase2_block_attribution.py

What this does
--------------
1. Loads the saved Phase 2 AttentionAggregator checkpoint.
2. Passes the Phase 1 block embeddings (N=997, B=174, d_in=16) through the frozen
   model to confirm baseline subject embeddings.
3. Fits PCA on the *saved* individual_embeddings.npy to define PC1/PC2 axes exactly
   as produced by the original run.
4. For each of the 174 blocks, masks that block by replacing its values with the
   cross-subject population mean for that block (mean masking, not zero masking,
   to minimise distributional shift to the model).
5. Runs a fresh forward pass and projects new embeddings onto the original PC1/PC2
   axes. Computes mean absolute delta and mean signed delta per block.
6. Optionally attributes log10Ige: merges phenotype, fits a RidgeCV on the baseline
   embedding, and computes delta in ridge-predicted score under each block mask.

Attribution targets
-------------------
- Phase 2 embedding PC1
- Phase 2 embedding PC2
- log10Ige ridge-prediction score (if subject ID alignment is confirmed)

Outputs  (results/analysis/phase2_block_attribution/)
--------
phase2_PC1_leave_one_block_out.csv
phase2_PC2_leave_one_block_out.csv
phase2_log10Ige_leave_one_block_out.csv  (only if phenotype attribution is valid)
top20_PC1_attribution.png
top20_PC2_attribution.png
top20_log10Ige_attribution.png           (only if valid)
README_phase2_block_attribution.md

Limitations
-----------
- Masking affects downstream transformer contextualization in the original model.
  A "perfect" attribution would re-run the model from Phase 1 genotype inputs
  with block b completely ablated. This script goes one level up: it masks the
  Phase 1 latent embedding for block b before feeding into the Phase 2 transformer.
  Residual information about block b may persist in positional embeddings and in
  the transformer's learned cross-block interactions.
- Attribution reflects embedding geometry change, not causal importance.
- Do not interpret high attribution as evidence that a block causes the phenotype.
"""

import os
import sys
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import yaml
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit(
        "ERROR: torch not found.\n"
        "Run in the vae conda environment:\n"
        "  KMP_DUPLICATE_LIB_OK=TRUE /path/to/vae/bin/python "
        "scripts/analysis/09_phase2_block_attribution.py"
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# ── add repo root to sys.path so we can import AttentionAggregator ────────────
def find_repo_root(start: Path) -> Path:
    """Walk upward until repo root is found."""
    for p in [start] + list(start.parents):
        if (p / "scripts" / "core" / "attention_phase2.py").exists():
            return p
    raise RuntimeError(
        "Could not find repo root containing scripts/core/attention_phase2.py"
    )


_REPO_ROOT = find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(_REPO_ROOT))

from scripts.core.attention_phase2 import AttentionAggregator

# ── default paths (relative to repo root) ─────────────────────────────────────
P1_EMBEDDINGS    = "results/output_regions/ORD_W_Scaled/embeddings/all_blocks.npy"
P1_LATENT_DIMS   = "results/output_regions/ORD_W_Scaled/embeddings/all_blocks_latent_dims.npy"
P2_CONFIG        = "configs/config_phase2.yaml"
P2_CHECKPOINT    = "results/output_regions2/ORD_W_Scaled/models/attention_aggregator.pt"
P2_EMBEDDINGS    = "results/output_regions2/ORD_W_Scaled/embeddings/individual_embeddings.npy"
P2_EMB_CSV       = "results/output_regions2/ORD_W_Scaled/embeddings/individual_embeddings.csv"
BLOCK_ORDER      = "results/output_regions/block_order.csv"
SUBJECTS_CSV     = "results/output_regions/subjects.csv"
PHENO_FILE       = "metadata/COS_TRIO_pheno_1165.csv"
DEFAULT_OUT      = "results/analysis/phase2_block_attribution"

BATCH_SIZE       = 256
PHENO_TARGET     = "log10Ige"

# ── no-HLA run paths ──────────────────────────────────────────────────────────
P2_CHECKPOINT_NOHLA = "results/output_regions2_noHLA/ORD_W_Scaled/models/attention_aggregator.pt"
P2_EMBEDDINGS_NOHLA = "results/output_regions2_noHLA/ORD_W_Scaled/embeddings/individual_embeddings.npy"
P2_EMB_CSV_NOHLA    = "results/output_regions2_noHLA/ORD_W_Scaled/embeddings/individual_embeddings.csv"
HLA_PATTERN         = "HLA_classII"

# Full-run CSVs needed when building the comparison table
FULL_PC1_CSV = "results/analysis/phase2_block_attribution/phase2_PC1_leave_one_block_out.csv"
FULL_PC2_CSV = "results/analysis/phase2_block_attribution/phase2_PC2_leave_one_block_out.csv"


# ── config and block-dim helpers ──────────────────────────────────────────────
def load_phase2_config(config_path: str) -> dict:
    """Return the [attention] section of config_phase2.yaml, or {} if not found."""
    p = Path(config_path)
    if not p.exists():
        warnings.warn(f"Phase 2 config not found: {p} — using hardcoded defaults")
        return {}
    with open(p) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("attention", {})


def load_block_dims(latent_dims_path: str, block_meta: pd.DataFrame):
    """Return per-block latent dim list, or None for legacy shared projection.

    Tries all_blocks_latent_dims.npy first (authoritative actual dims after
    clamping), then falls back to block_order.csv['latent_dim'].
    """
    p = Path(latent_dims_path)
    if p.exists():
        dims = np.load(p).astype(int).tolist()
        counts = dict(pd.Series(dims).value_counts().sort_index())
        print(f"[model]  using BlockProjector with per-block dims: {counts}")
        return dims
    if "latent_dim" in block_meta.columns:
        dims = block_meta["latent_dim"].astype(int).tolist()
        counts = dict(pd.Series(dims).value_counts().sort_index())
        print(f"[model]  using BlockProjector (from block_order.csv) with per-block dims: {counts}")
        return dims
    print("[model]  using legacy shared input projection")
    return None


# ── model loader ──────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, n_blocks: int, d_in: int,
               d_model: int = 64, n_heads: int = 4,
               n_layers: int = 2, d_ff: int = 128,
               dropout: float = 0.0,
               block_dims: list = None,
               n_pool_tokens: int = 1) -> AttentionAggregator:
    """Load AttentionAggregator from checkpoint.

    Pass block_dims to instantiate with BlockProjector (heterogeneous per-block
    latent dims, checkpoint keys input_proj.projectors.*).  Omit for legacy
    shared projection (checkpoint keys input_proj.0.weight).
    dropout=0.0 at inference — no randomness during attribution.
    """
    model = AttentionAggregator(
        n_blocks=n_blocks, d_in=d_in, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        dropout=dropout,
        block_dims=block_dims,
        n_pool_tokens=n_pool_tokens,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── batched inference ─────────────────────────────────────────────────────────
@torch.no_grad()
def encode_batched(model: AttentionAggregator, x_np: np.ndarray,
                   batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Run model.encode on x_np (N, B, d_in) in mini-batches.
    Returns subject embeddings (N, emb_dim), where emb_dim = n_pool_tokens * d_model.
    """
    N = x_np.shape[0]
    results = []
    for start in range(0, N, batch_size):
        batch = torch.tensor(x_np[start:start + batch_size], dtype=torch.float32)
        emb, _, _ = model.encode(batch, return_self_attn=False)
        results.append(emb.numpy())
    return np.concatenate(results, axis=0)


# ── PC projection ─────────────────────────────────────────────────────────────
def fit_pca(embeddings_np: np.ndarray, n_components: int = 2):
    """
    Fit StandardScaler + PCA on embeddings. Returns (scaler, pca, scores).
    """
    scaler = StandardScaler()
    z = scaler.fit_transform(embeddings_np)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(z)
    return scaler, pca, scores


def project_embeddings(embeddings_np: np.ndarray,
                       scaler: StandardScaler, pca: PCA) -> np.ndarray:
    """Project embeddings onto pre-fitted PCA axes. Returns (N, n_components)."""
    z = scaler.transform(embeddings_np)
    return pca.transform(z)


# ── phenotype loader ──────────────────────────────────────────────────────────
def load_phenotype(pheno_path: str, subject_iids: np.ndarray,
                   target: str = PHENO_TARGET):
    """
    Merge phenotype file on subject IIDs.

    Returns
    -------
    pheno_mask : bool array (N,) — subjects with non-NA phenotype
    y          : float array (N_pheno,) — phenotype values
    """
    pheno = pd.read_csv(pheno_path, low_memory=False)
    if "S_SUBJECTID" not in pheno.columns:
        raise ValueError("Expected 'S_SUBJECTID' in phenotype file.")
    pheno["IID"] = pheno["S_SUBJECTID"].astype(str).str.strip()

    if target not in pheno.columns:
        raise ValueError(f"Target column '{target}' not in phenotype file.")

    pheno_sub = pheno[["IID", target]].dropna(subset=[target])
    iid_to_pheno = dict(zip(pheno_sub["IID"], pheno_sub[target]))

    pheno_vals = np.array([iid_to_pheno.get(iid, np.nan) for iid in subject_iids])
    pheno_mask = np.isfinite(pheno_vals)

    n_with = pheno_mask.sum()
    print(f"[phenotype]  {target}: {n_with}/{len(subject_iids)} subjects have data")
    if n_with < 50:
        warnings.warn(f"Only {n_with} subjects with {target} — phenotype attribution unreliable.")

    return pheno_mask, pheno_vals[pheno_mask]


# ── ridge regression for phenotype ────────────────────────────────────────────
def fit_ridge(base_emb: np.ndarray, y: np.ndarray) -> RidgeCV:
    """
    Fit RidgeCV on baseline embeddings for phenotype attribution.
    Uses 5-fold CV over log-spaced alphas.
    """
    alphas = np.logspace(-3, 4, 50)
    scaler = StandardScaler()
    X = scaler.fit_transform(base_emb)
    ridge = RidgeCV(alphas=alphas, scoring="r2", cv=5)
    ridge.fit(X, y)
    # attach scaler so we can use it later
    ridge._feature_scaler = scaler
    r2_cv = ridge.best_score_
    print(f"[ridge]      {PHENO_TARGET} baseline: best α={ridge.alpha_:.4g}, CV R²={r2_cv:.3f}")
    return ridge


def ridge_predict(ridge: RidgeCV, emb: np.ndarray) -> np.ndarray:
    X = ridge._feature_scaler.transform(emb)
    return ridge.predict(X)


# ── leave-one-block-out loop ──────────────────────────────────────────────────
def run_lobo(
    model: AttentionAggregator,
    x_np: np.ndarray,
    base_emb: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    base_scores: np.ndarray,
    block_names: list,
    batch_size: int = BATCH_SIZE,
    include_block_indices=None,
    ridge_PC1=None,
    ridge_PC2=None,
    pheno_mask=None,
    pheno_y=None,
    ridge_ige=None,
):
    """
    Leave-one-block-out masking attribution.

    For each block b in include_block_indices (default: all blocks):
      - Replace x[:, b, :] with the cross-subject population mean for block b.
      - Run encode → masked subject embeddings.
      - Project onto PCA axes, compute per-subject delta vs baseline.

    Parameters
    ----------
    include_block_indices : list of int or None
        Block indices to include in LOBO. None = all blocks.
        Use to skip HLA blocks in a no-HLA run.

    Returns
    -------
    results_PC1  : list of dicts
    results_PC2  : list of dicts
    results_PC3  : list of dicts
    results_PC4  : list of dicts
    results_PC5  : list of dicts
    results_ige  : list of dicts or None
    """
    B = x_np.shape[1]
    n_pc = base_scores.shape[1]   # number of PCA components available
    assert len(block_names) == B, f"block_names length {len(block_names)} != B={B}"
    indices_to_run = list(range(B)) if include_block_indices is None else list(include_block_indices)

    # pre-compute per-block population means for mean-masking
    block_means = x_np.mean(axis=0)   # (B, d_in)

    do_ige = (ridge_ige is not None and pheno_mask is not None and pheno_y is not None)
    base_ige_pred = ridge_predict(ridge_ige, base_emb[pheno_mask]) if do_ige else None

    results_PC1, results_PC2, results_PC3, results_PC4, results_PC5, results_ige = \
        [], [], [], [], [], []

    print(f"\n[lobo]  running {len(indices_to_run)}/{B} blocks "
          f"(mean masking, batch_size={batch_size}, n_pc={n_pc}) ...")
    for loop_idx, b in enumerate(indices_to_run):
        x_masked = x_np.copy()
        x_masked[:, b, :] = block_means[b]           # replace block b with population mean

        masked_emb = encode_batched(model, x_masked, batch_size)   # (N, emb_dim)
        masked_scores = project_embeddings(masked_emb, scaler, pca)  # (N, n_pc)

        for pc_idx, (results_list, label) in enumerate([
            (results_PC1, "emb_PC1"),
            (results_PC2, "emb_PC2"),
            (results_PC3, "emb_PC3"),
            (results_PC4, "emb_PC4"),
            (results_PC5, "emb_PC5"),
        ]):
            if pc_idx >= n_pc:
                continue
            delta = masked_scores[:, pc_idx] - base_scores[:, pc_idx]
            results_list.append(dict(
                block_index=b,
                block_id=block_names[b],
                target=label,
                mean_abs_delta_score=float(np.abs(delta).mean()),
                mean_signed_delta_score=float(delta.mean()),
                n_subjects=len(delta),
            ))

        if do_ige:
            masked_ige_pred = ridge_predict(ridge_ige, masked_emb[pheno_mask])
            delta_ige = masked_ige_pred - base_ige_pred
            results_ige.append(dict(
                block_index=b,
                block_id=block_names[b],
                target="log10Ige_ridge_pred",
                mean_abs_delta_score=float(np.abs(delta_ige).mean()),
                mean_signed_delta_score=float(delta_ige.mean()),
                n_subjects=int(pheno_mask.sum()),
            ))

        if (loop_idx + 1) % 25 == 0 or loop_idx == len(indices_to_run) - 1:
            n_run = len(indices_to_run)
            print(f"  block {loop_idx+1:3d}/{n_run} done")

    return results_PC1, results_PC2, results_PC3, results_PC4, results_PC5, \
        (results_ige if do_ige else None)


# ── output helpers ────────────────────────────────────────────────────────────
def finalise_df(records: list, block_meta: pd.DataFrame, method: str) -> pd.DataFrame:
    """Add rank, method, and block metadata columns."""
    df = pd.DataFrame(records)
    df = df.sort_values("mean_abs_delta_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["method"] = method
    df["notes"] = (
        "LOBO masking: block replaced by cross-subject population mean of Phase 1 "
        "latent embedding; full Phase 2 forward pass re-run; checkpoint-based."
    )
    # merge block metadata (pos, gene, n_snps) if available
    meta_cols = [
        c for c in [
            "pos", "gene", "region_id", "region", "group",
            "chr", "n_snps", "latent_dim"
        ]
        if c in block_meta.columns
    ]

    if meta_cols:
        df = df.merge(block_meta[["block_id"] + meta_cols], on="block_id", how="left")

    # If latent_dim was not in block_order.csv, infer it from padded Phase 1 dim later if needed.
    df = add_normalized_attribution_columns(df)

    # Add normalized ranks
    for col in [
        "mean_abs_delta_score",
        "attr_per_snp",
        "attr_per_sqrt_snp",
        "attr_per_latent_dim",
        "attr_per_sqrt_latent_dim",
        "attr_per_snp_x_latent",
    ]:
        if col in df.columns:
            rank_col = col + "_rank"
            df[rank_col] = df[col].rank(ascending=False, method="min")

    return df

def add_normalized_attribution_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add size/capacity-normalized attribution metrics.

    Raw LOBO favors large/high-capacity blocks. These columns help separate:
      - raw model dependence
      - per-SNP impact
      - per-latent-dimension impact
    """
    df = df.copy()

    if "n_snps" in df.columns:
        df["attr_per_snp"] = df["mean_abs_delta_score"] / df["n_snps"].replace(0, np.nan)
        df["attr_per_sqrt_snp"] = df["mean_abs_delta_score"] / np.sqrt(df["n_snps"].replace(0, np.nan))

    if "latent_dim" in df.columns:
        df["attr_per_latent_dim"] = df["mean_abs_delta_score"] / df["latent_dim"].replace(0, np.nan)
        df["attr_per_sqrt_latent_dim"] = df["mean_abs_delta_score"] / np.sqrt(
            df["latent_dim"].replace(0, np.nan)
        )

    if "n_snps" in df.columns and "latent_dim" in df.columns:
        df["attr_per_snp_x_latent"] = df["mean_abs_delta_score"] / (
            df["n_snps"].replace(0, np.nan) * df["latent_dim"].replace(0, np.nan)
        )

    return df

def infer_region_id(block_id: str) -> str:
    """
    Convert subblock IDs into region IDs.
    Examples:
      region_6p21_HLA_classII_sb4 -> region_6p21_HLA_classII
      region_5q21_PDE4D_sb14      -> region_5q21_PDE4D
      control_CFTR_sb3            -> control_CFTR
    """
    s = str(block_id)
    if "_sb" in s:
        return s.rsplit("_sb", 1)[0]
    return s


def write_region_summary(df: pd.DataFrame, out_path: Path, target_label: str):
    """
    Summarize attribution at region level.

    sum_raw favors large regions.
    mean/median/top3 are better for comparing regions with different subblock counts.
    """
    d = df.copy()

    if "region_id" not in d.columns:
        d["region_id"] = d["block_id"].apply(infer_region_id)

    agg_dict = {
        "n_subblocks": ("block_id", "count"),
        "sum_raw_attr": ("mean_abs_delta_score", "sum"),
        "mean_raw_attr": ("mean_abs_delta_score", "mean"),
        "median_raw_attr": ("mean_abs_delta_score", "median"),
        "max_raw_attr": ("mean_abs_delta_score", "max"),
    }

    for col in [
        "attr_per_snp",
        "attr_per_sqrt_snp",
        "attr_per_latent_dim",
        "attr_per_sqrt_latent_dim",
    ]:
        if col in d.columns:
            agg_dict[f"mean_{col}"] = (col, "mean")
            agg_dict[f"max_{col}"] = (col, "max")

    region = (
        d.groupby("region_id")
        .agg(**agg_dict)
        .reset_index()
    )

    # Top-3 average raw attribution per region
    top3 = (
        d.sort_values(["region_id", "mean_abs_delta_score"], ascending=[True, False])
        .groupby("region_id")
        .head(3)
        .groupby("region_id")["mean_abs_delta_score"]
        .mean()
        .reset_index(name="top3_mean_raw_attr")
    )

    region = region.merge(top3, on="region_id", how="left")

    # ranks
    for col in [
        "sum_raw_attr",
        "mean_raw_attr",
        "median_raw_attr",
        "max_raw_attr",
        "top3_mean_raw_attr",
    ]:
        region[col + "_rank"] = region[col].rank(ascending=False, method="min")

    region = region.sort_values("mean_raw_attr", ascending=False)
    region["target"] = target_label
    region.to_csv(out_path, index=False)
    print(f"[output] {out_path.name}  ({len(region)} regions)")
    return region

def plot_top_blocks(df: pd.DataFrame, target_label: str,
                    out_path: Path, top_n: int = 20,
                    score_col: str = "mean_abs_delta_score",
                    focus_patterns=("HLA", "17q21", "IL1RL1", "FCER1A", "PDE4D")):
    if score_col not in df.columns:
        print(f"[plot]  WARNING: {score_col} not found; using mean_abs_delta_score")
        score_col = "mean_abs_delta_score"

    top = df.sort_values(score_col, ascending=False).head(top_n).copy()
    colors = []
    for bid in top["block_id"]:
        bid_s = str(bid)
        if any(p in bid_s for p in focus_patterns):
            colors.append("tomato")
        else:
            colors.append("steelblue")

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    y = np.arange(len(top))
    ax.barh(y, top[score_col].values, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top["block_id"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(score_col)
    ax.set_title(f"Top {top_n} blocks — {target_label}\n"
                 "(red = asthma-associated or control locus)")

    # signed delta as annotation
    for i, (_, row) in enumerate(top.iterrows()):
        signed = row["mean_signed_delta_score"]
        ax.text(row[score_col] + 1e-4, i,
                f"  signed={signed:+.4f}", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot]  saved {out_path}")


def write_readme(out_dir: Path, method_note: str):
    text = f"""# Phase 2 Block Attribution — README

## What is this?

This directory contains post-hoc **Phase 2 block-level attribution** results
for the block-based genotype embedding model.

## What was done?

Leave-one-block-out (LOBO) masking was applied to identify which of the 174
genomic blocks changes the Phase 2 subject embedding axes (PC1, PC2) most
when that block's Phase 1 latent embedding is replaced by its cross-subject
population mean.

## Method: Leave-One-Block-Out

For each block b (b = 0 … 173):
1. The Phase 1 block embedding for block b across all 997 subjects is replaced
   by the population mean of that block (mean masking).
2. The full Phase 2 AttentionAggregator forward pass is re-run on the masked
   input (checkpoint-based — no retraining).
3. The new subject embeddings are projected onto the PC1/PC2 axes fitted on
   the original (unmasked) embeddings.
4. Attribution = mean absolute change in PC score across subjects.
5. Mean signed change is also reported (direction of effect).

{method_note}

## Attribution Targets

| File | Target |
|------|--------|
| phase2_PC1_leave_one_block_out.csv | Phase 2 embedding PC1 |
| phase2_PC2_leave_one_block_out.csv | Phase 2 embedding PC2 |
| phase2_PC3_leave_one_block_out.csv | Phase 2 embedding PC3 |
| phase2_PC4_leave_one_block_out.csv | Phase 2 embedding PC4 |
| phase2_PC5_leave_one_block_out.csv | Phase 2 embedding PC5 |
| phase2_log10Ige_leave_one_block_out.csv | Ridge-predicted log10IgE (if computed) |

## What this is NOT

- This is **not SNP-level attribution** — it operates at the block/region level.
- This is **not causal evidence** — masking a block reduces its information, but
  a block may rank highly due to its correlation with other blocks or global
  embedding geometry, not biological importance.
- This is **not retraining** — the Phase 2 model weights are frozen throughout.
- The phenotype target (log10IgE) uses a **ridge regression prediction score**,
  not a direct phenotype-embedding association.

## Why Leave-One-Block-Out?

LOBO is simple, interpretable, and directly aligned with the block-based model
structure. It requires no gradient computation (post-hoc, frozen model) and
produces a single summary statistic per block.

## Residual Limitations

- Phase 1 embedding for a masked block is replaced with the population mean,
  not zero. This avoids extreme distribution shift, but the positional embedding
  added by the Transformer still encodes block position, so complete ablation
  is not achieved.
- Blocks with correlated Phase 1 embeddings will show similar attribution
  patterns.

## Environment

Requires PyTorch (`vae` conda environment).
Run: `KMP_DUPLICATE_LIB_OK=TRUE python scripts/analysis/09_phase2_block_attribution.py`
"""
    (out_dir / "README_phase2_block_attribution.md").write_text(text)
    print(f"[readme]  saved README_phase2_block_attribution.md")


# ── comparison table ──────────────────────────────────────────────────────────
def build_comparison_table(
    full_pc1_csv: str, full_pc2_csv: str,
    df_noHLA_PC1: pd.DataFrame, df_noHLA_PC2: pd.DataFrame,
    all_block_names: list, out_path: Path,
) -> pd.DataFrame:
    """
    Join full-run and no-HLA LOBO results into a per-block rank comparison table.

    Columns
    -------
    block_id, full_PC1_rank, full_PC2_rank, noHLA_PC1_rank, noHLA_PC2_rank,
    rank_change_PC1 (noHLA - full, negative = rose in rank after HLA removal),
    rank_change_PC2, is_PDE4D, is_HLA, is_17q21, is_IL1RL1, is_FCER1A
    """
    df_full_PC1 = pd.read_csv(full_pc1_csv)[["block_id", "rank"]].rename(
        columns={"rank": "full_PC1_rank"})
    df_full_PC2 = pd.read_csv(full_pc2_csv)[["block_id", "rank"]].rename(
        columns={"rank": "full_PC2_rank"})
    npc1 = df_noHLA_PC1[["block_id", "rank"]].rename(columns={"rank": "noHLA_PC1_rank"})
    npc2 = df_noHLA_PC2[["block_id", "rank"]].rename(columns={"rank": "noHLA_PC2_rank"})

    comp = (pd.DataFrame({"block_id": all_block_names})
            .merge(df_full_PC1, on="block_id", how="left")
            .merge(df_full_PC2, on="block_id", how="left")
            .merge(npc1, on="block_id", how="left")
            .merge(npc2, on="block_id", how="left"))

    comp["rank_change_PC1"] = comp["noHLA_PC1_rank"] - comp["full_PC1_rank"]
    comp["rank_change_PC2"] = comp["noHLA_PC2_rank"] - comp["full_PC2_rank"]

    for pat, col in [("PDE4D", "is_PDE4D"), ("HLA", "is_HLA"),
                     ("17q21", "is_17q21"), ("IL1RL1", "is_IL1RL1"), ("FCER1A", "is_FCER1A")]:
        comp[col] = comp["block_id"].str.contains(pat, na=False)

    comp.to_csv(out_path, index=False)
    print(f"[output] {out_path.name}  ({len(comp)} blocks)")
    return comp


def print_comparison_report(
    comp: pd.DataFrame,
    df_noHLA_PC1: pd.DataFrame, df_noHLA_PC2: pd.DataFrame,
    n_hla: int, n_non_hla: int,
):
    """Print the required post-run comparison report."""
    focus = ["HLA", "17q21", "IL1RL1", "FCER1A", "PDE4D"]

    def flag(bid):
        return "YES" if any(p in str(bid) for p in focus) else "-"

    print("\n══ no-HLA LOBO summary ════════════════════════════════════")
    print(f"  HLA blocks excluded from LOBO   : {n_hla}")
    print(f"  Non-HLA blocks in LOBO          : {n_non_hla}")

    # Verify HLA truly absent from no-HLA results
    hla_in_results = df_noHLA_PC1["block_id"].str.contains(HLA_PATTERN, na=False).sum()
    if hla_in_results == 0:
        print("  HLA blocks in no-HLA results    : CONFIRMED ABSENT")
    else:
        print(f"  WARNING: {hla_in_results} HLA block(s) found in no-HLA results — check logic")

    print("\n══ Top 10 no-HLA PC1 blocks ═══════════════════════════════")
    top10_nPC1 = df_noHLA_PC1.head(10)[
        ["rank", "block_id", "mean_abs_delta_score", "mean_signed_delta_score"]].copy()
    top10_nPC1["focus"] = top10_nPC1["block_id"].apply(flag)
    print(top10_nPC1.to_string(index=False))

    print("\n══ Top 10 no-HLA PC2 blocks ═══════════════════════════════")
    top10_nPC2 = df_noHLA_PC2.head(10)[
        ["rank", "block_id", "mean_abs_delta_score", "mean_signed_delta_score"]].copy()
    top10_nPC2["focus"] = top10_nPC2["block_id"].apply(flag)
    print(top10_nPC2.to_string(index=False))

    print("\n══ PDE4D rank comparison ═══════════════════════════════════")
    pde4d_rows = comp[comp["is_PDE4D"]]
    if pde4d_rows.empty:
        print("  PDE4D: no matching blocks found in comparison table")
    else:
        for _, row in pde4d_rows.iterrows():
            full_r1  = int(row["full_PC1_rank"])  if pd.notna(row["full_PC1_rank"])  else None
            full_r2  = int(row["full_PC2_rank"])  if pd.notna(row["full_PC2_rank"])  else None
            noHLA_r1 = int(row["noHLA_PC1_rank"]) if pd.notna(row["noHLA_PC1_rank"]) else None
            noHLA_r2 = int(row["noHLA_PC2_rank"]) if pd.notna(row["noHLA_PC2_rank"]) else None
            chg1 = row["rank_change_PC1"]
            chg2 = row["rank_change_PC2"]
            print(f"  {row['block_id']}")
            print(f"    PC1: full={full_r1}  noHLA={noHLA_r1}  "
                  f"change={int(chg1) if pd.notna(chg1) else 'n/a'}")
            print(f"    PC2: full={full_r2}  noHLA={noHLA_r2}  "
                  f"change={int(chg2) if pd.notna(chg2) else 'n/a'}")

    # Interpretation
    pde4d_pc1_full  = pde4d_rows["full_PC1_rank"].min()  if not pde4d_rows.empty else None
    pde4d_pc1_noHLA = pde4d_rows["noHLA_PC1_rank"].min() if not pde4d_rows.empty else None
    pde4d_pc2_full  = pde4d_rows["full_PC2_rank"].min()  if not pde4d_rows.empty else None
    pde4d_pc2_noHLA = pde4d_rows["noHLA_PC2_rank"].min() if not pde4d_rows.empty else None

    print("\n══ Interpretation ══════════════════════════════════════════")
    if pde4d_pc1_noHLA is not None and pde4d_pc1_full is not None:
        if pde4d_pc1_noHLA < pde4d_pc1_full:
            print(f"  PDE4D rose in PC1 after HLA removal "
                  f"(full rank {int(pde4d_pc1_full)} → no-HLA rank {int(pde4d_pc1_noHLA)}). "
                  "This is consistent with PDE4D being a secondary organizing axis "
                  "masked by HLA dominance in the full embedding space. "
                  "Do not interpret as evidence of causal importance.")
        else:
            print(f"  PDE4D did not rise in PC1 after HLA removal "
                  f"(full rank {int(pde4d_pc1_full)}, no-HLA rank {int(pde4d_pc1_noHLA)}). "
                  "PDE4D does not clearly emerge as a dominant axis when HLA is removed.")
    if pde4d_pc2_noHLA is not None and pde4d_pc2_full is not None:
        if pde4d_pc2_noHLA < pde4d_pc2_full:
            print(f"  PDE4D rose in PC2 after HLA removal "
                  f"(full rank {int(pde4d_pc2_full)} → no-HLA rank {int(pde4d_pc2_noHLA)}).")
        else:
            print(f"  PDE4D did not rise in PC2 after HLA removal "
                  f"(full rank {int(pde4d_pc2_full)}, no-HLA rank {int(pde4d_pc2_noHLA)}).")

def print_region_diagnostic(region_df: pd.DataFrame, label: str):
    focus_regions = ["HLA", "PDE4D", "17q21", "TNFSF", "SH2B3", "IL1RL1", "IL33", "FCER1A"]

    sub = region_df[
        region_df["region_id"].apply(lambda x: any(p in str(x) for p in focus_regions))
    ].copy()

    if sub.empty:
        return

    cols = [
        "region_id",
        "n_subblocks",
        "sum_raw_attr",
        "mean_raw_attr",
        "max_raw_attr",
        "top3_mean_raw_attr",
        "sum_raw_attr_rank",
        "mean_raw_attr_rank",
        "top3_mean_raw_attr_rank",
    ]
    cols = [c for c in cols if c in sub.columns]

    print(f"\n══ Region-level attribution diagnostic — {label} ═════════════")
    print(sub[cols].sort_values("mean_raw_attr_rank").to_string(index=False))

# ── validation helpers ────────────────────────────────────────────────────────
def check_baseline_agreement(base_emb_model: np.ndarray,
                              base_emb_saved: np.ndarray, tol: float = 1e-4):
    """
    Compare model-recomputed embeddings with saved embeddings.
    Max absolute difference should be < tol; warn if larger.
    """
    diff = np.abs(base_emb_model - base_emb_saved)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"[check]  baseline vs saved embeddings: max|Δ|={max_diff:.6f}, mean|Δ|={mean_diff:.6f}")
    if max_diff > tol:
        warnings.warn(
            f"Baseline embeddings differ from saved by {max_diff:.4f} (> tol={tol}). "
            "PC axes are fitted on saved embeddings, which is the reference. "
            "This may indicate a version mismatch in the model or random seed drift."
        )


# ── no-HLA mode ──────────────────────────────────────────────────────────────
def run_noHLA_mode(args):
    """
    Run leave-one-block-out attribution on the no-HLA Phase 2 model.

    The no-HLA Phase 2 model was trained on the full 174-block architecture
    with HLA_classII block embeddings zeroed out before input. This function:
    1. Replicates that zero-masking on the Phase 1 embeddings.
    2. Verifies the no-HLA baseline matches saved no-HLA embeddings.
    3. Runs LOBO only on the non-HLA blocks (HLA blocks are always zero
       and therefore not informative to attribute).
    4. Saves results and generates a full-vs-noHLA rank comparison table.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. Load Phase 2 config (hyperparameters) ──────────────────────────────
    print(f"[config] Phase 2 config: {args.phase2_config}")
    attn_cfg = load_phase2_config(args.phase2_config)
    d_model  = attn_cfg.get("d_model",  64)
    n_heads  = attn_cfg.get("n_heads",   4)
    n_layers = attn_cfg.get("n_layers",  2)
    d_ff     = attn_cfg.get("d_ff",    128)
    n_pool_tokens = int(attn_cfg.get("n_pool_tokens", 1))
    if n_pool_tokens < 1:
        raise ValueError("attention.n_pool_tokens must be >= 1")
    emb_dim = n_pool_tokens * d_model
    print(f"         d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, d_ff={d_ff}, "
          f"n_pool_tokens={n_pool_tokens}, emb_dim={emb_dim}")

    # ── 1. Load Phase 1 input embeddings ──────────────────────────────────────
    print(f"\n[load]   Phase 1 ORD embeddings: {args.p1_embeddings}")
    x_np = np.load(args.p1_embeddings)
    N, B, d_in = x_np.shape
    print(f"         shape: N={N}, B={B}, d_in={d_in}")
    if B != 174:
        raise ValueError(f"Expected B=174 blocks, got B={B}. Check {args.p1_embeddings}.")

    # ── 2. Load block order and per-block latent dims ─────────────────────────
    print(f"[load]   block_order: {args.block_order}")
    block_meta = pd.read_csv(args.block_order)
    if len(block_meta) != B:
        raise ValueError(
            f"block_order.csv has {len(block_meta)} rows but Phase 1 embeddings have {B} blocks."
        )
    block_names = block_meta["block_id"].tolist()

    block_dims = load_block_dims(args.p1_latent_dims, block_meta)
    if block_dims is not None:
        block_meta = block_meta.copy()
        block_meta["latent_dim"] = block_dims

    # ── 3. Identify HLA vs non-HLA block indices ──────────────────────────────
    hla_indices = [i for i, name in enumerate(block_names) if HLA_PATTERN in name]
    non_hla_indices = [i for i in range(B) if i not in set(hla_indices)]
    print(f"         HLA blocks: {len(hla_indices)} (indices {hla_indices[:3]}…)")
    print(f"         Non-HLA blocks for LOBO: {len(non_hla_indices)}")
    if len(hla_indices) == 0:
        raise RuntimeError(
            f"No blocks matching '{HLA_PATTERN}' found in block_order.csv. "
            "Cannot identify HLA blocks to zero out."
        )

    # ── 4. Zero-mask HLA blocks (replicates no-HLA training preprocessing) ────
    x_np_zeroed = x_np.copy()
    x_np_zeroed[:, hla_indices, :] = 0.0
    print(f"[mask]   zeroed {len(hla_indices)} HLA blocks in Phase 1 embeddings")

    # ── 5. Load subject IIDs ──────────────────────────────────────────────────
    print(f"[load]   subjects: {args.subjects}")
    subjects_csv_df = pd.read_csv(args.subjects)
    if "IID" not in subjects_csv_df.columns:
        raise ValueError(f"Expected 'IID' column in {args.subjects}.")
    subject_iids = subjects_csv_df["IID"].astype(str).values
    if len(subject_iids) != N:
        raise ValueError(
            f"subjects.csv has {len(subject_iids)} rows but embeddings have N={N}."
        )

    # ── 6. Verify IID order vs no-HLA embedding CSV ───────────────────────────
    print(f"[load]   no-HLA embeddings CSV: {args.noHLA_emb_csv}")
    emb_csv = pd.read_csv(args.noHLA_emb_csv)
    if "IID" not in emb_csv.columns:
        raise ValueError(f"Expected 'IID' column in {args.noHLA_emb_csv}.")
    emb_iids = emb_csv["IID"].astype(str).values
    if len(emb_iids) != N:
        raise ValueError(
            f"no-HLA individual_embeddings.csv has {len(emb_iids)} rows but N={N}."
        )
    if not (subject_iids == emb_iids).all():
        raise ValueError(
            "subjects.csv and no-HLA individual_embeddings.csv IID order does not match. "
            "Cannot safely align row indices."
        )
    print(f"         {N} subjects confirmed, IID order verified.")

    # ── 7. Load saved no-HLA embeddings ──────────────────────────────────────
    print(f"[load]   saved no-HLA Phase 2 embeddings: {args.noHLA_embeddings}")
    base_emb_saved = np.load(args.noHLA_embeddings)
    if base_emb_saved.shape != (N, emb_dim):
        raise ValueError(
            f"Expected shape ({N}, {emb_dim}), got {base_emb_saved.shape}. "
            "Check attention.d_model and attention.n_pool_tokens in the Phase 2 config."
        )

    # ── 8. Load no-HLA model ──────────────────────────────────────────────────
    print(f"[model]  loading no-HLA checkpoint: {args.noHLA_checkpoint}")
    model = load_model(args.noHLA_checkpoint, n_blocks=B, d_in=d_in,
                       d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                       d_ff=d_ff, dropout=0.0, block_dims=block_dims,
                       n_pool_tokens=n_pool_tokens)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"         AttentionAggregator (no-HLA): n_blocks={B}, d_in={d_in}, "
          f"d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, "
          f"n_pool_tokens={n_pool_tokens}, emb_dim={emb_dim} — {n_params:,} params")

    # ── 9. Verify baseline (zeroed input → no-HLA checkpoint vs saved embeddings)
    print("[check]  verifying no-HLA baseline forward pass ...")
    base_emb_model = encode_batched(model, x_np_zeroed, args.batch_size)
    check_baseline_agreement(base_emb_model, base_emb_saved)

    # ── 10. Fit PCA on saved no-HLA embeddings (defines reference axes) ───────
    print("[PCA]    fitting PCA on saved no-HLA embeddings ...")
    scaler, pca, base_scores = fit_pca(base_emb_saved, n_components=5)
    var_exp = pca.explained_variance_ratio_
    print("         no-HLA variance explained: " +
          "  ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(var_exp)))

    # ── 11. LOBO on non-HLA blocks (mean masking computed from zeroed input) ──
    res_PC1, res_PC2, res_PC3, res_PC4, res_PC5, _ = run_lobo(
        model=model,
        x_np=x_np_zeroed,           # HLA blocks permanently zeroed
        base_emb=base_emb_saved,
        scaler=scaler,
        pca=pca,
        base_scores=base_scores,
        block_names=block_names,
        batch_size=args.batch_size,
        include_block_indices=non_hla_indices,  # skip HLA blocks in LOBO
    )

    # ── 12. Build and save result DataFrames ──────────────────────────────────
    df_noHLA_PC1 = finalise_df(res_PC1, block_meta, method="LOBO_noHLA_checkpoint_mean_mask")
    df_noHLA_PC2 = finalise_df(res_PC2, block_meta, method="LOBO_noHLA_checkpoint_mean_mask")
    df_noHLA_PC3 = finalise_df(res_PC3, block_meta, method="LOBO_noHLA_checkpoint_mean_mask")
    df_noHLA_PC4 = finalise_df(res_PC4, block_meta, method="LOBO_noHLA_checkpoint_mean_mask")
    df_noHLA_PC5 = finalise_df(res_PC5, block_meta, method="LOBO_noHLA_checkpoint_mean_mask")

    for pc_label, df_noHLA in [
        ("PC1", df_noHLA_PC1), ("PC2", df_noHLA_PC2),
        ("PC3", df_noHLA_PC3), ("PC4", df_noHLA_PC4), ("PC5", df_noHLA_PC5),
    ]:
        out_path = out_dir / f"phase2_noHLA_{pc_label}_leave_one_block_out.csv"
        df_noHLA.to_csv(out_path, index=False)
        print(f"[output] {out_path.name}  ({len(df_noHLA)} blocks)")

    # ── 13. Plots ─────────────────────────────────────────────────────────────
    for pc_label, df_noHLA in [
        ("PC1", df_noHLA_PC1), ("PC2", df_noHLA_PC2),
        ("PC3", df_noHLA_PC3), ("PC4", df_noHLA_PC4), ("PC5", df_noHLA_PC5),
    ]:
        plot_top_blocks(df_noHLA, f"no-HLA Phase 2 emb {pc_label}",
                        out_dir / f"top20_noHLA_{pc_label}_attribution.png",
                        top_n=args.top_n)

    # ── 14. Comparison table ──────────────────────────────────────────────────
    comp_path = out_dir / "phase2_full_vs_noHLA_rank_comparison.csv"
    full_pc1 = args.full_pc1_csv
    full_pc2 = args.full_pc2_csv
    if Path(full_pc1).exists() and Path(full_pc2).exists():
        comp = build_comparison_table(
            full_pc1, full_pc2,
            df_noHLA_PC1, df_noHLA_PC2,
            block_names, comp_path,
        )
        print_comparison_report(
            comp, df_noHLA_PC1, df_noHLA_PC2,
            n_hla=len(hla_indices), n_non_hla=len(non_hla_indices),
        )
    else:
        missing = [p for p in [full_pc1, full_pc2] if not Path(p).exists()]
        print(f"\n[WARNING] Full-run LOBO CSVs not found; comparison table skipped.")
        for p in missing:
            print(f"  missing: {p}")
        print("  Run first with --mode full to generate the full-run attribution.")

    print(f"\n[done]  All no-HLA outputs written to: {out_dir}/")


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["full", "noHLA"], default="full",
                   help="'full' = full-run LOBO (default); 'noHLA' = no-HLA run + comparison")

    # shared paths
    p.add_argument("--p1-embeddings", default=P1_EMBEDDINGS,
                   help="Phase 1 ORD all_blocks.npy (N, B, d_in)")
    p.add_argument("--p1-latent-dims", default=P1_LATENT_DIMS,
                   help="all_blocks_latent_dims.npy — actual per-block latent dims "
                        "(used to build BlockProjector; falls back to block_order.csv)")
    p.add_argument("--phase2-config", default=P2_CONFIG,
                   help="Phase 2 YAML config — reads attention hyperparameters "
                        "(d_model, n_heads, n_layers, d_ff, n_pool_tokens)")
    p.add_argument("--block-order",   default=BLOCK_ORDER,
                   help="block_order.csv")
    p.add_argument("--subjects",      default=SUBJECTS_CSV,
                   help="subjects.csv (IID order matching Phase 1 embeddings)")
    p.add_argument("--out-dir",       default=DEFAULT_OUT)
    p.add_argument("--batch-size",    type=int, default=BATCH_SIZE)
    p.add_argument("--top-n",         type=int, default=20,
                   help="Number of top blocks to plot")
    p.add_argument(
        "--rank-by",
        default="mean_abs_delta_score",
        choices=[
            "mean_abs_delta_score",
            "attr_per_snp",
            "attr_per_sqrt_snp",
            "attr_per_latent_dim",
            "attr_per_sqrt_latent_dim",
            "attr_per_snp_x_latent",
        ],
        help="Column used for top-block plots. Default = raw LOBO attribution."
    )

    # full-mode paths
    p.add_argument("--checkpoint",    default=P2_CHECKPOINT,
                   help="Phase 2 attention_aggregator.pt (full run)")
    p.add_argument("--p2-embeddings", default=P2_EMBEDDINGS,
                   help="Saved Phase 2 individual_embeddings.npy (full run)")
    p.add_argument("--p2-emb-csv",    default=P2_EMB_CSV,
                   help="individual_embeddings.csv (full run, for IID order)")
    p.add_argument("--pheno",         default=PHENO_FILE,
                   help="Phenotype CSV with S_SUBJECTID and log10Ige")
    p.add_argument("--skip-phenotype", action="store_true",
                   help="Skip log10Ige phenotype attribution (full mode only)")

    # no-HLA mode paths
    p.add_argument("--noHLA-checkpoint",  default=P2_CHECKPOINT_NOHLA,
                   help="no-HLA Phase 2 attention_aggregator.pt")
    p.add_argument("--noHLA-embeddings",  default=P2_EMBEDDINGS_NOHLA,
                   help="Saved no-HLA individual_embeddings.npy")
    p.add_argument("--noHLA-emb-csv",     default=P2_EMB_CSV_NOHLA,
                   help="no-HLA individual_embeddings.csv (for IID order)")
    p.add_argument("--full-pc1-csv",      default=FULL_PC1_CSV,
                   help="Full-run PC1 LOBO CSV (for comparison table)")
    p.add_argument("--full-pc2-csv",      default=FULL_PC2_CSV,
                   help="Full-run PC2 LOBO CSV (for comparison table)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "noHLA":
        run_noHLA_mode(args)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. Load Phase 2 config (hyperparameters) ──────────────────────────────
    print(f"[config] Phase 2 config: {args.phase2_config}")
    attn_cfg = load_phase2_config(args.phase2_config)
    d_model  = attn_cfg.get("d_model",  64)
    n_heads  = attn_cfg.get("n_heads",   4)
    n_layers = attn_cfg.get("n_layers",  2)
    d_ff     = attn_cfg.get("d_ff",    128)
    n_pool_tokens = int(attn_cfg.get("n_pool_tokens", 1))
    if n_pool_tokens < 1:
        raise ValueError("attention.n_pool_tokens must be >= 1")
    emb_dim = n_pool_tokens * d_model
    print(f"         d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, d_ff={d_ff}, "
          f"n_pool_tokens={n_pool_tokens}, emb_dim={emb_dim}")

    # ── 1. Load Phase 1 input embeddings ──────────────────────────────────────
    print(f"\n[load]   Phase 1 ORD embeddings: {args.p1_embeddings}")
    x_np = np.load(args.p1_embeddings)
    N, B, d_in = x_np.shape
    print(f"         shape: N={N}, B={B}, d_in={d_in}")
    if B != 174:
        raise ValueError(f"Expected B=174 blocks, got B={B}. Check {args.p1_embeddings}.")

    # ── 2. Load block order and per-block latent dims ─────────────────────────
    print(f"[load]   block_order: {args.block_order}")
    block_meta = pd.read_csv(args.block_order)
    if len(block_meta) != B:
        raise ValueError(
            f"block_order.csv has {len(block_meta)} rows but Phase 1 embeddings have {B} blocks."
        )
    block_names = block_meta["block_id"].tolist()
    print(f"         {B} blocks confirmed. Sample: {block_names[:3]}")

    block_dims = load_block_dims(args.p1_latent_dims, block_meta)
    if block_dims is not None:
        block_meta = block_meta.copy()
        block_meta["latent_dim"] = block_dims

    # ── 3. Load subject IIDs ──────────────────────────────────────────────────
    print(f"[load]   subjects: {args.subjects}")
    subjects_csv = pd.read_csv(args.subjects)
    if "IID" not in subjects_csv.columns:
        raise ValueError(f"Expected 'IID' column in {args.subjects}.")
    subject_iids = subjects_csv["IID"].astype(str).values
    if len(subject_iids) != N:
        raise ValueError(f"subjects.csv has {len(subject_iids)} rows but embeddings have N={N}.")

    # Verify IID order matches embedding CSV
    emb_csv = pd.read_csv(args.p2_emb_csv)
    if "IID" not in emb_csv.columns:
        raise ValueError(f"Expected 'IID' column in {args.p2_emb_csv}.")
    emb_iids = emb_csv["IID"].astype(str).values
    if len(emb_iids) != N:
        raise ValueError(
            f"individual_embeddings.csv has {len(emb_iids)} rows but N={N}."
        )
    if not (subject_iids == emb_iids).all():
        raise ValueError(
            "subjects.csv and individual_embeddings.csv IID order does not match. "
            "Cannot safely align row indices."
        )
    print(f"         {N} subjects confirmed, IID order verified.")

    # ── 4. Load saved baseline embeddings ─────────────────────────────────────
    print(f"[load]   saved Phase 2 embeddings: {args.p2_embeddings}")
    base_emb_saved = np.load(args.p2_embeddings)
    if base_emb_saved.shape != (N, emb_dim):
        raise ValueError(
            f"Expected shape ({N}, {emb_dim}), got {base_emb_saved.shape}. "
            "Check attention.d_model and attention.n_pool_tokens in the Phase 2 config."
        )

    # ── 5. Load model ─────────────────────────────────────────────────────────
    print(f"[model]  loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, n_blocks=B, d_in=d_in,
                       d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                       d_ff=d_ff, dropout=0.0, block_dims=block_dims,
                       n_pool_tokens=n_pool_tokens)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"         AttentionAggregator: n_blocks={B}, d_in={d_in}, d_model={d_model}, "
          f"n_heads={n_heads}, n_layers={n_layers}, "
          f"n_pool_tokens={n_pool_tokens}, emb_dim={emb_dim} — {n_params:,} params")

    # ── 6. Verify baseline (model vs saved) ───────────────────────────────────
    print("[check]  verifying baseline forward pass ...")
    base_emb_model = encode_batched(model, x_np, args.batch_size)
    check_baseline_agreement(base_emb_model, base_emb_saved)

    # ── 7. Fit PCA on saved embeddings (defines the reference axes) ───────────
    print("[PCA]    fitting PCA on saved embeddings ...")
    scaler, pca, base_scores = fit_pca(base_emb_saved, n_components=5)
    var_exp = pca.explained_variance_ratio_
    print("         variance explained: " +
          "  ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(var_exp)))

    # ── 8. Phenotype loading + ridge ──────────────────────────────────────────
    do_pheno = not args.skip_phenotype
    pheno_mask = pheno_y = ridge_ige = None

    if do_pheno:
        try:
            pheno_mask, pheno_y = load_phenotype(args.pheno, subject_iids)
            if pheno_mask.sum() >= 50:
                ridge_ige = fit_ridge(base_emb_saved[pheno_mask], pheno_y)
            else:
                print(f"[phenotype]  too few subjects ({pheno_mask.sum()}) — skipping")
                do_pheno = False
        except Exception as e:
            print(f"[phenotype]  WARNING: {e} — skipping phenotype attribution")
            do_pheno = False

    # ── 9. Leave-one-block-out ────────────────────────────────────────────────
    res_PC1, res_PC2, res_PC3, res_PC4, res_PC5, res_ige = run_lobo(
        model=model,
        x_np=x_np,
        base_emb=base_emb_saved,
        scaler=scaler,
        pca=pca,
        base_scores=base_scores,
        block_names=block_names,
        batch_size=args.batch_size,
        ridge_ige=ridge_ige if do_pheno else None,
        pheno_mask=pheno_mask,
        pheno_y=pheno_y,
    )

    method_str = (
        "Method: checkpoint-based forward pass; mean masking (population mean per block)."
    )

    # ── 10. Build and save result DataFrames ──────────────────────────────────
    df_PC1 = finalise_df(res_PC1, block_meta, method="LOBO_checkpoint_mean_mask")
    df_PC2 = finalise_df(res_PC2, block_meta, method="LOBO_checkpoint_mean_mask")
    df_PC3 = finalise_df(res_PC3, block_meta, method="LOBO_checkpoint_mean_mask")
    df_PC4 = finalise_df(res_PC4, block_meta, method="LOBO_checkpoint_mean_mask")
    df_PC5 = finalise_df(res_PC5, block_meta, method="LOBO_checkpoint_mean_mask")

    df_PC1.to_csv(out_dir / "phase2_PC1_leave_one_block_out.csv", index=False)
    df_PC2.to_csv(out_dir / "phase2_PC2_leave_one_block_out.csv", index=False)
    df_PC3.to_csv(out_dir / "phase2_PC3_leave_one_block_out.csv", index=False)
    df_PC4.to_csv(out_dir / "phase2_PC4_leave_one_block_out.csv", index=False)
    df_PC5.to_csv(out_dir / "phase2_PC5_leave_one_block_out.csv", index=False)
    print(f"\n[output] phase2_PC1_leave_one_block_out.csv  ({len(df_PC1)} blocks)")
    print(f"[output] phase2_PC2_leave_one_block_out.csv  ({len(df_PC2)} blocks)")
    print(f"[output] phase2_PC3_leave_one_block_out.csv  ({len(df_PC3)} blocks)")
    print(f"[output] phase2_PC4_leave_one_block_out.csv  ({len(df_PC4)} blocks)")
    print(f"[output] phase2_PC5_leave_one_block_out.csv  ({len(df_PC5)} blocks)")

        # ── 10b. Optional phenotype attribution DataFrame ─────────────────────────
    df_ige = None
    if do_pheno and res_ige is not None:
        df_ige = finalise_df(
            res_ige,
            block_meta,
            method="LOBO_checkpoint_mean_mask"
        )
        df_ige.to_csv(
            out_dir / "phase2_log10Ige_leave_one_block_out.csv",
            index=False
        )
        print(f"[output] phase2_log10Ige_leave_one_block_out.csv  ({len(df_ige)} blocks)")

    # ── 10c. Region-level summaries ───────────────────────────────────────────
    region_summaries = {}

    for pc_label, df_pc in [
        ("PC1", df_PC1), ("PC2", df_PC2),
        ("PC3", df_PC3), ("PC4", df_PC4), ("PC5", df_PC5),
    ]:
        region_summaries[pc_label] = write_region_summary(
            df_pc,
            out_dir / f"region_summary_{pc_label}_attribution.csv",
            target_label=pc_label,
        )

    if df_ige is not None:
        region_summaries["log10Ige"] = write_region_summary(
            df_ige,
            out_dir / "region_summary_log10Ige_attribution.csv",
            target_label="log10Ige",
        )

    # Print region diagnostics
    for label, region_df in region_summaries.items():
        print_region_diagnostic(region_df, label)

    if do_pheno and res_ige is not None:
        df_ige = finalise_df(res_ige, block_meta, method="LOBO_checkpoint_mean_mask")
        df_ige.to_csv(out_dir / "phase2_log10Ige_leave_one_block_out.csv", index=False)
        print(f"[output] phase2_log10Ige_leave_one_block_out.csv  ({len(df_ige)} blocks)")

    # ── 11. Plots ─────────────────────────────────────────────────────────────
    plot_top_blocks(
        df_PC1,
        "Phase 2 emb PC1",
        out_dir / f"top20_PC1_attribution_by_{args.rank_by}.png",
        top_n=args.top_n,
        score_col=args.rank_by,
    )
    plot_top_blocks(
        df_PC2,
        "Phase 2 emb PC2",
        out_dir / f"top20_PC2_attribution_by_{args.rank_by}.png",
        top_n=args.top_n,
        score_col=args.rank_by,
    )
    # plot_top_blocks(df_PC2, "Phase 2 emb PC2",
    #                 out_dir / "top20_PC2_attribution.png", top_n=args.top_n)
    plot_top_blocks(df_PC3, "Phase 2 emb PC3",
                    out_dir / "top20_PC3_attribution.png", top_n=args.top_n)
    plot_top_blocks(df_PC4, "Phase 2 emb PC4",
                    out_dir / "top20_PC4_attribution.png", top_n=args.top_n)
    plot_top_blocks(df_PC5, "Phase 2 emb PC5",
                    out_dir / "top20_PC5_attribution.png", top_n=args.top_n)
    if do_pheno and res_ige is not None:
        plot_top_blocks(df_ige, "log10IgE ridge-pred",
                        out_dir / "top20_log10Ige_attribution.png", top_n=args.top_n)

    # ── 12. README ────────────────────────────────────────────────────────────
    write_readme(out_dir, method_str)

    # ── 13. Print top-10 summary ──────────────────────────────────────────────
    focus = ["HLA", "17q21", "IL1RL1", "FCER1A", "PDE4D"]

    def flag_focus(bid):
        return "YES" if any(p in str(bid) for p in focus) else "-"

    for pc_label, df_pc in [
        ("PC1", df_PC1), ("PC2", df_PC2),
        ("PC3", df_PC3), ("PC4", df_PC4), ("PC5", df_PC5),
    ]:
        print(f"\n══ Top 10 blocks — {pc_label} ════════════════════════════════")
        top10 = df_pc.head(10)[
            ["rank", "block_id", "mean_abs_delta_score", "mean_signed_delta_score"]
        ].copy()
        top10["focus"] = top10["block_id"].apply(flag_focus)
        print(top10.to_string(index=False))

    if do_pheno and res_ige is not None:
        print("\n══ Top 10 blocks — log10IgE (ridge) ══════════════════")
        top10_ige = df_ige.head(10)[["rank", "block_id", "mean_abs_delta_score", "mean_signed_delta_score"]].copy()
        top10_ige["focus"] = top10_ige["block_id"].apply(flag_focus)
        print(top10_ige.to_string(index=False))
    else:
        print("\n[phenotype]  log10Ige attribution was skipped.")

    print(f"\n[done]  All outputs written to: {out_dir}/")


if __name__ == "__main__":
    main()
