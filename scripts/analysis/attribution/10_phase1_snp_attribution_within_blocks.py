#!/usr/bin/env python3
"""
10_phase1_snp_attribution_within_blocks.py

Hierarchical SNP-level attribution within selected Phase 1 LD blocks.

Concept
-------
Phase 2 LOBO (09_phase2_block_attribution.py) identifies which genomic blocks
drive the Phase 2 embedding axes (PC1, PC2) and IgE ridge prediction.
This script goes one level deeper: for each selected block, it identifies
SNPs whose variation contributes most to the learned block representation
or downstream embedding signal.

Method
------
For each selected block and each SNP j within that block:
  1. Mean-mask SNP j: replace the dosage of SNP j across all subjects with
     its population mean dosage.
  2. Re-run the Phase 1 beta-VAE encoder (frozen) to get a new block embedding.
  3. Pad to Phase 2 input dimension (16) and insert into a copy of all_blocks.npy.
  4. Re-run the frozen Phase 2 AttentionAggregator.
  5. Project new subject embeddings onto the fixed PC axes fitted on saved
     individual_embeddings.npy.
  6. Compute mean_abs_delta_score and mean_signed_delta_score vs baseline.
  7. Optionally compute delta in ridge-predicted log10IgE.

Attribution reflects embedding geometry change, not causal importance.
Do not interpret high attribution as evidence that a SNP causes disease.

Limitations
-----------
- Mean masking replaces SNP variation with population mean, not ablation.
- Phase 1 VAE bottleneck limits what SNP-level signal is captured.
- Phase 2 contextualizes across blocks; SNP effects are filtered through
  the block embedding before entering the transformer.
- log10IgE ridge attribution is exploratory (CV R2 ~0.016 baseline).

Outputs  (results/analysis/phase1_snp_attribution/)
--------
{block_id}_snp_attribution.csv           per-block, all targets
all_selected_blocks_snp_attribution.csv  combined
top{k}_emb_pc1_{block_id}.png
top{k}_emb_pc2_{block_id}.png
top{k}_log10ige_ridge_pred_{block_id}.png  (if phenotype not skipped)

Run as:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/analysis/10_phase1_snp_attribution_within_blocks.py \\
        --selected-blocks region_6p21_HLA_classII_sb15 \\
        --max-snps-per-block 5 --skip-phenotype
"""

import sys
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    sys.exit(
        "ERROR: torch not found.\n"
        "Run in the vae conda environment:\n"
        "  KMP_DUPLICATE_LIB_OK=TRUE /path/to/vae/bin/python "
        "scripts/analysis/10_phase1_snp_attribution_within_blocks.py"
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# ── add repo root so we can import core modules ───────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from scripts.core.attention_phase2 import AttentionAggregator
from scripts.core.VAE_phase1 import BlockVAE

# ── default paths ─────────────────────────────────────────────────────────────
P1_ALL_BLOCKS    = "results/output_regions/ORD/embeddings/all_blocks.npy"
P1_LATENT_DIMS   = "results/output_regions/ORD/embeddings/all_blocks_latent_dims.npy"
P1_MODELS_DIR    = "results/output_regions/ORD/models"
P1_EMB_DIR       = "results/output_regions/ORD/embeddings"
RAW_DIR          = "data/region_blocks"
BLOCK_ORDER      = "results/output_regions/block_order.csv"
SUBJECTS_CSV     = "results/output_regions/subjects.csv"

P2_CHECKPOINT       = "results/output_regions2/ORD/models/attention_aggregator.pt"
P2_EMBEDDINGS       = "results/output_regions2/ORD/embeddings/individual_embeddings.npy"
P2_EMB_CSV          = "results/output_regions2/ORD/embeddings/individual_embeddings.csv"

P2_CHECKPOINT_NOHLA = "results/output_regions2_noHLA/ORD/models/attention_aggregator.pt"
P2_EMBEDDINGS_NOHLA = "results/output_regions2_noHLA/ORD/embeddings/individual_embeddings.npy"
P2_EMB_CSV_NOHLA    = "results/output_regions2_noHLA/ORD/embeddings/individual_embeddings.csv"

PHENO_FILE       = "metadata/COS_TRIO_pheno_1165.csv"
PHENO_TARGET     = "log10Ige"
HLA_PATTERN      = "HLA_classII"
DEFAULT_OUT      = "results/analysis/phase1_snp_attribution"

BATCH_SIZE   = 256
P2_D_MODEL   = 64
P2_D_IN      = 16


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers reused verbatim from 09_phase2_block_attribution.py
# ═══════════════════════════════════════════════════════════════════════════════

def load_p2_model(ckpt_path: str, n_blocks: int, d_in: int = P2_D_IN,
                  d_model: int = P2_D_MODEL, n_heads: int = 4,
                  n_layers: int = 2, d_ff: int = 128,
                  dropout: float = 0.0) -> AttentionAggregator:
    model = AttentionAggregator(
        n_blocks=n_blocks, d_in=d_in, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        dropout=dropout,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_batched(model: AttentionAggregator, x_np: np.ndarray,
                   batch_size: int = BATCH_SIZE) -> np.ndarray:
    results = []
    for start in range(0, x_np.shape[0], batch_size):
        batch = torch.tensor(x_np[start:start + batch_size], dtype=torch.float32)
        emb, _, _ = model.encode(batch, return_self_attn=False)
        results.append(emb.numpy())
    return np.concatenate(results, axis=0)


def fit_pca(embeddings_np: np.ndarray, n_components: int = 2):
    scaler = StandardScaler()
    z = scaler.fit_transform(embeddings_np)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(z)
    return scaler, pca, scores


def project_embeddings(embeddings_np: np.ndarray,
                       scaler: StandardScaler, pca: PCA) -> np.ndarray:
    return pca.transform(scaler.transform(embeddings_np))


def load_phenotype(pheno_path: str, subject_iids: np.ndarray,
                   target: str = PHENO_TARGET):
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


def fit_ridge(base_emb: np.ndarray, y: np.ndarray) -> RidgeCV:
    alphas = np.logspace(-3, 4, 50)
    scaler = StandardScaler()
    X = scaler.fit_transform(base_emb)
    ridge = RidgeCV(alphas=alphas, scoring="r2", cv=5)
    ridge.fit(X, y)
    ridge._feature_scaler = scaler
    print(f"[ridge]      {PHENO_TARGET} baseline: best α={ridge.alpha_:.4g}, "
          f"CV R²={ridge.best_score_:.3f}  "
          f"(exploratory — weak signal expected at this sample size)")
    return ridge


def ridge_predict(ridge: RidgeCV, emb: np.ndarray) -> np.ndarray:
    return ridge.predict(ridge._feature_scaler.transform(emb))


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline agreement check (adapted from 09; adds label + shape guard)
# ═══════════════════════════════════════════════════════════════════════════════

def check_baseline_agreement(computed: np.ndarray, reference: np.ndarray,
                              tol: float = 1e-4, label: str = "baseline") -> float:
    if computed.shape != reference.shape:
        raise RuntimeError(
            f"[{label}] shape mismatch: computed={computed.shape} "
            f"vs reference={reference.shape}"
        )
    diff = np.abs(computed - reference)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print(f"[check]      {label}: max|Δ|={max_diff:.6f}, mean|Δ|={mean_diff:.6f}")
    if max_diff > tol:
        warnings.warn(
            f"[{label}] max|Δ|={max_diff:.4f} exceeds tol={tol}. "
            "Possible model version mismatch or random seed drift."
        )
    return max_diff


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 VAE loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_p1_vae(ckpt_path: str):
    """
    Load BlockVAE from a state_dict checkpoint.
    Infers p (n_snps), d (latent_dim), and loss_type from weight shapes.
    drop=0.0 is safe regardless of training dropout because model.eval()
    disables all Dropout layers.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    p = int(ckpt["enc.0.weight"].shape[1])
    d = int(ckpt["mu.weight"].shape[0])
    dec_w_keys = sorted(
        [k for k in ckpt if k.startswith("dec.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    if not dec_w_keys:
        raise RuntimeError(f"No decoder weight keys found in {ckpt_path}")
    out_dim = int(ckpt[dec_w_keys[-1]].shape[0])
    if out_dim == p * 3:
        loss_type = "CAT"
    elif out_dim == p * 2:
        loss_type = "ORD"
    else:
        loss_type = "MSE"
    model = BlockVAE(p=p, d=d, drop=0.0, loss_type=loss_type)
    model.load_state_dict(ckpt)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[p1_vae]     {Path(ckpt_path).name}: p={p}, d={d}, loss_type={loss_type}")
    return model, p, d, loss_type


# ═══════════════════════════════════════════════════════════════════════════════
# .raw loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw(raw_path: str, expected_iids: np.ndarray):
    """
    Load a PLINK .raw dosage file.
    Columns: FID IID PAT MAT SEX PHENOTYPE <SNP_1> ... <SNP_P>
    Fails loudly on count mismatch, duplicate IIDs, or IID set mismatch.
    If IID sets match but order differs, reorders rows to match expected_iids.
    Genotype files are expected to be clean with no missing values.
    """
    df = pd.read_csv(raw_path, sep=r"\s+", engine="python")
    raw_iids = df["IID"].astype(str).str.strip().values
    fname = Path(raw_path).name

    if len(raw_iids) != len(expected_iids):
        raise RuntimeError(
            f"{raw_path}: subject count mismatch: "
            f"raw={len(raw_iids)} vs expected={len(expected_iids)}"
        )

    raw_series = pd.Series(raw_iids)
    if raw_series.duplicated().any():
        dups = raw_series[raw_series.duplicated()].tolist()
        raise RuntimeError(f"{raw_path}: duplicate IIDs in .raw file: {dups[:5]}")

    exp_series = pd.Series(expected_iids)
    if exp_series.duplicated().any():
        raise RuntimeError(f"{raw_path}: duplicate IIDs in subjects.csv.")

    raw_set = set(raw_iids)
    exp_set = set(expected_iids)
    if raw_set != exp_set:
        only_raw = sorted(raw_set - exp_set)[:5]
        only_exp = sorted(exp_set - raw_set)[:5]
        raise RuntimeError(
            f"{raw_path}: IID set mismatch. "
            f"Only in raw (sample): {only_raw}. Only in expected (sample): {only_exp}."
        )

    if not np.array_equal(raw_iids, expected_iids):
        print(f"[raw]        {fname}: IID set matches but order differed; "
              "reordered to subjects.csv")
        iid_to_pos = {iid: i for i, iid in enumerate(raw_iids)}
        new_order = [iid_to_pos[iid] for iid in expected_iids]
        df = df.iloc[new_order].reset_index(drop=True)
        raw_iids = df["IID"].astype(str).str.strip().values
        assert np.array_equal(raw_iids, expected_iids), \
            f"{raw_path}: reorder failed — IID mismatch after reorder"
    snp_cols = list(df.columns[6:])
    if not snp_cols:
        raise RuntimeError(f"{raw_path}: no SNP columns found after PHENOTYPE column.")
    geno = df[snp_cols].values.astype(np.float32)
    n_nan = int(np.isnan(geno).sum())
    if n_nan > 0:
        raise RuntimeError(
            f"{raw_path}: found {n_nan} NaN values in SNP dosage matrix. "
            "Expected clean genotype data with no missing values."
        )
    print(f"[raw]        {fname}: {len(snp_cols)} SNPs, "
          f"{len(raw_iids)} subjects, 0 NaN")
    return geno, snp_cols


# ═══════════════════════════════════════════════════════════════════════════════
# SNP locus parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_snp_locus(col_name: str) -> str:
    """
    Strip the trailing PLINK allele suffix (e.g. _A, _G) if safe.
    '12:111742373:A:G_A' -> '12:111742373:A:G'
    Falls back to col_name unchanged if format does not match.
    """
    parts = col_name.rsplit("_", 1)
    if len(parts) == 2 and 1 <= len(parts[1]) <= 2 and parts[1].isalpha():
        return parts[0]
    return col_name


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 baseline recomputation and validation
# ═══════════════════════════════════════════════════════════════════════════════

def recompute_p1_baseline(block_id: str, block_index: int,
                          model_p1, geno: np.ndarray,
                          all_blocks: np.ndarray,
                          latent_dims: np.ndarray,
                          p1_emb_dir: str,
                          tol: float = 1e-4,
                          check_all_blocks_slice: bool = True):
    """
    Re-encode the block with the frozen Phase 1 VAE and validate against:
      (a) saved {block_id}.npy
      (b) all_blocks[:, block_index, :true_dim]  (skipped if check_all_blocks_slice=False)
    Fails loudly on shape mismatch. Warns if max|Δ| > tol.
    Returns (recomputed_emb (N, true_dim), true_dim, snp_means (P,)).
    """
    true_dim = int(latent_dims[block_index])
    N = geno.shape[0]

    x = torch.tensor(geno, dtype=torch.float32)
    with torch.no_grad():
        mu, _ = model_p1.encode(x)
    recomputed = mu.numpy()   # (N, true_dim)

    if recomputed.shape != (N, true_dim):
        raise RuntimeError(
            f"{block_id}: recomputed shape {recomputed.shape} "
            f"!= expected ({N}, {true_dim})"
        )

    # validate against saved per-block embedding
    saved_path = Path(p1_emb_dir) / f"{block_id}.npy"
    saved_emb = np.load(str(saved_path))
    if saved_emb.shape != recomputed.shape:
        raise RuntimeError(
            f"{block_id}: saved .npy shape {saved_emb.shape} "
            f"!= recomputed shape {recomputed.shape}"
        )
    check_baseline_agreement(recomputed, saved_emb, tol=tol,
                             label=f"{block_id} recomputed vs saved .npy")

    # validate against all_blocks slice (skip when block is zeroed in noHLA mode)
    if check_all_blocks_slice:
        ab_slice = all_blocks[:, block_index, :true_dim]
        if ab_slice.shape != recomputed.shape:
            raise RuntimeError(
                f"{block_id}: all_blocks slice shape {ab_slice.shape} "
                f"!= recomputed shape {recomputed.shape}"
            )
        check_baseline_agreement(recomputed, ab_slice, tol=tol,
                                 label=f"{block_id} recomputed vs all_blocks slice")

    snp_means = geno.mean(axis=0)   # (P,) population mean dosage per SNP
    return recomputed, true_dim, snp_means


# ═══════════════════════════════════════════════════════════════════════════════
# SNP attribution loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_snp_attribution(
    block_id: str,
    block_index: int,
    model_p1,
    model_p2,
    geno: np.ndarray,
    snp_cols: list,
    snp_means: np.ndarray,
    true_dim: int,
    x_p2_base: np.ndarray,       # (N, B, 16); HLA-zeroed in noHLA mode
    base_p2_scores: np.ndarray,  # (N, 2)
    scaler_p2: StandardScaler,
    pca_p2: PCA,
    run_label: str,
    batch_size: int = BATCH_SIZE,
    max_snps: int = None,
    pheno_mask: np.ndarray = None,
    base_ige_pred: np.ndarray = None,
    ridge_ige=None,
) -> list:
    """
    For each SNP j in the block:
      - Replace dosage of SNP j with its population mean across all subjects.
      - Re-run Phase 1 VAE encoder (frozen) -> new block embedding (N, true_dim).
      - Pad to (N, 16), insert into a copy of x_p2_base at block_index.
      - Re-run Phase 2 AttentionAggregator (frozen).
      - Project onto fixed PC axes; compute delta vs baseline scores.
      - Optionally compute delta in ridge-predicted log10IgE.

    This identifies SNPs whose variation contributes most to the learned block
    representation or downstream embedding signal.
    Attribution reflects embedding geometry change, not causal importance.
    """
    N, P = geno.shape
    MAX_DIM = x_p2_base.shape[2]   # 16
    indices = list(range(P)) if max_snps is None else list(range(min(P, max_snps)))
    do_ige = (ridge_ige is not None and pheno_mask is not None
              and base_ige_pred is not None)

    print(f"\n[snp_attr]   {block_id}: {len(indices)}/{P} SNPs, run={run_label}")
    records = []

    for j_idx, j in enumerate(indices):
        geno_masked = geno.copy()
        geno_masked[:, j] = snp_means[j]

        x = torch.tensor(geno_masked, dtype=torch.float32)
        with torch.no_grad():
            mu, _ = model_p1.encode(x)
        recomp = mu.numpy()   # (N, true_dim)

        x_p2 = x_p2_base.copy()
        x_p2[:, block_index, :true_dim] = recomp
        x_p2[:, block_index, true_dim:] = 0.0

        masked_emb = encode_batched(model_p2, x_p2, batch_size)
        masked_scores = project_embeddings(masked_emb, scaler_p2, pca_p2)

        snp_locus = parse_snp_locus(snp_cols[j])

        for tgt_idx, tgt_name in enumerate(["emb_PC1", "emb_PC2"]):
            delta = masked_scores[:, tgt_idx] - base_p2_scores[:, tgt_idx]
            records.append(dict(
                block_id=block_id,
                block_index=block_index,
                snp_column=snp_cols[j],
                snp_locus=snp_locus,
                snp_id=snp_locus,
                target=tgt_name,
                mean_abs_delta_score=float(np.abs(delta).mean()),
                mean_signed_delta_score=float(delta.mean()),
                n_subjects=N,
                true_latent_dim=true_dim,
                phase2_input_dim=MAX_DIM,
                run=run_label,
            ))

        if do_ige:
            masked_ige = ridge_predict(ridge_ige, masked_emb[pheno_mask])
            delta_ige = masked_ige - base_ige_pred
            records.append(dict(
                block_id=block_id,
                block_index=block_index,
                snp_column=snp_cols[j],
                snp_locus=snp_locus,
                snp_id=snp_locus,
                target="log10Ige_ridge_pred",
                mean_abs_delta_score=float(np.abs(delta_ige).mean()),
                mean_signed_delta_score=float(delta_ige.mean()),
                n_subjects=int(pheno_mask.sum()),
                true_latent_dim=true_dim,
                phase2_input_dim=MAX_DIM,
                run=run_label,
            ))

        if (j_idx + 1) % 20 == 0 or j_idx == len(indices) - 1:
            print(f"  SNP {j_idx + 1:4d}/{len(indices)} done")

    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ═══════════════════════════════════════════════════════════════════════════════

def finalise_snp_df(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["rank_within_block"] = (
        df.groupby(["block_id", "target"])["mean_abs_delta_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    max_per_group = df.groupby(["block_id", "target"])["mean_abs_delta_score"].transform("max")
    df["normalized_snp_score_within_block"] = df["mean_abs_delta_score"] / max_per_group.where(
        max_per_group > 0, other=1.0
    )
    return df


def add_block_weighted_priority(df: pd.DataFrame,
                                lobo_pc1_csv: str = None,
                                lobo_pc2_csv: str = None,
                                lobo_ige_csv: str = None) -> pd.DataFrame:
    """
    Heuristic block-weighted SNP priority.
    block_weighted_snp_priority = normalized_block_score * normalized_snp_score_within_block
    Labeled as a prioritization heuristic. Does not imply causal importance.
    Only added for (block_id, target) pairs present in both df and the LOBO CSVs.
    """
    lobo_norm = {}
    for tgt, path in [
        ("emb_PC1",             lobo_pc1_csv),
        ("emb_PC2",             lobo_pc2_csv),
        ("log10Ige_ridge_pred", lobo_ige_csv),
    ]:
        if path is None or not Path(path).exists():
            continue
        try:
            lobo = pd.read_csv(path)[["block_id", "mean_abs_delta_score"]]
            max_s = lobo["mean_abs_delta_score"].max()
            lobo["norm_block"] = lobo["mean_abs_delta_score"] / max_s if max_s > 0 else 0.0
            lobo_norm[tgt] = lobo.set_index("block_id")["norm_block"].to_dict()
        except Exception as e:
            print(f"[priority]   WARNING: could not load LOBO CSV for {tgt}: {e}")

    if not lobo_norm:
        print("[priority]   no LOBO CSVs loaded — block-weighted priority not computed")
        return df

    def _block_score(row):
        return lobo_norm.get(row["target"], {}).get(row["block_id"], np.nan)

    def _priority(row):
        nb = _block_score(row)
        ns = row.get("normalized_snp_score_within_block", np.nan)
        return float(nb) * float(ns) if (pd.notna(nb) and pd.notna(ns)) else np.nan

    df["source_block_attribution_score"] = df.apply(_block_score, axis=1)
    df["block_weighted_snp_priority"] = df.apply(_priority, axis=1)
    df["priority_type"] = df["block_weighted_snp_priority"].apply(
        lambda v: "heuristic_block_weighted" if pd.notna(v) else np.nan
    )
    return df


def plot_top_snps(df_sub: pd.DataFrame, block_id: str, target_label: str,
                  out_path: Path, top_k: int = 20):
    top = df_sub.sort_values("mean_abs_delta_score", ascending=False).head(top_k).copy()
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.38)))
    y = np.arange(len(top))
    ax.barh(y, top["mean_abs_delta_score"].values, color="steelblue", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top["snp_locus"].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |Δ score| under SNP mean-masking")
    ax.set_title(
        f"Top {len(top)} SNPs — {block_id}\n"
        f"Target: {target_label}\n"
        "SNPs whose variation contributes most to the learned embedding signal.\n"
        "Do not interpret as causal evidence.",
        fontsize=9,
    )
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row["mean_abs_delta_score"] + 1e-6, i,
                f"  {row['mean_signed_delta_score']:+.5f}", va="center", fontsize=6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot]       saved {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Argparse
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["full", "noHLA"], default="full",
                   help="'full' = full Phase 2 model (default); "
                        "'noHLA' = no-HLA Phase 2 model with HLA blocks zeroed")
    p.add_argument("--selected-blocks", required=True,
                   help="Comma-separated block IDs to attribute, e.g. "
                        "region_6p21_HLA_classII_sb15,region_9p24_IL33")

    # Phase 1 paths
    p.add_argument("--p1-all-blocks",  default=P1_ALL_BLOCKS)
    p.add_argument("--p1-latent-dims", default=P1_LATENT_DIMS)
    p.add_argument("--p1-models-dir",  default=P1_MODELS_DIR)
    p.add_argument("--p1-emb-dir",     default=P1_EMB_DIR)
    p.add_argument("--raw-dir",        default=RAW_DIR)
    p.add_argument("--block-order",    default=BLOCK_ORDER)
    p.add_argument("--subjects",       default=SUBJECTS_CSV)

    # Phase 2 full paths
    p.add_argument("--checkpoint",    default=P2_CHECKPOINT)
    p.add_argument("--p2-embeddings", default=P2_EMBEDDINGS)
    p.add_argument("--p2-emb-csv",    default=P2_EMB_CSV)

    # Phase 2 no-HLA paths
    p.add_argument("--noHLA-checkpoint",  default=P2_CHECKPOINT_NOHLA)
    p.add_argument("--noHLA-embeddings",  default=P2_EMBEDDINGS_NOHLA)
    p.add_argument("--noHLA-emb-csv",     default=P2_EMB_CSV_NOHLA)

    # Phenotype
    p.add_argument("--pheno",          default=PHENO_FILE)
    p.add_argument("--skip-phenotype", action="store_true")

    # Optional block-weighted priority (requires script 09 LOBO CSVs)
    p.add_argument("--lobo-pc1-csv", default=None,
                   help="phase2_PC1_leave_one_block_out.csv from script 09 "
                        "(enables heuristic block-weighted SNP priority)")
    p.add_argument("--lobo-pc2-csv", default=None)
    p.add_argument("--lobo-ige-csv", default=None)

    # Attribution controls
    p.add_argument("--max-snps-per-block", type=int, default=None,
                   help="Truncate SNP loop per block — for debugging only. "
                        "Default None = all SNPs in block.")
    p.add_argument("--top-k-pc1",  type=int, default=20)
    p.add_argument("--top-k-pc2",  type=int, default=20)
    p.add_argument("--top-k-ige",  type=int, default=20)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--out-dir",    default=DEFAULT_OUT)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_label = args.mode

    selected_blocks = [b.strip() for b in args.selected_blocks.split(",") if b.strip()]
    if not selected_blocks:
        sys.exit("ERROR: --selected-blocks is empty.")
    print(f"\n[config]     mode={run_label}")
    print(f"             {len(selected_blocks)} selected block(s): {selected_blocks}")

    # ── 1. Block order ────────────────────────────────────────────────────────
    print(f"\n[load]       block_order: {args.block_order}")
    block_meta = pd.read_csv(args.block_order)
    if "block_id" not in block_meta.columns or "pos" not in block_meta.columns:
        raise RuntimeError("block_order.csv must have 'pos' and 'block_id' columns.")
    block_id_to_index = dict(zip(block_meta["block_id"], block_meta["pos"]))
    block_names = block_meta["block_id"].tolist()
    B_total = len(block_names)
    print(f"             {B_total} blocks total")

    for bid in selected_blocks:
        if bid not in block_id_to_index:
            raise RuntimeError(
                f"Block '{bid}' not found in block_order.csv. "
                f"Sample: {block_names[:3]}"
            )

    # ── 2. Subject IIDs ───────────────────────────────────────────────────────
    print(f"[load]       subjects: {args.subjects}")
    subj_df = pd.read_csv(args.subjects)
    if "IID" not in subj_df.columns:
        raise ValueError(f"Expected 'IID' column in {args.subjects}.")
    subject_iids = subj_df["IID"].astype(str).values
    N = len(subject_iids)
    print(f"             {N} subjects")

    # ── 3. Phase 2 paths by mode ──────────────────────────────────────────────
    if run_label == "noHLA":
        p2_ckpt    = args.noHLA_checkpoint
        p2_emb_npy = args.noHLA_embeddings
        p2_emb_csv = args.noHLA_emb_csv
    else:
        p2_ckpt    = args.checkpoint
        p2_emb_npy = args.p2_embeddings
        p2_emb_csv = args.p2_emb_csv

    # ── 4. Verify Phase 2 IID order ───────────────────────────────────────────
    print(f"[load]       Phase 2 embeddings CSV: {p2_emb_csv}")
    emb_csv_df = pd.read_csv(p2_emb_csv)
    if "IID" not in emb_csv_df.columns:
        raise ValueError(f"Expected 'IID' column in {p2_emb_csv}.")
    emb_iids = emb_csv_df["IID"].astype(str).values
    if len(emb_iids) != N:
        raise ValueError(
            f"Phase 2 CSV has {len(emb_iids)} rows but subjects.csv has {N}."
        )
    if not np.array_equal(subject_iids, emb_iids):
        raise ValueError(
            "subjects.csv and Phase 2 individual_embeddings.csv IID order mismatch."
        )
    print(f"             IID order verified ({N} subjects)")

    # ── 5. Load saved Phase 2 embeddings ──────────────────────────────────────
    print(f"[load]       saved Phase 2 embeddings: {p2_emb_npy}")
    base_emb_saved = np.load(p2_emb_npy)
    if base_emb_saved.shape != (N, P2_D_MODEL):
        raise ValueError(
            f"Expected Phase 2 embeddings shape ({N}, {P2_D_MODEL}), "
            f"got {base_emb_saved.shape}."
        )

    # ── 6. Load all_blocks.npy ────────────────────────────────────────────────
    print(f"[load]       Phase 1 all_blocks: {args.p1_all_blocks}")
    all_blocks_orig = np.load(args.p1_all_blocks)
    N_b, B_b, d_in = all_blocks_orig.shape
    print(f"             shape: N={N_b}, B={B_b}, d_in={d_in}")
    if N_b != N:
        raise ValueError(f"all_blocks.npy N={N_b} != subjects.csv N={N}")
    if B_b != B_total:
        raise ValueError(f"all_blocks.npy B={B_b} != block_order.csv B={B_total}")
    if d_in != P2_D_IN:
        raise ValueError(f"all_blocks.npy d_in={d_in} != expected {P2_D_IN}")

    # ── 7. Load latent dims ───────────────────────────────────────────────────
    print(f"[load]       latent_dims: {args.p1_latent_dims}")
    latent_dims = np.load(args.p1_latent_dims)
    if len(latent_dims) != B_total:
        raise ValueError(f"latent_dims length {len(latent_dims)} != B={B_total}")
    print(f"             unique true dims: {sorted(set(latent_dims.astype(int)))}")

    # ── 8. noHLA: zero HLA blocks in all_blocks ───────────────────────────────
    hla_indices_set = set()
    if run_label == "noHLA":
        hla_indices = [i for i, name in enumerate(block_names) if HLA_PATTERN in name]
        if not hla_indices:
            raise RuntimeError(
                f"No blocks matching '{HLA_PATTERN}' found — cannot run noHLA mode."
            )
        hla_indices_set = set(hla_indices)
        all_blocks = all_blocks_orig.copy()
        all_blocks[:, hla_indices, :] = 0.0
        print(f"[noHLA]      zeroed {len(hla_indices)} HLA blocks in all_blocks")
        # Reject any selected block that is an HLA block — inserting a recomputed
        # HLA embedding into x_p2 would reintroduce HLA into the no-HLA model.
        hla_selected = [b for b in selected_blocks
                        if int(block_id_to_index[b]) in hla_indices_set]
        if hla_selected:
            raise RuntimeError(
                f"Cannot run SNP attribution for HLA block(s) {hla_selected} in "
                "noHLA mode because noHLA mode zeroes HLA inputs. "
                "Use --mode full for HLA SNP attribution."
            )
        print(f"[noHLA]      selected blocks confirmed non-HLA: {selected_blocks}")
    else:
        all_blocks = all_blocks_orig   # no copy needed; not mutated
        print(f"[full]       all selected blocks allowed: {selected_blocks}")

    # ── 9. Load Phase 2 model ─────────────────────────────────────────────────
    print(f"\n[model]      loading Phase 2 checkpoint: {p2_ckpt}")
    model_p2 = load_p2_model(p2_ckpt, n_blocks=B_total, d_in=d_in)
    n_params = sum(p.numel() for p in model_p2.parameters())
    print(f"             AttentionAggregator: n_blocks={B_total}, d_in={d_in}, "
          f"d_model={P2_D_MODEL} — {n_params:,} params")

    # ── 10. Phase 2 baseline verification ─────────────────────────────────────
    print("[check]      verifying Phase 2 baseline forward pass ...")
    base_emb_recomp = encode_batched(model_p2, all_blocks, args.batch_size)
    if base_emb_recomp.shape != base_emb_saved.shape:
        raise RuntimeError(
            f"Phase 2 recomputed shape {base_emb_recomp.shape} "
            f"!= saved shape {base_emb_saved.shape}"
        )
    max_diff = check_baseline_agreement(
        base_emb_recomp, base_emb_saved, tol=1e-4,
        label="Phase 2 recomputed vs saved"
    )
    if max_diff > 1e-3:
        raise RuntimeError(
            f"Phase 2 baseline mismatch too large (max|Δ|={max_diff:.4f} > 1e-3). "
            "Verify that --checkpoint and --p2-embeddings are from the same run."
        )

    # ── 11. PCA on saved Phase 2 embeddings (defines fixed reference axes) ────
    print("[PCA]        fitting PCA on saved Phase 2 embeddings ...")
    scaler_p2, pca_p2, base_scores = fit_pca(base_emb_saved, n_components=2)
    var_exp = pca_p2.explained_variance_ratio_
    print(f"             PC1={var_exp[0]*100:.1f}%  "
          f"PC2={var_exp[1]*100:.1f}% of embedding variance")

    # ── 12. Phenotype + ridge (fitted once on saved embeddings) ───────────────
    do_pheno = not args.skip_phenotype
    pheno_mask = pheno_y = ridge_ige = base_ige_pred = None

    if do_pheno:
        try:
            pheno_mask, pheno_y = load_phenotype(args.pheno, subject_iids)
            if pheno_mask.sum() >= 50:
                ridge_ige = fit_ridge(base_emb_saved[pheno_mask], pheno_y)
                base_ige_pred = ridge_predict(ridge_ige, base_emb_saved[pheno_mask])
            else:
                print(f"[phenotype]  too few subjects ({pheno_mask.sum()}) — skipping")
                do_pheno = False
        except Exception as e:
            print(f"[phenotype]  WARNING: {e} — skipping IgE attribution")
            do_pheno = False

    # ── 13. Per-block SNP attribution ─────────────────────────────────────────
    all_records = []

    for bid in selected_blocks:
        block_index = int(block_id_to_index[bid])
        is_hla = block_index in hla_indices_set   # always False after noHLA validation
        print(f"\n{'='*60}")
        print(f"[block]      {bid}  (index={block_index})")
        print(f"{'='*60}")

        # Load Phase 1 VAE from checkpoint
        ckpt_path = Path(args.p1_models_dir) / f"{bid}.pt"
        if not ckpt_path.exists():
            raise RuntimeError(f"Phase 1 checkpoint not found: {ckpt_path}")
        model_p1, p_snps, d_latent, loss_type = load_p1_vae(str(ckpt_path))

        # Load .raw genotype
        raw_path = Path(args.raw_dir) / f"{bid}.raw"
        if not raw_path.exists():
            raise RuntimeError(f".raw file not found: {raw_path}")
        geno, snp_cols = load_raw(str(raw_path), subject_iids)

        if geno.shape[1] != p_snps:
            raise RuntimeError(
                f"{bid}: checkpoint p={p_snps} != .raw SNP count={geno.shape[1]}"
            )

        # Recompute Phase 1 baseline + validate
        # Skip all_blocks slice check for HLA blocks in noHLA mode (they are zeroed)
        recomp_emb, true_dim, snp_means = recompute_p1_baseline(
            bid, block_index, model_p1, geno,
            all_blocks_orig,    # always use unmodified all_blocks for validation
            latent_dims, args.p1_emb_dir,
            check_all_blocks_slice=not is_hla or run_label == "full",
        )
        print(f"             true_dim={true_dim}, n_snps={geno.shape[1]}, "
              f"loss_type={loss_type}")

        # Run SNP attribution (x_p2_base = HLA-zeroed all_blocks in noHLA mode)
        block_records = run_snp_attribution(
            block_id=bid,
            block_index=block_index,
            model_p1=model_p1,
            model_p2=model_p2,
            geno=geno,
            snp_cols=snp_cols,
            snp_means=snp_means,
            true_dim=true_dim,
            x_p2_base=all_blocks,
            base_p2_scores=base_scores,
            scaler_p2=scaler_p2,
            pca_p2=pca_p2,
            run_label=run_label,
            batch_size=args.batch_size,
            max_snps=args.max_snps_per_block,
            pheno_mask=pheno_mask if do_pheno else None,
            base_ige_pred=base_ige_pred if do_pheno else None,
            ridge_ige=ridge_ige if do_pheno else None,
        )
        all_records.extend(block_records)

        # Per-block CSV (all targets)
        block_df = finalise_snp_df(block_records)
        per_block_csv = out_dir / f"{bid}_snp_attribution.csv"
        block_df.to_csv(per_block_csv, index=False)
        print(f"[output]     {per_block_csv.name}  ({len(block_df)} rows)")

        # Per-block plots
        for tgt, top_k, tgt_label in [
            ("emb_PC1",             args.top_k_pc1, "Phase 2 emb PC1"),
            ("emb_PC2",             args.top_k_pc2, "Phase 2 emb PC2"),
            ("log10Ige_ridge_pred", args.top_k_ige,
             "log10IgE ridge-pred (exploratory)"),
        ]:
            sub = block_df[block_df["target"] == tgt]
            if sub.empty:
                continue
            safe_tgt = tgt.lower().replace(" ", "_").replace("/", "_")
            fname = f"top{top_k}_{safe_tgt}_{bid}.png"
            plot_top_snps(sub, bid, tgt_label, out_dir / fname, top_k=top_k)

    # ── 14. Combined CSV ──────────────────────────────────────────────────────
    combined_df = finalise_snp_df(all_records)

    if args.lobo_pc1_csv or args.lobo_pc2_csv:
        combined_df = add_block_weighted_priority(
            combined_df,
            lobo_pc1_csv=args.lobo_pc1_csv,
            lobo_pc2_csv=args.lobo_pc2_csv,
            lobo_ige_csv=args.lobo_ige_csv,
        )

    combined_csv = out_dir / "all_selected_blocks_snp_attribution.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"\n[output]     {combined_csv.name}  "
          f"({len(combined_df)} rows, "
          f"{combined_df['block_id'].nunique() if not combined_df.empty else 0} blocks)")

    # ── 15. Top-10 summary ────────────────────────────────────────────────────
    for tgt, tgt_label in [
        ("emb_PC1",             "Phase 2 emb PC1"),
        ("emb_PC2",             "Phase 2 emb PC2"),
        ("log10Ige_ridge_pred", "log10IgE ridge-pred"),
    ]:
        sub = combined_df[combined_df["target"] == tgt] if not combined_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        print(f"\n══ Top 10 SNPs — {tgt_label} ({'all selected blocks'}) "
              f"══════════════")
        cols = ["rank_within_block", "block_id", "snp_locus",
                "mean_abs_delta_score", "mean_signed_delta_score", "n_subjects"]
        print(sub.sort_values("mean_abs_delta_score", ascending=False)
              .head(10)[cols].to_string(index=False))

    print(f"\n[done]       All outputs written to: {out_dir}/")


if __name__ == "__main__":
    main()
