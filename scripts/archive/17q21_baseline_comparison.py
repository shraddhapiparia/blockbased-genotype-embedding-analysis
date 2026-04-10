#!/usr/bin/env python3
"""
17q21_baseline_comparison.py

Compares four models for each 17q21 subblock x pctpred_fev1_pre_BD:

  Model 1a: pheno ~ geno_PC1  + covariates
  Model 1b: pheno ~ geno_norm3 + covariates
  Model 2a: pheno ~ emb_PC1   + covariates
  Model 2b: pheno ~ emb_norm3  + covariates

Also runs the reduced model (covariates only) once to get incremental R².

Prints a 2x2 comparison table per subblock. No files saved.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = "data/region_blocks"
EMB_BLOCK_NPY = "results/output_regions2/ORD/embeddings/block_contextual_repr.npy"
ATTN_CSV      = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
BLOCK_ORDER   = "results/output_regions/block_order.csv"
EIGENVEC_FILE = "metadata/ldpruned_997subs.eigenvec"
PHENO_FILE    = "metadata/COS_TRIO_pheno_1165.csv"

SUBBLOCKS  = [f"region_17q21_core_sb{i}" for i in range(1, 6)]
PHENO      = "pctpred_fev1_pre_BD"
PLINK_META = {"FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"}
PC_COLS    = [f"PC{i}" for i in range(1, 11)]
COVARS     = PC_COLS + ["gender_num", "smkexp_current"]
N_PCS      = 3   # for norm3


# ── helpers ───────────────────────────────────────────────────────────────────
def load_block_names(path, expected_B):
    bo  = pd.read_csv(path)
    col = next((c for c in ["block","block_id","region","name"]
                if c in bo.columns), None)
    if col is None:
        raise ValueError(f"No block-name column in {path}")
    names = bo[col].astype(str).tolist()
    if len(names) != expected_B:
        raise ValueError(f"block_order {len(names)} != {expected_B}")
    return names


def load_raw_plink(path):
    df = pd.read_csv(path, sep=r"\s+", low_memory=False)
    df["IID"] = df["IID"].astype(str)
    snp_cols  = [c for c in df.columns if c not in PLINK_META]
    return df[["IID"] + snp_cols].copy(), snp_cols


def load_eigenvec(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lstrip("#")
    df = df.rename(columns={"FID": "FID_ev"})
    df["IID"] = df["IID"].astype(str)
    return df[["IID"] + PC_COLS]


def top_k_pcs(matrix, k):
    """PCA on (N, S) matrix — returns (N, k) scores."""
    X = SimpleImputer(strategy="mean").fit_transform(matrix.astype(float))
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=k, random_state=42)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def compute_norm3(pcs):
    """L2 norm of top-k PC scores, z-scored. Shape (N,) → (N,)."""
    norm = np.sqrt((pcs ** 2).sum(axis=1))
    return (norm - norm.mean()) / (norm.std() + 1e-12)


def sign_align(emb, geno):
    r, _ = stats.spearmanr(emb, geno)
    return emb * (-1 if r < 0 else 1)


def run_ols(formula, data):
    try:
        sub = data.dropna()
        if len(sub) < 20:
            return None
        return smf.ols(formula, data=sub).fit()
    except Exception:
        return None


def extract(res, predictor):
    if res is None:
        return dict(beta=np.nan, pval=np.nan, r2=np.nan, n=np.nan)
    return dict(
        beta = float(res.params.get(predictor, np.nan)),
        pval = float(res.pvalues.get(predictor, np.nan)),
        r2   = float(res.rsquared),
        n    = int(len(res.model.data.frame.dropna())),
    )


def sig_stars(p):
    if np.isnan(p): return "  "
    if p < 0.001:   return "***"
    if p < 0.01:    return "** "
    if p < 0.05:    return "*  "
    return "   "


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # ── load shared data ──────────────────────────────────────────────────────
    print("Loading data …")
    emb_block   = np.load(EMB_BLOCK_NPY)
    N, B, D     = emb_block.shape
    attn_df     = pd.read_csv(ATTN_CSV)
    subj_iids   = attn_df["IID"].astype(str).values
    block_names = load_block_names(BLOCK_ORDER, B)
    iid_to_idx  = {iid: i for i, iid in enumerate(subj_iids)}

    pcs_df = load_eigenvec(EIGENVEC_FILE)

    pheno_raw = pd.read_csv(PHENO_FILE, low_memory=False)
    pheno_raw = pheno_raw.rename(columns={"S_SUBJECTID": "IID"})
    pheno_raw["IID"] = pheno_raw["IID"].astype(str)

    if "gender" in pheno_raw.columns:
        s = pheno_raw["gender"].astype(str).str.strip().str.lower()
        pheno_raw["gender_num"] = s.map(
            {"male":1.,"m":1.,"1":1.,"female":0.,"f":0.,"2":0.}
        )

    keep = ["IID", PHENO] + [c for c in ["gender_num","smkexp_current","age"]
                              if c in pheno_raw.columns]
    pheno_df = pheno_raw[keep].copy()

    master = (
        pd.DataFrame({"IID": subj_iids})
        .merge(pcs_df,   on="IID", how="left")
        .merge(pheno_df, on="IID", how="left")
    )

    covar_cols  = [c for c in COVARS if c in master.columns]
    covar_str   = " + ".join(covar_cols)
    safe_pheno  = PHENO.replace("-","_")
    master[safe_pheno] = master[PHENO]

    # reduced model R² (covariates only) — same for all blocks
    res_reduced = run_ols(f"{safe_pheno} ~ {covar_str}", master)
    r2_reduced  = res_reduced.rsquared if res_reduced else np.nan
    n_reduced   = len(res_reduced.model.data.frame.dropna()) if res_reduced else 0
    print(f"  Reduced model R² (covariates only): {r2_reduced:.6f}  n={n_reduced}")
    print(f"  Phenotype: {PHENO}")
    print(f"  Covariates: {covar_str}\n")

    # ── per-subblock loop ─────────────────────────────────────────────────────
    # collect rows for final summary table
    summary_rows = []

    for sb_name in SUBBLOCKS:
        raw_path = Path(RAW_DIR) / f"{sb_name}.raw"
        sb_short = sb_name.replace("region_17q21_core_", "").upper()

        print(f"{'━'*72}")
        print(f"  {sb_name}")
        print(f"{'━'*72}")

        if not raw_path.exists():
            print(f"  [SKIP] raw file not found")
            continue

        try:
            b_idx = block_names.index(sb_name)
        except ValueError:
            print(f"  [SKIP] not in block_order")
            continue

        # load genotypes
        raw_df, snp_cols = load_raw_plink(raw_path)
        common_iids = [iid for iid in raw_df["IID"] if iid in iid_to_idx]
        if len(common_iids) < 50:
            print(f"  [SKIP] only {len(common_iids)} matched subjects")
            continue

        raw_aligned = raw_df.set_index("IID").loc[common_iids]
        emb_rows    = [iid_to_idx[iid] for iid in common_iids]

        # ── genotype summaries ────────────────────────────────────────────────
        dosage = raw_aligned[snp_cols].values.astype(float)
        dosage[dosage == -9] = np.nan

        k = min(N_PCS, len(snp_cols))
        geno_pcs, geno_var = top_k_pcs(dosage, k)
        geno_pc1_vec  = geno_pcs[:, 0]
        geno_norm3_vec = compute_norm3(geno_pcs)

        print(f"  Genotype: {len(snp_cols)} SNPs  "
              f"PC1 var={100*geno_var[0]:.1f}%  "
              f"PC1-3 cum={100*geno_var.sum():.1f}%")

        # ── embedding summaries ───────────────────────────────────────────────
        emb_sub = emb_block[emb_rows, b_idx, :]   # (n_common, 64)
        emb_sub_scaled = StandardScaler().fit_transform(emb_sub)
        emb_pca  = PCA(n_components=N_PCS, random_state=42)
        emb_pcs_arr = emb_pca.fit_transform(emb_sub_scaled)
        emb_var  = emb_pca.explained_variance_ratio_

        emb_pc1_vec   = sign_align(emb_pcs_arr[:, 0], geno_pc1_vec)
        emb_norm3_vec = compute_norm3(emb_pcs_arr)

        print(f"  Embedding: PC1 var={100*emb_var[0]:.1f}%  "
              f"PC1-3 cum={100*emb_var.sum():.1f}%")

        # genotype ↔ embedding alignment check
        r_pc1, _   = stats.spearmanr(geno_pc1_vec, emb_pc1_vec)
        r_norm3, _ = stats.spearmanr(geno_norm3_vec, emb_norm3_vec)
        print(f"  Geno ↔ Emb alignment:  PC1 r={r_pc1:+.3f}  norm3 r={r_norm3:+.3f}")

        # ── build regression dataframe ────────────────────────────────────────
        # subset master to common_iids
        master_sub = master.set_index("IID").loc[common_iids].reset_index()
        master_sub["geno_pc1"]   = geno_pc1_vec
        master_sub["geno_norm3"] = geno_norm3_vec
        master_sub["emb_pc1"]    = emb_pc1_vec
        master_sub["emb_norm3"]  = emb_norm3_vec

        # ── run four models ───────────────────────────────────────────────────
        models = {
            "geno_PC1":   ("geno_pc1",   f"{safe_pheno} ~ geno_pc1   + {covar_str}"),
            "geno_norm3": ("geno_norm3", f"{safe_pheno} ~ geno_norm3 + {covar_str}"),
            "emb_PC1":    ("emb_pc1",    f"{safe_pheno} ~ emb_pc1    + {covar_str}"),
            "emb_norm3":  ("emb_norm3",  f"{safe_pheno} ~ emb_norm3  + {covar_str}"),
        }

        results = {}
        for label, (pred, formula) in models.items():
            res = run_ols(formula, master_sub)
            ex  = extract(res, pred)
            ex["incr_r2"] = ex["r2"] - r2_reduced if not np.isnan(ex["r2"]) else np.nan
            results[label] = ex

        # ── print 2×2 table ───────────────────────────────────────────────────
        print(f"\n  {'':20s} {'geno_PC1':>12s} {'geno_norm3':>12s} "
              f"{'emb_PC1':>12s} {'emb_norm3':>12s}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for metric, fmt in [("beta", "{:>+12.4f}"), ("pval", "{:>12.4g}"),
                             ("incr_r2", "{:>12.5f}"), ("n", "{:>12.0f}")]:
            row_str = f"  {metric:<20s}"
            for label in ["geno_PC1", "geno_norm3", "emb_PC1", "emb_norm3"]:
                val = results[label][metric]
                row_str += fmt.format(val) if not np.isnan(val) else f"{'NA':>12s}"
            print(row_str)

        # significance stars row
        star_str = f"  {'sig':20s}"
        for label in ["geno_PC1", "geno_norm3", "emb_PC1", "emb_norm3"]:
            stars = sig_stars(results[label]["pval"])
            star_str += f"{stars:>12s}"
        print(star_str)

        # ── which model wins? ─────────────────────────────────────────────────
        best_label = min(results, key=lambda x: results[x]["pval"]
                         if not np.isnan(results[x]["pval"]) else 1.0)
        best_p     = results[best_label]["pval"]
        best_incr  = results[best_label]["incr_r2"]

        print(f"\n  Best model: {best_label}  "
              f"p={best_p:.4g}  incr_R²={best_incr:.5f} "
              f"({100*best_incr:.3f}%)")

        # emb vs geno comparison at same summarisation level
        for sfx in ["PC1", "norm3"]:
            p_g = results[f"geno_{sfx}"]["pval"]
            p_e = results[f"emb_{sfx}"]["pval"]
            if not (np.isnan(p_g) or np.isnan(p_e)):
                delta = (-np.log10(p_e + 1e-300)) - (-np.log10(p_g + 1e-300))
                direction = "emb better" if delta > 0 else "geno better"
                print(f"  {sfx}: emb p={p_e:.4g}  geno p={p_g:.4g}  "
                      f"Δ-log10p={delta:+.3f}  → {direction}")

        summary_rows.append({
            "subblock":          sb_short,
            "geno_pc1_beta":     results["geno_PC1"]["beta"],
            "geno_pc1_p":        results["geno_PC1"]["pval"],
            "geno_pc1_incr_r2":  results["geno_PC1"]["incr_r2"],
            "geno_norm3_beta":   results["geno_norm3"]["beta"],
            "geno_norm3_p":      results["geno_norm3"]["pval"],
            "geno_norm3_incr_r2":results["geno_norm3"]["incr_r2"],
            "emb_pc1_beta":      results["emb_PC1"]["beta"],
            "emb_pc1_p":         results["emb_PC1"]["pval"],
            "emb_pc1_incr_r2":   results["emb_PC1"]["incr_r2"],
            "emb_norm3_beta":    results["emb_norm3"]["beta"],
            "emb_norm3_p":       results["emb_norm3"]["pval"],
            "emb_norm3_incr_r2": results["emb_norm3"]["incr_r2"],
            "best_model":        best_label,
            "geno_emb_r_pc1":    round(float(r_pc1), 4),
            "geno_emb_r_norm3":  round(float(r_norm3), 4),
        })

    # ── final summary across all subblocks ────────────────────────────────────
    if not summary_rows:
        print("\nNo results to summarise.")
        return

    print(f"\n{'═'*72}")
    print(f"  SUMMARY: {PHENO}  —  genotype vs embedding comparison")
    print(f"  Reduced model R² (covariates only): {r2_reduced:.5f}")
    print(f"{'═'*72}")
    print(f"  {'SB':<6s} {'geno_PC1_p':>12s} {'emb_PC1_p':>12s} "
          f"{'Δ-log10p':>10s} {'geno_n3_p':>12s} {'emb_n3_p':>12s} "
          f"{'Δ-log10p':>10s} {'best':>12s}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} "
          f"{'-'*12} {'-'*12} {'-'*10} {'-'*12}")

    for row in summary_rows:
        p_gp1 = row["geno_pc1_p"]
        p_ep1 = row["emb_pc1_p"]
        p_gn3 = row["geno_norm3_p"]
        p_en3 = row["emb_norm3_p"]

        d_pc1  = ((-np.log10(p_ep1 + 1e-300)) -
                  (-np.log10(p_gp1 + 1e-300))) if not (np.isnan(p_ep1) or np.isnan(p_gp1)) else np.nan
        d_norm3 = ((-np.log10(p_en3 + 1e-300)) -
                   (-np.log10(p_gn3 + 1e-300))) if not (np.isnan(p_en3) or np.isnan(p_gn3)) else np.nan

        print(
            f"  {row['subblock']:<6s} "
            f"{p_gp1:>12.4g} {p_ep1:>12.4g} {d_pc1:>+10.3f} "
            f"{p_gn3:>12.4g} {p_en3:>12.4g} {d_norm3:>+10.3f} "
            f"{row['best_model']:>12s}"
        )

    # incremental R² comparison
    print(f"\n  Incremental R² (beyond covariates):")
    print(f"  {'SB':<6s} {'geno_PC1':>10s} {'emb_PC1':>10s} "
          f"{'geno_norm3':>12s} {'emb_norm3':>12s}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
    for row in summary_rows:
        print(
            f"  {row['subblock']:<6s} "
            f"{row['geno_pc1_incr_r2']:>10.5f} "
            f"{row['emb_pc1_incr_r2']:>10.5f} "
            f"{row['geno_norm3_incr_r2']:>12.5f} "
            f"{row['emb_norm3_incr_r2']:>12.5f}"
        )

    print(f"\n{'═'*72}")


if __name__ == "__main__":
    main()