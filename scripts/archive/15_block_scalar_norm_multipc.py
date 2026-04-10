#!/usr/bin/env python3
"""
block_scalar_norm_multipc.py

Tests phenotype association of block-level embeddings using a SCALAR SUMMARY
of the top-3 block PCs (L2 norm of PC1+PC2+PC3 scores per subject), keeping
the same single-β regression structure as the original PC1-only analysis.

For each block the scalar summary is:
    norm3 = sqrt(bPC1² + bPC2² + bPC3²)   (per subject, standardised)

This is then compared directly against the original PC1-only result.

For each (block × phenotype):
  model_unadj : pheno ~ norm3
  model_adj   : pheno ~ norm3 + gPC1 + … + gPC10 + covars

Prints top 20 hits per phenotype (unadj and adj), plus a side-by-side
comparison with the PC1-only baseline for the same block/phenotype.
No files are saved.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
EMB_BLOCK_NPY = "results/output_regions2/ORD/embeddings/block_contextual_repr.npy"
ATTN_CSV      = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
BLOCK_ORDER   = "results/output_regions/block_order.csv"
EIGENVEC_FILE = "metadata/ldpruned_997subs.eigenvec"
PHENO_FILE    = "metadata/COS_TRIO_pheno_1165.csv"

PC_COLS = [f"PC{i}" for i in range(1, 11)]

CONTINUOUS_PHENOS  = ["G19B", "log10eos", "pctpred_fev1_pre_BD"]
CATEGORICAL_PHENOS = ["G20D"]
ALL_PHENOS         = CONTINUOUS_PHENOS + CATEGORICAL_PHENOS

N_BLOCK_PCS = 3


# ── helpers ───────────────────────────────────────────────────────────────────
def load_eigenvec(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lstrip("#")
    df = df.rename(columns={"FID": "FID_ev"})
    df["IID"] = df["IID"].astype(str)
    return df[["IID"] + PC_COLS]


def load_block_names(block_order_path: str, expected_B: int):
    bo = pd.read_csv(block_order_path)
    possible_cols = ["block", "block_id", "region", "name"]
    name_col = next((c for c in possible_cols if c in bo.columns), None)
    if name_col is None:
        raise ValueError(f"No block-name column in {block_order_path}")
    block_names = bo[name_col].astype(str).tolist()
    if len(block_names) != expected_B:
        raise ValueError(f"block_order length {len(block_names)} != {expected_B}")
    return block_names


def recode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Hospitalized_Asthma_Last_Yr", "smkexp_current"]:
        if col not in df.columns:
            continue
        mask = df[col].isin([1.0, 2.0])
        df.loc[~mask, col] = np.nan
        df[col] = df[col].map({1.0: 0.0, 2.0: 1.0})
    if "G20D" in df.columns:
        df["G20D"] = pd.to_numeric(df["G20D"], errors="coerce")
        df["G20D"] = (df["G20D"] > 0).astype(float)
    return df


def normalize_gender(series: pd.Series) -> pd.Series:
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str).str.strip().str.lower()
    return s.map({"male": 1.0, "m": 1.0, "1": 1.0, "female": 0.0, "f": 0.0, "2": 0.0})


def get_covars(pheno: str, available_cols) -> list:
    covars = PC_COLS.copy()
    if pheno in ["log10eos", "G20D", "Hospitalized_Asthma_Last_Yr"]:
        covars += ["age", "gender_num"]
    elif pheno in ["G19B", "BMI", "age"]:
        covars += ["gender_num"]
    elif pheno == "pctpred_fev1_pre_BD":
        covars += ["gender_num", "smkexp_current"]
    return [c for c in covars if c in available_cols]


def block_top_k_pcs(block_repr: np.ndarray, block_idx: int, k: int) -> np.ndarray:
    """Return top-k PCs of (N, 64) block representation. Shape: (N, k)."""
    X = block_repr[:, block_idx, :]
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=k, random_state=42)
    return pca.fit_transform(X)   # (N, k)


def compute_norm3(pcs: np.ndarray) -> np.ndarray:
    """
    L2 norm across top-k PC scores per subject.
    pcs: (N, k)  → returns (N,), then z-scored so it's on unit scale.
    """
    norm = np.sqrt((pcs ** 2).sum(axis=1))   # (N,)
    # z-score so β is on a comparable scale to the original PC1-only analysis
    norm = (norm - norm.mean()) / (norm.std() + 1e-12)
    return norm


def run_ols(formula: str, data: pd.DataFrame):
    try:
        sub = data.dropna()
        if len(sub) < 20:
            return None
        return smf.ols(formula, data=sub).fit()
    except Exception:
        return None


def safe_logit(formula: str, data: pd.DataFrame):
    try:
        sub = data.dropna()
        if sub[formula.split("~")[0].strip()].nunique() < 2:
            return None
        if len(sub) < 20:
            return None
        return smf.logit(formula, data=sub).fit(disp=False, maxiter=200)
    except Exception:
        return None


def extract_result(res, predictor: str) -> dict:
    if res is None:
        return dict(beta=np.nan, pval=np.nan, r2=np.nan, n=np.nan)
    b  = res.params.get(predictor, np.nan)
    pv = res.pvalues.get(predictor, np.nan)
    r2 = getattr(res, "rsquared", None)
    if r2 is None:
        r2 = getattr(res, "prsquared", np.nan)
    return dict(
        beta=float(b), pval=float(pv),
        r2=float(r2) if pd.notna(r2) else np.nan,
        n=len(res.model.data.frame)
    )


def add_fdr(df: pd.DataFrame, p_col: str, out_col: str) -> pd.DataFrame:
    df = df.copy()
    df[out_col] = np.nan
    for pheno, idx in df.groupby("phenotype").groups.items():
        pvals = df.loc[idx, p_col].astype(float)
        mask  = pvals.notna()
        if mask.sum() == 0:
            continue
        _, qvals, _, _ = multipletests(pvals[mask], method="fdr_bh")
        df.loc[pvals[mask].index, out_col] = qvals
    return df


def print_top_hits(assoc_df: pd.DataFrame, p_col: str, fdr_col: str,
                   beta_col: str, label: str, top_n: int = 20):
    print(f"\n{'═'*80}")
    print(f"  {label}")
    print(f"{'═'*80}")
    phenos = sorted(assoc_df["phenotype"].dropna().unique())
    for pheno in phenos:
        sub = (
            assoc_df[assoc_df["phenotype"] == pheno]
            .dropna(subset=[p_col])
            .sort_values(p_col)
            .head(top_n)
        )
        print(f"\n  ── {pheno} (top {top_n}) ──")
        print(f"  {'block':<50s} {'group':<10s} {'β_norm3':>9s} "
              f"{'p':>10s} {'FDR':>10s} {'r2':>8s} {'n':>5s}")
        print(f"  {'-'*50} {'-'*10} {'-'*9} {'-'*10} {'-'*10} {'-'*8} {'-'*5}")
        for _, row in sub.iterrows():
            fdr_v = row.get(fdr_col, np.nan)
            fdr_s = f"{fdr_v:.4g}" if pd.notna(fdr_v) else "NA"
            sig   = "**" if pd.notna(fdr_v) and fdr_v < 0.05 else (
                    "*"  if pd.notna(fdr_v) and fdr_v < 0.10 else "  ")
            print(
                f"  {row['block']:<50s} "
                f"{row['block_group']:<10s} "
                f"{row[beta_col]:>+9.4f} "
                f"{row[p_col]:>10.4g} "
                f"{fdr_s:>10s} "
                f"{row['r2_norm3']:>8.4f} "
                f"{int(row['n']):>5d} "
                f"{sig}"
            )


def print_comparison(assoc_df: pd.DataFrame, p_col_norm: str, p_col_pc1: str,
                     b_col_norm: str, b_col_pc1: str, label: str, top_n: int = 20):
    """Side-by-side norm3 vs PC1-only for the top hits (by norm3 p-value)."""
    print(f"\n{'═'*90}")
    print(f"  COMPARISON: norm3 vs PC1-only  |  {label}")
    print(f"{'═'*90}")
    phenos = sorted(assoc_df["phenotype"].dropna().unique())
    for pheno in phenos:
        sub = (
            assoc_df[assoc_df["phenotype"] == pheno]
            .dropna(subset=[p_col_norm])
            .sort_values(p_col_norm)
            .head(top_n)
        )
        print(f"\n  ── {pheno} ──")
        print(f"  {'block':<50s} {'group':<10s} "
              f"{'β_norm3':>9s} {'p_norm3':>10s}   "
              f"{'β_PC1':>9s} {'p_PC1':>10s}   {'Δ-logp':>8s}")
        print(f"  {'-'*50} {'-'*10} {'-'*9} {'-'*10}   {'-'*9} {'-'*10}   {'-'*8}")
        for _, row in sub.iterrows():
            p_n  = row[p_col_norm]
            p_p1 = row[p_col_pc1]
            delta = (
                (-np.log10(p_n + 1e-300)) - (-np.log10(p_p1 + 1e-300))
                if pd.notna(p_n) and pd.notna(p_p1) else np.nan
            )
            direction = "↑" if pd.notna(delta) and delta > 0 else ("↓" if pd.notna(delta) else " ")
            print(
                f"  {row['block']:<50s} "
                f"{row['block_group']:<10s} "
                f"{row[b_col_norm]:>+9.4f} "
                f"{p_n:>10.4g}   "
                f"{row[b_col_pc1]:>+9.4f} "
                f"{p_p1:>10.4g}   "
                f"{delta:>+7.2f} {direction}"
            )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load data
    print("Loading embeddings …")
    emb_block = np.load(EMB_BLOCK_NPY)
    N, B, D   = emb_block.shape
    print(f"  block repr: {emb_block.shape}  ({B} blocks × {D} dims)")

    attn_df     = pd.read_csv(ATTN_CSV)
    subj_iids   = attn_df["IID"].astype(str).values
    block_names = load_block_names(BLOCK_ORDER, B)

    print("Loading genotype PCs …")
    pcs_df = load_eigenvec(EIGENVEC_FILE)

    print("Loading phenotypes …")
    pheno_raw = pd.read_csv(PHENO_FILE, low_memory=False)
    pheno_raw = pheno_raw.rename(columns={"S_SUBJECTID": "IID"})
    pheno_raw["IID"] = pheno_raw["IID"].astype(str)
    pheno_raw = recode_categorical(pheno_raw)
    if "gender" in pheno_raw.columns:
        pheno_raw["gender_num"] = normalize_gender(pheno_raw["gender"])

    keep = ["IID"] + [c for c in ALL_PHENOS if c in pheno_raw.columns]
    for c in ["age", "gender_num", "smkexp_current"]:
        if c in pheno_raw.columns:
            keep.append(c)
    pheno_df = pheno_raw[list(dict.fromkeys(keep))].copy()

    master = (
        pd.DataFrame({"IID": subj_iids})
        .merge(pcs_df,   on="IID", how="left")
        .merge(pheno_df, on="IID", how="left")
    )

    n_pc_ok = master[PC_COLS].notna().all(axis=1).sum()
    print(f"  subjects with PCs: {n_pc_ok}/{N}")
    for ph in ALL_PHENOS:
        if ph in master.columns:
            print(f"    {ph:<35s}  n={master[ph].notna().sum()}")

    # 2. Extract PC1 and norm3 per block
    print(f"\nExtracting PC1 and norm(PC1-3) for all {B} blocks …")
    pc1_arr   = np.zeros((N, B), dtype=np.float32)
    norm3_arr = np.zeros((N, B), dtype=np.float32)

    for b in range(B):
        pcs           = block_top_k_pcs(emb_block, b, N_BLOCK_PCS)
        pc1_arr[:, b] = pcs[:, 0]
        norm3_arr[:, b] = compute_norm3(pcs)
    print("  done")

    # 3. Regressions
    pheno_cols = [p for p in ALL_PHENOS if p in master.columns]
    print(f"\nRunning regressions ({B} blocks × {len(pheno_cols)} phenotypes) …")

    def get_block_group(name: str) -> str:
        return "control" if "control_" in name.lower() else "asthma"

    rows = []

    for b_idx, bname in enumerate(block_names):
        reg_df = master.copy()
        reg_df["bPC1"]  = pc1_arr[:, b_idx]
        reg_df["norm3"] = norm3_arr[:, b_idx]
        bgroup = get_block_group(bname)

        for pheno in pheno_cols:
            safe_p  = pheno.replace("-", "_")
            reg_df[safe_p] = reg_df[pheno]
            is_cat  = pheno in CATEGORICAL_PHENOS
            fit_fn  = safe_logit if is_cat else run_ols
            covars  = get_covars(pheno, reg_df.columns)
            covar_f = " + ".join(covars)

            # norm3 models
            res_nu = fit_fn(f"{safe_p} ~ norm3", reg_df)
            res_na = fit_fn(f"{safe_p} ~ norm3 + {covar_f}", reg_df) if covar_f else res_nu

            # PC1-only baseline (for comparison)
            res_p1u = fit_fn(f"{safe_p} ~ bPC1", reg_df)
            res_p1a = fit_fn(f"{safe_p} ~ bPC1 + {covar_f}", reg_df) if covar_f else res_p1u

            eu  = extract_result(res_nu,  "norm3")
            ea  = extract_result(res_na,  "norm3")
            ep1u = extract_result(res_p1u, "bPC1")
            ep1a = extract_result(res_p1a, "bPC1")

            rows.append({
                "block": bname,
                "block_group": bgroup,
                "phenotype": pheno,
                # norm3 results
                "beta_norm3_unadj": eu["beta"],
                "pval_norm3_unadj": eu["pval"],
                "r2_norm3": eu["r2"],
                "n": eu["n"],
                "beta_norm3_adj": ea["beta"],
                "pval_norm3_adj": ea["pval"],
                "r2_norm3_adj": ea["r2"],
                "n_adj": ea["n"],
                # PC1-only baseline
                "beta_pc1_unadj": ep1u["beta"],
                "pval_pc1_unadj": ep1u["pval"],
                "beta_pc1_adj": ep1a["beta"],
                "pval_pc1_adj": ep1a["pval"],
                "covars_adj": ";".join(covars),
            })

        if (b_idx + 1) % 20 == 0:
            print(f"  block {b_idx+1}/{B} …")

    assoc_df = pd.DataFrame(rows)

    assoc_df = add_fdr(assoc_df, "pval_norm3_unadj", "FDR_norm3_unadj")
    assoc_df = add_fdr(assoc_df, "pval_norm3_adj",   "FDR_norm3_adj")
    assoc_df = add_fdr(assoc_df, "pval_pc1_unadj",   "FDR_pc1_unadj")
    assoc_df = add_fdr(assoc_df, "pval_pc1_adj",     "FDR_pc1_adj")

    # 4. Print results
    print_top_hits(assoc_df, "pval_norm3_unadj", "FDR_norm3_unadj", "beta_norm3_unadj",
                   "Scalar norm3  |  UNADJUSTED")
    print_top_hits(assoc_df, "pval_norm3_adj",   "FDR_norm3_adj",   "beta_norm3_adj",
                   "Scalar norm3  |  ADJUSTED (PC1-10 + covars)")

    print_comparison(assoc_df,
                     p_col_norm="pval_norm3_unadj", p_col_pc1="pval_pc1_unadj",
                     b_col_norm="beta_norm3_unadj", b_col_pc1="beta_pc1_unadj",
                     label="UNADJUSTED")
    print_comparison(assoc_df,
                     p_col_norm="pval_norm3_adj",   p_col_pc1="pval_pc1_adj",
                     b_col_norm="beta_norm3_adj",   b_col_pc1="beta_pc1_adj",
                     label="ADJUSTED")

    # 5. Summary
    alpha = 0.05
    print(f"\n{'═'*80}")
    print("  Summary — norm3 vs PC1-only")
    print(f"{'═'*80}")
    for label, p_n, p_p1, fdr_n, fdr_p1 in [
        ("Unadjusted",
         "pval_norm3_unadj", "pval_pc1_unadj",
         "FDR_norm3_unadj",  "FDR_pc1_unadj"),
        ("Adjusted",
         "pval_norm3_adj",   "pval_pc1_adj",
         "FDR_norm3_adj",    "FDR_pc1_adj"),
    ]:
        nom_n  = (assoc_df[p_n]   < alpha).sum()
        nom_p1 = (assoc_df[p_p1]  < alpha).sum()
        fdr_nv = (assoc_df[fdr_n] < alpha).sum()
        fdr_p1v= (assoc_df[fdr_p1]< alpha).sum()
        print(f"\n  {label}")
        print(f"    norm3  : nominal p<0.05 = {nom_n:4d}   FDR<0.05 = {fdr_nv:4d}")
        print(f"    PC1    : nominal p<0.05 = {nom_p1:4d}   FDR<0.05 = {fdr_p1v:4d}")

    # Blocks where norm3 clearly beats PC1 (Δ-log10p > 1)
    assoc_df["delta_logp_unadj"] = (
        -np.log10(assoc_df["pval_norm3_unadj"].clip(lower=1e-300)) -
        (-np.log10(assoc_df["pval_pc1_unadj"].clip(lower=1e-300)))
    )
    assoc_df["delta_logp_adj"] = (
        -np.log10(assoc_df["pval_norm3_adj"].clip(lower=1e-300)) -
        (-np.log10(assoc_df["pval_pc1_adj"].clip(lower=1e-300)))
    )

    improved = assoc_df[assoc_df["delta_logp_adj"] > 1].sort_values("delta_logp_adj", ascending=False)
    if len(improved) > 0:
        print(f"\n  Blocks where norm3 gains >1 -log10(p) over PC1 (adjusted):")
        print(f"  {'block':<50s} {'phenotype':<25s} {'Δ-log10p':>10s}")
        for _, row in improved.head(20).iterrows():
            print(f"  {row['block']:<50s} {row['phenotype']:<25s} {row['delta_logp_adj']:>+10.3f}")
    else:
        print("\n  No blocks with >1 -log10(p) gain from norm3 over PC1 (adjusted).")


if __name__ == "__main__":
    main()