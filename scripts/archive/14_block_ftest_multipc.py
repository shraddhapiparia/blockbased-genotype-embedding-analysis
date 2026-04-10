#!/usr/bin/env python3
"""
block_ftest_multipc.py

Tests phenotype association of block-level embeddings using JOINT F-test / LRT
across block PC1+PC2+PC3 (rather than PC1 alone).

For each (block × phenotype):
  - Unadjusted:  full model  (pheno ~ bPC1 + bPC2 + bPC3)
                 vs null     (pheno ~ 1)
  - Adjusted:    full model  (pheno ~ bPC1 + bPC2 + bPC3 + covars)
                 vs reduced  (pheno ~ covars)
  → F-test (OLS) or LRT (logit) gives one p-value per block per phenotype

Prints top 20 hits per phenotype (unadjusted and adjusted), sorted by p-value.
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

# ── paths (mirrored from main script) ─────────────────────────────────────────
EMB_BLOCK_NPY = "results/output_regions2/ORD/embeddings/block_contextual_repr.npy"
ATTN_CSV      = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
BLOCK_ORDER   = "results/output_regions/block_order.csv"
EIGENVEC_FILE = "metadata/ldpruned_997subs.eigenvec"
PHENO_FILE    = "metadata/COS_TRIO_pheno_1165.csv"

PC_COLS = [f"PC{i}" for i in range(1, 11)]

CONTINUOUS_PHENOS  = ["G19B", "log10eos", "pctpred_fev1_pre_BD"]
CATEGORICAL_PHENOS = ["G20D"]
ALL_PHENOS         = CONTINUOUS_PHENOS + CATEGORICAL_PHENOS

N_BLOCK_PCS = 3   # PC1, PC2, PC3 extracted per block


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
    return pca.fit_transform(X)          # (N, k)


def ftest_ols(full_formula: str, reduced_formula: str, data: pd.DataFrame):
    """
    F-test comparing full vs reduced OLS model.
    Returns (f_stat, p_value, df_num, df_den, n).
    """
    try:
        sub = data.dropna()
        if len(sub) < 20:
            return None
        full    = smf.ols(full_formula,    data=sub).fit()
        reduced = smf.ols(reduced_formula, data=sub).fit()
        # manual F-test: (RSS_r - RSS_f) / q  /  (RSS_f / df_res_f)
        rss_f = full.ssr
        rss_r = reduced.ssr
        df_num = full.df_model - reduced.df_model
        if df_num <= 0:
            return None
        df_den = full.df_resid
        f_stat = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
        p_val  = 1 - stats.f.cdf(f_stat, df_num, df_den)
        return dict(stat=float(f_stat), pval=float(p_val),
                    df_num=df_num, df_den=df_den, n=len(sub),
                    r2_full=float(full.rsquared),
                    r2_reduced=float(reduced.rsquared))
    except Exception:
        return None


def lrt_logit(full_formula: str, reduced_formula: str, data: pd.DataFrame):
    """
    Likelihood-ratio test comparing full vs reduced logit model.
    Returns dict with chi2, pval, df, n.
    """
    try:
        sub = data.dropna()
        if sub[full_formula.split("~")[0].strip()].nunique() < 2:
            return None
        if len(sub) < 20:
            return None
        full    = smf.logit(full_formula,    data=sub).fit(disp=False, maxiter=200)
        reduced = smf.logit(reduced_formula, data=sub).fit(disp=False, maxiter=200)
        chi2  = 2 * (full.llf - reduced.llf)
        df    = full.df_model - reduced.df_model
        if df <= 0:
            return None
        p_val = 1 - stats.chi2.cdf(chi2, df)
        return dict(stat=float(chi2), pval=float(p_val),
                    df_num=df, df_den=None, n=len(sub),
                    r2_full=float(full.prsquared),
                    r2_reduced=float(reduced.prsquared))
    except Exception:
        return None


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
                   label: str, top_n: int = 20):
    print(f"\n{'═'*78}")
    print(f"  {label}")
    print(f"{'═'*78}")

    phenos = sorted(assoc_df["phenotype"].dropna().unique())
    for pheno in phenos:
        sub = (
            assoc_df[assoc_df["phenotype"] == pheno]
            .dropna(subset=[p_col])
            .sort_values(p_col)
            .head(top_n)
        )
        print(f"\n  ── {pheno} (top {top_n}) ──")
        print(f"  {'block':<50s} {'group':<10s} {'stat':>8s} {'df_num':>6s} "
              f"{'n':>5s} {'p':>10s} {'FDR':>10s} {'r2_full':>8s}")
        print(f"  {'-'*50} {'-'*10} {'-'*8} {'-'*6} {'-'*5} {'-'*10} {'-'*10} {'-'*8}")
        for _, row in sub.iterrows():
            fdr_v = row.get(fdr_col, np.nan)
            fdr_s = f"{fdr_v:.4g}" if pd.notna(fdr_v) else "NA"
            sig   = "**" if pd.notna(fdr_v) and fdr_v < 0.05 else (
                    "*"  if pd.notna(fdr_v) and fdr_v < 0.10 else "  ")
            print(
                f"  {row['block']:<50s} "
                f"{row['block_group']:<10s} "
                f"{row['stat']:>8.3f} "
                f"{int(row['df_num']):>6d} "
                f"{int(row['n']):>5d} "
                f"{row[p_col]:>10.4g} "
                f"{fdr_s:>10s} "
                f"{row['r2_full']:>8.4f} "
                f"{sig}"
            )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load data
    print("Loading embeddings …")
    emb_block = np.load(EMB_BLOCK_NPY)
    N, B, D   = emb_block.shape
    print(f"  block repr: {emb_block.shape}  ({B} blocks × {D} dims)")

    attn_df   = pd.read_csv(ATTN_CSV)
    subj_iids = attn_df["IID"].astype(str).values
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

    # 2. Extract top-K PCs per block
    print(f"\nExtracting top {N_BLOCK_PCS} PCs for all {B} blocks …")
    block_pcs = np.zeros((N, B, N_BLOCK_PCS), dtype=np.float32)
    for b in range(B):
        block_pcs[:, b, :] = block_top_k_pcs(emb_block, b, N_BLOCK_PCS)
    print("  done")

    # 3. Run regressions
    pheno_cols = [p for p in ALL_PHENOS if p in master.columns]
    print(f"\nRunning F-test / LRT regressions ({B} blocks × {len(pheno_cols)} phenotypes) …")

    def get_block_group(name: str) -> str:
        return "control" if "control_" in name.lower() else "asthma"

    rows_u, rows_a = [], []

    for b_idx, bname in enumerate(block_names):
        reg_df = master.copy()

        # Add block PC columns with formula-safe names
        bp_cols = []
        for k in range(N_BLOCK_PCS):
            col = f"bPC{k+1}"
            reg_df[col] = block_pcs[:, b_idx, k]
            bp_cols.append(col)

        bp_formula = " + ".join(bp_cols)   # bPC1 + bPC2 + bPC3
        bgroup     = get_block_group(bname)

        for pheno in pheno_cols:
            safe_p  = pheno.replace("-", "_")
            reg_df[safe_p] = reg_df[pheno]
            is_cat  = pheno in CATEGORICAL_PHENOS
            test_fn = lrt_logit if is_cat else ftest_ols
            covars  = get_covars(pheno, reg_df.columns)
            covar_f = " + ".join(covars)

            # ── unadjusted ──
            res_u = test_fn(
                f"{safe_p} ~ {bp_formula}",
                f"{safe_p} ~ 1",
                reg_df,
            )
            if res_u:
                rows_u.append({
                    "block": bname, "block_group": bgroup,
                    "phenotype": pheno,
                    **{k: res_u[k] for k in ["stat", "pval", "df_num", "n",
                                              "r2_full", "r2_reduced"]},
                })

            # ── adjusted ──
            if covar_f:
                res_a = test_fn(
                    f"{safe_p} ~ {bp_formula} + {covar_f}",
                    f"{safe_p} ~ {covar_f}",
                    reg_df,
                )
                if res_a:
                    rows_a.append({
                        "block": bname, "block_group": bgroup,
                        "phenotype": pheno,
                        **{k: res_a[k] for k in ["stat", "pval", "df_num", "n",
                                                  "r2_full", "r2_reduced"]},
                        "covars": ";".join(covars),
                    })

        if (b_idx + 1) % 20 == 0:
            print(f"  block {b_idx+1}/{B} …")

    assoc_u = pd.DataFrame(rows_u)
    assoc_a = pd.DataFrame(rows_a)

    assoc_u = add_fdr(assoc_u, "pval", "FDR_BH")
    assoc_a = add_fdr(assoc_a, "pval", "FDR_BH")

    # 4. Print results
    model_label = "logit LRT" if any(p in CATEGORICAL_PHENOS for p in pheno_cols) else "OLS F-test"
    print_top_hits(assoc_u, "pval", "FDR_BH",
                   f"F-test / LRT  |  block PC1+PC2+PC3  |  UNADJUSTED  (df_num=3)")
    print_top_hits(assoc_a, "pval", "FDR_BH",
                   f"F-test / LRT  |  block PC1+PC2+PC3  |  ADJUSTED (PC1-10 + covars)")

    # Quick summary counts
    alpha = 0.05
    print(f"\n{'═'*78}")
    print("  Summary")
    print(f"{'═'*78}")
    for label, df in [("Unadjusted", assoc_u), ("Adjusted", assoc_a)]:
        nom = (df["pval"] < alpha).sum()
        fdr = (df["FDR_BH"] < alpha).sum()
        print(f"  {label:<12s}  nominal p<0.05: {nom:4d}   FDR<0.05: {fdr:4d}")


if __name__ == "__main__":
    main()