#!/usr/bin/env python3
"""
06_attention_confounder_analysis.py

For each candidate block:
  1) OLS: attention ~ asthma
  2) OLS: attention ~ asthma + PC1..PC10
  3) Pearson r with each PC

Saves:
  results/output_regions2/ORD/confounder_analysis/block_asthma_pc_models.csv
  results/output_regions2/ORD/confounder_analysis/block_pc_correlations.csv

Plots:
  pc_correlation_heatmap.png
  asthma_effect_before_after_pc_adjustment.png
  scatter_<block>_PC1.png  (for 4 selected blocks)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ──────────────────────────────────────────────────────────────────────
ATTENTION_CSV = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
EIGENVEC_FILE = "metadata/ldpruned_997subs.eigenvec"
PHENO_FILE    = "metadata/COS_TRIO_pheno_1165.csv"
OUT_DIR       = "results/output_regions2/ORD/confounder_analysis"

CANDIDATE_BLOCKS = [
    "region_17q21_core_sb4",
    "region_2q12_IL1RL1_cluster_sb3",
    "region_2q12_IL1RL1_cluster_sb4",
    "region_1q31_TNFSF_cluster_sb19",
    "region_1q31_TNFSF_cluster_sb16",
    "region_5q31_type2_cytokine_sb6",
    "control_OCA2_sb10",
    "control_EDAR_sb2",
    "control_ALDH2",
]

# Candidate asthma column names, tried in order
ASTHMA_CANDIDATES = [
    "Dr_Dx_Asthma",
    "Persistent_asthma",
    "Asthma_Wheeze",
    "ENDOTYPE_AK",
]

PC_COLS = [f"PC{i}" for i in range(1, 11)]


# ── helpers ────────────────────────────────────────────────────────────────────
def load_eigenvec(path: str) -> pd.DataFrame:
    """Tab-separated; first col may be '#FID', second is IID."""
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lstrip("#")   # strip leading # from FID if present
    df = df.rename(columns={"FID": "FID_ev"})
    df["IID"] = df["IID"].astype(str)
    return df[["IID"] + PC_COLS]


def pick_asthma_col(pheno: pd.DataFrame) -> str:
    for col in ASTHMA_CANDIDATES:
        if col in pheno.columns:
            non_null = pheno[col].notna().sum()
            if non_null > 10:
                print(f"  [asthma] using column '{col}'  ({non_null} non-null rows)")
                return col
    # fallback: look for any column containing 'asthma' (case-insensitive)
    hits = [c for c in pheno.columns if "asthma" in c.lower()]
    if hits:
        col = hits[0]
        print(f"  [asthma] fallback to column '{col}'")
        return col
    raise ValueError(
        f"Could not find an asthma column. Tried: {ASTHMA_CANDIDATES}\n"
        f"Available columns: {list(pheno.columns[:40])}"
    )


def binarize_asthma(series: pd.Series) -> pd.Series:
    """
    Map to 0/1. Common codings: 1=yes/2=no, or 0=no/1=yes, or string 'Y'/'N'.
    Detect automatically.
    """
    s = series.copy()
    unique_vals = set(s.dropna().unique())

    if unique_vals <= {0, 1}:
        return s.astype(float)

    if unique_vals <= {1, 2}:
        # 1=yes, 2=no is common in this dataset
        return (s == 1).astype(float)

    # string fallback
    s_str = s.astype(str).str.upper().str.strip()
    if unique_vals <= {"Y", "N", "NAN"}:
        return (s_str == "Y").astype(float)

    # last resort: treat anything > median as 1
    med = s.median()
    print(f"  [asthma] binarize: using > {med} as positive")
    return (s > med).astype(float)


def run_ols(formula: str, data: pd.DataFrame):
    """Return fitted OLS result or None on failure."""
    try:
        return smf.ols(formula, data=data.dropna()).fit()
    except Exception as e:
        print(f"  [OLS] failed: {formula!r}  — {e}")
        return None


def scatter_block_pc(df: pd.DataFrame, block: str, pc: str, out_path: Path,
                     hue_col: str = "asthma"):
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {0.0: "#4878cf", 1.0: "#d65f5f", np.nan: "#aaaaaa"}
    for grp, sub in df.groupby(hue_col, dropna=False):
        c = colors.get(grp, "#888888")
        label = {0.0: "No asthma", 1.0: "Asthma"}.get(grp, f"grp={grp}")
        ax.scatter(sub[pc], sub[block], s=14, alpha=0.5, c=c, label=label)

    r, p = stats.pearsonr(df[[pc, block]].dropna()[pc], df[[pc, block]].dropna()[block])
    ax.set_xlabel(pc, fontsize=12)
    ax.set_ylabel(f"Pooling attention\n{block}", fontsize=10)
    ax.set_title(f"{block}\n{pc}  r={r:.3f}  p={p:.3g}", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-csv", default=ATTENTION_CSV)
    parser.add_argument("--eigenvec",      default=EIGENVEC_FILE)
    parser.add_argument("--pheno-file",    default=PHENO_FILE)
    parser.add_argument("--out-dir",       default=OUT_DIR)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
    print("Loading attention weights …")
    attn = pd.read_csv(args.attention_csv)
    attn["IID"] = attn["IID"].astype(str)

    print("Loading eigenvec …")
    pcs = load_eigenvec(args.eigenvec)

    print("Loading phenotype …")
    pheno = pd.read_csv(args.pheno_file, low_memory=False)
    if "S_SUBJECTID" in pheno.columns:
        pheno = pheno.rename(columns={"S_SUBJECTID": "IID"})
    pheno["IID"] = pheno["IID"].astype(str)

    asthma_col = pick_asthma_col(pheno)
    pheno["asthma"] = binarize_asthma(pheno[asthma_col])

    # ── merge ─────────────────────────────────────────────────────────────────
    df = (
        attn
        .merge(pcs,                        on="IID", how="inner")
        .merge(pheno[["IID", "asthma"]],   on="IID", how="inner")
    )
    print(f"  Merged N = {len(df)}  (asthma 1: {int(df['asthma'].sum())}, 0: {int((df['asthma']==0).sum())})")

    # check which candidate blocks are actually present
    available_blocks = [b for b in CANDIDATE_BLOCKS if b in df.columns]
    missing_blocks   = [b for b in CANDIDATE_BLOCKS if b not in df.columns]
    if missing_blocks:
        print(f"  [warn] blocks not in attention CSV (skipped): {missing_blocks}")

    # ── model loop ────────────────────────────────────────────────────────────
    model_rows = []
    corr_rows  = []

    for block in available_blocks:
        safe = block.replace("-", "_")    # statsmodels formula-safe name
        df[safe] = df[block]

        # 1) OLS: attention ~ asthma
        res0 = run_ols(f"{safe} ~ asthma", df)
        # 2) OLS: attention ~ asthma + PC1..PC10
        pc_terms = " + ".join(PC_COLS)
        res1 = run_ols(f"{safe} ~ asthma + {pc_terms}", df)

        row = {"block": block}

        if res0 is not None:
            row["beta_asthma_unadj"]   = round(res0.params.get("asthma", np.nan), 6)
            row["pval_asthma_unadj"]   = round(res0.pvalues.get("asthma", np.nan), 6)
            row["r2_unadj"]            = round(res0.rsquared, 6)

        if res1 is not None:
            row["beta_asthma_adj"]     = round(res1.params.get("asthma", np.nan), 6)
            row["pval_asthma_adj"]     = round(res1.pvalues.get("asthma", np.nan), 6)
            row["r2_adj"]              = round(res1.rsquared, 6)

        model_rows.append(row)

        # 3) Pearson correlations with each PC
        sub = df[[block] + PC_COLS].dropna()
        for pc in PC_COLS:
            r, p = stats.pearsonr(sub[block], sub[pc])
            corr_rows.append({
                "block": block,
                "PC":    pc,
                "pearson_r": round(r, 6),
                "pval":      round(p, 6),
            })

    models_df = pd.DataFrame(model_rows)
    corr_df   = pd.DataFrame(corr_rows)

    models_df.to_csv(out / "block_asthma_pc_models.csv", index=False)
    corr_df.to_csv(  out / "block_pc_correlations.csv",  index=False)
    print(f"\nSaved CSVs to {out}/")

    # ── summary print ─────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  Block association summary (unadjusted vs PC-adjusted)")
    print(f"{'═'*65}")
    alpha = 0.05
    for _, row in models_df.iterrows():
        p0 = row.get("pval_asthma_unadj", np.nan)
        p1 = row.get("pval_asthma_adj",   np.nan)
        sig0 = "**" if p0 < alpha else "  "
        sig1 = "**" if p1 < alpha else "  "
        print(
            f"  {row['block']:<45s}"
            f"  unadj p={p0:.3g}{sig0}"
            f"  adj p={p1:.3g}{sig1}"
        )
    print(f"{'═'*65}\n")
    print("Blocks that REMAIN significant after PC adjustment:")
    still_sig = models_df[models_df["pval_asthma_adj"] < alpha]["block"].tolist()
    for b in still_sig:
        print(f"  • {b}")
    if not still_sig:
        print("  (none at p < 0.05 after PC adjustment)")

    # ── plot 1: PC correlation heatmap ────────────────────────────────────────
    pivot = corr_df.pivot(index="block", columns="PC", values="pearson_r")
    pivot = pivot[[f"PC{i}" for i in range(1, 11)]]  # ordered

    fig, ax = plt.subplots(figsize=(12, max(4, len(available_blocks) * 0.55)))
    sns.heatmap(
        pivot,
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("Pearson r: block pooling attention vs PC1–PC10", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out / "pc_correlation_heatmap.png", dpi=160)
    plt.close()
    print("Saved pc_correlation_heatmap.png")

    # ── plot 2: asthma effect before/after PC adjustment ─────────────────────
    plot_df = models_df.dropna(subset=["beta_asthma_unadj", "beta_asthma_adj"])
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.55)))
    y = np.arange(len(plot_df))
    h = 0.3
    ax.barh(y + h/2, plot_df["beta_asthma_unadj"], h, color="#4878cf", alpha=0.85, label="Unadjusted")
    ax.barh(y - h/2, plot_df["beta_asthma_adj"],   h, color="#d65f5f", alpha=0.85, label="PC-adjusted")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["block"], fontsize=9)
    ax.set_xlabel("OLS β (asthma effect on pooling attention)")
    ax.set_title("Asthma association before and after PC1–PC10 adjustment")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "asthma_effect_before_after_pc_adjustment.png", dpi=160)
    plt.close()
    print("Saved asthma_effect_before_after_pc_adjustment.png")

    # ── plot 3–6: scatter block ~ PC1 ────────────────────────────────────────
    scatter_pairs = [
        "control_OCA2_sb10",
        "control_EDAR_sb2",
        "region_17q21_core_sb4",
        "region_2q12_IL1RL1_cluster_sb4",
    ]
    for block in scatter_pairs:
        if block not in df.columns:
            print(f"  [scatter] skipping {block} (not in merged data)")
            continue
        fname = f"scatter_{block}_PC1.png"
        scatter_block_pc(df, block, "PC1", out / fname)
        print(f"Saved {fname}")

    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
