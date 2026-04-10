#!/usr/bin/env python3
"""07_17q21_validation.py — 17q21 subblock validation and baseline comparison.

Purpose  : Two-stage validation for 17q21 (sb1–sb5):
           Step A — full validation: geno↔embedding alignment, SNP-level
                    correlations, 4 named publication figures, rsID lookup.
           Step B — baseline comparison: OLS geno_PC1 / geno_norm3 / emb_PC1 /
                    emb_norm3 ~ FEV1; incremental R² and 2×2 model tables
                    (terminal output only, no files saved).
Inputs   : data/region_blocks/region_17q21_core_sb*.raw,
           results/output_regions2/ORD/embeddings/block_contextual_repr.npy,
           results/output_regions2/ORD/embeddings/pooling_attention_weights.csv,
           results/output_regions/block_order.csv,
           metadata/ldpruned_997subs.eigenvec,
           metadata/COS_TRIO_pheno_1165.csv
           (Step A only) results/output_regions2/ORD/all_blocks_pheno_analysis/
                         phenotype_block_associations.tsv
Outputs  : Step A → results/output_regions2/ORD/17q21_validation/{figures/,tables/}
           Step B → terminal output only
Workflow : Active analysis — Step 7; final validation / publication figures.

Usage
-----
  python scripts/analysis/07_17q21_validation.py --mode all        # default
  python scripts/analysis/07_17q21_validation.py --mode validation
  python scripts/analysis/07_17q21_validation.py --mode baseline
"""

# ══════════════════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════════════════
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# Shared path constants
# ══════════════════════════════════════════════════════════════════════════════

RAW_DIR       = "data/region_blocks"
EMB_BLOCK_NPY = "results/output_regions2/ORD/embeddings/block_contextual_repr.npy"
ATTN_CSV      = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
BLOCK_ORDER   = "results/output_regions/block_order.csv"
EIGENVEC_FILE = "metadata/ldpruned_997subs.eigenvec"
PHENO_FILE    = "metadata/COS_TRIO_pheno_1165.csv"

# Step A only
ASSOC_TSV     = "results/output_regions2/ORD/all_blocks_pheno_analysis/phenotype_block_associations.tsv"
OUT_DIR_A     = "results/output_regions2/ORD/17q21_validation"

SUBBLOCKS     = [f"region_17q21_core_sb{i}" for i in range(1, 6)]
PLINK_META    = {"FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"}
PC_COLS       = [f"PC{i}" for i in range(1, 11)]
N_BLOCK_PCS   = 3

# Step B phenotype
PHENO_B   = "pctpred_fev1_pre_BD"
COVARS_B  = PC_COLS + ["gender_num", "smkexp_current"]

# Step A plot style
PALETTE_POS = "#2166ac"
PALETTE_NEG = "#d6604d"
PALETTE_HL  = "#1a1a2e"
GREY_LIGHT  = "#cccccc"
FONT_TITLE  = 13
FONT_LABEL  = 11
FONT_TICK   = 9
DPI         = 180
ENSEMBL_SLEEP = 0.34

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_block_names(path, expected_B):
    bo = pd.read_csv(path)
    col = next((c for c in ["block", "block_id", "region", "name"] if c in bo.columns), None)
    if col is None:
        raise ValueError(f"No block-name column in {path}")
    names = bo[col].astype(str).tolist()
    if len(names) != expected_B:
        raise ValueError(f"block_order {len(names)} != {expected_B}")
    return names


def load_raw_plink(path):
    df = pd.read_csv(path, sep=r"\s+", low_memory=False)
    df["IID"] = df["IID"].astype(str)
    snp_cols = [c for c in df.columns if c not in PLINK_META]
    return df[["IID"] + snp_cols].copy(), snp_cols


def load_eigenvec(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lstrip("#")
    df["IID"] = df["IID"].astype(str)
    return df[["IID"] + PC_COLS]


def sign_align_scalar(emb, geno):
    """Return sign-flipped emb so it correlates positively with geno. Returns scalar."""
    r, _ = stats.spearmanr(emb, geno)
    return emb * (-1 if r < 0 else 1)


# ══════════════════════════════════════════════════════════════════════════════
# Step A helpers  (from 17q21_genotype_embedding_validation.py)
# ══════════════════════════════════════════════════════════════════════════════

def _geno_pc1_vec(dosage_matrix):
    X = SimpleImputer(strategy="mean").fit_transform(dosage_matrix.astype(float))
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1, random_state=42)
    return pca.fit_transform(X).ravel(), pca.explained_variance_ratio_[0]


def _emb_top_pcs(block_repr, block_idx, k):
    X = block_repr[:, block_idx, :]
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=k, random_state=42)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def _sign_align_tuple(emb, geno):
    """Returns (aligned_emb, was_flipped)."""
    r, _ = stats.spearmanr(emb, geno)
    return emb * (-1 if r < 0 else 1), r < 0


def _spearman_ci(x, y, n_boot=1000):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 20:
        return dict(r=np.nan, p=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=len(x))
    r, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(42)
    boot = [stats.spearmanr(x[idx := rng.integers(0, len(x), len(x))],
                            y[idx])[0]
            for _ in range(n_boot) if len(np.unique(x[rng.integers(0, len(x), len(x))])) > 1]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return dict(r=float(r), p=float(p), ci_lo=float(ci_lo), ci_hi=float(ci_hi), n=len(x))


def _parse_snp_name(snp_raw):
    core = snp_raw.rsplit("_", 1)[0] if "_" in snp_raw else snp_raw
    parts = core.split(":")
    if len(parts) >= 4:
        return dict(chr=parts[0], pos=int(parts[1]), ref=parts[2], alt=parts[3])
    elif len(parts) == 3:
        return dict(chr=parts[0], pos=int(parts[1]), ref=parts[2], alt=".")
    return dict(chr=".", pos=0, ref=".", alt=".")


def _lookup_rsids_ensembl(positions):
    unique_pos = list({(p["chr"], p["pos"]) for p in positions})
    result = {}
    print(f"\n  Looking up rsIDs for {len(unique_pos)} unique positions via Ensembl …")
    for done, (chrom, pos) in enumerate(unique_pos, 1):
        key = (str(chrom), int(pos))
        try:
            url = (f"https://rest.ensembl.org/overlap/region/human/"
                   f"{chrom}:{pos}-{pos}?feature=variation&content-type=application/json")
            resp = requests.get(url, timeout=15)
            if resp.status_code == 429:
                time.sleep(5)
                resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                rs_ids = [v["id"] for v in resp.json()
                          if isinstance(v.get("id"), str) and v["id"].startswith("rs")]
                result[key] = rs_ids[0] if rs_ids else "NA"
            else:
                result[key] = "NA"
        except Exception:
            result[key] = "NA"
        time.sleep(ENSEMBL_SLEEP)
        if done % 20 == 0 or done == len(unique_pos):
            print(f"    {done}/{len(unique_pos)} ({100*done/len(unique_pos):.0f}%) …", end="\r")
    found = sum(1 for v in result.values() if v != "NA")
    print(f"\n  rsID lookup complete — found {found}/{len(unique_pos)}")
    return result


def _plot_block_level_bar(block_results, out_path):
    labels = [r["name"].replace("region_17q21_core_", "").upper() for r in block_results]
    rs = [r["r"] for r in block_results]
    ci_lo = [r["ci_lo"] for r in block_results]; ci_hi = [r["ci_hi"] for r in block_results]
    colors = [PALETTE_POS if r >= 0 else PALETTE_NEG for r in rs]
    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = range(len(labels))
    ax.barh(y_pos, rs, color=colors, alpha=0.85, height=0.55, zorder=3)
    ax.errorbar(rs, list(y_pos),
                xerr=[[abs(r - lo) for r, lo in zip(rs, ci_lo)],
                      [abs(hi - r) for r, hi in zip(rs, ci_hi)]],
                fmt="none", color="#333333", capsize=4, linewidth=1.2, zorder=4)
    for i, r_val in enumerate(rs):
        ax.text(r_val + (0.03 if r_val >= 0 else -0.03), i, f"{r_val:+.3f}",
                va="center", ha="left" if r_val >= 0 else "right",
                fontsize=FONT_TICK, color=PALETTE_HL, fontweight="bold")
    ax.set_yticks(list(y_pos)); ax.set_yticklabels(labels, fontsize=FONT_LABEL)
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Spearman r  (Genotype PC1 ↔ Embedding PC1)", fontsize=FONT_LABEL)
    ax.set_title("17q21 Subblocks: Genotype–Embedding Alignment\n"
                 "Alternating signs reflect known inversion LD structure",
                 fontsize=FONT_TITLE, pad=10)
    ax.set_xlim(-1.05, 1.15)
    ax.legend(handles=[mpatches.Patch(color=PALETTE_POS, label="Positive correlation"),
                       mpatches.Patch(color=PALETTE_NEG, label="Negative correlation (sign flip)")],
              fontsize=8, loc="lower right", framealpha=0.9)
    plt.tight_layout(); plt.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close()
    print(f"  saved {out_path.name}")


def _plot_snp_scatter(scatter_data, out_path):
    n = len(scatter_data); ncols = min(3, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), sharey=False)
    axes = np.array(axes).flatten()
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax, item in zip(axes, scatter_data):
        dos = item["dosage"]; epc1 = item["epc1"]; r = item["r"]
        name = item["name"].replace("region_17q21_core_", "").upper()
        rng = np.random.default_rng(0)
        x_jit = dos + rng.uniform(-0.15, 0.15, size=len(dos))
        colors_dot = np.where(dos == 2, PALETTE_POS, np.where(dos == 0, GREY_LIGHT, "#74a9cf"))
        ax.scatter(x_jit, epc1, c=colors_dot, s=10, alpha=0.5, zorder=2)
        for d_val in [0, 1, 2]:
            mask = dos == d_val
            if mask.sum() > 0:
                ax.hlines(epc1[mask].mean(), d_val - 0.3, d_val + 0.3,
                          colors=PALETTE_HL, linewidth=2.5, zorder=3)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["0\n(hom ref)", "1\n(het)", "2\n(hom alt)"], fontsize=FONT_TICK)
        ax.set_xlabel("SNP Dosage", fontsize=FONT_LABEL)
        ax.set_ylabel("Embedding PC1", fontsize=FONT_LABEL)
        rsid = item.get("rsid", "NA")
        snp_label = rsid if rsid != "NA" else item["snp"].split("_")[0]
        ax.set_title(f"{name}\n{snp_label}\nSpearman r = {r:+.3f}", fontsize=FONT_TITLE - 1, pad=8)
        ymin = epc1.min() - 0.3
        for d_val in [0, 1, 2]:
            ax.text(d_val, ymin, f"n={(dos == d_val).sum()}", ha="center",
                    fontsize=7, color="#666666")
    fig.suptitle("SNP Dosage vs Embedding PC1  (group means = horizontal lines)",
                 fontsize=FONT_TITLE, y=1.01)
    plt.tight_layout(); plt.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close()
    print(f"  saved {out_path.name}")


def _plot_interblock_heatmap(epc1_matrix, subblock_names, out_path):
    labels = [n.replace("region_17q21_core_", "").upper() for n in subblock_names]
    n = len(labels)
    rmat = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                rmat[i, j], _ = stats.spearmanr(epc1_matrix[:, i], epc1_matrix[:, j])
    rdf = pd.DataFrame(rmat, index=labels, columns=labels)
    mask = np.eye(n, dtype=bool)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(rdf, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="#dddddd", annot_kws={"size": 10, "weight": "bold"},
                ax=ax, mask=mask, cbar_kws={"label": "Spearman r"})
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="#f0f0f0", zorder=2))
        ax.text(i + 0.5, i + 0.5, "1.00", ha="center", va="center",
                fontsize=10, color="#aaaaaa")
    ax.set_title("Inter-Subblock Embedding PC1 Correlations\n"
                 "(Alternating signs = 17q21 inversion LD structure)",
                 fontsize=FONT_TITLE, pad=10)
    ax.tick_params(axis="both", labelsize=FONT_LABEL)
    plt.tight_layout(); plt.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close()
    print(f"  saved {out_path.name}")


def _plot_fev1_volcano(assoc_path, out_path):
    assoc = pd.read_csv(assoc_path, sep="\t")
    fev = assoc[assoc["phenotype"] == "pctpred_fev1_pre_BD"].copy()
    fev = fev.dropna(subset=["beta_adj", "pval_adj"])
    fev["neg_log10_p"] = -np.log10(fev["pval_adj"].clip(lower=1e-30))
    fev["is_17q21"] = fev["block"].str.contains("17q21", case=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    other = fev[~fev["is_17q21"]]
    ax.scatter(other["beta_adj"], other["neg_log10_p"],
               s=14, alpha=0.3, color=GREY_LIGHT, zorder=2, label="Other blocks")
    q21 = fev[fev["is_17q21"]]
    ax.scatter(q21["beta_adj"], q21["neg_log10_p"],
               s=60, alpha=0.9, color=PALETTE_NEG,
               edgecolors=PALETTE_HL, linewidth=0.6, zorder=4, label="17q21 subblocks")
    for _, row in q21.iterrows():
        ax.annotate(row["block"].replace("region_17q21_core_", "sb"),
                    xy=(row["beta_adj"], row["neg_log10_p"]),
                    xytext=(6, 3), textcoords="offset points", fontsize=7.5, color=PALETTE_HL,
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6))
    ax.axhline(-np.log10(0.05), color="#888888", linewidth=0.9, linestyle="--", label="p = 0.05")
    ax.axvline(0, color="#cccccc", linewidth=0.7)
    ax.set_xlabel("β  (block PC1 effect, ancestry-adjusted)", fontsize=FONT_LABEL)
    ax.set_ylabel("−log₁₀(p adjusted)", fontsize=FONT_LABEL)
    ax.set_title("Association with Predicted FEV1 (pctpred_fev1_pre_BD)\n"
                 "All blocks — 17q21 subblocks highlighted", fontsize=FONT_TITLE, pad=10)
    ax.legend(fontsize=9, framealpha=0.9); ax.tick_params(labelsize=FONT_TICK)
    plt.tight_layout(); plt.savefig(out_path, dpi=DPI, bbox_inches="tight"); plt.close()
    print(f"  saved {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Step B helpers  (from 17q21_baseline_comparison.py)
# ══════════════════════════════════════════════════════════════════════════════

def _top_k_pcs(matrix, k):
    X = SimpleImputer(strategy="mean").fit_transform(matrix.astype(float))
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=k, random_state=42)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def _compute_norm3(pcs):
    norm = np.sqrt((pcs ** 2).sum(axis=1))
    return (norm - norm.mean()) / (norm.std() + 1e-12)


def _run_ols(formula, data):
    try:
        sub = data.dropna()
        return smf.ols(formula, data=sub).fit() if len(sub) >= 20 else None
    except Exception:
        return None


def _extract(res, predictor):
    if res is None:
        return dict(beta=np.nan, pval=np.nan, r2=np.nan, n=np.nan)
    return dict(beta=float(res.params.get(predictor, np.nan)),
                pval=float(res.pvalues.get(predictor, np.nan)),
                r2=float(res.rsquared),
                n=int(len(res.model.data.frame.dropna())))


def _sig_stars(p):
    if np.isnan(p): return "  "
    if p < 0.001:   return "***"
    if p < 0.01:    return "** "
    if p < 0.05:    return "*  "
    return "   "


# ══════════════════════════════════════════════════════════════════════════════
# Step A — full validation with figures and tables
# (logic from 17q21_genotype_embedding_validation.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_validation():
    """Step A: 17q21 subblock validation — 4 publication figures + SNP correlation tables."""
    fig_dir = Path(OUT_DIR_A) / "figures"
    tbl_dir = Path(OUT_DIR_A) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    print("Loading block embeddings …")
    emb_block   = np.load(EMB_BLOCK_NPY)
    N, B, D     = emb_block.shape
    print(f"  shape: {emb_block.shape}")

    attn_df     = pd.read_csv(ATTN_CSV)
    subj_iids   = attn_df["IID"].astype(str).values
    block_names = load_block_names(BLOCK_ORDER, B)
    iid_to_idx  = {iid: i for i, iid in enumerate(subj_iids)}

    block_results = []   # fig 1
    scatter_data  = []   # fig 2
    epc1_raw_cols = []   # fig 3
    present_names = []
    all_snp_rows  = []
    all_parsed    = []

    print("\nProcessing subblocks …")

    for sb_name in SUBBLOCKS:
        raw_path = Path(RAW_DIR) / f"{sb_name}.raw"
        print(f"\n  {sb_name}")

        if not raw_path.exists():
            print("    [SKIP] not found"); continue
        try:
            b_idx = block_names.index(sb_name)
        except ValueError:
            print("    [SKIP] not in block_order"); continue

        raw_df, snp_cols = load_raw_plink(raw_path)
        common_iids = [iid for iid in raw_df["IID"] if iid in iid_to_idx]
        if len(common_iids) < 50:
            print(f"    [SKIP] n={len(common_iids)}"); continue

        raw_aligned = raw_df.set_index("IID").loc[common_iids]
        emb_rows    = [iid_to_idx[iid] for iid in common_iids]

        k = min(N_BLOCK_PCS, D)
        emb_pcs, emb_var = _emb_top_pcs(emb_block, b_idx, k)
        emb_sub = emb_pcs[emb_rows, :]

        dosage = raw_aligned[snp_cols].values.astype(float)
        dosage[dosage == -9] = np.nan
        gpc1, gpc1_var = _geno_pc1_vec(dosage)

        print(f"    SNPs={len(snp_cols)}  subjects={len(common_iids)}")
        print(f"    Geno PC1 var={100*gpc1_var:.1f}%  "
              f"Emb PC var: {' '.join(f'ePC{i+1}={100*v:.1f}%' for i, v in enumerate(emb_var))}")

        res = _spearman_ci(gpc1, emb_sub[:, 0])
        print(f"    geno PC1 ↔ ePC1  r={res['r']:+.4f}  p={res['p']:.3g}")
        block_results.append({"name": sb_name, **res})

        epc1_aligned, flipped = _sign_align_tuple(emb_sub[:, 0], gpc1)
        if flipped:
            print("    [sign flipped for scatter]")

        epc1_raw_cols.append(emb_pcs[:, 0])
        present_names.append(sb_name)

        snp_rs = []
        for snp in snp_cols:
            dos = raw_aligned[snp].values.astype(float)
            dos[dos == -9] = np.nan
            mask = ~np.isnan(dos)
            if mask.sum() < 50 or np.nanstd(dos[mask]) < 1e-6:
                continue
            r_s, p_s  = stats.spearmanr(epc1_aligned[mask], dos[mask])
            mean_dos  = float(np.nanmean(dos))
            maf       = min(mean_dos / 2, 1 - mean_dos / 2)
            parsed    = _parse_snp_name(snp)
            all_parsed.append(parsed)
            snp_rs.append({
                "subblock": sb_name, "snp_raw": snp,
                "chr": parsed["chr"], "pos": parsed["pos"],
                "ref": parsed["ref"], "alt": parsed["alt"],
                "r": float(r_s), "p": float(p_s),
                "mean_dos": round(mean_dos, 4), "maf": round(maf, 4), "n_valid": int(mask.sum()),
            })

        all_snp_rows.extend(snp_rs)

        if snp_rs:
            best = sorted(snp_rs, key=lambda x: -abs(x["r"]))[0]
            best_dos = raw_aligned[best["snp_raw"]].values.astype(float)
            best_dos[best_dos == -9] = np.nan
            scatter_data.append({"name": sb_name, "snp": best["snp_raw"],
                                  "r": best["r"], "dosage": best_dos,
                                  "epc1": epc1_aligned, "rsid": "NA"})

    # rsID lookup
    rsid_map = _lookup_rsids_ensembl(all_parsed)
    for row in all_snp_rows:
        row["rsid"] = rsid_map.get((str(row["chr"]), int(row["pos"])), "NA")
    for item in scatter_data:
        parsed = _parse_snp_name(item["snp"])
        item["rsid"] = rsid_map.get((str(parsed["chr"]), int(parsed["pos"])), "NA")

    # Save SNP tables
    print("\nSaving SNP correlation tables …")
    col_order = ["subblock", "snp_raw", "chr", "pos", "ref", "alt", "rsid",
                 "r", "p", "fdr", "mean_dos", "maf", "n_valid"]
    combined_rows = []
    for sb_name in SUBBLOCKS:
        rows = [r for r in all_snp_rows if r["subblock"] == sb_name]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        _, fdr_vals, _, _ = multipletests(df["p"], method="fdr_bh")
        df["fdr"] = fdr_vals
        df = df[col_order].sort_values("r", key=abs, ascending=False)
        sb_short = sb_name.replace("region_17q21_core_", "")
        out_path = tbl_dir / f"snp_correlations_{sb_short}.tsv"
        df.to_csv(out_path, sep="\t", index=False, float_format="%.6e")
        print(f"  saved {out_path.name}  ({len(df)} SNPs)")
        n_fdr = (df["fdr"] < 0.05).sum(); n_rs = (df["rsid"] != "NA").sum()
        print(f"    FDR<0.05={n_fdr}  rsIDs found={n_rs}")
        combined_rows.append(df)

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        comb_path = tbl_dir / "snp_correlations_all_subblocks.tsv"
        combined.to_csv(comb_path, sep="\t", index=False, float_format="%.6e")
        print(f"\n  saved combined: {comb_path.name}  ({len(combined)} rows)")

    # Save figures
    print("\nSaving figures …")
    if block_results:
        _plot_block_level_bar(block_results, fig_dir / "fig1_block_level_bar.png")
    if scatter_data:
        _plot_snp_scatter(scatter_data, fig_dir / "fig2_snp_scatter_all_sb.png")
    if len(epc1_raw_cols) >= 2:
        _plot_interblock_heatmap(np.column_stack(epc1_raw_cols), present_names,
                                 fig_dir / "fig3_interblock_heatmap.png")
    if Path(ASSOC_TSV).exists():
        _plot_fev1_volcano(ASSOC_TSV, fig_dir / "fig4_fev1_volcano_17q21.png")
    else:
        print(f"  [warn] ASSOC_TSV not found — skipping fig4")

    # Final summary
    combined_lookup = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    print(f"\n{'═'*70}"); print("  Final summary"); print(f"{'═'*70}")
    print(f"  {'subblock':<10s} {'r':>8s} {'95% CI':>20s} {'SNPs':>6s} {'FDR<0.05':>9s} {'rsIDs':>7s}")
    for br in block_results:
        sb_s = br["name"].replace("region_17q21_core_", "")
        ci   = f"[{br['ci_lo']:+.3f}, {br['ci_hi']:+.3f}]"
        sub  = combined_lookup[combined_lookup["subblock"] == br["name"]] if len(combined_lookup) > 0 else pd.DataFrame()
        print(f"  {sb_s:<10s} {br['r']:>+8.4f} {ci:>20s} "
              f"{len(sub):>6d} {(sub['fdr'] < 0.05).sum() if len(sub) > 0 else 0:>9d} "
              f"{(sub['rsid'] != 'NA').sum() if len(sub) > 0 else 0:>7d}")
    print(f"\n  Figures → {fig_dir}/\n  Tables  → {tbl_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Step B — geno vs embedding baseline comparison (terminal output only)
# (logic from 17q21_baseline_comparison.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline_comparison():
    """Step B: OLS comparison of geno_PC1/norm3 vs emb_PC1/norm3 for FEV1. Prints only."""
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
        pheno_raw["gender_num"] = s.map({"male": 1., "m": 1., "1": 1.,
                                         "female": 0., "f": 0., "2": 0.})
    keep = (["IID", PHENO_B] +
            [c for c in ["gender_num", "smkexp_current", "age"] if c in pheno_raw.columns])
    pheno_df = pheno_raw[keep].copy()

    master = (pd.DataFrame({"IID": subj_iids})
              .merge(pcs_df, on="IID", how="left")
              .merge(pheno_df, on="IID", how="left"))

    covar_cols = [c for c in COVARS_B if c in master.columns]
    covar_str  = " + ".join(covar_cols)
    safe_pheno = PHENO_B.replace("-", "_")
    master[safe_pheno] = master[PHENO_B]

    res_reduced = _run_ols(f"{safe_pheno} ~ {covar_str}", master)
    r2_reduced  = res_reduced.rsquared if res_reduced else np.nan
    n_reduced   = len(res_reduced.model.data.frame.dropna()) if res_reduced else 0
    print(f"  Reduced model R² (covariates only): {r2_reduced:.6f}  n={n_reduced}")
    print(f"  Phenotype: {PHENO_B}\n  Covariates: {covar_str}\n")

    summary_rows = []

    for sb_name in SUBBLOCKS:
        raw_path = Path(RAW_DIR) / f"{sb_name}.raw"
        sb_short = sb_name.replace("region_17q21_core_", "").upper()
        print(f"{'━'*72}\n  {sb_name}\n{'━'*72}")

        if not raw_path.exists():
            print("  [SKIP] raw file not found"); continue
        try:
            b_idx = block_names.index(sb_name)
        except ValueError:
            print("  [SKIP] not in block_order"); continue

        raw_df, snp_cols = load_raw_plink(raw_path)
        common_iids = [iid for iid in raw_df["IID"] if iid in iid_to_idx]
        if len(common_iids) < 50:
            print(f"  [SKIP] only {len(common_iids)} matched subjects"); continue

        raw_aligned = raw_df.set_index("IID").loc[common_iids]
        emb_rows    = [iid_to_idx[iid] for iid in common_iids]

        dosage = raw_aligned[snp_cols].values.astype(float)
        dosage[dosage == -9] = np.nan

        k = min(N_BLOCK_PCS, len(snp_cols))
        geno_pcs, geno_var = _top_k_pcs(dosage, k)
        geno_pc1_vec  = geno_pcs[:, 0]
        geno_norm3_vec = _compute_norm3(geno_pcs)
        print(f"  Genotype: {len(snp_cols)} SNPs  "
              f"PC1 var={100*geno_var[0]:.1f}%  PC1-3 cum={100*geno_var.sum():.1f}%")

        emb_sub = emb_block[emb_rows, b_idx, :]
        emb_sub_scaled = StandardScaler().fit_transform(emb_sub)
        emb_pca = PCA(n_components=N_BLOCK_PCS, random_state=42)
        emb_pcs_arr = emb_pca.fit_transform(emb_sub_scaled)
        emb_var = emb_pca.explained_variance_ratio_
        emb_pc1_vec   = sign_align_scalar(emb_pcs_arr[:, 0], geno_pc1_vec)
        emb_norm3_vec = _compute_norm3(emb_pcs_arr)
        print(f"  Embedding: PC1 var={100*emb_var[0]:.1f}%  PC1-3 cum={100*emb_var.sum():.1f}%")

        r_pc1, _   = stats.spearmanr(geno_pc1_vec, emb_pc1_vec)
        r_norm3, _ = stats.spearmanr(geno_norm3_vec, emb_norm3_vec)
        print(f"  Geno ↔ Emb alignment:  PC1 r={r_pc1:+.3f}  norm3 r={r_norm3:+.3f}")

        master_sub = master.set_index("IID").loc[common_iids].reset_index()
        master_sub["geno_pc1"]   = geno_pc1_vec
        master_sub["geno_norm3"] = geno_norm3_vec
        master_sub["emb_pc1"]    = emb_pc1_vec
        master_sub["emb_norm3"]  = emb_norm3_vec

        models = {
            "geno_PC1":   ("geno_pc1",   f"{safe_pheno} ~ geno_pc1   + {covar_str}"),
            "geno_norm3": ("geno_norm3", f"{safe_pheno} ~ geno_norm3 + {covar_str}"),
            "emb_PC1":    ("emb_pc1",    f"{safe_pheno} ~ emb_pc1    + {covar_str}"),
            "emb_norm3":  ("emb_norm3",  f"{safe_pheno} ~ emb_norm3  + {covar_str}"),
        }
        results = {}
        for label, (pred, formula) in models.items():
            res = _run_ols(formula, master_sub)
            ex  = _extract(res, pred)
            ex["incr_r2"] = ex["r2"] - r2_reduced if not np.isnan(ex["r2"]) else np.nan
            results[label] = ex

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

        star_str = f"  {'sig':20s}"
        for label in ["geno_PC1", "geno_norm3", "emb_PC1", "emb_norm3"]:
            star_str += f"{_sig_stars(results[label]['pval']):>12s}"
        print(star_str)

        best_label = min(results, key=lambda x: results[x]["pval"]
                         if not np.isnan(results[x]["pval"]) else 1.0)
        best_p = results[best_label]["pval"]
        best_incr = results[best_label]["incr_r2"]
        print(f"\n  Best model: {best_label}  p={best_p:.4g}  incr_R²={best_incr:.5f}")

        for sfx in ["PC1", "norm3"]:
            p_g = results[f"geno_{sfx}"]["pval"]
            p_e = results[f"emb_{sfx}"]["pval"]
            if not (np.isnan(p_g) or np.isnan(p_e)):
                delta = (-np.log10(p_e + 1e-300)) - (-np.log10(p_g + 1e-300))
                print(f"  {sfx}: emb p={p_e:.4g}  geno p={p_g:.4g}  "
                      f"Δ-log10p={delta:+.3f}  → {'emb better' if delta > 0 else 'geno better'}")

        summary_rows.append({
            "subblock": sb_short,
            "geno_pc1_p": results["geno_PC1"]["pval"],
            "emb_pc1_p":  results["emb_PC1"]["pval"],
            "geno_norm3_p": results["geno_norm3"]["pval"],
            "emb_norm3_p":  results["emb_norm3"]["pval"],
            "geno_pc1_incr_r2":  results["geno_PC1"]["incr_r2"],
            "emb_pc1_incr_r2":   results["emb_PC1"]["incr_r2"],
            "geno_norm3_incr_r2": results["geno_norm3"]["incr_r2"],
            "emb_norm3_incr_r2":  results["emb_norm3"]["incr_r2"],
            "best_model":    best_label,
            "geno_emb_r_pc1":    round(float(r_pc1), 4),
            "geno_emb_r_norm3":  round(float(r_norm3), 4),
        })

    if not summary_rows:
        print("\nNo results to summarise."); return

    print(f"\n{'═'*72}")
    print(f"  SUMMARY: {PHENO_B}  —  genotype vs embedding comparison")
    print(f"  Reduced model R² (covariates only): {r2_reduced:.5f}")
    print(f"{'═'*72}")
    print(f"  {'SB':<6s} {'geno_PC1_p':>12s} {'emb_PC1_p':>12s} "
          f"{'Δ-log10p':>10s} {'geno_n3_p':>12s} {'emb_n3_p':>12s} "
          f"{'Δ-log10p':>10s} {'best':>12s}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} "
          f"{'-'*12} {'-'*12} {'-'*10} {'-'*12}")
    for row in summary_rows:
        p_gp1 = row["geno_pc1_p"]; p_ep1 = row["emb_pc1_p"]
        p_gn3 = row["geno_norm3_p"]; p_en3 = row["emb_norm3_p"]
        d_pc1  = ((-np.log10(p_ep1 + 1e-300)) - (-np.log10(p_gp1 + 1e-300))
                  if not (np.isnan(p_ep1) or np.isnan(p_gp1)) else np.nan)
        d_norm3 = ((-np.log10(p_en3 + 1e-300)) - (-np.log10(p_gn3 + 1e-300))
                   if not (np.isnan(p_en3) or np.isnan(p_gn3)) else np.nan)
        print(f"  {row['subblock']:<6s} {p_gp1:>12.4g} {p_ep1:>12.4g} {d_pc1:>+10.3f} "
              f"{p_gn3:>12.4g} {p_en3:>12.4g} {d_norm3:>+10.3f} {row['best_model']:>12s}")

    print(f"\n  Incremental R² (beyond covariates):")
    for row in summary_rows:
        print(f"  {row['subblock']:<6s} "
              f"geno_PC1={row['geno_pc1_incr_r2']:>10.5f}  "
              f"emb_PC1={row['emb_pc1_incr_r2']:>10.5f}  "
              f"geno_norm3={row['geno_norm3_incr_r2']:>12.5f}  "
              f"emb_norm3={row['emb_norm3_incr_r2']:>12.5f}")
    print(f"\n{'═'*72}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI dispatch
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="07_17q21_validation — 17q21 subblock validation and baseline comparison"
    )
    ap.add_argument(
        "--mode", choices=["all", "validation", "baseline"], default="all",
        help=("Which analysis to run. 'all' runs both (default). "
              "'validation' = figures + tables (Step A). "
              "'baseline' = OLS geno vs emb comparison, terminal output only (Step B).")
    )
    args = ap.parse_args()

    if args.mode in ("all", "validation"):
        run_validation()

    if args.mode in ("all", "baseline"):
        run_baseline_comparison()


if __name__ == "__main__":
    main()
