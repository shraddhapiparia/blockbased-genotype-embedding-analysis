#!/usr/bin/env python3
"""
17q21_validation_combined.py

Combined validation script for 17q21 subblocks (sb1-sb5).

Produces:
  Figures (saved to OUT_DIR/figures/):
    fig1_block_level_bar.png        — geno PC1 ↔ emb PC1 Spearman r per subblock
    fig2_snp_scatter_all_sb.png     — SNP dosage vs embedding PC1 for all 5 subblocks
    fig3_interblock_heatmap.png     — inter-subblock embedding PC1 correlations
    fig4_fev1_volcano_17q21.png     — FEV1 volcano with 17q21 highlighted

  Tables (saved to OUT_DIR/tables/):
    snp_correlations_sb1.tsv        — all SNPs for each subblock
    snp_correlations_sb2.tsv
    ...
    snp_correlations_all_subblocks.tsv

Prints terminal summary throughout.
"""

import time
import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = "data/region_blocks"
EMB_BLOCK_NPY = "results/output_regions2/ORD/embeddings/block_contextual_repr.npy"
ATTN_CSV      = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
BLOCK_ORDER   = "results/output_regions/block_order.csv"
ASSOC_TSV     = "results/output_regions2/ORD/all_blocks_pheno_analysis/phenotype_block_associations.tsv"
OUT_DIR       = "results/output_regions2/ORD/17q21_validation"

SUBBLOCKS     = [f"region_17q21_core_sb{i}" for i in range(1, 6)]
PLINK_META    = {"FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"}
N_BLOCK_PCS   = 3

# Ensembl
ENSEMBL_SLEEP = 0.34
ENSEMBL_BATCH = 200

# ── plot style ────────────────────────────────────────────────────────────────
PALETTE_POS = "#2166ac"
PALETTE_NEG = "#d6604d"
PALETTE_HL  = "#1a1a2e"
GREY_LIGHT  = "#cccccc"
FONT_TITLE  = 13
FONT_LABEL  = 11
FONT_TICK   = 9
DPI         = 180

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})


# ── helpers ───────────────────────────────────────────────────────────────────
def load_block_names(path, expected_B):
    bo = pd.read_csv(path)
    col = next((c for c in ["block","block_id","region","name"] if c in bo.columns), None)
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


def geno_pc1_vec(dosage_matrix):
    X = SimpleImputer(strategy="mean").fit_transform(dosage_matrix.astype(float))
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1, random_state=42)
    return pca.fit_transform(X).ravel(), pca.explained_variance_ratio_[0]


def emb_top_pcs(block_repr, block_idx, k):
    X = block_repr[:, block_idx, :]
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=k, random_state=42)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def sign_align(emb, geno):
    r, _ = stats.spearmanr(emb, geno)
    return emb * (-1 if r < 0 else 1), r < 0


def spearman_ci(x, y, n_boot=1000):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 20:
        return dict(r=np.nan, p=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=len(x))
    r, p = stats.spearmanr(x, y)
    rng  = np.random.default_rng(42)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) > 1:
            br, _ = stats.spearmanr(x[idx], y[idx])
            boot.append(br)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return dict(r=float(r), p=float(p),
                ci_lo=float(ci_lo), ci_hi=float(ci_hi), n=len(x))


def parse_snp_name(snp_raw):
    core  = snp_raw.rsplit("_", 1)[0] if "_" in snp_raw else snp_raw
    parts = core.split(":")
    if len(parts) >= 4:
        return dict(chr=parts[0], pos=int(parts[1]), ref=parts[2], alt=parts[3])
    elif len(parts) == 3:
        return dict(chr=parts[0], pos=int(parts[1]), ref=parts[2], alt=".")
    return dict(chr=".", pos=0, ref=".", alt=".")


# ── Ensembl rsID lookup ───────────────────────────────────────────────────────
def lookup_rsids_ensembl(positions):
    unique_pos = list({(p["chr"], p["pos"]) for p in positions})
    total      = len(unique_pos)
    result     = {}
    print(f"\n  Looking up rsIDs for {total} unique positions via Ensembl …")

    for done, (chrom, pos) in enumerate(unique_pos, 1):
        key = (str(chrom), int(pos))
        try:
            url  = (f"https://rest.ensembl.org/overlap/region/human/"
                    f"{chrom}:{pos}-{pos}"
                    f"?feature=variation&content-type=application/json")
            resp = requests.get(url, timeout=15)

            if resp.status_code == 429:
                print(f"\n    [rate limit] backing off 5s …")
                time.sleep(5)
                resp = requests.get(url, timeout=15)

            if resp.status_code == 200:
                rs_ids = [v["id"] for v in resp.json()
                          if isinstance(v.get("id"), str)
                          and v["id"].startswith("rs")]
                result[key] = rs_ids[0] if rs_ids else "NA"
            else:
                result[key] = "NA"
        except Exception:
            result[key] = "NA"

        time.sleep(ENSEMBL_SLEEP)
        if done % 20 == 0 or done == total:
            print(f"    {done}/{total} ({100*done/total:.0f}%) …", end="\r")

    found = sum(1 for v in result.values() if v != "NA")
    print(f"\n  rsID lookup complete — found {found}/{total}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_block_level_bar(block_results, out_path):
    labels = [r["name"].replace("region_17q21_core_", "").upper()
              for r in block_results]
    rs     = [r["r"]     for r in block_results]
    ci_lo  = [r["ci_lo"] for r in block_results]
    ci_hi  = [r["ci_hi"] for r in block_results]
    colors = [PALETTE_POS if r >= 0 else PALETTE_NEG for r in rs]
    xerr_lo = [abs(r - lo) for r, lo in zip(rs, ci_lo)]
    xerr_hi = [abs(hi - r) for r, hi in zip(rs, ci_hi)]

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos   = range(len(labels))

    ax.barh(y_pos, rs, color=colors, alpha=0.85, height=0.55, zorder=3)
    ax.errorbar(rs, list(y_pos), xerr=[xerr_lo, xerr_hi],
                fmt="none", color="#333333", capsize=4, linewidth=1.2, zorder=4)

    for i, r_val in enumerate(rs):
        x_text = r_val + (0.03 if r_val >= 0 else -0.03)
        ha     = "left" if r_val >= 0 else "right"
        ax.text(x_text, i, f"{r_val:+.3f}", va="center", ha=ha,
                fontsize=FONT_TICK, color=PALETTE_HL, fontweight="bold")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=FONT_LABEL)
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Spearman r  (Genotype PC1 ↔ Embedding PC1)", fontsize=FONT_LABEL)
    ax.set_title("17q21 Subblocks: Genotype–Embedding Alignment\n"
                 "Alternating signs reflect known inversion LD structure",
                 fontsize=FONT_TITLE, pad=10)
    ax.set_xlim(-1.05, 1.15)

    pos_patch = mpatches.Patch(color=PALETTE_POS, label="Positive correlation")
    neg_patch = mpatches.Patch(color=PALETTE_NEG, label="Negative correlation (sign flip)")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8,
              loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


def plot_snp_scatter(scatter_data, out_path):
    n     = len(scatter_data)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows),
                             sharey=False)
    axes = np.array(axes).flatten()

    # hide unused panels
    for ax in axes[n:]:
        ax.set_visible(False)

    for ax, item in zip(axes, scatter_data):
        dos  = item["dosage"]
        epc1 = item["epc1"]
        r    = item["r"]
        snp  = item["snp"]
        name = item["name"].replace("region_17q21_core_", "").upper()

        rng   = np.random.default_rng(0)
        x_jit = dos + rng.uniform(-0.15, 0.15, size=len(dos))

        colors_dot = np.where(dos == 2, PALETTE_POS,
                     np.where(dos == 0, GREY_LIGHT, "#74a9cf"))

        ax.scatter(x_jit, epc1, c=colors_dot, s=10, alpha=0.5, zorder=2)

        for d_val in [0, 1, 2]:
            mask = dos == d_val
            if mask.sum() > 0:
                ax.hlines(epc1[mask].mean(), d_val - 0.3, d_val + 0.3,
                          colors=PALETTE_HL, linewidth=2.5, zorder=3)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["0\n(hom ref)", "1\n(het)", "2\n(hom alt)"],
                           fontsize=FONT_TICK)
        ax.set_xlabel("SNP Dosage", fontsize=FONT_LABEL)
        ax.set_ylabel("Embedding PC1", fontsize=FONT_LABEL)

        # parse SNP display name — use rsid if available else chr:pos
        rsid = item.get("rsid", "NA")
        snp_label = rsid if rsid != "NA" else snp.split("_")[0]

        ax.set_title(f"{name}\n{snp_label}\nSpearman r = {r:+.3f}",
                     fontsize=FONT_TITLE - 1, pad=8)

        ymin = epc1.min() - 0.3
        for d_val in [0, 1, 2]:
            n_d = (dos == d_val).sum()
            ax.text(d_val, ymin, f"n={n_d}", ha="center",
                    fontsize=7, color="#666666")

    fig.suptitle("SNP Dosage vs Embedding PC1  (group means = horizontal lines)",
                 fontsize=FONT_TITLE, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


def plot_interblock_heatmap(epc1_matrix, subblock_names, out_path):
    labels = [n.replace("region_17q21_core_", "").upper() for n in subblock_names]
    n      = len(labels)
    rmat   = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                r, _ = stats.spearmanr(epc1_matrix[:, i], epc1_matrix[:, j])
                rmat[i, j] = r

    rdf  = pd.DataFrame(rmat, index=labels, columns=labels)
    mask = np.eye(n, dtype=bool)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(rdf, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="#dddddd",
                annot_kws={"size": 10, "weight": "bold"},
                ax=ax, mask=mask, cbar_kws={"label": "Spearman r"})

    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                   fill=True, color="#f0f0f0", zorder=2))
        ax.text(i + 0.5, i + 0.5, "1.00", ha="center", va="center",
                fontsize=10, color="#aaaaaa")

    ax.set_title("Inter-Subblock Embedding PC1 Correlations\n"
                 "(Alternating signs = 17q21 inversion LD structure)",
                 fontsize=FONT_TITLE, pad=10)
    ax.tick_params(axis="both", labelsize=FONT_LABEL)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


def plot_fev1_volcano(assoc_path, out_path):
    assoc = pd.read_csv(assoc_path, sep="\t")
    fev   = assoc[assoc["phenotype"] == "pctpred_fev1_pre_BD"].copy()
    fev   = fev.dropna(subset=["beta_adj", "pval_adj"])
    fev["neg_log10_p"] = -np.log10(fev["pval_adj"].clip(lower=1e-30))
    fev["is_17q21"]    = fev["block"].str.contains("17q21", case=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    other = fev[~fev["is_17q21"]]
    ax.scatter(other["beta_adj"], other["neg_log10_p"],
               s=14, alpha=0.3, color=GREY_LIGHT, zorder=2, label="Other blocks")

    q21 = fev[fev["is_17q21"]]
    ax.scatter(q21["beta_adj"], q21["neg_log10_p"],
               s=60, alpha=0.9, color=PALETTE_NEG,
               edgecolors=PALETTE_HL, linewidth=0.6,
               zorder=4, label="17q21 subblocks")

    for _, row in q21.iterrows():
        short = row["block"].replace("region_17q21_core_", "sb")
        ax.annotate(short,
                    xy=(row["beta_adj"], row["neg_log10_p"]),
                    xytext=(6, 3), textcoords="offset points",
                    fontsize=7.5, color=PALETTE_HL,
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6))

    ax.axhline(-np.log10(0.05), color="#888888", linewidth=0.9,
               linestyle="--", label="p = 0.05")
    ax.axvline(0, color="#cccccc", linewidth=0.7)
    ax.set_xlabel("β  (block PC1 effect, ancestry-adjusted)", fontsize=FONT_LABEL)
    ax.set_ylabel("−log₁₀(p adjusted)", fontsize=FONT_LABEL)
    ax.set_title("Association with Predicted FEV1 (pctpred_fev1_pre_BD)\n"
                 "All blocks — 17q21 subblocks highlighted",
                 fontsize=FONT_TITLE, pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.tick_params(labelsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    fig_dir = Path(OUT_DIR) / "figures"
    tbl_dir = Path(OUT_DIR) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # ── load embeddings ───────────────────────────────────────────────────────
    print("Loading block embeddings …")
    emb_block   = np.load(EMB_BLOCK_NPY)
    N, B, D     = emb_block.shape
    print(f"  shape: {emb_block.shape}")

    attn_df     = pd.read_csv(ATTN_CSV)
    subj_iids   = attn_df["IID"].astype(str).values
    block_names = load_block_names(BLOCK_ORDER, B)
    iid_to_idx  = {iid: i for i, iid in enumerate(subj_iids)}

    # ── per-subblock processing ───────────────────────────────────────────────
    block_results = []   # fig 1
    scatter_data  = []   # fig 2
    epc1_raw_cols = []   # fig 3 — raw (no sign flip)
    present_names = []
    all_snp_rows  = []   # tables
    all_parsed    = []   # for rsID lookup

    print("\nProcessing subblocks …")

    for sb_name in SUBBLOCKS:
        raw_path = Path(RAW_DIR) / f"{sb_name}.raw"
        print(f"\n  {sb_name}")

        if not raw_path.exists():
            print(f"    [SKIP] not found")
            continue

        try:
            b_idx = block_names.index(sb_name)
        except ValueError:
            print(f"    [SKIP] not in block_order")
            continue

        raw_df, snp_cols = load_raw_plink(raw_path)
        common_iids = [iid for iid in raw_df["IID"] if iid in iid_to_idx]
        if len(common_iids) < 50:
            print(f"    [SKIP] n={len(common_iids)}")
            continue

        raw_aligned = raw_df.set_index("IID").loc[common_iids]
        emb_rows    = [iid_to_idx[iid] for iid in common_iids]

        # embedding PCs
        k = min(N_BLOCK_PCS, D)
        emb_pcs, emb_var = emb_top_pcs(emb_block, b_idx, k)
        emb_sub          = emb_pcs[emb_rows, :]

        # genotype PC1
        dosage = raw_aligned[snp_cols].values.astype(float)
        dosage[dosage == -9] = np.nan
        gpc1, gpc1_var = geno_pc1_vec(dosage)

        print(f"    SNPs={len(snp_cols)}  subjects={len(common_iids)}")
        print(f"    Geno PC1 var={100*gpc1_var:.1f}%  "
              f"Emb PC var: {' '.join(f'ePC{i+1}={100*v:.1f}%' for i,v in enumerate(emb_var))}")

        # block-level correlation (used for fig 1)
        res = spearman_ci(gpc1, emb_sub[:, 0])
        print(f"    geno PC1 ↔ ePC1  r={res['r']:+.4f}  p={res['p']:.3g}")

        block_results.append({
            "name":  sb_name,
            "r":     res["r"],
            "ci_lo": res["ci_lo"],
            "ci_hi": res["ci_hi"],
            "p":     res["p"],
        })

        # sign-aligned ePC1 for scatter
        epc1_aligned, flipped = sign_align(emb_sub[:, 0], gpc1)
        if flipped:
            print(f"    [sign flipped for scatter]")

        # raw ePC1 (no sign flip) for heatmap
        epc1_raw_cols.append(emb_pcs[:, 0])   # full N subjects
        present_names.append(sb_name)

        # SNP-level correlations
        snp_rs = []
        for snp in snp_cols:
            dos  = raw_aligned[snp].values.astype(float)
            dos[dos == -9] = np.nan
            mask    = ~np.isnan(dos)
            n_valid = mask.sum()
            if n_valid < 50 or np.nanstd(dos[mask]) < 1e-6:
                continue
            r_s, p_s  = stats.spearmanr(epc1_aligned[mask], dos[mask])
            mean_dos  = float(np.nanmean(dos))
            maf       = min(mean_dos / 2, 1 - mean_dos / 2)
            parsed    = parse_snp_name(snp)
            all_parsed.append(parsed)
            snp_rs.append({
                "subblock": sb_name,
                "snp_raw":  snp,
                "chr":      parsed["chr"],
                "pos":      parsed["pos"],
                "ref":      parsed["ref"],
                "alt":      parsed["alt"],
                "r":        float(r_s),
                "p":        float(p_s),
                "mean_dos": round(mean_dos, 4),
                "maf":      round(maf, 4),
                "n_valid":  int(n_valid),
            })

        all_snp_rows.extend(snp_rs)

        # scatter data (top SNP per subblock)
        if snp_rs:
            snp_rs_sorted = sorted(snp_rs, key=lambda x: -abs(x["r"]))
            best = snp_rs_sorted[0]
            best_dos = raw_aligned[best["snp_raw"]].values.astype(float)
            best_dos[best_dos == -9] = np.nan
            scatter_data.append({
                "name":    sb_name,
                "snp":     best["snp_raw"],
                "r":       best["r"],
                "dosage":  best_dos,
                "epc1":    epc1_aligned,
                "rsid":    "NA",   # filled in after lookup
            })

    # ── rsID lookup ───────────────────────────────────────────────────────────
    rsid_map = lookup_rsids_ensembl(all_parsed)

    # attach rsIDs to snp rows
    for row in all_snp_rows:
        row["rsid"] = rsid_map.get((str(row["chr"]), int(row["pos"])), "NA")

    # attach rsIDs to scatter data (top SNP per subblock)
    for item in scatter_data:
        parsed = parse_snp_name(item["snp"])
        item["rsid"] = rsid_map.get((str(parsed["chr"]), int(parsed["pos"])), "NA")

    # ── save SNP tables ───────────────────────────────────────────────────────
    print("\nSaving SNP correlation tables …")

    combined_rows = []
    col_order = ["subblock","snp_raw","chr","pos","ref","alt","rsid",
                 "r","p","fdr","mean_dos","maf","n_valid"]

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

        # terminal top 10
        n_fdr = (df["fdr"] < 0.05).sum()
        n_rs  = (df["rsid"] != "NA").sum()
        print(f"    FDR<0.05={n_fdr}  rsIDs found={n_rs}")
        print(f"    {'rsid':<15s} {'pos':>10s} {'r':>8s} {'p':>12s} "
              f"{'fdr':>12s} {'maf':>7s}")
        print(f"    {'-'*15} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*7}")
        for _, row in df.head(10).iterrows():
            print(f"    {row['rsid']:<15s} {int(row['pos']):>10d} "
                  f"{row['r']:>+8.4f} {row['p']:>12.4e} "
                  f"{row['fdr']:>12.4e} {row['maf']:>7.4f}")

        combined_rows.append(df)

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        comb_path = tbl_dir / "snp_correlations_all_subblocks.tsv"
        combined.to_csv(comb_path, sep="\t", index=False, float_format="%.6e")
        print(f"\n  saved combined: {comb_path.name}  ({len(combined)} rows total)")

    # ── save figures ──────────────────────────────────────────────────────────
    print("\nSaving figures …")

    if block_results:
        plot_block_level_bar(block_results, fig_dir / "fig1_block_level_bar.png")

    if scatter_data:
        plot_snp_scatter(scatter_data, fig_dir / "fig2_snp_scatter_all_sb.png")

    if len(epc1_raw_cols) >= 2:
        epc1_matrix = np.column_stack(epc1_raw_cols)
        plot_interblock_heatmap(epc1_matrix, present_names,
                                fig_dir / "fig3_interblock_heatmap.png")

    if Path(ASSOC_TSV).exists():
        plot_fev1_volcano(ASSOC_TSV, fig_dir / "fig4_fev1_volcano_17q21.png")
    else:
        print(f"  [warn] ASSOC_TSV not found — skipping fig4")

    # ── final summary ─────────────────────────────────────────────────────────
    # build lookup from combined df which has fdr computed
    if combined_rows:
        combined_lookup = pd.concat(combined_rows, ignore_index=True)
    else:
        combined_lookup = pd.DataFrame()

    print(f"\n{'═'*70}")
    print("  Final summary")
    print(f"{'═'*70}")
    print(f"  {'subblock':<10s} {'r':>8s} {'95% CI':>20s} "
        f"{'SNPs':>6s} {'FDR<0.05':>9s} {'rsIDs':>7s}")
    print(f"  {'-'*10} {'-'*8} {'-'*20} {'-'*6} {'-'*9} {'-'*7}")

    for br in block_results:
        sb_s = br["name"].replace("region_17q21_core_", "")
        ci   = f"[{br['ci_lo']:+.3f}, {br['ci_hi']:+.3f}]"

        if len(combined_lookup) > 0:
            sub   = combined_lookup[combined_lookup["subblock"] == br["name"]]
            n_snp = len(sub)
            n_fdr = (sub["fdr"] < 0.05).sum()
            n_rs  = (sub["rsid"] != "NA").sum()
        else:
            n_snp = n_fdr = n_rs = 0

        print(f"  {sb_s:<10s} {br['r']:>+8.4f} {ci:>20s} "
            f"{n_snp:>6d} {n_fdr:>9d} {n_rs:>7d}")

    print(f"\n  Figures → {fig_dir}/")
    print(f"  Tables  → {tbl_dir}/")


if __name__ == "__main__":
    main()