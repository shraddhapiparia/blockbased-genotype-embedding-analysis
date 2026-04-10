#!/usr/bin/env python3
"""
10_hla_cluster_analysis.py

Goal
----
Cluster subjects using global subject embeddings, then test whether
top HLA class II block-level embedding summaries (block_PC1) differ by cluster.

Added in this version
---------------------
1. Load genotype PCs from eigenvec
2. Test whether subject clusters are largely explained by genotype PCs
3. Save cluster-vs-genotype-PC summaries and plots

Interpretation
--------------
- HLA block_PC1 ~ cluster tells us which HLA sub-blocks organize the learned geometry
- genotype PC ~ cluster tells us whether those clusters mostly reflect ancestry-like structure
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── paths ─────────────────────────────────────────────────────────────
EMB_SUBJ_NPY  = "results/output_regions2/ORD/embeddings/individual_embeddings.npy"
EMB_BLOCK_NPY = "results/output_regions2/ORD/embeddings/block_contextual_repr.npy"
ATTN_CSV      = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
BLOCK_ORDER   = "results/output_regions/block_order.csv"
EIGENVEC_FILE = "metadata/ldpruned_997subs.eigenvec"
OUT_DIR       = "results/output_regions2/ORD/cytokine_cluster_analysis"

PC_COLS = [f"PC{i}" for i in range(1, 11)]

# Top HLA blocks from gradient analysis
# HLA_BLOCKS = [
#     "region_6p21_HLA_classII_sb15",
#     "region_6p21_HLA_classII_sb10",
#     "region_6p21_HLA_classII_sb4",
#     "region_6p21_HLA_classII_sb12",
#     "region_6p21_HLA_classII_sb9",
#     "region_6p21_HLA_classII_sb11",
#     "region_6p21_HLA_classII_sb14",
#     "region_6p21_HLA_classII_sb7",
#     "region_6p21_HLA_classII_sb5",
#     "region_6p21_HLA_classII_sb8",
#     "region_6p21_HLA_classII_sb1",
#     "region_6p21_HLA_classII_sb13",
#     "region_6p21_HLA_classII_sb17",
#     "region_6p21_HLA_classII_sb6",
#     "region_6p21_HLA_classII_sb3",
#     "region_6p21_HLA_classII_sb2",
# ]
HLA_BLOCKS = [
    "region_17q21_core_sb1",
"region_17q21_core_sb2",
"region_17q21_core_sb3",
"region_17q21_core_sb4",
"region_17q21_core_sb5",
]

# ── helpers ───────────────────────────────────────────────────────────
def load_block_names(block_order_path: str, expected_B: int):
    bo = pd.read_csv(block_order_path)
    possible_cols = ["block", "block_id", "region", "name"]
    name_col = next((c for c in possible_cols if c in bo.columns), None)
    if name_col is None:
        raise ValueError(
            f"No block-name column found in {block_order_path}. "
            f"Available columns: {list(bo.columns)}"
        )
    block_names = bo[name_col].astype(str).tolist()
    if len(block_names) != expected_B:
        raise ValueError(
            f"block_order length {len(block_names)} != embedding blocks {expected_B}"
        )
    return block_names

def load_eigenvec(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lstrip("#")
    df["IID"] = df["IID"].astype(str)
    return df[["IID"] + PC_COLS]

def block_first_pc(block_repr: np.ndarray, block_idx: int) -> np.ndarray:
    X = block_repr[:, block_idx, :]
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1, random_state=42)
    pc = pca.fit_transform(X).ravel()

    # consistent sign anchoring
    if pca.components_[0, 0] < 0:
        pc = -pc
    return pc

def eta_squared_from_anova(groups):
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean) ** 2)

    if ss_total == 0:
        return np.nan
    return float(ss_between / ss_total)

def test_continuous_vs_cluster(values: np.ndarray, cluster: np.ndarray):
    df = pd.DataFrame({"value": values, "cluster": cluster}).dropna()
    groups = [g["value"].values for _, g in df.groupby("cluster")]

    if len(groups) < 2 or any(len(g) < 5 for g in groups):
        return {
            "n": len(df),
            "anova_F": np.nan,
            "anova_p": np.nan,
            "kruskal_H": np.nan,
            "kruskal_p": np.nan,
            "eta2": np.nan,
        }

    F, p_anova = stats.f_oneway(*groups)
    H, p_kw = stats.kruskal(*groups)
    eta2 = eta_squared_from_anova(groups)

    return {
        "n": len(df),
        "anova_F": float(F),
        "anova_p": float(p_anova),
        "kruskal_H": float(H),
        "kruskal_p": float(p_kw),
        "eta2": float(eta2),
    }

def cramers_v_from_chi2(chi2, n, r, k):
    denom = n * max(1, min(r - 1, k - 1))
    if denom <= 0:
        return np.nan
    return float(np.sqrt(chi2 / denom))

def test_cluster_vs_binned_pc(values: np.ndarray, cluster: np.ndarray, q=4):
    df = pd.DataFrame({"value": values, "cluster": cluster}).dropna()
    if len(df) < 20:
        return {"chi2": np.nan, "chi2_p": np.nan, "cramers_v": np.nan}

    try:
        df["bin"] = pd.qcut(df["value"], q=q, duplicates="drop")
    except Exception:
        return {"chi2": np.nan, "chi2_p": np.nan, "cramers_v": np.nan}

    tab = pd.crosstab(df["cluster"], df["bin"])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return {"chi2": np.nan, "chi2_p": np.nan, "cramers_v": np.nan}

    chi2, p, _, _ = stats.chi2_contingency(tab)
    cv = cramers_v_from_chi2(chi2, tab.values.sum(), *tab.shape)
    return {"chi2": float(chi2), "chi2_p": float(p), "cramers_v": float(cv)}

def save_boxplot(df: pd.DataFrame, col: str, out_path: Path, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.boxplot(data=df, x="cluster", y=col, ax=ax, color="white", width=0.55)
    sns.stripplot(data=df, x="cluster", y=col, ax=ax, alpha=0.45, size=3)

    means = df.groupby("cluster")[col].mean().round(3).to_dict()
    counts = df.groupby("cluster")[col].size().to_dict()

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Cluster")
    ax.set_ylabel(ylabel)

    text = " | ".join(
        [f"C{k}: mean={means.get(k, np.nan):.3f}, n={counts.get(k, 0)}"
         for k in sorted(df["cluster"].dropna().unique())]
    )
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=8, va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_subject_cluster_plot(df: pd.DataFrame, xcol: str, ycol: str, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df,
        x=xcol,
        y=ycol,
        hue="cluster",
        palette="tab10",
        s=28,
        alpha=0.8,
        ax=ax,
    )
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_colored_scatter(df: pd.DataFrame, xcol: str, ycol: str, color_col: str, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        df[xcol], df[ycol],
        c=df[color_col],
        cmap="plasma",
        s=20,
        alpha=0.8
    )
    plt.colorbar(sc, ax=ax, label=color_col)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ── main ──────────────────────────────────────────────────────────────
def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("Loading embeddings...")
    emb_subj = np.load(EMB_SUBJ_NPY)
    emb_block = np.load(EMB_BLOCK_NPY)
    N, B, D = emb_block.shape
    print(f"  emb_subj : {emb_subj.shape}")
    print(f"  emb_block: {emb_block.shape}")

    attn_df = pd.read_csv(ATTN_CSV)
    subj_iids = attn_df["IID"].astype(str).values

    block_names = load_block_names(BLOCK_ORDER, B)
    block_to_idx = {b: i for i, b in enumerate(block_names)}

    missing = [b for b in HLA_BLOCKS if b not in block_to_idx]
    if missing:
        raise ValueError(f"These HLA blocks were not found in block_order.csv:\n{missing}")

    print("Loading genotype PCs...")
    pcs_df = load_eigenvec(EIGENVEC_FILE)

    print("Computing subject PCA...")
    Zs = StandardScaler().fit_transform(emb_subj)
    subj_pca_model = PCA(n_components=10, random_state=42)
    subj_pca = subj_pca_model.fit_transform(Zs)
    explained = subj_pca_model.explained_variance_ratio_ * 100

    print("Clustering subjects with KMeans(k=3) on subject PCA(1:10)...")
    km = KMeans(n_clusters=3, random_state=42, n_init=20)
    clusters_raw = km.fit_predict(subj_pca[:, :10])

    # relabel clusters by mean PCA1 for consistency: left -> middle -> right
    tmp = pd.DataFrame({"cluster_raw": clusters_raw, "PC1": subj_pca[:, 0]})
    order = tmp.groupby("cluster_raw")["PC1"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    clusters = np.array([remap[c] for c in clusters_raw], dtype=int)

    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    print("\nCluster sizes:")
    for k, v in cluster_counts.items():
        print(f"  cluster {k}: n={v}")

    subject_df = pd.DataFrame({
        "IID": subj_iids,
        "cluster": clusters,
        "embedPC1": subj_pca[:, 0],
        "embedPC2": subj_pca[:, 1],
        "embedPC3": subj_pca[:, 2],
    })

    # merge genotype PCs
    subject_df = subject_df.merge(pcs_df, on="IID", how="left")
    subject_df.to_csv(out / "subject_clusters.tsv", sep="\t", index=False)

    save_subject_cluster_plot(
        subject_df,
        xcol="embedPC1",
        ycol="embedPC2",
        out_path=fig_dir / "subject_clusters_pca.png",
        title=f"Subject clusters on embedding PCA\nPC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%"
    )

    print("\nTesting genotype PCs ~ cluster ...")
    geno_rows = []
    for pc in PC_COLS:
        vals = pd.to_numeric(subject_df[pc], errors="coerce").values

        cont_res = test_continuous_vs_cluster(vals, subject_df["cluster"].values)
        chi_res = test_cluster_vs_binned_pc(vals, subject_df["cluster"].values, q=4)

        means = subject_df.groupby("cluster")[pc].mean().to_dict()
        medians = subject_df.groupby("cluster")[pc].median().to_dict()

        geno_rows.append({
            "geno_pc": pc,
            **cont_res,
            **chi_res,
            "cluster0_mean": float(means.get(0, np.nan)),
            "cluster1_mean": float(means.get(1, np.nan)),
            "cluster2_mean": float(means.get(2, np.nan)),
            "cluster0_median": float(medians.get(0, np.nan)),
            "cluster1_median": float(medians.get(1, np.nan)),
            "cluster2_median": float(medians.get(2, np.nan)),
        })

        save_boxplot(
            subject_df[["cluster", pc]].copy(),
            col=pc,
            out_path=fig_dir / f"{pc}_by_cluster.png",
            title=f"{pc} by subject cluster",
            ylabel=pc,
        )

        save_colored_scatter(
            subject_df,
            xcol="embedPC1",
            ycol="embedPC2",
            color_col=pc,
            out_path=fig_dir / f"subject_pca_colored_by_{pc}.png",
            title=f"Subject embedding PCA colored by {pc}",
        )

    geno_summary = pd.DataFrame(geno_rows)

    for pcol, qcol in [
        ("anova_p", "anova_fdr_bh"),
        ("kruskal_p", "kruskal_fdr_bh"),
        ("chi2_p", "chi2_fdr_bh"),
    ]:
        pvals = geno_summary[pcol].values
        mask = np.isfinite(pvals)
        qvals = np.full(len(geno_summary), np.nan)
        if mask.sum() > 0:
            _, qtmp, _, _ = multipletests(pvals[mask], method="fdr_bh")
            qvals[mask] = qtmp
        geno_summary[qcol] = qvals

    geno_summary = geno_summary.sort_values(
        ["eta2", "kruskal_p"],
        ascending=[False, True]
    ).reset_index(drop=True)

    geno_summary.to_csv(out / "genotype_pc_cluster_association.tsv", sep="\t", index=False)

    print("\nComputing HLA block_PC1 values...")
    hla_df = subject_df.copy()

    for block in HLA_BLOCKS:
        idx = block_to_idx[block]
        hla_df[block] = block_first_pc(emb_block, idx)

    hla_df.to_csv(out / "hla_block_pc1_by_subject.tsv", sep="\t", index=False)

    

    # heatmap of genotype PC cluster means
    geno_heat = geno_summary.set_index("geno_pc")[["cluster0_mean", "cluster1_mean", "cluster2_mean"]]
    fig, ax = plt.subplots(figsize=(6, max(4, len(geno_heat) * 0.45)))
    sns.heatmap(
        geno_heat,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Genotype PC mean by subject cluster", fontsize=12)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(fig_dir / "genotype_pc_cluster_mean_heatmap.png", dpi=160)
    plt.close()

    print("\nTop genotype PCs separating clusters:")
    show_cols = [
        "geno_pc", "eta2", "anova_p", "anova_fdr_bh",
        "kruskal_p", "kruskal_fdr_bh",
        "chi2_p", "chi2_fdr_bh", "cramers_v",
        "cluster0_mean", "cluster1_mean", "cluster2_mean"
    ]
    print(geno_summary[show_cols].head(10).to_string(index=False))

    
    print("Testing block_PC1 ~ cluster ...")
    rows = []
    for block in HLA_BLOCKS:
        res = test_continuous_vs_cluster(hla_df[block].values, hla_df["cluster"].values)

        cluster_means = hla_df.groupby("cluster")[block].mean().to_dict()
        cluster_medians = hla_df.groupby("cluster")[block].median().to_dict()

        row = {
            "block": block,
            **res,
            "cluster0_mean": float(cluster_means.get(0, np.nan)),
            "cluster1_mean": float(cluster_means.get(1, np.nan)),
            "cluster2_mean": float(cluster_means.get(2, np.nan)),
            "cluster0_median": float(cluster_medians.get(0, np.nan)),
            "cluster1_median": float(cluster_medians.get(1, np.nan)),
            "cluster2_median": float(cluster_medians.get(2, np.nan)),
        }
        rows.append(row)

        save_boxplot(
            hla_df[["cluster", block]].copy(),
            col=block,
            out_path=fig_dir / f"{block}_by_cluster.png",
            title=f"{block}\nblock_PC1 by subject cluster",
            ylabel="block_PC1",
        )

    summary = pd.DataFrame(rows)

    for pcol, qcol in [("anova_p", "anova_fdr_bh"), ("kruskal_p", "kruskal_fdr_bh")]:
        pvals = summary[pcol].values
        mask = np.isfinite(pvals)
        qvals = np.full(len(summary), np.nan)
        if mask.sum() > 0:
            _, qtmp, _, _ = multipletests(pvals[mask], method="fdr_bh")
            qvals[mask] = qtmp
        summary[qcol] = qvals

    summary = summary.sort_values(
        ["eta2", "kruskal_p"],
        ascending=[False, True]
    ).reset_index(drop=True)

    summary.to_csv(out / "hla_block_cluster_association.tsv", sep="\t", index=False)

    print("\nComputing HLA block_PC1 residuals after regressing out genotype PCs...")
    pc_matrix = subject_df[PC_COLS].values.astype(float)
    pc_matrix_scaled = StandardScaler().fit_transform(pc_matrix)
    valid_mask = np.isfinite(pc_matrix_scaled).all(axis=1)

    resid_rows = []
    for block in HLA_BLOCKS:
        y = hla_df.loc[valid_mask, block].values
        X = pc_matrix_scaled[valid_mask]
        lr = LinearRegression().fit(X, y)
        residuals = y - lr.predict(X)
        cluster_valid = hla_df.loc[valid_mask, "cluster"].values
        res = test_continuous_vs_cluster(residuals, cluster_valid)
        resid_rows.append({"block": block, **res})

    resid_summary = pd.DataFrame(resid_rows).sort_values(
        ["eta2", "kruskal_p"], ascending=[False, True]
    ).reset_index(drop=True)
    resid_summary.to_csv(
        out / "hla_block_residual_cluster_association.tsv", sep="\t", index=False
    )

    # ── comparison table: raw eta2 vs residual eta2
    compare = summary[["block", "eta2"]].rename(columns={"eta2": "eta2_raw"}).merge(
        resid_summary[["block", "eta2"]].rename(columns={"eta2": "eta2_residual"}),
        on="block"
    )
    compare["eta2_drop"] = compare["eta2_raw"] - compare["eta2_residual"]
    compare["pct_retained"] = (compare["eta2_residual"] / compare["eta2_raw"] * 100).round(1)
    compare = compare.sort_values("eta2_raw", ascending=False).reset_index(drop=True)
    compare.to_csv(out / "hla_eta2_raw_vs_residual.tsv", sep="\t", index=False)

    print("\nRaw vs residual eta² (after regressing out genotype PCs):")
    print(compare.to_string(index=False))

    print("\nComputing inter-block PC1 correlations...")
    hla_pc1_cols = [b for b in HLA_BLOCKS if b in hla_df.columns]
    corr_matrix = hla_df[hla_pc1_cols].corr(method="spearman")
    corr_matrix.to_csv(out / "hla_block_pc1_spearman_corr.tsv", sep="\t")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.3,
        annot=True,
        fmt=".2f",
        ax=ax
    )
    ax.set_title(
        "Spearman correlation between HLA block_PC1 values\n"
        "(high r = same LD axis, not independent signals)",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "hla_block_pc1_correlation_heatmap.png", dpi=160)
    plt.close()

    # heatmap of HLA cluster means
    heat_cols = ["cluster0_mean", "cluster1_mean", "cluster2_mean"]
    heat_df = summary.set_index("block")[heat_cols]

    fig, ax = plt.subplots(figsize=(7, max(5, len(heat_df) * 0.35)))
    sns.heatmap(
        heat_df,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        ax=ax
    )
    ax.set_title("HLA block_PC1 mean by subject cluster", fontsize=12)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(fig_dir / "hla_block_cluster_mean_heatmap.png", dpi=160)
    plt.close()

    print("\nTop HLA blocks separating clusters:")
    show_cols = [
        "block", "eta2", "anova_p", "anova_fdr_bh",
        "kruskal_p", "kruskal_fdr_bh",
        "cluster0_mean", "cluster1_mean", "cluster2_mean"
    ]
    print(summary[show_cols].head(12).to_string(index=False))

    print(f"\nSaved outputs in: {out}")
    print("Main files:")
    print("  subject_clusters.tsv")
    print("  genotype_pc_cluster_association.tsv")
    print("  hla_block_residual_cluster_association.tsv")
    print("  hla_eta2_raw_vs_residual.tsv")
    print("  hla_block_pc1_by_subject.tsv")
    print("  hla_block_cluster_association.tsv")
    print("  figures/subject_clusters_pca.png")
    print("  figures/genotype_pc_cluster_mean_heatmap.png")
    print("  figures/hla_block_cluster_mean_heatmap.png")
    print("  figures/PC*_by_cluster.png")
    print("  figures/<block>_by_cluster.png")

if __name__ == "__main__":
    main()