#!/usr/bin/env python3
"""02_subject_cluster_analysis.py — subject-level embedding clustering and HLA analysis.

Purpose  : Three-stage subject-level analysis of Phase 2 embeddings.
           Step A — phenotype-free HDBSCAN clustering, PCA/UMAP, block-driver analysis.
           Step B — UMAP colored by attention weights; top-block identification.
           Step C — KMeans(k=3) clustering; HLA block_PC1 and genotype-PC association
                    per cluster; raw vs residual eta² comparison.
Inputs   : individual_embeddings.npy, block_contextual_repr.npy,
           pooling_attention_weights.csv, block_order.csv,
           metadata/ldpruned_997subs.eigenvec,
           (step B only) clustering/cluster_labels.csv, clustering/clustering_metrics.csv
Outputs  : subject_cluster_assignments.csv, PCA/UMAP plots, HLA cluster TSVs,
           genotype_pc_cluster_association.tsv, hla_block_cluster_association.tsv
Workflow : Active analysis — Step 2; cluster assignments feed 03_ (leave-HLA-out)
           and 04_ (cluster stability).

Usage
-----
  # run all three steps in sequence (default):
  python scripts/analysis/02_subject_cluster_analysis.py --step all

  # run individual steps:
  python scripts/analysis/02_subject_cluster_analysis.py --step clustering
  python scripts/analysis/02_subject_cluster_analysis.py --step umap_hla
  python scripts/analysis/02_subject_cluster_analysis.py --step hla_analysis
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

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    sns = None
    HAS_SNS = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    umap = None
    HAS_UMAP = False

try:
    import hdbscan as _hdbscan
    HAS_HDBSCAN = True
except ImportError:
    _hdbscan = None
    HAS_HDBSCAN = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════════════════════
# Shared path constants
# ══════════════════════════════════════════════════════════════════════════════

# Step A / B (09_ / 05_) — parameterised at CLI
# Step C (10_) — hardcoded paths below

# Step C paths
_EMB_SUBJ_NPY  = "results/output_regions2/ORD/embeddings/individual_embeddings.npy"
_EMB_BLOCK_NPY = "results/output_regions2/ORD/embeddings/block_contextual_repr.npy"
_ATTN_CSV      = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
_BLOCK_ORDER   = "results/output_regions/block_order.csv"
_EIGENVEC_FILE = "metadata/ldpruned_997subs.eigenvec"
_OUT_DIR_C     = "results/output_regions2/ORD/cytokine_cluster_analysis"

_PC_COLS = [f"PC{i}" for i in range(1, 11)]

# Top HLA blocks from gradient analysis
HLA_BLOCKS = [
    "region_6p21_HLA_classII_sb15",
    "region_6p21_HLA_classII_sb10",
    "region_6p21_HLA_classII_sb4",
    "region_6p21_HLA_classII_sb12",
    "region_6p21_HLA_classII_sb9",
    "region_6p21_HLA_classII_sb11",
    "region_6p21_HLA_classII_sb14",
    "region_6p21_HLA_classII_sb7",
    "region_6p21_HLA_classII_sb5",
    "region_6p21_HLA_classII_sb8",
    "region_6p21_HLA_classII_sb1",
    "region_6p21_HLA_classII_sb13",
    "region_6p21_HLA_classII_sb17",
    "region_6p21_HLA_classII_sb6",
    "region_6p21_HLA_classII_sb3",
    "region_6p21_HLA_classII_sb2",
]


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_block_names(block_order_csv, expected_n_blocks: int):
    """Load block names from block_order.csv; fallback to generic names."""
    if block_order_csv is None:
        return [f"block_{i}" for i in range(expected_n_blocks)]
    df = pd.read_csv(block_order_csv)
    possible_cols = ["block", "block_id", "block_name", "region", "name"]
    col = next((c for c in possible_cols if c in df.columns), df.columns[0])
    names = df[col].astype(str).tolist()
    if len(names) != expected_n_blocks:
        raise ValueError(
            f"block_order.csv has {len(names)} names but expected {expected_n_blocks}"
        )
    return names


def load_eigenvec(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lstrip("#")
    df["IID"] = df["IID"].astype(str)
    return df[["IID"] + _PC_COLS]


# ══════════════════════════════════════════════════════════════════════════════
# Step A helpers  (from 09_unsupervised_subject_cluster_analysis.py)
# ══════════════════════════════════════════════════════════════════════════════

def _load_array(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    arr = np.load(path, allow_pickle=True)
    print(f"[load] {name}: {path}  shape={arr.shape}")
    return arr


def _standardize(X):
    return StandardScaler().fit_transform(X)


def _run_pca_umap(Xs, random_state=42, n_neighbors=30, min_dist=0.1):
    pca_model = PCA(n_components=2)
    X_pca = pca_model.fit_transform(Xs)
    pca_var = pca_model.explained_variance_ratio_ * 100.0
    umap_model = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors,
        min_dist=min_dist, metric="euclidean", random_state=random_state,
    )
    X_umap = umap_model.fit_transform(Xs)
    return X_pca, pca_var, X_umap


def _run_hdbscan(Xs, min_cluster_size=25, min_samples=None):
    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        metric="euclidean", cluster_selection_method="eom", prediction_data=True,
    )
    labels = clusterer.fit_predict(Xs)
    probs = getattr(clusterer, "probabilities_", None)
    valid = labels >= 0
    sil = (silhouette_score(Xs[valid], labels[valid])
           if valid.sum() >= 2 and len(np.unique(labels[valid])) >= 2 else np.nan)
    return clusterer, labels, probs, sil


def _make_color_map(labels):
    unique = sorted(np.unique(labels))
    cmap = plt.cm.get_cmap("tab20", max(len(unique), 1))
    return {lab: (0.7, 0.7, 0.7, 0.8) if lab == -1 else cmap(i)
            for i, lab in enumerate(unique)}


def _plot_embedding(ax, X2d, labels, title, xlabel, ylabel):
    color_map = _make_color_map(labels)
    for lab in sorted(np.unique(labels)):
        mask = labels == lab
        ax.scatter(X2d[mask, 0], X2d[mask, 1], s=16, alpha=0.8,
                   color=color_map[lab],
                   label=f"cluster {lab}" if lab != -1 else "noise")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="best", markerscale=1.2)


def _save_cluster_plots_ab(X_pca, pca_var, X_umap, labels, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _plot_embedding(axes[0], X_pca, labels,
                    title=f"Subject embeddings (PCA)\nPC1={pca_var[0]:.1f}%, PC2={pca_var[1]:.1f}%",
                    xlabel=f"PC1 ({pca_var[0]:.1f}% var)", ylabel=f"PC2 ({pca_var[1]:.1f}% var)")
    _plot_embedding(axes[1], X_umap, labels,
                    title="UMAP of subject embeddings", xlabel="UMAP-1", ylabel="UMAP-2")
    plt.tight_layout()
    out = outdir / "subject_clusters_pca_umap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print(f"[save] {out}")


def _block_scores_from_attention(attn):
    if attn.ndim != 2:
        raise ValueError(f"Expected (N, B), got {attn.shape}")
    return attn


def _block_scores_from_embeddings(block_emb, method="norm"):
    if block_emb.ndim != 3:
        raise ValueError(f"Expected (N, B, D), got {block_emb.shape}")
    if method == "norm":
        return np.linalg.norm(block_emb, axis=2)
    elif method == "pc1":
        n, b, d = block_emb.shape
        out = np.zeros((n, b), dtype=float)
        for j in range(b):
            out[:, j] = PCA(n_components=1, random_state=42).fit_transform(
                block_emb[:, j, :]).ravel()
        return out
    elif method == "mean":
        return block_emb.mean(axis=2)
    raise ValueError(f"Unknown block score method: {method}")


def _zscore_cols(X):
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True); sd[sd == 0] = 1.0
    return (X - mu) / sd


def _cluster_block_diffs(block_scores, labels, block_names, top_k=10):
    clusters = sorted([c for c in np.unique(labels) if c != -1])
    if not clusters:
        raise ValueError("No non-noise clusters found.")
    Bz = _zscore_cols(block_scores)
    diff_rows, top_rows = [], []
    for c in clusters:
        in_c, out_c = labels == c, labels != c
        diff = Bz[in_c].mean(axis=0) - Bz[out_c].mean(axis=0)
        diff_rows.append(pd.Series(diff, index=block_names, name=f"cluster_{c}"))
        for rank, idx in enumerate(np.argsort(-np.abs(diff))[:top_k], 1):
            top_rows.append({"cluster": c, "rank": rank, "block": block_names[idx],
                             "diff_z": diff[idx],
                             "mean_in_z": Bz[in_c].mean(axis=0)[idx],
                             "mean_out_z": Bz[out_c].mean(axis=0)[idx]})
    return pd.DataFrame(diff_rows), pd.DataFrame(top_rows)


def _save_block_heatmap(diff_df, top_blocks_df, outdir: Path, max_blocks=20):
    selected = []
    for blk in top_blocks_df["block"]:
        if blk not in selected:
            selected.append(blk)
        if len(selected) >= max_blocks:
            break
    plot_df = diff_df[selected].copy()
    plt.figure(figsize=(max(10, 0.6 * len(selected)), 4 + 0.5 * len(plot_df)))
    if HAS_SNS:
        sns.heatmap(plot_df, cmap="coolwarm", center=0, annot=False)
    else:
        plt.imshow(plot_df.values, aspect="auto")
        plt.colorbar(label="mean z-score difference")
    plt.title("Cluster × LD block contribution heatmap")
    plt.xlabel("LD blocks"); plt.ylabel("Subject clusters"); plt.tight_layout()
    out = outdir / "cluster_block_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print(f"[save] {out}")


def _cluster_blocks_by_corr(block_scores, block_names, outdir: Path, n_block_clusters=6):
    corr = np.nan_to_num(np.corrcoef(block_scores.T), nan=0.0)
    dist = 1.0 - corr; np.fill_diagonal(dist, 0.0)
    labels = AgglomerativeClustering(
        n_clusters=n_block_clusters, metric="precomputed", linkage="average"
    ).fit_predict(dist)
    pd.DataFrame({"block": block_names, "block_cluster": labels}).sort_values(
        ["block_cluster", "block"]).to_csv(outdir / "ld_block_clusters.csv", index=False)
    order = np.argsort(labels)
    plt.figure(figsize=(12, 10))
    if HAS_SNS:
        sns.heatmap(corr[order][:, order], cmap="vlag", center=0,
                    xticklabels=False, yticklabels=False)
    else:
        plt.imshow(corr[order][:, order], aspect="auto"); plt.colorbar()
    plt.title("LD block correlation heatmap (reordered by block clusters)")
    plt.tight_layout()
    out = outdir / "ld_block_correlation_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print(f"[save] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Step B helpers  (from 05_umap_hla_interpretation.py)
# ══════════════════════════════════════════════════════════════════════════════

def _pick_best_k(metrics_df):
    km = metrics_df[metrics_df["method"] == "KMeans"].sort_values("silhouette", ascending=False)
    if len(km) == 0:
        raise ValueError("No KMeans rows found in clustering_metrics.csv")
    row = km.iloc[0]
    return f"kmeans_k{int(row['k'])}", int(row["k"]), float(row["silhouette"])


# ══════════════════════════════════════════════════════════════════════════════
# Step C helpers  (from 10_hla_cluster_analysis.py)
# ══════════════════════════════════════════════════════════════════════════════

def _block_first_pc(block_repr: np.ndarray, block_idx: int) -> np.ndarray:
    X = block_repr[:, block_idx, :]
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1, random_state=42)
    pc = pca.fit_transform(X).ravel()
    if pca.components_[0, 0] < 0:
        pc = -pc
    return pc


def _eta_squared(groups):
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    return np.nan if ss_total == 0 else float(ss_between / ss_total)


def _test_continuous_vs_cluster(values: np.ndarray, cluster: np.ndarray):
    df = pd.DataFrame({"value": values, "cluster": cluster}).dropna()
    groups = [g["value"].values for _, g in df.groupby("cluster")]
    if len(groups) < 2 or any(len(g) < 5 for g in groups):
        return {"n": len(df), "anova_F": np.nan, "anova_p": np.nan,
                "kruskal_H": np.nan, "kruskal_p": np.nan, "eta2": np.nan}
    F, p_a = stats.f_oneway(*groups)
    H, p_k = stats.kruskal(*groups)
    return {"n": len(df), "anova_F": float(F), "anova_p": float(p_a),
            "kruskal_H": float(H), "kruskal_p": float(p_k),
            "eta2": _eta_squared(groups)}


def _cramers_v(chi2, n, r, k):
    denom = n * max(1, min(r - 1, k - 1))
    return np.nan if denom <= 0 else float(np.sqrt(chi2 / denom))


def _test_cluster_vs_binned_pc(values: np.ndarray, cluster: np.ndarray, q=4):
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
    return {"chi2": float(chi2), "chi2_p": float(p),
            "cramers_v": _cramers_v(chi2, tab.values.sum(), *tab.shape)}


def _save_boxplot(df, col, out_path, title, ylabel):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if HAS_SNS:
        sns.boxplot(data=df, x="cluster", y=col, ax=ax, color="white", width=0.55)
        sns.stripplot(data=df, x="cluster", y=col, ax=ax, alpha=0.45, size=3)
    means = df.groupby("cluster")[col].mean().round(3).to_dict()
    counts = df.groupby("cluster")[col].size().to_dict()
    ax.set_title(title, fontsize=10); ax.set_xlabel("Cluster"); ax.set_ylabel(ylabel)
    text = " | ".join([f"C{k}: mean={means.get(k, np.nan):.3f}, n={counts.get(k, 0)}"
                       for k in sorted(df["cluster"].dropna().unique())])
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=8, va="bottom")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


def _save_subject_cluster_plot(df, xcol, ycol, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    if HAS_SNS:
        sns.scatterplot(data=df, x=xcol, y=ycol, hue="cluster",
                        palette="tab10", s=28, alpha=0.8, ax=ax)
    else:
        for c in df["cluster"].unique():
            m = df["cluster"] == c
            ax.scatter(df.loc[m, xcol], df.loc[m, ycol], s=28, alpha=0.8, label=f"C{c}")
        ax.legend()
    ax.set_title(title, fontsize=11); plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()


def _save_colored_scatter(df, xcol, ycol, color_col, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df[xcol], df[ycol], c=df[color_col], cmap="plasma", s=20, alpha=0.8)
    plt.colorbar(sc, ax=ax, label=color_col)
    ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(title, fontsize=11)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


def _apply_fdr(df, p_cols):
    for pcol, qcol in p_cols:
        pvals = df[pcol].values
        mask = np.isfinite(pvals)
        qvals = np.full(len(df), np.nan)
        if mask.sum() > 0:
            _, qtmp, _, _ = multipletests(pvals[mask], method="fdr_bh")
            qvals[mask] = qtmp
        df[qcol] = qvals
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step A — phenotype-free HDBSCAN clustering + block-driver analysis
# (logic from 09_unsupervised_subject_cluster_analysis.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_unsupervised_clustering(
    subject_embeddings: Path,
    outdir: Path,
    block_embeddings: Path = None,
    attention_weights: Path = None,
    block_order_csv: Path = None,
    min_cluster_size: int = 25,
    min_samples: int = None,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    top_k_blocks: int = 10,
    block_score_method: str = "norm",
    do_cluster_blocks: bool = False,
    n_block_clusters: int = 6,
):
    """Step A: HDBSCAN subject clustering + PCA/UMAP + block-driver heatmap."""
    if not HAS_UMAP:
        raise ImportError("umap-learn required: pip install umap-learn")
    if not HAS_HDBSCAN:
        raise ImportError("hdbscan required: pip install hdbscan")

    ensure_dir(outdir)
    subj_emb = _load_array(subject_embeddings, "subject_embeddings")
    if subj_emb.ndim != 2:
        raise ValueError(f"Subject embeddings must be (N, D), got {subj_emb.shape}")

    block_emb = _load_array(block_embeddings, "block_embeddings") if block_embeddings else None
    attn = _load_array(attention_weights, "attention_weights") if attention_weights else None

    if block_emb is None and attn is None:
        warnings.warn("No block embeddings or attention weights — block-driver analysis skipped.")

    Xs = _standardize(subj_emb)
    X_pca, pca_var, X_umap = _run_pca_umap(Xs, n_neighbors=umap_n_neighbors,
                                            min_dist=umap_min_dist)
    _, labels, probs, sil = _run_hdbscan(Xs, min_cluster_size, min_samples)

    uniq, counts = np.unique(labels, return_counts=True)
    pd.DataFrame({"cluster": uniq, "n_subjects": counts,
                  "is_noise": uniq == -1}).to_csv(
        outdir / "subject_cluster_summary.csv", index=False)
    pd.DataFrame([{
        "n_subjects": int(subj_emb.shape[0]),
        "embedding_dim": int(subj_emb.shape[1]),
        "n_clusters_excluding_noise": int(len([x for x in uniq if x != -1])),
        "n_noise_subjects": int((labels == -1).sum()),
        "silhouette_excluding_noise": float(sil) if not np.isnan(sil) else np.nan,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
    }]).to_csv(outdir / "clustering_metrics.csv", index=False)

    _save_cluster_plots_ab(X_pca, pca_var, X_umap, labels, outdir)

    subj_df = pd.DataFrame({
        "subject_idx": np.arange(subj_emb.shape[0]), "cluster": labels,
        "cluster_prob": probs if probs is not None else np.nan,
        "pca1": X_pca[:, 0], "pca2": X_pca[:, 1],
        "umap1": X_umap[:, 0], "umap2": X_umap[:, 1],
    })
    subj_df.to_csv(outdir / "subject_cluster_assignments.csv", index=False)
    print(f"[save] {outdir / 'subject_cluster_assignments.csv'}")

    if block_emb is not None or attn is not None:
        block_scores = (_block_scores_from_attention(attn) if attn is not None
                        else _block_scores_from_embeddings(block_emb, block_score_method))
        block_names = load_block_names(block_order_csv, block_scores.shape[1])
        diff_df, top_blocks_df = _cluster_block_diffs(block_scores, labels,
                                                      block_names, top_k_blocks)
        diff_df.to_csv(outdir / "cluster_block_mean_differences.csv")
        top_blocks_df.to_csv(outdir / "top_blocks_per_cluster.csv", index=False)
        _save_block_heatmap(diff_df, top_blocks_df, outdir)
        if do_cluster_blocks:
            _cluster_blocks_by_corr(block_scores, block_names, outdir, n_block_clusters)

    print("\n[done] Step A — phenotype-free clustering complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Step B — UMAP + attention coloring + cluster overlay
# (logic from 05_umap_hla_interpretation.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_umap_hla(
    phase2_loss_dir: Path,
    phase1_dir: Path,
    outdir: Path,
    pheno_file=None,
    iid_col: str = "IID",
    umap_n_neighbors: int = 20,
    umap_min_dist: float = 0.1,
    umap_metric: str = "cosine",
):
    """Step B: UMAP colored by attention + cluster overlay + optional phenotype merge."""
    if not HAS_UMAP:
        raise ImportError("umap-learn required: pip install umap-learn")

    expected = [
        phase2_loss_dir / "embeddings" / "individual_embeddings.npy",
        phase2_loss_dir / "embeddings" / "pooling_attention_weights.csv",
        phase2_loss_dir / "clustering" / "cluster_labels.csv",
        phase2_loss_dir / "clustering" / "clustering_metrics.csv",
        phase1_dir / "block_order.csv",
    ]
    for p in expected:
        if not p.exists():
            raise FileNotFoundError(f"Missing expected file: {p}")

    ensure_dir(outdir)
    t0 = time.time()

    embeddings = np.load(phase2_loss_dir / "embeddings" / "individual_embeddings.npy")
    pool_df = pd.read_csv(phase2_loss_dir / "embeddings" / "pooling_attention_weights.csv")
    cluster_df = pd.read_csv(phase2_loss_dir / "clustering" / "cluster_labels.csv")
    metrics_df = pd.read_csv(phase2_loss_dir / "clustering" / "clustering_metrics.csv")

    best_key, best_k, best_sil = _pick_best_k(metrics_df)
    if best_key not in cluster_df.columns:
        raise KeyError(f"Best key {best_key} not found in cluster_labels.csv")
    labels = cluster_df[best_key].values

    Z = StandardScaler().fit_transform(embeddings)
    reducer = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist,
                        metric=umap_metric, random_state=42)
    Z2d = reducer.fit_transform(Z)
    np.save(outdir / "umap_2d.npy", Z2d)

    plt.figure(figsize=(7, 6))
    palette = (sns.color_palette("tab10", n_colors=max(10, best_k)) if HAS_SNS
               else [plt.cm.tab10(i) for i in range(max(10, best_k))])
    for c in sorted(np.unique(labels)):
        mask = labels == c
        plt.scatter(Z2d[mask, 0], Z2d[mask, 1], s=8, alpha=0.7,
                    label=f"Cluster {c}", c=[palette[c % len(palette)]])
    plt.title(f"UMAP of ORD embeddings (k={best_k}, sil={best_sil:.3f})")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(markerscale=2, fontsize=8, ncol=2); plt.tight_layout()
    plt.savefig(outdir / "umap_clusters.png", dpi=160); plt.close()

    block_cols = [c for c in pool_df.columns if c != iid_col]
    top_blocks = pool_df[block_cols].mean(axis=0).sort_values(ascending=False).head(10)
    top_blocks.to_csv(outdir / "top10_blocks_by_attention.csv", header=["mean_attn"])

    for block_id in top_blocks.index.tolist():
        vals = pool_df[block_id].values
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(Z2d[:, 0], Z2d[:, 1], c=vals, cmap="plasma", s=8, alpha=0.75)
        plt.colorbar(sc, label="attention")
        plt.title(f"UMAP colored by attention to {block_id}")
        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.tight_layout()
        plt.savefig(outdir / f"umap_block_{block_id}_attention.png", dpi=160)
        plt.close()

    if pheno_file:
        pheno = pd.read_csv(pheno_file)
        if iid_col not in pheno.columns and "S_SUBJECTID" in pheno.columns:
            pheno = pheno.rename(columns={"S_SUBJECTID": iid_col})
        if iid_col not in pheno.columns:
            raise KeyError(f"Phenotype file missing ID column: {iid_col} or S_SUBJECTID")
        merged = pd.DataFrame({iid_col: pool_df[iid_col].astype(str)}).merge(
            pheno.astype({iid_col: str}), on=iid_col, how="left")
        unmatched = merged[merged.isnull().any(axis=1)].shape[0]
        print(f"[umap_hla] phenotype merge: {unmatched}/{merged.shape[0]} missing")
        merged.to_csv(outdir / "merged_pheno.csv", index=False)

    pd.DataFrame([{
        "phase2_loss_dir": str(phase2_loss_dir), "phase1_dir": str(phase1_dir),
        "outdir": str(outdir), "n_subjects": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1], "best_k": best_k, "best_sil": best_sil,
        "top_hla_candidates": ",".join(top_blocks.index.tolist()),
    }]).to_csv(outdir / "umap_hla_analysis_summary.csv", index=False)

    print(f"[done] Step B — UMAP/HLA complete in {time.time() - t0:.1f}s, output in {outdir}")


# ══════════════════════════════════════════════════════════════════════════════
# Step C — KMeans(k=3) + HLA block_PC1 + genotype-PC per cluster
# (logic from 10_hla_cluster_analysis.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_hla_cluster_analysis():
    """Step C: KMeans(k=3) on subject embeddings; test HLA block_PC1 and genotype PCs by cluster."""
    out = Path(_OUT_DIR_C)
    ensure_dir(out)
    fig_dir = out / "figures"
    ensure_dir(fig_dir)

    print("Loading embeddings...")
    emb_subj  = np.load(_EMB_SUBJ_NPY)
    emb_block = np.load(_EMB_BLOCK_NPY)
    N, B, D   = emb_block.shape
    print(f"  emb_subj : {emb_subj.shape}")
    print(f"  emb_block: {emb_block.shape}")

    attn_df   = pd.read_csv(_ATTN_CSV)
    subj_iids = attn_df["IID"].astype(str).values
    block_names = load_block_names(_BLOCK_ORDER, B)
    block_to_idx = {b: i for i, b in enumerate(block_names)}

    missing = [b for b in _HLA_BLOCKS if b not in block_to_idx]
    if missing:
        raise ValueError(f"HLA blocks not found in block_order.csv:\n{missing}")

    print("Loading genotype PCs...")
    pcs_df = load_eigenvec(_EIGENVEC_FILE)

    print("Computing subject PCA...")
    Zs = StandardScaler().fit_transform(emb_subj)
    subj_pca_model = PCA(n_components=10, random_state=42)
    subj_pca = subj_pca_model.fit_transform(Zs)
    explained = subj_pca_model.explained_variance_ratio_ * 100

    print("Clustering subjects with KMeans(k=3)...")
    km = KMeans(n_clusters=3, random_state=42, n_init=20)
    clusters_raw = km.fit_predict(subj_pca[:, :10])

    # relabel by mean PC1 for consistency across runs
    tmp = pd.DataFrame({"cluster_raw": clusters_raw, "PC1": subj_pca[:, 0]})
    order = tmp.groupby("cluster_raw")["PC1"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    clusters = np.array([remap[c] for c in clusters_raw], dtype=int)

    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    print("\nCluster sizes:")
    for k, v in cluster_counts.items():
        print(f"  cluster {k}: n={v}")

    subject_df = pd.DataFrame({
        "IID": subj_iids, "cluster": clusters,
        "embedPC1": subj_pca[:, 0], "embedPC2": subj_pca[:, 1], "embedPC3": subj_pca[:, 2],
    }).merge(pcs_df, on="IID", how="left")
    subject_df.to_csv(out / "subject_clusters.tsv", sep="\t", index=False)

    _save_subject_cluster_plot(
        subject_df, xcol="embedPC1", ycol="embedPC2",
        out_path=fig_dir / "subject_clusters_pca.png",
        title=f"Subject clusters on embedding PCA\nPC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%")

    # ── genotype PCs vs cluster ───────────────────────────────────────────────
    print("\nTesting genotype PCs ~ cluster ...")
    geno_rows = []
    for pc in _PC_COLS:
        vals = pd.to_numeric(subject_df[pc], errors="coerce").values
        row = {"geno_pc": pc,
               **_test_continuous_vs_cluster(vals, subject_df["cluster"].values),
               **_test_cluster_vs_binned_pc(vals, subject_df["cluster"].values, q=4)}
        for stat, agg in [("mean", lambda x: x.mean()), ("median", lambda x: x.median())]:
            grp = subject_df.groupby("cluster")[pc]
            for k in [0, 1, 2]:
                row[f"cluster{k}_{stat}"] = float(agg(grp.get_group(k)) if k in grp.groups else np.nan)
        geno_rows.append(row)
        _save_boxplot(subject_df[["cluster", pc]].copy(), col=pc,
                      out_path=fig_dir / f"{pc}_by_cluster.png",
                      title=f"{pc} by subject cluster", ylabel=pc)
        _save_colored_scatter(subject_df, xcol="embedPC1", ycol="embedPC2",
                              color_col=pc,
                              out_path=fig_dir / f"subject_pca_colored_by_{pc}.png",
                              title=f"Subject embedding PCA colored by {pc}")

    geno_summary = _apply_fdr(pd.DataFrame(geno_rows),
                              [("anova_p", "anova_fdr_bh"),
                               ("kruskal_p", "kruskal_fdr_bh"),
                               ("chi2_p", "chi2_fdr_bh")])
    geno_summary = geno_summary.sort_values(["eta2", "kruskal_p"],
                                            ascending=[False, True]).reset_index(drop=True)
    geno_summary.to_csv(out / "genotype_pc_cluster_association.tsv", sep="\t", index=False)

    # heatmap of genotype PC cluster means
    heat_cols = ["cluster0_mean", "cluster1_mean", "cluster2_mean"]
    if all(c in geno_summary.columns for c in heat_cols):
        geno_heat = geno_summary.set_index("geno_pc")[heat_cols]
        fig, ax = plt.subplots(figsize=(6, max(4, len(geno_heat) * 0.45)))
        if HAS_SNS:
            sns.heatmap(geno_heat, cmap="coolwarm", center=0, linewidths=0.5,
                        annot=True, fmt=".2f", ax=ax)
        ax.set_title("Genotype PC mean by subject cluster", fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_dir / "genotype_pc_cluster_mean_heatmap.png", dpi=160)
        plt.close()

    # ── HLA block_PC1 vs cluster ──────────────────────────────────────────────
    print("\nComputing HLA block_PC1 values...")
    hla_df = subject_df.copy()
    for block in _HLA_BLOCKS:
        hla_df[block] = _block_first_pc(emb_block, block_to_idx[block])
    hla_df.to_csv(out / "hla_block_pc1_by_subject.tsv", sep="\t", index=False)

    print("Testing block_PC1 ~ cluster ...")
    hla_rows = []
    for block in _HLA_BLOCKS:
        res = _test_continuous_vs_cluster(hla_df[block].values, hla_df["cluster"].values)
        for stat, agg in [("mean", lambda x: x.mean()), ("median", lambda x: x.median())]:
            grp = hla_df.groupby("cluster")[block]
            for k in [0, 1, 2]:
                res[f"cluster{k}_{stat}"] = float(agg(grp.get_group(k)) if k in grp.groups else np.nan)
        hla_rows.append({"block": block, **res})
        _save_boxplot(hla_df[["cluster", block]].copy(), col=block,
                      out_path=fig_dir / f"{block}_by_cluster.png",
                      title=f"{block}\nblock_PC1 by subject cluster", ylabel="block_PC1")

    summary = _apply_fdr(pd.DataFrame(hla_rows),
                         [("anova_p", "anova_fdr_bh"), ("kruskal_p", "kruskal_fdr_bh")])
    summary = summary.sort_values(["eta2", "kruskal_p"],
                                  ascending=[False, True]).reset_index(drop=True)
    summary.to_csv(out / "hla_block_cluster_association.tsv", sep="\t", index=False)

    # HLA cluster heatmap
    heat_cols = ["cluster0_mean", "cluster1_mean", "cluster2_mean"]
    if all(c in summary.columns for c in heat_cols):
        heat_df = summary.set_index("block")[heat_cols]
        fig, ax = plt.subplots(figsize=(7, max(5, len(heat_df) * 0.35)))
        if HAS_SNS:
            sns.heatmap(heat_df, cmap="coolwarm", center=0, linewidths=0.5,
                        annot=True, fmt=".2f", ax=ax)
        ax.set_title("HLA block_PC1 mean by subject cluster", fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_dir / "hla_block_cluster_mean_heatmap.png", dpi=160)
        plt.close()

    # ── HLA block_PC1 residuals after regressing out genotype PCs ─────────────
    print("\nComputing HLA block_PC1 residuals after regressing out genotype PCs...")
    pc_matrix = subject_df[_PC_COLS].values.astype(float)
    pc_scaled  = StandardScaler().fit_transform(pc_matrix)
    valid_mask = np.isfinite(pc_scaled).all(axis=1)

    resid_rows = []
    for block in _HLA_BLOCKS:
        y = hla_df.loc[valid_mask, block].values
        X = pc_scaled[valid_mask]
        lr = LinearRegression().fit(X, y)
        residuals = y - lr.predict(X)
        resid_rows.append({"block": block,
                           **_test_continuous_vs_cluster(residuals,
                               hla_df.loc[valid_mask, "cluster"].values)})

    resid_summary = pd.DataFrame(resid_rows).sort_values(
        ["eta2", "kruskal_p"], ascending=[False, True]).reset_index(drop=True)
    resid_summary.to_csv(out / "hla_block_residual_cluster_association.tsv",
                         sep="\t", index=False)

    compare = (summary[["block", "eta2"]].rename(columns={"eta2": "eta2_raw"})
               .merge(resid_summary[["block", "eta2"]].rename(columns={"eta2": "eta2_residual"}),
                      on="block"))
    compare["eta2_drop"] = compare["eta2_raw"] - compare["eta2_residual"]
    compare["pct_retained"] = (compare["eta2_residual"] / compare["eta2_raw"] * 100).round(1)
    compare = compare.sort_values("eta2_raw", ascending=False).reset_index(drop=True)
    compare.to_csv(out / "hla_eta2_raw_vs_residual.tsv", sep="\t", index=False)
    print("\nRaw vs residual eta²:")
    print(compare.to_string(index=False))

    # ── inter-block PC1 correlations ──────────────────────────────────────────
    hla_pc1_cols = [b for b in _HLA_BLOCKS if b in hla_df.columns]
    corr_matrix = hla_df[hla_pc1_cols].corr(method="spearman")
    corr_matrix.to_csv(out / "hla_block_pc1_spearman_corr.tsv", sep="\t")

    fig, ax = plt.subplots(figsize=(10, 8))
    if HAS_SNS:
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    linewidths=0.3, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Spearman correlation between HLA block_PC1 values", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "hla_block_pc1_correlation_heatmap.png", dpi=160)
    plt.close()

    print(f"\n[done] Step C — HLA cluster analysis complete. Outputs in: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI dispatch
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="02_subject_cluster_analysis — three-step subject clustering pipeline"
    )
    ap.add_argument(
        "--step", choices=["all", "clustering", "umap_hla", "hla_analysis"],
        default="hla_analysis",
        help=("Which step to run. 'all' runs A → B → C in sequence. "
              "'clustering' = Step A (HDBSCAN). "
              "'umap_hla' = Step B (UMAP + attention). "
              "'hla_analysis' = Step C (KMeans + HLA; default, uses hardcoded paths).")
    )
    # Step A args
    ap.add_argument("--subject-embeddings", type=Path,
                    help="[Step A] Path to individual_embeddings.npy")
    ap.add_argument("--block-embeddings", type=Path, default=None,
                    help="[Step A] Optional block_contextual_repr.npy")
    ap.add_argument("--attention-weights", type=Path, default=None,
                    help="[Step A] Optional attention weights .npy or .csv")
    ap.add_argument("--block-order-csv", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None,
                    help="[Step A/B] Output directory")
    ap.add_argument("--min-cluster-size", type=int, default=25)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--umap-n-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--top-k-blocks", type=int, default=10)
    ap.add_argument("--block-score-method", choices=["norm", "pc1", "mean"], default="norm")
    ap.add_argument("--cluster-blocks", action="store_true")
    ap.add_argument("--n-block-clusters", type=int, default=6)
    # Step B args
    ap.add_argument("--phase2-loss-dir", type=Path, default=None,
                    help="[Step B] Phase2 loss directory, e.g. results/output_regions2/ORD")
    ap.add_argument("--phase1-dir", type=Path, default=Path("results/output_regions"),
                    help="[Step B] Phase1 directory with block_order.csv")
    ap.add_argument("--pheno-file", default=None,
                    help="[Step B] Optional phenotype CSV")
    ap.add_argument("--iid-col", default="IID")
    ap.add_argument("--umap-metric", default="cosine")
    args = ap.parse_args()

    run_a = args.step in ("all", "clustering")
    run_b = args.step in ("all", "umap_hla")
    run_c = args.step in ("all", "hla_analysis")

    if run_a:
        if args.subject_embeddings is None or args.outdir is None:
            ap.error("--subject-embeddings and --outdir are required for Step A (clustering)")
        run_unsupervised_clustering(
            subject_embeddings=args.subject_embeddings,
            outdir=args.outdir,
            block_embeddings=args.block_embeddings,
            attention_weights=args.attention_weights,
            block_order_csv=args.block_order_csv,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            top_k_blocks=args.top_k_blocks,
            block_score_method=args.block_score_method,
            do_cluster_blocks=args.cluster_blocks,
            n_block_clusters=args.n_block_clusters,
        )

    if run_b:
        if args.phase2_loss_dir is None or args.outdir is None:
            ap.error("--phase2-loss-dir and --outdir are required for Step B (umap_hla)")
        run_umap_hla(
            phase2_loss_dir=args.phase2_loss_dir,
            phase1_dir=args.phase1_dir,
            outdir=args.outdir,
            pheno_file=args.pheno_file,
            iid_col=args.iid_col,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_metric=args.umap_metric,
        )

    if run_c:
        run_hla_cluster_analysis()


if __name__ == "__main__":
    main()
