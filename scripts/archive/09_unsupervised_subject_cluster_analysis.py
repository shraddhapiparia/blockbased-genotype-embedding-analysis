#!/usr/bin/env python3
"""09_unsupervised_subject_cluster_analysis.py — unsupervised subject-level clustering.

Purpose : Cluster subjects using global Phase 2 embeddings via PCA → UMAP →
          HDBSCAN and agglomerative clustering; generate cluster-level summaries.
Inputs  : results/output_regions2/<loss>/embeddings/individual_embeddings.npy,
          metadata/ldpruned_997subs.eigenvec, metadata/COS_TRIO_pheno_1165.csv
Outputs : results/output_regions2/<loss>/subject_clusters/ (TSVs + PNGs)
Workflow: Active analysis — Step 9; cluster assignments feed into 10_, 12_, 13_.
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

try:
    import umap
except ImportError:
    raise ImportError("Please install umap-learn: pip install umap-learn")

try:
    import hdbscan
except ImportError:
    raise ImportError("Please install hdbscan: pip install hdbscan")

try:
    import seaborn as sns
except ImportError:
    sns = None


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_array(path: Path, name: str):
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    arr = np.load(path, allow_pickle=True)
    print(f"[load] {name}: {path}  shape={arr.shape}")
    return arr


def load_block_names(block_order_csv: Path, expected_n_blocks: int):
    if block_order_csv is None:
        return [f"block_{i}" for i in range(expected_n_blocks)]

    df = pd.read_csv(block_order_csv)
    possible_cols = ["block", "block_name", "region", "name"]
    col = None
    for c in possible_cols:
        if c in df.columns:
            col = c
            break

    if col is None:
        # fallback: first column
        col = df.columns[0]

    names = df[col].astype(str).tolist()
    if len(names) != expected_n_blocks:
        raise ValueError(
            f"block_order.csv has {len(names)} names but expected {expected_n_blocks} blocks"
        )
    return names


def standardize_subject_embeddings(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs


def run_pca_umap(Xs, random_state=42, n_neighbors=30, min_dist=0.1):
    pca_model = PCA(n_components=2)
    X_pca = pca_model.fit_transform(Xs)
    pca_var = pca_model.explained_variance_ratio_ * 100.0

    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    X_umap = umap_model.fit_transform(Xs)

    return X_pca, pca_var, X_umap


def run_hdbscan(Xs, min_cluster_size=25, min_samples=None):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(Xs)
    probs = getattr(clusterer, "probabilities_", None)

    valid = labels >= 0
    if valid.sum() >= 2 and len(np.unique(labels[valid])) >= 2:
        sil = silhouette_score(Xs[valid], labels[valid])
    else:
        sil = np.nan

    return clusterer, labels, probs, sil


def make_cluster_color_map(labels):
    unique = sorted(np.unique(labels))
    cmap = plt.cm.get_cmap("tab20", max(len(unique), 1))
    color_map = {}
    for i, lab in enumerate(unique):
        if lab == -1:
            color_map[lab] = (0.7, 0.7, 0.7, 0.8)  # gray for noise
        else:
            color_map[lab] = cmap(i)
    return color_map


def plot_embedding(ax, X2d, labels, title, xlabel, ylabel):
    color_map = make_cluster_color_map(labels)
    for lab in sorted(np.unique(labels)):
        mask = labels == lab
        ax.scatter(
            X2d[mask, 0],
            X2d[mask, 1],
            s=16,
            alpha=0.8,
            color=color_map[lab],
            label=f"cluster {lab}" if lab != -1 else "noise",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="best", markerscale=1.2)


def save_cluster_plots(X_pca, pca_var, X_umap, labels, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_embedding(
        axes[0],
        X_pca,
        labels,
        title=f"Subject embeddings (PCA)\nPC1={pca_var[0]:.1f}%, PC2={pca_var[1]:.1f}%",
        xlabel=f"PC1 ({pca_var[0]:.1f}% var)",
        ylabel=f"PC2 ({pca_var[1]:.1f}% var)",
    )

    plot_embedding(
        axes[1],
        X_umap,
        labels,
        title="UMAP of subject embeddings",
        xlabel="UMAP-1",
        ylabel="UMAP-2",
    )

    plt.tight_layout()
    outpath = outdir / "subject_clusters_pca_umap.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {outpath}")


# ------------------------------------------------------------
# Block scoring
# ------------------------------------------------------------
def block_scores_from_attention(attn):
    """
    attn: (N, B)
    Returns per-subject per-block score matrix: (N, B)
    """
    if attn.ndim != 2:
        raise ValueError(f"Expected attention weights shape (N, B), got {attn.shape}")
    return attn


def block_scores_from_block_embeddings(block_emb, method="norm"):
    """
    block_emb: (N, B, D)
    Returns per-subject per-block score matrix: (N, B)
    """
    if block_emb.ndim != 3:
        raise ValueError(f"Expected block embeddings shape (N, B, D), got {block_emb.shape}")

    if method == "norm":
        return np.linalg.norm(block_emb, axis=2)

    elif method == "pc1":
        n, b, d = block_emb.shape
        out = np.zeros((n, b), dtype=float)
        for j in range(b):
            pca = PCA(n_components=1, random_state=42)
            out[:, j] = pca.fit_transform(block_emb[:, j, :]).ravel()
        return out

    elif method == "mean":
        return block_emb.mean(axis=2)

    else:
        raise ValueError(f"Unknown block score method: {method}")


def zscore_columns(X):
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def compute_cluster_block_differences(block_scores, labels, block_names, top_k=10):
    """
    block_scores: (N, B)
    labels: (N,)
    Returns:
      diff_df: rows=clusters, cols=blocks, values = mean(cluster)-mean(others)
      top_blocks_df: long-form table of top_k blocks per cluster
    """
    clusters = sorted([c for c in np.unique(labels) if c != -1])
    if len(clusters) == 0:
        raise ValueError("No non-noise clusters found.")

    block_scores_z = zscore_columns(block_scores)

    diff_rows = []
    top_rows = []

    for c in clusters:
        in_c = labels == c
        out_c = labels != c

        mean_in = block_scores_z[in_c].mean(axis=0)
        mean_out = block_scores_z[out_c].mean(axis=0)
        diff = mean_in - mean_out

        diff_rows.append(pd.Series(diff, index=block_names, name=f"cluster_{c}"))

        order = np.argsort(-np.abs(diff))[:top_k]
        for rank, idx in enumerate(order, start=1):
            top_rows.append(
                {
                    "cluster": c,
                    "rank": rank,
                    "block": block_names[idx],
                    "diff_z": diff[idx],
                    "mean_in_z": mean_in[idx],
                    "mean_out_z": mean_out[idx],
                }
            )

    diff_df = pd.DataFrame(diff_rows)
    top_blocks_df = pd.DataFrame(top_rows)
    return diff_df, top_blocks_df


def save_top_blocks_table(top_blocks_df, outdir: Path):
    outpath = outdir / "top_blocks_per_cluster.csv"
    top_blocks_df.to_csv(outpath, index=False)
    print(f"[save] {outpath}")


def plot_cluster_block_heatmap(diff_df, top_blocks_df, outdir: Path, max_blocks=20):
    """
    Build heatmap using union of top blocks across clusters.
    """
    selected_blocks = []
    for blk in top_blocks_df["block"].tolist():
        if blk not in selected_blocks:
            selected_blocks.append(blk)
        if len(selected_blocks) >= max_blocks:
            break

    plot_df = diff_df[selected_blocks].copy()

    plt.figure(figsize=(max(10, 0.6 * len(selected_blocks)), 4 + 0.5 * len(plot_df)))

    if sns is not None:
        sns.heatmap(plot_df, cmap="coolwarm", center=0, annot=False)
    else:
        plt.imshow(plot_df.values, aspect="auto")
        plt.colorbar(label="mean z-score difference")
        plt.xticks(range(len(selected_blocks)), selected_blocks, rotation=90)
        plt.yticks(range(len(plot_df.index)), plot_df.index)

    plt.title("Cluster × LD block contribution heatmap")
    plt.xlabel("LD blocks")
    plt.ylabel("Subject clusters")
    plt.tight_layout()

    outpath = outdir / "cluster_block_heatmap.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {outpath}")


# ------------------------------------------------------------
# Optional block clustering
# ------------------------------------------------------------
def cluster_blocks(block_scores, block_names, outdir: Path, n_block_clusters=6):
    """
    block_scores: (N, B)
    Cluster blocks by correlation across subjects.
    """
    corr = np.corrcoef(block_scores.T)
    corr = np.nan_to_num(corr, nan=0.0)

    # Convert correlation to distance
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)

    model = AgglomerativeClustering(
        n_clusters=n_block_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(dist)

    cluster_df = pd.DataFrame(
        {
            "block": block_names,
            "block_cluster": labels,
        }
    ).sort_values(["block_cluster", "block"])

    out_csv = outdir / "ld_block_clusters.csv"
    cluster_df.to_csv(out_csv, index=False)
    print(f"[save] {out_csv}")

    # Plot reordered correlation heatmap
    order = np.argsort(labels)
    corr_ord = corr[order][:, order]
    names_ord = [block_names[i] for i in order]

    plt.figure(figsize=(12, 10))
    if sns is not None:
        sns.heatmap(corr_ord, cmap="vlag", center=0, xticklabels=False, yticklabels=False)
    else:
        plt.imshow(corr_ord, aspect="auto")
        plt.colorbar(label="block-block correlation")

    plt.title("LD block correlation heatmap (reordered by block clusters)")
    plt.tight_layout()
    out_png = outdir / "ld_block_correlation_heatmap.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {out_png}")

    # Save ordered names for inspection
    out_order = outdir / "ld_block_cluster_order.csv"
    pd.DataFrame(
        {
            "order": np.arange(len(names_ord)),
            "block": names_ord,
            "block_cluster": labels[order],
        }
    ).to_csv(out_order, index=False)
    print(f"[save] {out_order}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phenotype-free subject clustering and LD block driver analysis"
    )

    parser.add_argument(
        "--subject-embeddings",
        type=Path,
        required=True,
        help="Path to subject embeddings .npy, shape (N, D)",
    )
    parser.add_argument(
        "--block-embeddings",
        type=Path,
        default=None,
        help="Optional path to block embeddings .npy, shape (N, B, d)",
    )
    parser.add_argument(
        "--attention-weights",
        type=Path,
        default=None,
        help="Optional path to attention weights .npy, shape (N, B)",
    )
    parser.add_argument(
        "--block-order-csv",
        type=Path,
        default=None,
        help="Optional block_order.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory",
    )

    parser.add_argument("--min-cluster-size", type=int, default=25)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--top-k-blocks", type=int, default=10)
    parser.add_argument(
        "--block-score-method",
        choices=["norm", "pc1", "mean"],
        default="norm",
        help="Used only if block embeddings are supplied",
    )
    parser.add_argument(
        "--cluster-blocks",
        action="store_true",
        help="Also cluster LD blocks using subject-level block scores",
    )
    parser.add_argument(
        "--n-block-clusters",
        type=int,
        default=6,
        help="Number of LD block clusters for optional block clustering",
    )

    args = parser.parse_args()
    ensure_dir(args.outdir)

    # --------------------
    # Load data
    # --------------------
    subj_emb = load_array(args.subject_embeddings, "subject_embeddings")
    if subj_emb.ndim != 2:
        raise ValueError(f"Subject embeddings must have shape (N, D), got {subj_emb.shape}")

    block_emb = load_array(args.block_embeddings, "block_embeddings") if args.block_embeddings else None
    attn = load_array(args.attention_weights, "attention_weights") if args.attention_weights else None

    if block_emb is None and attn is None:
        warnings.warn(
            "Neither block embeddings nor attention weights were provided. "
            "Clustering and PCA/UMAP plots will run, but block-driver analysis will be skipped."
        )

    if block_emb is not None and attn is not None:
        print("[info] Both block embeddings and attention weights provided. "
              "Using attention weights for block-driver analysis.")

    # --------------------
    # Subject clustering
    # --------------------
    Xs = standardize_subject_embeddings(subj_emb)
    X_pca, pca_var, X_umap = run_pca_umap(
        Xs,
        random_state=42,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
    )

    clusterer, labels, probs, sil = run_hdbscan(
        Xs,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    uniq, counts = np.unique(labels, return_counts=True)
    cluster_summary = pd.DataFrame({"cluster": uniq, "n_subjects": counts})
    cluster_summary["is_noise"] = cluster_summary["cluster"] == -1
    cluster_summary.to_csv(args.outdir / "subject_cluster_summary.csv", index=False)
    print(f"[save] {args.outdir / 'subject_cluster_summary.csv'}")

    metrics = {
        "n_subjects": int(subj_emb.shape[0]),
        "embedding_dim": int(subj_emb.shape[1]),
        "n_clusters_excluding_noise": int(len([x for x in uniq if x != -1])),
        "n_noise_subjects": int((labels == -1).sum()),
        "silhouette_excluding_noise": float(sil) if not np.isnan(sil) else np.nan,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
    }
    pd.DataFrame([metrics]).to_csv(args.outdir / "clustering_metrics.csv", index=False)
    print(f"[save] {args.outdir / 'clustering_metrics.csv'}")

    save_cluster_plots(X_pca, pca_var, X_umap, labels, args.outdir)

    # Save per-subject coordinates + labels
    subj_df = pd.DataFrame(
        {
            "subject_idx": np.arange(subj_emb.shape[0]),
            "cluster": labels,
            "cluster_prob": probs if probs is not None else np.nan,
            "pca1": X_pca[:, 0],
            "pca2": X_pca[:, 1],
            "umap1": X_umap[:, 0],
            "umap2": X_umap[:, 1],
        }
    )
    subj_df.to_csv(args.outdir / "subject_cluster_assignments.csv", index=False)
    print(f"[save] {args.outdir / 'subject_cluster_assignments.csv'}")

    # --------------------
    # Block-driver analysis
    # --------------------
    if block_emb is not None or attn is not None:
        if attn is not None:
            block_scores = block_scores_from_attention(attn)
            print("[info] Using attention weights as subject-level block scores")
        else:
            block_scores = block_scores_from_block_embeddings(
                block_emb, method=args.block_score_method
            )
            print(f"[info] Using block embeddings with method='{args.block_score_method}'")

        n_blocks = block_scores.shape[1]
        block_names = load_block_names(args.block_order_csv, n_blocks)

        diff_df, top_blocks_df = compute_cluster_block_differences(
            block_scores=block_scores,
            labels=labels,
            block_names=block_names,
            top_k=args.top_k_blocks,
        )

        diff_df.to_csv(args.outdir / "cluster_block_mean_differences.csv")
        print(f"[save] {args.outdir / 'cluster_block_mean_differences.csv'}")

        save_top_blocks_table(top_blocks_df, args.outdir)
        plot_cluster_block_heatmap(diff_df, top_blocks_df, args.outdir, max_blocks=20)

        # --------------------
        # Optional block clustering
        # --------------------
        if args.cluster_blocks:
            cluster_blocks(
                block_scores=block_scores,
                block_names=block_names,
                outdir=args.outdir,
                n_block_clusters=args.n_block_clusters,
            )

    print("\n[done] Phenotype-free clustering analysis complete.")


if __name__ == "__main__":
    main()