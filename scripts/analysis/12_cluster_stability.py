#!/usr/bin/env python3
"""
12_cluster_stability.py

Goal
----
Test whether the 3 subject clusters from KMeans are stable before proceeding
to phenotype testing.

Three checks
------------
1. KMeans seed stability  — rerun KMeans(k=3) across 50 random seeds,
   compute pairwise ARI between all runs and against the reference run.
2. Algorithm robustness  — compare KMeans clusters to GMM and HDBSCAN
   via ARI.
3. Silhouette / Elbow    — confirm k=3 is the right choice (k=2..6).

Outputs
-------
results/.../cluster_stability/
  ari_kmeans_seeds.tsv          — pairwise ARI for 50 seed runs
  ari_algorithm_comparison.tsv  — KMeans vs GMM vs HDBSCAN
  elbow_silhouette.tsv          — inertia + silhouette for k=2..6
  subject_cluster_stability.tsv — per-subject: reference cluster, mode
                                   cluster across seeds, % agreement
  figures/
    ari_distribution_seeds.png
    elbow_plot.png
    silhouette_plot.png
    algorithm_cluster_scatter.png
    subject_stability_histogram.png
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score

warnings.filterwarnings("ignore")

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("hdbscan not installed — skipping HDBSCAN check. "
          "Install with: pip install hdbscan")

# ── paths (mirror script 10) ──────────────────────────────────────────
EMB_SUBJ_NPY = "results/output_regions2/ORD/embeddings/individual_embeddings.npy"
ATTN_CSV     = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
OUT_DIR      = "results/output_regions2/ORD/cluster_stability"

N_SEEDS      = 50
K_REF        = 3          # reference k from script 10
K_RANGE      = range(2, 7)
HDBSCAN_MIN  = 15         # min_cluster_size for HDBSCAN


# ── helper: relabel clusters by PC1 mean (same rule as script 10) ─────
def relabel_by_pc1(labels: np.ndarray, pc1: np.ndarray) -> np.ndarray:
    """Relabel so cluster with lowest mean PC1 = 0, highest = 2."""
    tmp = pd.DataFrame({"c": labels, "pc1": pc1})
    order = tmp.groupby("c")["pc1"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap.get(c, -1) for c in labels], dtype=int)


def main():
    out = Path(OUT_DIR)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    # ── load & embed ─────────────────────────────────────────────────
    print("Loading embeddings...")
    emb_subj = np.load(EMB_SUBJ_NPY)
    attn_df  = pd.read_csv(ATTN_CSV)
    subj_iids = attn_df["IID"].astype(str).values

    Zs = StandardScaler().fit_transform(emb_subj)
    pca_model = PCA(n_components=10, random_state=42)
    subj_pca  = pca_model.fit_transform(Zs)
    pc1       = subj_pca[:, 0]
    X         = subj_pca[:, :10]       # same feature space as script 10

    # ── reference clustering (seed=42, same as script 10) ────────────
    print("Computing reference clustering (seed=42)...")
    km_ref = KMeans(n_clusters=K_REF, random_state=42, n_init=20)
    ref_raw = km_ref.fit_predict(X)
    ref_labels = relabel_by_pc1(ref_raw, pc1)

    # ══════════════════════════════════════════════════════════════════
    # CHECK 1: KMeans seed stability
    # ══════════════════════════════════════════════════════════════════
    print(f"\nRunning KMeans across {N_SEEDS} seeds...")
    seed_labels = {}   # seed -> labels array
    ari_vs_ref  = {}

    rng = np.random.default_rng(0)
    seeds = rng.integers(0, 100_000, size=N_SEEDS).tolist()

    for seed in seeds:
        km = KMeans(n_clusters=K_REF, random_state=int(seed), n_init=20)
        raw = km.fit_predict(X)
        lab = relabel_by_pc1(raw, pc1)
        seed_labels[seed] = lab
        ari_vs_ref[seed] = adjusted_rand_score(ref_labels, lab)

    ari_ref_series = pd.Series(ari_vs_ref)
    print(f"  ARI vs reference — mean: {ari_ref_series.mean():.4f}  "
          f"min: {ari_ref_series.min():.4f}  max: {ari_ref_series.max():.4f}")

    # pairwise ARI sample (50×50 = 2450 pairs)
    seed_list = list(seed_labels.keys())
    pairwise = []
    for i in range(len(seed_list)):
        for j in range(i + 1, len(seed_list)):
            a = adjusted_rand_score(seed_labels[seed_list[i]],
                                    seed_labels[seed_list[j]])
            pairwise.append(a)

    pairwise = np.array(pairwise)
    print(f"  Pairwise ARI — mean: {pairwise.mean():.4f}  "
          f"min: {pairwise.min():.4f}  max: {pairwise.max():.4f}")

    ari_seed_df = pd.DataFrame({
        "seed": seed_list,
        "ari_vs_reference": [ari_vs_ref[s] for s in seed_list],
    })
    ari_seed_df.to_csv(out / "ari_kmeans_seeds.tsv", sep="\t", index=False)

    # plot ARI distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ari_ref_series.values, bins=20, edgecolor="white")
    ax.axvline(ari_ref_series.mean(), color="red", linestyle="--",
               label=f"mean={ari_ref_series.mean():.3f}")
    ax.set_xlabel("ARI vs reference clustering (seed=42)")
    ax.set_ylabel("Count")
    ax.set_title(f"KMeans(k={K_REF}) seed stability across {N_SEEDS} seeds")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "ari_distribution_seeds.png", dpi=160)
    plt.close()

    # per-subject stability: what fraction of seeds agrees with reference?
    all_labels_matrix = np.stack(
        [seed_labels[s] for s in seed_list], axis=1
    )   # shape (N, N_SEEDS)
    subj_mode = np.apply_along_axis(
        lambda row: np.bincount(row[row >= 0], minlength=K_REF).argmax(),
        axis=1,
        arr=all_labels_matrix
    )
    subj_pct_agree = np.mean(
        all_labels_matrix == ref_labels[:, None], axis=1
    ) * 100

    subj_stability_df = pd.DataFrame({
        "IID": subj_iids,
        "ref_cluster": ref_labels,
        "mode_cluster": subj_mode,
        "pct_seed_agreement": subj_pct_agree.round(1),
        "stable": (subj_pct_agree >= 90).astype(int),
    })
    subj_stability_df.to_csv(
        out / "subject_cluster_stability.tsv", sep="\t", index=False
    )

    n_stable = subj_stability_df["stable"].sum()
    pct_stable = 100 * n_stable / len(subj_stability_df)
    print(f"\n  Subjects stable (≥90% seed agreement): "
          f"{n_stable}/{len(subj_stability_df)} ({pct_stable:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(subj_pct_agree, bins=20, edgecolor="white")
    ax.axvline(90, color="red", linestyle="--", label="90% threshold")
    ax.set_xlabel("% seeds agreeing with reference cluster assignment")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Per-subject cluster stability across seeds")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "subject_stability_histogram.png", dpi=160)
    plt.close()

    # ══════════════════════════════════════════════════════════════════
    # CHECK 2: Algorithm comparison — GMM and (optionally) HDBSCAN
    # ══════════════════════════════════════════════════════════════════
    print("\nRunning GMM(k=3)...")
    gmm = GaussianMixture(n_components=K_REF, random_state=42, n_init=10)
    gmm_raw = gmm.fit_predict(X)
    gmm_labels = relabel_by_pc1(gmm_raw, pc1)
    ari_gmm = adjusted_rand_score(ref_labels, gmm_labels)
    print(f"  ARI KMeans vs GMM: {ari_gmm:.4f}")

    algo_rows = [
        {"comparison": "KMeans_vs_GMM",    "ari": ari_gmm, "n_clusters_algo2": K_REF},
    ]

    if HAS_HDBSCAN:
        print(f"Running HDBSCAN(min_cluster_size={HDBSCAN_MIN})...")
        hdb = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN, prediction_data=True)
        hdb_raw = hdb.fit_predict(X)
        n_hdb_clusters = len(set(hdb_raw)) - (1 if -1 in hdb_raw else 0)
        n_noise = (hdb_raw == -1).sum()
        print(f"  HDBSCAN found {n_hdb_clusters} clusters, {n_noise} noise points")

        # ARI on non-noise points only
        mask_valid = hdb_raw != -1
        if mask_valid.sum() > 10 and n_hdb_clusters >= 2:
            ari_hdb = adjusted_rand_score(ref_labels[mask_valid], hdb_raw[mask_valid])
        else:
            ari_hdb = np.nan
        print(f"  ARI KMeans vs HDBSCAN (non-noise): {ari_hdb:.4f}" if np.isfinite(ari_hdb)
              else "  ARI not computable")

        algo_rows.append({
            "comparison": "KMeans_vs_HDBSCAN",
            "ari": ari_hdb,
            "n_clusters_algo2": n_hdb_clusters,
        })

    algo_df = pd.DataFrame(algo_rows)
    algo_df.to_csv(out / "ari_algorithm_comparison.tsv", sep="\t", index=False)

    # scatter: KMeans vs GMM coloring
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (labels, title) in zip(
        axes,
        [(ref_labels, f"KMeans(k={K_REF})  [reference]"),
         (gmm_labels, f"GMM(k={K_REF})  ARI={ari_gmm:.3f}")]
    ):
        sns.scatterplot(
            x=subj_pca[:, 0], y=subj_pca[:, 1],
            hue=labels, palette="tab10",
            s=25, alpha=0.8, ax=ax, legend="full"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Embedding PC1")
        ax.set_ylabel("Embedding PC2")
    plt.suptitle("Algorithm comparison on embedding PCA", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "algorithm_cluster_scatter.png", dpi=160)
    plt.close()

    # ══════════════════════════════════════════════════════════════════
    # CHECK 3: Elbow + silhouette to confirm k=3
    # ══════════════════════════════════════════════════════════════════
    print("\nComputing elbow / silhouette for k=2..6...")
    elbow_rows = []
    for k in K_RANGE:
        km_k = KMeans(n_clusters=k, random_state=42, n_init=20)
        labs_k = km_k.fit_predict(X)
        inertia = km_k.inertia_
        sil = silhouette_score(X, labs_k) if k > 1 else np.nan
        elbow_rows.append({"k": k, "inertia": inertia, "silhouette": sil})
        print(f"  k={k}  inertia={inertia:.1f}  silhouette={sil:.4f}")

    elbow_df = pd.DataFrame(elbow_rows)
    elbow_df.to_csv(out / "elbow_silhouette.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(elbow_df["k"], elbow_df["inertia"], marker="o", color="steelblue")
    axes[0].axvline(K_REF, color="red", linestyle="--", label=f"k={K_REF}")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia (within-cluster SS)")
    axes[0].set_title("Elbow plot")
    axes[0].legend()

    axes[1].plot(elbow_df["k"], elbow_df["silhouette"], marker="o", color="darkorange")
    axes[1].axvline(K_REF, color="red", linestyle="--", label=f"k={K_REF}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette by k")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "elbow_silhouette.png", dpi=160)
    plt.close()

    # ── summary printout ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STABILITY SUMMARY")
    print("=" * 60)
    print(f"\nKMeans seed stability ({N_SEEDS} seeds):")
    print(f"  Mean ARI vs reference : {ari_ref_series.mean():.4f}")
    print(f"  Min  ARI vs reference : {ari_ref_series.min():.4f}")
    print(f"  Pairwise ARI mean     : {pairwise.mean():.4f}")
    print(f"  Subjects ≥90% stable  : {n_stable}/{len(subj_stability_df)} "
          f"({pct_stable:.1f}%)")
    print(f"\nAlgorithm comparison:")
    print(algo_df.to_string(index=False))
    print(f"\nk selection (silhouette):")
    print(elbow_df.to_string(index=False))
    best_k = elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"]
    print(f"  Best k by silhouette: {best_k}")

    verdict = "STABLE" if (
        ari_ref_series.mean() > 0.95 and ari_gmm > 0.80 and pct_stable > 85
    ) else "UNSTABLE — review before proceeding"
    print(f"\nVerdict: {verdict}")
    print(f"\nOutputs saved to: {out}")

if __name__ == "__main__":
    main()