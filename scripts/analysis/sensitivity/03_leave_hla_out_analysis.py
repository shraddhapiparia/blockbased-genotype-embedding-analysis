#!/usr/bin/env python3
"""
03_leave_hla_out_analysis.py

Leave-HLA-out validation.

What this does
--------------
1. Load Phase 1 block embeddings (all_blocks.npy) — frozen VAE outputs
2. Identify HLA class II block indices from block_order.csv
3. Zero-mask those blocks in the embedding matrix
4. Re-run Phase 2 attention model on masked embeddings
5. Cluster subjects using the new (no-HLA) subject embeddings
6. Test whether ORIGINAL HLA block_PC1 values still separate the new clusters

Why this breaks circularity
---------------------------
The new clusters are built without seeing HLA at all.
If HLA block_PC1 still separates them, HLA is aligned with
broader genomic structure — not just defining its own test space.
"""

from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.core.attention_phase2 import load_config, run_phase2

# ── paths ──────────────────────────────────────────────────────────────
PHASE2_CONFIG_NOHLA  = "configs/config_phase2_noHLA.yaml"
ORIGINAL_BLOCK_NPY   = "results/output_regions2/ORD/embeddings/block_contextual_repr.npy"
ORIGINAL_ATTN_CSV    = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
BLOCK_ORDER          = "results/output_regions/block_order.csv"
OUT_DIR              = "results/output_regions2/ORD/hla_cluster_analysis/leave_hla_out"

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

# ── helpers (same as script 10) ────────────────────────────────────────
def block_first_pc(block_repr: np.ndarray, block_idx: int) -> np.ndarray:
    X = block_repr[:, block_idx, :]
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1, random_state=42)
    pc = pca.fit_transform(X).ravel()
    if pca.components_[0, 0] < 0:
        pc = -pc
    return pc

def eta_squared(groups):
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    return float(ss_between / ss_total) if ss_total > 0 else np.nan

def test_vs_cluster(values, clusters):
    df = pd.DataFrame({"v": values, "c": clusters}).dropna()
    groups = [g["v"].values for _, g in df.groupby("c")]
    if len(groups) < 2 or any(len(g) < 5 for g in groups):
        return {"eta2": np.nan, "kruskal_p": np.nan}
    _, p_kw = stats.kruskal(*groups)
    return {"eta2": eta_squared(groups), "kruskal_p": float(p_kw)}

# ── main ───────────────────────────────────────────────────────────────
def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── Step 1: check if noHLA Phase 2 already ran ─────────────────────
    cfg = load_config(PHASE2_CONFIG_NOHLA)
    noHLA_emb_path = Path(cfg["output_dir"]) / "ORD/embeddings/individual_embeddings.npy"

    if not noHLA_emb_path.exists():
        print("No-HLA Phase 2 embeddings not found. Running Phase 2 with HLA blocks masked...")
        print("  This will take a while — same duration as original Phase 2.")

        # Patch: zero out HLA blocks in all_blocks.npy before phase2 reads it
        # We do this by temporarily writing a masked version
        phase1_dir = Path(cfg["phase1_dir"])
        all_blocks_path = phase1_dir / "ORD/embeddings/all_blocks.npy"

        print(f"  Loading Phase 1 blocks from {all_blocks_path}")
        all_blocks = np.load(all_blocks_path)   # (N, B, D_vae)

        bo = pd.read_csv(BLOCK_ORDER)
        name_col = next(c for c in ["block", "block_id", "region", "name"] if c in bo.columns)
        block_names = bo[name_col].astype(str).tolist()

        exclude_patterns = cfg.get("exclude_blocks_containing", ["HLA_classII"])
        hla_indices = [
            i for i, name in enumerate(block_names)
            if any(pat in name for pat in exclude_patterns)
        ]
        print(f"  Masking {len(hla_indices)} HLA blocks: indices {hla_indices[:5]}...")

        all_blocks_masked = all_blocks.copy()
        all_blocks_masked[:, hla_indices, :] = 0.0

        # Save masked version temporarily
        masked_path = phase1_dir / "ORD/embeddings/all_blocks_noHLA.npy"
        np.save(masked_path, all_blocks_masked)
        print(f"  Saved masked blocks to {masked_path}")

        orig = phase1_dir / "ORD/embeddings/all_blocks.npy"
        shutil.copy(masked_path, orig)
        print(f"  Overwrote all_blocks.npy with masked version")

        run_phase2(cfg)
        print(f"  Phase 2 complete. Embeddings saved to {noHLA_emb_path}")
    else:
        print(f"Found existing no-HLA embeddings at {noHLA_emb_path}, skipping Phase 2 rerun.")

    # ── Step 2: load noHLA subject embeddings and cluster ──────────────
    print("\nLoading no-HLA subject embeddings...")
    emb_noHLA = np.load(noHLA_emb_path)          # (N, d)
    print(f"  shape: {emb_noHLA.shape}")

    attn_df = pd.read_csv(ORIGINAL_ATTN_CSV)
    subj_iids = attn_df["IID"].astype(str).values

    print("Computing subject PCA on no-HLA embeddings...")
    Zs = StandardScaler().fit_transform(emb_noHLA)
    pca_model = PCA(n_components=10, random_state=42)
    subj_pca = pca_model.fit_transform(Zs)
    explained = pca_model.explained_variance_ratio_ * 100

    print("Clustering subjects (KMeans k=3) on no-HLA embedding PCA...")
    km = KMeans(n_clusters=3, random_state=42, n_init=20)
    clusters_raw = km.fit_predict(subj_pca[:, :10])

    tmp = pd.DataFrame({"cr": clusters_raw, "PC1": subj_pca[:, 0]})
    order = tmp.groupby("cr")["PC1"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    clusters_noHLA = np.array([remap[c] for c in clusters_raw], dtype=int)

    print("\nNo-HLA cluster sizes:")
    for k, v in pd.Series(clusters_noHLA).value_counts().sort_index().items():
        print(f"  cluster {k}: n={v}")

    noHLA_df = pd.DataFrame({
        "IID": subj_iids,
        "cluster_noHLA": clusters_noHLA,
        "noHLA_PC1": subj_pca[:, 0],
        "noHLA_PC2": subj_pca[:, 1],
    })
    noHLA_df.to_csv(out / "subject_clusters_noHLA.tsv", sep="\t", index=False)

    # scatter plot of noHLA clusters
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=noHLA_df, x="noHLA_PC1", y="noHLA_PC2",
        hue="cluster_noHLA", palette="tab10", s=28, alpha=0.8, ax=ax
    )
    ax.set_title(
        f"No-HLA subject clusters on embedding PCA\n"
        f"PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "subject_clusters_noHLA_pca.png", dpi=160)
    plt.close()

    # ── Step 3: compute original HLA block_PC1 values ─────────────────
    print("\nLoading original block contextual embeddings...")
    emb_block_orig = np.load(ORIGINAL_BLOCK_NPY)   # (N, B, D)

    bo = pd.read_csv(BLOCK_ORDER)
    name_col = next(c for c in ["block", "block_id", "region", "name"] if c in bo.columns)
    block_names = bo[name_col].astype(str).tolist()
    block_to_idx = {b: i for i, b in enumerate(block_names)}

    print("Computing original HLA block_PC1 values...")
    for block in HLA_BLOCKS:
        idx = block_to_idx[block]
        noHLA_df[block] = block_first_pc(emb_block_orig, idx)

    # ── Step 4: test HLA block_PC1 ~ noHLA clusters ───────────────────
    print("\nTesting original HLA block_PC1 ~ no-HLA clusters...")
    rows = []
    for block in HLA_BLOCKS:
        res = test_vs_cluster(noHLA_df[block].values, noHLA_df["cluster_noHLA"].values)
        means = noHLA_df.groupby("cluster_noHLA")[block].mean().to_dict()
        rows.append({
            "block": block,
            **res,
            "cluster0_mean": float(means.get(0, np.nan)),
            "cluster1_mean": float(means.get(1, np.nan)),
            "cluster2_mean": float(means.get(2, np.nan)),
        })

    results_df = pd.DataFrame(rows).sort_values(
        ["eta2", "kruskal_p"], ascending=[False, True]
    ).reset_index(drop=True)
    results_df.to_csv(out / "hla_block_vs_noHLA_clusters.tsv", sep="\t", index=False)

    print("\nHLA block_PC1 ~ no-HLA cluster results:")
    print(results_df[["block", "eta2", "kruskal_p",
                       "cluster0_mean", "cluster1_mean", "cluster2_mean"
                       ]].to_string(index=False))

    # ── Step 5: heatmap ────────────────────────────────────────────────
    heat_df = results_df.set_index("block")[
        ["cluster0_mean", "cluster1_mean", "cluster2_mean"]
    ]
    fig, ax = plt.subplots(figsize=(7, max(5, len(heat_df) * 0.35)))
    sns.heatmap(
        heat_df, cmap="coolwarm", center=0,
        linewidths=0.5, annot=True, fmt=".2f", ax=ax
    )
    ax.set_title(
        "Original HLA block_PC1 mean\nby no-HLA subject cluster",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "hla_block_vs_noHLA_cluster_heatmap.png", dpi=160)
    plt.close()

# find which non-HLA blocks best separate the noHLA clusters
    NON_HLA_BLOCKS = [b for b in block_names if "HLA_classII" not in b]

    print("\nTesting non-HLA blocks ~ no-HLA clusters (top 20)...")
    nonhla_rows = []
    for block in NON_HLA_BLOCKS:
        idx = block_to_idx[block]
        pc1 = block_first_pc(emb_block_orig, idx)
        res = test_vs_cluster(pc1, noHLA_df["cluster_noHLA"].values)
        nonhla_rows.append({"block": block, **res})

    nonhla_df = pd.DataFrame(nonhla_rows).sort_values(
        ["eta2", "kruskal_p"], ascending=[False, True]
    ).reset_index(drop=True)
    nonhla_df.to_csv(out / "nonHLA_block_vs_noHLA_clusters.tsv", sep="\t", index=False)
    print(nonhla_df.head(20).to_string(index=False))

    # ── PDE4D cross-cluster check ──────────────────────────────────────
    # load original clusters now — needed for both PDE4D check and HLA spread check
    orig_clusters_df = pd.read_csv(
        "results/output_regions2/ORD/hla_cluster_analysis/subject_clusters.tsv",
        sep="\t"
    )
    noHLA_df = noHLA_df.merge(
        orig_clusters_df[["IID", "cluster"]], on="IID", how="left"
    )
    # align original cluster labels to same subject order as noHLA_df
    orig_cluster_labels = noHLA_df["cluster"].values

    top_pde4d = "region_5q21_PDE4D_sb32"
    idx = block_to_idx[top_pde4d]
    pde4d_pc1 = block_first_pc(emb_block_orig, idx)

    res_orig  = test_vs_cluster(pde4d_pc1, orig_cluster_labels)
    res_noHLA = test_vs_cluster(pde4d_pc1, noHLA_df["cluster_noHLA"].values)

    print(f"\nPDE4D sb32 vs ORIGINAL HLA clusters: eta2={res_orig['eta2']:.4f},  p={res_orig['kruskal_p']:.2e}")
    print(f"PDE4D sb32 vs NO-HLA clusters:       eta2={res_noHLA['eta2']:.4f},  p={res_noHLA['kruskal_p']:.2e}")

    # ── PDE4D size confound check ──────────────────────────────────────
    print("\nChecking PDE4D signal distribution (size confound check)...")
    pde4d_blocks = [b for b in block_names if "PDE4D" in b]
    pde4d_rows = []
    for block in pde4d_blocks:
        idx = block_to_idx[block]
        pc1 = block_first_pc(emb_block_orig, idx)
        res = test_vs_cluster(pc1, noHLA_df["cluster_noHLA"].values)
        pde4d_rows.append({"block": block, **res})

    pde4d_df = pd.DataFrame(pde4d_rows).sort_values("eta2", ascending=False)
    pde4d_df.to_csv(out / "pde4d_block_vs_noHLA_clusters.tsv", sep="\t", index=False)

    print(f"PDE4D blocks with eta2 > 0.05: {(pde4d_df['eta2'] > 0.05).sum()} / {len(pde4d_df)}")
    print(f"PDE4D median eta2:             {pde4d_df['eta2'].median():.4f}")
    print(f"PDE4D max eta2:                {pde4d_df['eta2'].max():.4f}")

    # ── non-HLA non-PDE4D blocks ───────────────────────────────────────
    print("\nChecking non-HLA non-PDE4D blocks (true size confound control)...")
    other_blocks = [b for b in block_names if "PDE4D" not in b and "HLA" not in b]
    other_rows = []
    for block in other_blocks:
        idx = block_to_idx[block]
        pc1 = block_first_pc(emb_block_orig, idx)
        res = test_vs_cluster(pc1, noHLA_df["cluster_noHLA"].values)
        other_rows.append({"block": block, **res})

    other_df = pd.DataFrame(other_rows).sort_values(
        "eta2", ascending=False
    ).reset_index(drop=True)
    other_df.to_csv(out / "nonHLA_nonPDE4D_block_vs_noHLA_clusters.tsv", sep="\t", index=False)

    print("\nTop 10 non-HLA non-PDE4D blocks vs no-HLA clusters:")
    print(other_df.head(10)[["block", "eta2", "kruskal_p"]].to_string(index=False))

    # ── HLA sub-block eta2 spread vs ORIGINAL clusters (size confound defense)
    print("\nHLA sub-block eta2 spread vs ORIGINAL clusters (size confound check)...")
    hla_eta2_rows = []
    for b in HLA_BLOCKS:
        idx = block_to_idx[b]
        pc1 = block_first_pc(emb_block_orig, idx)
        res = test_vs_cluster(pc1, orig_cluster_labels)
        hla_eta2_rows.append({"block": b, "eta2": res["eta2"]})

    hla_eta2_df = pd.DataFrame(hla_eta2_rows).sort_values("eta2", ascending=False)
    hla_eta2_df.to_csv(out / "hla_eta2_spread_vs_original_clusters.tsv", sep="\t", index=False)

    print(f"HLA eta2 range: {hla_eta2_df['eta2'].min():.4f} "
          f"to {hla_eta2_df['eta2'].max():.4f}")
    print(f"HLA eta2 std:   {hla_eta2_df['eta2'].std():.4f}")
    print("(high std = signal concentrated in specific sub-blocks = not size-driven)")
    print(hla_eta2_df.to_string(index=False))

    # ── final outputs summary ──────────────────────────────────────────
    print(f"\nOutputs saved to: {out}")
    print("  subject_clusters_noHLA.tsv")
    print("  hla_block_vs_noHLA_clusters.tsv")
    print("  nonHLA_block_vs_noHLA_clusters.tsv")
    print("  pde4d_block_vs_noHLA_clusters.tsv")
    print("  nonHLA_nonPDE4D_block_vs_noHLA_clusters.tsv")
    print("  hla_eta2_spread_vs_original_clusters.tsv")
    print("  figures/subject_clusters_noHLA_pca.png")
    print("  figures/hla_block_vs_noHLA_cluster_heatmap.png")

if __name__ == "__main__":
    main()