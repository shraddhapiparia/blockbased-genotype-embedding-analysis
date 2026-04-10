#!/usr/bin/env python3
"""05_umap_hla_interpretation.py — UMAP embedding and HLA cluster interpretation.

Purpose : Reduce subject embeddings via PCA then UMAP; run KMeans clustering;
          overlay HLA block_PC1 values to interpret cluster geometry.
Inputs  : results/output_regions2/<loss>/embeddings/{individual_embeddings.npy,
          block_contextual_repr.npy}, results/output_regions/block_order.csv,
          metadata/ldpruned_997subs.eigenvec
Outputs : results/output_regions2/<loss>/umap_hla/ (PNGs, cluster_labels.csv)
Workflow: Active analysis — Step 5; precedes confounder and phenotype analyses.
"""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples

try:
    import umap
except ImportError:
    umap = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
    print('[umap_hla] missing seaborn; fallback to matplotlib only for plotting')


def pick_best_k(metrics_df):
    km = metrics_df[metrics_df['method'] == 'KMeans'].copy()
    km = km.sort_values('silhouette', ascending=False)
    if len(km) == 0:
        raise ValueError('No KMeans rows found in clustering_metrics.csv')
    best_row = km.iloc[0]
    best_k = int(best_row['k'])
    best_key = f'kmeans_k{best_k}'
    best_sil = float(best_row['silhouette'])
    return best_key, best_k, best_sil


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='UMAP-HLA analysis script')
    parser.add_argument('--phase2-loss-dir', required=True,
                        help='Phase2 loss directory (e.g. results/output_regions2/ORD)')
    parser.add_argument('--phase1-dir', default='results/output_regions',
                        help='Phase1 directory with block_order.csv')
    parser.add_argument('--pheno-file', default=None, help='Optional phenotype CSV to merge using IID/S_SUBJECTID')
    parser.add_argument('--iid-col', default='IID', help='Subject identifier column in phase2 outputs')
    parser.add_argument('--umap-n-neighbors', type=int, default=20)
    parser.add_argument('--umap-min-dist', type=float, default=0.1)
    parser.add_argument('--umap-metric', default='cosine')
    parser.add_argument('--outdir', default=None, help='Output directory for plots and tables')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.dry_run:
        print('[umap_hla] dry-run mode; no computation performed')
        print(vars(args))
        return

    phase2_dir = Path(args.phase2_loss_dir)
    phase1_dir = Path(args.phase1_dir)

    expected_files = [
        phase2_dir / 'embeddings' / 'individual_embeddings.npy',
        phase2_dir / 'embeddings' / 'pooling_attention_weights.csv',
        phase2_dir / 'clustering' / 'cluster_labels.csv',
        phase2_dir / 'clustering' / 'clustering_metrics.csv',
        phase1_dir / 'block_order.csv',
    ]

    for p in expected_files:
        if not p.exists():
            raise FileNotFoundError(f'Missing expected file: {p}')

    outdir = Path(args.outdir if args.outdir else phase2_dir / 'umap_hla')
    ensure_dir(outdir)

    log = []
    t0 = time.time()

    embeddings = np.load(phase2_dir / 'embeddings' / 'individual_embeddings.npy')
    pool_df = pd.read_csv(phase2_dir / 'embeddings' / 'pooling_attention_weights.csv')
    block_order = pd.read_csv(phase1_dir / 'block_order.csv')
    cluster_df = pd.read_csv(phase2_dir / 'clustering' / 'cluster_labels.csv')
    metrics_df = pd.read_csv(phase2_dir / 'clustering' / 'clustering_metrics.csv')

    best_key, best_k, best_sil = pick_best_k(metrics_df)
    if best_key not in cluster_df.columns:
        raise KeyError(f'Best key {best_key} not found in cluster_labels.csv')

    labels = cluster_df[best_key].values

    if umap is None:
        raise ImportError('umap-learn is required: pip install umap-learn')

    Z = StandardScaler().fit_transform(embeddings)
    reducer = umap.UMAP(n_neighbors=args.umap_n_neighbors,
                        min_dist=args.umap_min_dist,
                        metric=args.umap_metric,
                        random_state=42)
    Z2d = reducer.fit_transform(Z)

    np.save(outdir / 'umap_2d.npy', Z2d)

    # Plot cluster UMAP
    plt.figure(figsize=(7, 6))
    if sns is not None:
        palette = sns.color_palette('tab10', n_colors=max(10, best_k))
    else:
        cmap = plt.get_cmap('tab10')
        palette = [cmap(i) for i in range(max(10, best_k))]
    for c in sorted(np.unique(labels)):
        mask = labels == c
        plt.scatter(Z2d[mask, 0], Z2d[mask, 1], s=8, alpha=0.7, label=f'Cluster {c}', c=[palette[c % len(palette)]])
    plt.title(f'UMAP of ORD embeddings (k={best_k}, sil={best_sil:.3f})')
    plt.xlabel('UMAP-1'); plt.ylabel('UMAP-2')
    plt.legend(markerscale=2, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / 'umap_clusters.png', dpi=160)
    plt.close()

    # Attention -> top blocks list
    block_cols = [c for c in pool_df.columns if c != args.iid_col]
    mean_attn = pool_df[block_cols].mean(axis=0)
    top_blocks = mean_attn.sort_values(ascending=False).head(10)
    top_blocks.to_csv(outdir / 'top10_blocks_by_attention.csv', header=['mean_attn'])

    # UMAP color by top blocks again
    for block_id in top_blocks.index.tolist():
        vals = pool_df[block_id].values
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(Z2d[:, 0], Z2d[:, 1], c=vals, cmap='plasma', s=8, alpha=0.75)
        plt.colorbar(sc, label='attention')
        plt.title(f'UMAP colored by attention to {block_id}')
        plt.xlabel('UMAP-1'); plt.ylabel('UMAP-2')
        plt.tight_layout()
        plt.savefig(outdir / f'umap_block_{block_id}_attention.png', dpi=160)
        plt.close()

    # Optional phenotype merge
    if args.pheno_file:
        pheno = pd.read_csv(args.pheno_file)
        if args.iid_col not in pheno.columns and 'S_SUBJECTID' in pheno.columns:
            pheno = pheno.rename(columns={'S_SUBJECTID': args.iid_col})

        if args.iid_col not in pheno.columns:
            raise KeyError(f'Phenotype file missing required ID column: {args.iid_col} or S_SUBJECTID')

        merged = pd.DataFrame({args.iid_col: pool_df[args.iid_col].astype(str)})
        merged = merged.merge(pheno.astype({args.iid_col: str}), on=args.iid_col, how='left')
        unmatched = merged[merged.isnull().any(axis=1)].shape[0]
        print(f'[umap_hla] phenotype merge: {unmatched}/{merged.shape[0]} missing values due to ID mismatch')
        merged.to_csv(outdir / 'merged_pheno.csv', index=False)

    log.append({'phase2_loss_dir': str(phase2_dir), 'phase1_dir': str(phase1_dir), 'outdir': str(outdir), 'n_subjects': embeddings.shape[0], 'embedding_dim': embeddings.shape[1], 'best_k': best_k, 'best_sil': best_sil, 'top_hla_candidates': ','.join(top_blocks.index.tolist())})
    pd.DataFrame(log).to_csv(outdir / 'umap_hla_analysis_summary.csv', index=False)

    print(f'[umap_hla] complete in {time.time() - t0:.1f}s, output in {outdir}')


if __name__ == '__main__':
    main()
