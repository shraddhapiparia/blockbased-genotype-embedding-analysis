#!/usr/bin/env python3
"""analyze_phase2_blocks.py — post-Phase 2 block-level attention analysis.

Purpose : Load Phase 2 outputs; rank blocks by mean attention weight; compare
          asthma vs control block groups; produce summary tables and plots.
Inputs  : results/output_regions2/<loss>/embeddings/pooling_attention_weights.csv,
          per_block_recon_mse.csv; optionally results/output_regions/block_order.csv
Outputs : results/output_regions2/<loss>/block_analysis/ (CSV tables + PNGs)
Workflow: Core supporting — run after Phase 2; feeds downstream analysis scripts.
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Helpers
# ============================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def load_inputs(loss_dir: Path, phase1_dir: Path = None):
    """
    Reads earlier-produced Phase 2 files from one loss directory.

    Expected:
      loss_dir/
        embeddings/pooling_attention_weights.csv
        embeddings/per_block_recon_mse.csv

    Optional:
      phase1_dir/block_order.csv
    """
    emb_dir = loss_dir / "embeddings"

    attn_csv = emb_dir / "pooling_attention_weights.csv"
    recon_csv = emb_dir / "per_block_recon_mse.csv"

    if not attn_csv.exists():
        raise FileNotFoundError(f"Missing file: {attn_csv}")
    if not recon_csv.exists():
        raise FileNotFoundError(f"Missing file: {recon_csv}")

    attn_df = pd.read_csv(attn_csv)
    recon_df = pd.read_csv(recon_csv)

    # attention matrix: rows = subjects, columns = blocks
    if "IID" in attn_df.columns:
        block_cols = [c for c in attn_df.columns if c != "IID"]
    else:
        block_cols = list(attn_df.columns)

    attn_mat = attn_df[block_cols].to_numpy(dtype=float)

    # recon mse
    if not {"block_id", "recon_mse"}.issubset(recon_df.columns):
        raise ValueError("per_block_recon_mse.csv must contain columns: block_id, recon_mse")

    # optional metadata
    block_meta = None
    if phase1_dir is not None:
        meta_fp = phase1_dir / "block_order.csv"
        if meta_fp.exists():
            block_meta = pd.read_csv(meta_fp)

    return attn_df, attn_mat, block_cols, recon_df, block_meta


def summarize_blocks(attn_mat, block_cols, recon_df, block_meta=None):
    """
    Builds one block-level summary table.
    """
    s = pd.DataFrame({
        "block_id": block_cols,
        "mean_attn": attn_mat.mean(axis=0),
        "std_attn": attn_mat.std(axis=0),
        "min_attn": attn_mat.min(axis=0),
        "max_attn": attn_mat.max(axis=0),
        "q25_attn": np.quantile(attn_mat, 0.25, axis=0),
        "median_attn": np.quantile(attn_mat, 0.50, axis=0),
        "q75_attn": np.quantile(attn_mat, 0.75, axis=0),
    })

    out = s.merge(recon_df, on="block_id", how="left")

    if block_meta is not None and "block_id" in block_meta.columns:
        keep_cols = [c for c in block_meta.columns if c in {
            "block_id", "gene", "n_snps", "pos"
        }]
        out = out.merge(block_meta[keep_cols].drop_duplicates(), on="block_id", how="left")

    out["rank_mean_attn"] = out["mean_attn"].rank(ascending=False, method="min").astype(int)
    out["rank_std_attn"] = out["std_attn"].rank(ascending=False, method="min").astype(int)
    out["rank_recon_mse"] = out["recon_mse"].rank(ascending=False, method="min").astype(int)

    out["z_mean_attn"] = zscore(out["mean_attn"])
    out["z_std_attn"] = zscore(out["std_attn"])
    out["z_recon_mse"] = zscore(out["recon_mse"])

    # simple combined prioritization scores
    out["score_high_attn_high_err"] = out["z_mean_attn"] + out["z_recon_mse"]
    out["score_selective_and_hard"] = out["z_std_attn"] + out["z_recon_mse"]

    # flags
    out["top30_mean_attn"] = out["rank_mean_attn"] <= 30
    out["top30_std_attn"] = out["rank_std_attn"] <= 30
    out["top30_recon_mse"] = out["rank_recon_mse"] <= 30

    return out.sort_values("mean_attn", ascending=False).reset_index(drop=True)


def get_topk(df, by, k=30, ascending=False):
    return df.sort_values(by, ascending=ascending).head(k).copy()


def get_overlap_table(summary_df, k=30):
    mean_set = set(summary_df.nsmallest(k, "rank_mean_attn")["block_id"])
    std_set = set(summary_df.nsmallest(k, "rank_std_attn")["block_id"])
    mse_set = set(summary_df.nsmallest(k, "rank_recon_mse")["block_id"])

    all_blocks = sorted(mean_set | std_set | mse_set)
    rows = []
    for b in all_blocks:
        rows.append({
            "block_id": b,
            "top_mean_attn": b in mean_set,
            "top_std_attn": b in std_set,
            "top_recon_mse": b in mse_set,
            "n_lists": int((b in mean_set) + (b in std_set) + (b in mse_set)),
        })
    return pd.DataFrame(rows).sort_values(["n_lists", "block_id"], ascending=[False, True])


# ============================================================
# Plotting
# ============================================================
def plot_barh(df, xcol, ycol, title, xlabel, out_fp, color="#5b8fd1"):
    fig_h = max(5, 0.35 * len(df))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(len(df))
    ax.barh(y, df[xcol].values, color=color, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(df[ycol].values, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_fp, dpi=180)
    plt.close()


def plot_scatter(summary_df, out_fp):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        summary_df["mean_attn"].values,
        summary_df["recon_mse"].values,
        s=18,
        alpha=0.75
    )
    ax.set_xlabel("Mean pooling attention")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Mean attention vs reconstruction MSE")

    # annotate a few interesting points
    ann = summary_df.sort_values("score_high_attn_high_err", ascending=False).head(12)
    for _, r in ann.iterrows():
        ax.annotate(
            r["block_id"],
            (r["mean_attn"], r["recon_mse"]),
            fontsize=7,
            alpha=0.9
        )

    plt.tight_layout()
    plt.savefig(out_fp, dpi=180)
    plt.close()


def plot_filtered_heatmap(attn_mat, block_cols, selected_blocks, out_fp, max_subjects=120):
    block_to_idx = {b: i for i, b in enumerate(block_cols)}
    sel_idx = [block_to_idx[b] for b in selected_blocks if b in block_to_idx]
    if len(sel_idx) == 0:
        return

    A = attn_mat[:, sel_idx]

    # pick subset of subjects for readability
    n = A.shape[0]
    keep = min(max_subjects, n)
    if keep < n:
        # sort by mean attention over selected blocks and take evenly spaced sample
        order = np.argsort(A.mean(axis=1))
        take = np.linspace(0, n - 1, keep).astype(int)
        A = A[order][take]

    fig_w = max(8, 0.35 * len(sel_idx))
    fig_h = max(5, 0.06 * A.shape[0] + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(A, aspect="auto", interpolation="nearest", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(sel_idx)))
    ax.set_xticklabels([block_cols[i] for i in sel_idx], rotation=90, fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Selected blocks")
    ax.set_ylabel("Subjects (sampled)")
    ax.set_title("Filtered pooling-attention heatmap")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pooling attention")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=180)
    plt.close()


def plot_attn_vs_std(summary_df, out_fp):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        summary_df["mean_attn"].values,
        summary_df["std_attn"].values,
        s=18,
        alpha=0.75
    )
    ax.set_xlabel("Mean pooling attention")
    ax.set_ylabel("Attention SD across subjects")
    ax.set_title("Mean attention vs attention variability")

    ann = summary_df.sort_values("std_attn", ascending=False).head(12)
    for _, r in ann.iterrows():
        ax.annotate(
            r["block_id"],
            (r["mean_attn"], r["std_attn"]),
            fontsize=7,
            alpha=0.9
        )

    plt.tight_layout()
    plt.savefig(out_fp, dpi=180)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Analyze filtered block-level attention/reconstruction from Phase 2 outputs"
    )
    ap.add_argument(
        "--loss-dir",
        required=True,
        help="Path to one loss output folder, e.g. ../output_regions2/ORD"
    )
    ap.add_argument(
        "--phase1-dir",
        default=None,
        help="Optional path to Phase 1 dir containing block_order.csv"
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Top-k blocks to show in filtered tables/plots"
    )
    args = ap.parse_args()

    loss_dir = Path(args.loss_dir)
    phase1_dir = Path(args.phase1_dir) if args.phase1_dir else None
    topk = args.topk

    out_dir = loss_dir / "filtered_analysis"
    plot_dir = out_dir / "plots"
    table_dir = out_dir / "tables"
    ensure_dir(out_dir)
    ensure_dir(plot_dir)
    ensure_dir(table_dir)

    # ---------- load ----------
    attn_df, attn_mat, block_cols, recon_df, block_meta = load_inputs(loss_dir, phase1_dir)

    # ---------- summarize ----------
    summary_df = summarize_blocks(attn_mat, block_cols, recon_df, block_meta)
    summary_df.to_csv(table_dir / "block_summary_full.csv", index=False)

    top_mean = get_topk(summary_df, "mean_attn", k=topk, ascending=False)
    top_std = get_topk(summary_df, "std_attn", k=topk, ascending=False)
    top_mse = get_topk(summary_df, "recon_mse", k=topk, ascending=False)
    top_high_attn_high_err = get_topk(summary_df, "score_high_attn_high_err", k=topk, ascending=False)
    top_selective_and_hard = get_topk(summary_df, "score_selective_and_hard", k=topk, ascending=False)

    top_mean.to_csv(table_dir / f"top{topk}_mean_attention.csv", index=False)
    top_std.to_csv(table_dir / f"top{topk}_attention_sd.csv", index=False)
    top_mse.to_csv(table_dir / f"top{topk}_recon_mse.csv", index=False)
    top_high_attn_high_err.to_csv(table_dir / f"top{topk}_high_attention_high_error.csv", index=False)
    top_selective_and_hard.to_csv(table_dir / f"top{topk}_selective_and_hard.csv", index=False)

    overlap_df = get_overlap_table(summary_df, k=topk)
    overlap_df.to_csv(table_dir / f"top{topk}_overlap_table.csv", index=False)

    # ---------- plots ----------
    plot_barh(
        top_mean.sort_values("mean_attn", ascending=True),
        xcol="mean_attn",
        ycol="block_id",
        title=f"Top {topk} blocks by mean pooling attention",
        xlabel="Mean pooling attention",
        out_fp=plot_dir / f"top{topk}_mean_attention.png",
        color="#5b8fd1",
    )

    plot_barh(
        top_std.sort_values("std_attn", ascending=True),
        xcol="std_attn",
        ycol="block_id",
        title=f"Top {topk} blocks by attention variability",
        xlabel="Attention SD across subjects",
        out_fp=plot_dir / f"top{topk}_attention_sd.png",
        color="#e08a7a",
    )

    plot_barh(
        top_mse.sort_values("recon_mse", ascending=True),
        xcol="recon_mse",
        ycol="block_id",
        title=f"Top {topk} blocks by reconstruction MSE",
        xlabel="Reconstruction MSE",
        out_fp=plot_dir / f"top{topk}_recon_mse.png",
        color="#c76b6b",
    )

    plot_barh(
        top_high_attn_high_err.sort_values("score_high_attn_high_err", ascending=True),
        xcol="score_high_attn_high_err",
        ycol="block_id",
        title=f"Top {topk} blocks: high attention + high reconstruction error",
        xlabel="z(mean attention) + z(recon MSE)",
        out_fp=plot_dir / f"top{topk}_high_attention_high_error.png",
        color="#9c6bd3",
    )

    plot_barh(
        top_selective_and_hard.sort_values("score_selective_and_hard", ascending=True),
        xcol="score_selective_and_hard",
        ycol="block_id",
        title=f"Top {topk} blocks: selective + hard to reconstruct",
        xlabel="z(attention SD) + z(recon MSE)",
        out_fp=plot_dir / f"top{topk}_selective_and_hard.png",
        color="#8b5a3c",
    )

    plot_scatter(summary_df, plot_dir / "scatter_mean_attention_vs_recon_mse.png")
    plot_attn_vs_std(summary_df, plot_dir / "scatter_mean_attention_vs_attention_sd.png")

    # heatmap from top mean-attn blocks
    plot_filtered_heatmap(
        attn_mat=attn_mat,
        block_cols=block_cols,
        selected_blocks=top_mean["block_id"].tolist(),
        out_fp=plot_dir / f"heatmap_top{topk}_mean_attention_blocks.png",
        max_subjects=120,
    )

    # heatmap from top variable-attn blocks
    plot_filtered_heatmap(
        attn_mat=attn_mat,
        block_cols=block_cols,
        selected_blocks=top_std["block_id"].tolist(),
        out_fp=plot_dir / f"heatmap_top{topk}_attention_sd_blocks.png",
        max_subjects=120,
    )

    # ---------- quick console summary ----------
    print("\nSaved outputs to:")
    print(f"  {out_dir}")

    print("\nMain tables:")
    print(f"  {table_dir / 'block_summary_full.csv'}")
    print(f"  {table_dir / f'top{topk}_mean_attention.csv'}")
    print(f"  {table_dir / f'top{topk}_attention_sd.csv'}")
    print(f"  {table_dir / f'top{topk}_recon_mse.csv'}")
    print(f"  {table_dir / f'top{topk}_overlap_table.csv'}")

    print("\nMain plots:")
    print(f"  {plot_dir / f'top{topk}_mean_attention.png'}")
    print(f"  {plot_dir / f'top{topk}_attention_sd.png'}")
    print(f"  {plot_dir / f'top{topk}_recon_mse.png'}")
    print(f"  {plot_dir / 'scatter_mean_attention_vs_recon_mse.png'}")
    print(f"  {plot_dir / f'heatmap_top{topk}_mean_attention_blocks.png'}")

    print("\nTop 10 by mean attention:")
    print(top_mean[["block_id", "mean_attn", "std_attn", "recon_mse"]].head(10).to_string(index=False))

    print("\nTop 10 by attention SD:")
    print(top_std[["block_id", "mean_attn", "std_attn", "recon_mse"]].head(10).to_string(index=False))

    print("\nTop 10 by reconstruction MSE:")
    print(top_mse[["block_id", "mean_attn", "std_attn", "recon_mse"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()