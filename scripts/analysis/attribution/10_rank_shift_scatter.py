#!/usr/bin/env python3
"""
10_rank_shift_scatter.py

Rank-shift scatter plot: full Phase 2 model vs no-HLA Phase 2 model.

For each genomic block (non-HLA only) plots:
  x = rank in full Phase 2 PC1 attribution
  y = rank in no-HLA Phase 2 PC1 attribution

Blocks that rise in importance after HLA removal appear in the upper-right
region (high full rank, low no-HLA rank).  The diagonal y = x marks no change.

Outputs
-------
results/analysis/phase2_block_attribution/phase2_PC1_rank_shift_scatter.png
results/analysis/phase2_block_attribution/phase2_PC2_rank_shift_scatter.png

Run
---
  python scripts/analysis/10_rank_shift_scatter.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ─────────────────────────────────────────────────────────────────────
COMPARISON_CSV = "results/analysis/phase2_block_attribution/phase2_full_vs_noHLA_rank_comparison.csv"
FULL_PC1_CSV   = "results/analysis/phase2_block_attribution/phase2_PC1_leave_one_block_out.csv"
FULL_PC2_CSV   = "results/analysis/phase2_block_attribution/phase2_PC2_leave_one_block_out.csv"
NOHA_PC1_CSV   = "results/analysis/phase2_block_attribution/phase2_noHLA_PC1_leave_one_block_out.csv"
NOHA_PC2_CSV   = "results/analysis/phase2_block_attribution/phase2_noHLA_PC2_leave_one_block_out.csv"
OUT_DIR        = "results/analysis/phase2_block_attribution"


# ── data loading ──────────────────────────────────────────────────────────────
def load_comparison() -> pd.DataFrame:
    """
    Load the pre-built comparison table if available; otherwise build it from
    the individual attribution CSVs.  Only non-HLA blocks (those with finite
    noHLA ranks) are returned.
    """
    comp_path = Path(COMPARISON_CSV)
    if comp_path.exists():
        df = pd.read_csv(comp_path)
        # Derive is_SH2B3 flag if not already present
        if "is_SH2B3" not in df.columns:
            df["is_SH2B3"] = df["block_id"].str.contains("SH2B3", na=False)
        return df
    else:
        # Fallback: merge from individual CSVs on block_id
        print("[warn]  comparison CSV not found; building from individual CSVs")
        f_pc1 = pd.read_csv(FULL_PC1_CSV)[["block_id", "rank"]].rename(
            columns={"rank": "full_PC1_rank"})
        f_pc2 = pd.read_csv(FULL_PC2_CSV)[["block_id", "rank"]].rename(
            columns={"rank": "full_PC2_rank"})
        n_pc1 = pd.read_csv(NOHA_PC1_CSV)[["block_id", "rank"]].rename(
            columns={"rank": "noHLA_PC1_rank"})
        n_pc2 = pd.read_csv(NOHA_PC2_CSV)[["block_id", "rank"]].rename(
            columns={"rank": "noHLA_PC2_rank"})

        df = (f_pc1.merge(f_pc2, on="block_id", how="outer")
                   .merge(n_pc1, on="block_id", how="outer")
                   .merge(n_pc2, on="block_id", how="outer"))

        for pat, col in [("PDE4D", "is_PDE4D"), ("HLA", "is_HLA"),
                         ("17q21", "is_17q21"), ("IL1RL1", "is_IL1RL1"),
                         ("FCER1A", "is_FCER1A"), ("SH2B3", "is_SH2B3")]:
            df[col] = df["block_id"].str.contains(pat, na=False)

        df["rank_change_PC1"] = df["noHLA_PC1_rank"] - df["full_PC1_rank"]
        df["rank_change_PC2"] = df["noHLA_PC2_rank"] - df["full_PC2_rank"]
        return df


# ── plotting ──────────────────────────────────────────────────────────────────
def _assign_color(row) -> str:
    if row.get("is_PDE4D", False):
        return "tomato"
    if row.get("is_IL1RL1", False):
        return "darkorange"
    if row.get("is_SH2B3", False):
        return "mediumpurple"
    if row.get("is_17q21", False):
        return "forestgreen"
    return "steelblue"


def _assign_size(row) -> float:
    # Slightly larger dot for annotated loci
    if any(row.get(k, False) for k in
           ["is_PDE4D", "is_IL1RL1", "is_SH2B3", "is_17q21"]):
        return 28
    return 14


def _short_label(block_id: str) -> str:
    """Return a compact readable label from block_id."""
    # e.g. region_5q21_PDE4D_sb33 → PDE4D_sb33
    parts = block_id.split("_")
    # find locus token (contains letters + digits), skip 'region', 'control', 'chr', coords
    skip = {"region", "control", "cluster", "core"}
    label_parts = [p for p in parts if p.lower() not in skip
                   and not p[:2].isdigit()
                   and not (len(p) > 1 and p[0].isdigit() and p[-1].isalpha()
                            and len(p) <= 5)]
    return "_".join(label_parts) if label_parts else block_id


def make_rank_shift_plot(
    df: pd.DataFrame,
    full_rank_col: str,
    noHLA_rank_col: str,
    rank_change_col: str,
    pc_label: str,
    out_path: Path,
    top_n_pde4d_label: int = 5,
    top_n_improvers_label: int = 3,
) -> None:
    """
    Scatter: x = full rank, y = no-HLA rank, for non-HLA blocks only.

    Rank 1 is plotted at top-left (x-axis: 1 on left; y-axis inverted so 1 is
    at top).  Blocks above the diagonal improved after HLA removal.
    """
    # Filter to blocks with both ranks (excludes HLA blocks which have NaN noHLA rank)
    plot_df = df[df[full_rank_col].notna() & df[noHLA_rank_col].notna()].copy()
    plot_df[full_rank_col]  = plot_df[full_rank_col].astype(int)
    plot_df[noHLA_rank_col] = plot_df[noHLA_rank_col].astype(int)

    n_hla_excluded = df["is_HLA"].sum() if "is_HLA" in df.columns else 0
    n_total = len(plot_df)

    colors = [_assign_color(row) for _, row in plot_df.iterrows()]
    sizes  = [_assign_size(row)  for _, row in plot_df.iterrows()]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Background scatter (non-highlighted blocks)
    is_other = ~(plot_df["is_PDE4D"] | plot_df.get("is_IL1RL1", False) |
                 plot_df.get("is_SH2B3", False) | plot_df.get("is_17q21", False))
    ax.scatter(
        plot_df.loc[is_other, full_rank_col],
        plot_df.loc[is_other, noHLA_rank_col],
        s=12, color="steelblue", alpha=0.45, linewidths=0, zorder=2,
    )

    # Highlighted categories (plotted on top)
    for mask, color, label, zorder in [
        (plot_df.get("is_17q21",  pd.Series(False, index=plot_df.index)), "forestgreen", "17q21",  3),
        (plot_df.get("is_IL1RL1", pd.Series(False, index=plot_df.index)), "darkorange",  "IL1RL1", 4),
        (plot_df.get("is_SH2B3",  pd.Series(False, index=plot_df.index)), "mediumpurple","SH2B3",  4),
        (plot_df["is_PDE4D"],                                              "tomato",      "PDE4D",  5),
    ]:
        sub = plot_df[mask]
        if sub.empty:
            continue
        ax.scatter(
            sub[full_rank_col], sub[noHLA_rank_col],
            s=28, color=color, alpha=0.85, linewidths=0.4,
            edgecolors="white", zorder=zorder, label=label,
        )

    # Diagonal reference line (y = x, no rank change)
    max_rank = max(plot_df[full_rank_col].max(), plot_df[noHLA_rank_col].max())
    diag = np.arange(1, max_rank + 1)
    ax.plot(diag, diag, color="black", linewidth=0.8, linestyle="--",
            alpha=0.5, zorder=1, label="no change (y = x)")

    # Invert y-axis so rank 1 is at the top
    ax.invert_yaxis()

    # ── Annotate top-N PDE4D blocks (best no-HLA rank) ────────────────────────
    pde4d_df = plot_df[plot_df["is_PDE4D"]].nsmallest(top_n_pde4d_label, noHLA_rank_col)
    for _, row in pde4d_df.iterrows():
        lbl = _short_label(row["block_id"])
        x, y = row[full_rank_col], row[noHLA_rank_col]
        ax.annotate(
            lbl, xy=(x, y),
            xytext=(x + max_rank * 0.04, y - max_rank * 0.025),
            fontsize=6.5, color="tomato",
            arrowprops=dict(arrowstyle="-", color="tomato", lw=0.6),
        )

    # ── Annotate top-N largest rank improvements (any category) ───────────────
    # rank improvement = full_rank - noHLA_rank = -rank_change
    improvement = plot_df[full_rank_col] - plot_df[noHLA_rank_col]
    top_improvers = plot_df[improvement == improvement.nlargest(top_n_improvers_label).iloc[-1]].index
    # Use nlargest directly on the series
    top_improver_idx = improvement.nlargest(top_n_improvers_label).index
    # skip any already labelled by PDE4D annotation
    labelled_ids = set(pde4d_df["block_id"])
    for idx in top_improver_idx:
        row = plot_df.loc[idx]
        if row["block_id"] in labelled_ids:
            continue
        lbl = _short_label(row["block_id"])
        x, y = row[full_rank_col], row[noHLA_rank_col]
        ax.annotate(
            lbl, xy=(x, y),
            xytext=(x - max_rank * 0.10, y + max_rank * 0.025),
            fontsize=6.5, color="dimgray",
            arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.6),
        )

    # ── Axes and labels ───────────────────────────────────────────────────────
    ax.set_xlabel(f"Full model rank ({pc_label})", fontsize=11)
    ax.set_ylabel(f"No-HLA model rank ({pc_label})", fontsize=11)
    ax.set_title(
        f"Rank shift after HLA masking\n(Phase 2 {pc_label} attribution)",
        fontsize=12, fontweight="bold",
    )

    # Axis limits: 0.5 padding so rank-1 dot isn't clipped
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.set_ylim(max_rank + 0.5, 0.5)   # y inverted

    # ── Legend ────────────────────────────────────────────────────────────────
    # Add a dummy entry to note that HLA blocks are excluded
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    hla_patch = mpatches.Patch(
        color="lightgray",
        label=f"HLA blocks (n={n_hla_excluded}, no no-HLA rank)",
    )
    other_patch = mpatches.Patch(color="steelblue", alpha=0.6, label=f"other ({n_total} blocks shown)")
    handles = existing_handles + [other_patch, hla_patch]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.85)

    # ── Annotation: above-diagonal = improved ─────────────────────────────────
    ax.text(
        0.03, 0.03, "↑ improved after HLA removal",
        transform=ax.transAxes, fontsize=7.5, color="dimgray",
        va="bottom", style="italic",
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot]  saved {out_path}")


# ── summary table ─────────────────────────────────────────────────────────────
def print_top_improvers(df: pd.DataFrame, full_col: str, noHLA_col: str,
                        change_col: str, pc_label: str, top_n: int = 10) -> None:
    valid = df[df[full_col].notna() & df[noHLA_col].notna()].copy()
    valid["rank_improvement"] = valid[full_col] - valid[noHLA_col]
    top = valid.nlargest(top_n, "rank_improvement")[
        ["block_id", full_col, noHLA_col, "rank_improvement",
         "is_PDE4D", "is_HLA", "is_IL1RL1"]
    ].copy()
    top["is_PDE4D"]  = top["is_PDE4D"].map({True: "YES", False: "-"})
    top["is_IL1RL1"] = top["is_IL1RL1"].map({True: "YES", False: "-"})

    print(f"\n══ Top {top_n} rank improvements — {pc_label} "
          f"(full rank − noHLA rank, positive = rose) ══")
    print(top.to_string(index=False))

    pde4d = valid[valid["is_PDE4D"]]
    best_full  = pde4d[full_col].min()
    best_noHLA = pde4d[noHLA_col].min()
    print(f"\n  PDE4D best rank — full model: {int(best_full)}  |  "
          f"no-HLA model: {int(best_noHLA)}")
    if best_noHLA < best_full:
        print(f"  → PDE4D shifts toward rank 1 after HLA removal  "
              f"({int(best_full)} → {int(best_noHLA)})")
    else:
        print(f"  → PDE4D does not clearly improve after HLA removal")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_comparison()
    if "is_SH2B3" not in df.columns:
        df["is_SH2B3"] = df["block_id"].str.contains("SH2B3", na=False)

    n_overlapping_PC1 = df["full_PC1_rank"].notna() & df["noHLA_PC1_rank"].notna()
    n_overlapping_PC2 = df["full_PC2_rank"].notna() & df["noHLA_PC2_rank"].notna()
    print(f"[data]  loaded {len(df)} total blocks")
    print(f"        overlapping (both full + noHLA ranks): "
          f"PC1={n_overlapping_PC1.sum()}  PC2={n_overlapping_PC2.sum()}")
    print(f"        HLA blocks (excluded from scatter):   {df['is_HLA'].sum()}")

    # ── PC1 plot ─────────────────────────────────────────────────────────────
    make_rank_shift_plot(
        df=df,
        full_rank_col="full_PC1_rank",
        noHLA_rank_col="noHLA_PC1_rank",
        rank_change_col="rank_change_PC1",
        pc_label="PC1",
        out_path=out_dir / "phase2_PC1_rank_shift_scatter.png",
    )
    print_top_improvers(df, "full_PC1_rank", "noHLA_PC1_rank",
                        "rank_change_PC1", "PC1")

    # ── PC2 plot ─────────────────────────────────────────────────────────────
    make_rank_shift_plot(
        df=df,
        full_rank_col="full_PC2_rank",
        noHLA_rank_col="noHLA_PC2_rank",
        rank_change_col="rank_change_PC2",
        pc_label="PC2",
        out_path=out_dir / "phase2_PC2_rank_shift_scatter.png",
    )
    print_top_improvers(df, "full_PC2_rank", "noHLA_PC2_rank",
                        "rank_change_PC2", "PC2")

    print(f"\n[done]  outputs in {out_dir}/")


if __name__ == "__main__":
    main()
