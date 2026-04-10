#!/usr/bin/env python3
"""
Make Phase-1 summary plots from ./output_phase1 results.

Updated per your request:
- Includes BCE (no default exclusion).
- By default plots ALL loss types found in the summary CSV.
- You can still optionally pick a subset via --losses or --k/--prefer.

Assumptions (based on your run_phase1 code):
- Summary CSV is at: <phase1_dir>/vae_summary_ord.csv
- Columns include:
  loss, block, bal_acc_va, ld_corr_va, sec
  (optionally: conc_va, va_loss, n_snps, etc.)

Outputs PNGs to:
  /Users/shraddh_mac/geno_ld_attention/geno_ld_attention/results/plots
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_OUTDIR = "/Users/shraddh_mac/geno_ld_attention/geno_ld_attention/results/plots"


def choose_losses(
    all_losses: list[str],
    losses_arg: list[str] | None,
    prefer: list[str] | None,
    k: int | None,
) -> list[str]:
    """
    Selection rules:
    1) If --losses is provided => use that (in the order given), but only those present.
    2) Else if k is provided (and k>0):
         start with prefer (if provided), then fill with remaining present losses, until k
    3) Else => return all present losses, with prefer-first ordering if provided.
    """
    present = [x for x in all_losses if isinstance(x, str)]

    if losses_arg and len(losses_arg) > 0:
        chosen = [x for x in losses_arg if x in present]
        if not chosen:
            raise ValueError(f"--losses specified {losses_arg} but none were found in CSV. Present: {present}")
        return chosen

    prefer = prefer or []
    prefer_present = [x for x in prefer if x in present]
    remainder = [x for x in present if x not in prefer_present]
    ordered = prefer_present + remainder

    if k is not None and k > 0:
        return ordered[:k]

    return ordered


def plot_scatter_bal_vs_ld(df: pd.DataFrame, losses: list[str], outdir: Path) -> None:
    cm = plt.get_cmap("tab10")
    color_idx = {loss: i for i, loss in enumerate(losses)}

    plt.figure(figsize=(7.5, 5.5))
    for loss in losses:
        d = df[df["loss"] == loss]
        plt.scatter(
            d["ld_corr_va"],
            d["bal_acc_va"],
            label=loss,
            alpha=0.8,
            s=35,
            c=[cm(color_idx[loss] % 10)],
        )
    plt.xlabel("ld_corr_va (LD structure preservation)")
    plt.ylabel("bal_acc_va (mean per-class recall)")
    plt.title("Validation tradeoff: Balanced accuracy vs LD-correlation")
    plt.legend(frameon=False, ncol=min(3, len(losses)))
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / "scatter_bal_acc_vs_ld_corr.png", dpi=200)
    plt.close()


def plot_winner_bar(df: pd.DataFrame, losses: list[str], outdir: Path) -> None:
    """
    For each block: pick the loss row with max bal_acc_va.
    Tie-breakers (if present): ld_corr_va, conc_va, then lower va_loss.
    """
    d = df[df["loss"].isin(losses)].copy()

    # Build tie-break sort keys
    sort_cols = ["bal_acc_va", "ld_corr_va"]
    ascending = [False, False]

    if "conc_va" in d.columns:
        sort_cols.append("conc_va")
        ascending.append(False)

    if "va_loss" in d.columns:
        # lower is better => sort ascending on va_loss
        sort_cols.append("va_loss")
        ascending.append(True)

    d = d.sort_values(sort_cols, ascending=ascending)
    winners = d.groupby("block", as_index=False).head(1).copy()
    winners = winners.sort_values("bal_acc_va", ascending=False)

    cm = plt.get_cmap("tab10")
    color_idx = {loss: i for i, loss in enumerate(losses)}
    colors = [cm(color_idx[x] % 10) for x in winners["loss"].tolist()]

    plt.figure(figsize=(max(10, 0.45 * len(winners)), 5.5))
    plt.bar(winners["block"], winners["bal_acc_va"], color=colors)
    plt.xticks(rotation=75, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Best bal_acc_va")
    plt.title("Per-block winner (highest validation balanced accuracy)")

    # Legend
    handles = []
    labels = []
    for loss in losses:
        handles.append(plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, color=cm(color_idx[loss] % 10)))
        labels.append(loss)
    plt.legend(handles, labels, frameon=False, ncol=min(5, len(losses)))

    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / "winner_bar_by_block.png", dpi=200)
    plt.close()


def plot_box_by_loss(df: pd.DataFrame, losses: list[str], value_col: str, outpath: Path, title: str, ylabel: str) -> None:
    d = df[df["loss"].isin(losses)].copy()
    groups = [d.loc[d["loss"] == loss, value_col].dropna().astype(float).values for loss in losses]

    plt.figure(figsize=(max(7.5, 0.9 * len(losses)), 5.5))
    plt.boxplot(groups, labels=losses, showfliers=True)
    plt.ylabel(ylabel)
    plt.title(title)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_runtime_vs_perf(df: pd.DataFrame, losses: list[str], outdir: Path) -> None:
    cm = plt.get_cmap("tab10")
    color_idx = {loss: i for i, loss in enumerate(losses)}

    plt.figure(figsize=(7.5, 5.5))
    for loss in losses:
        d = df[df["loss"] == loss]
        plt.scatter(
            d["sec"],
            d["bal_acc_va"],
            label=loss,
            alpha=0.8,
            s=35,
            c=[cm(color_idx[loss] % 10)],
        )
    plt.xlabel("sec (wall-clock per run)")
    plt.ylabel("bal_acc_va")
    plt.title("Runtime vs performance")
    plt.legend(frameon=False, ncol=min(3, len(losses)))
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / "scatter_runtime_vs_bal_acc.png", dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase1_dir",
        type=str,
        default="./output_phase1",
        help="Path to Phase-1 output directory (contains vae_summary.csv).",
    )
    ap.add_argument("--summary_csv", type=str, default="vae_summary.csv",
                    help="Summary CSV filename inside phase1_dir (default: vae_summary.csv).")
    ap.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help="Directory to save plot PNGs.",
    )

    # NEW: allow either plot ALL or specify subset
    ap.add_argument(
        "--losses",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of losses to plot (e.g. --losses ORD MSE MSE_STD CAT BCE). "
             "If omitted, plots all losses found in the CSV.",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help="Optional: number of loss types to plot (uses prefer-first ordering). If omitted, uses all.",
    )
    ap.add_argument(
        "--prefer",
        type=str,
        nargs="*",
        default=["ORD", "MSE", "MSE_STD", "CAT", "BCE"],
        help="Preferred ordering (used when selecting all or when --k is set).",
    )

    args = ap.parse_args()

    phase1_dir = Path(args.phase1_dir)
    outdir = Path(args.outdir)

    summary_csv = phase1_dir / args.summary_csv
    if not summary_csv.exists():
        raise FileNotFoundError(f"Could not find summary CSV at: {summary_csv}")

    df = pd.read_csv(summary_csv)

    required = {"loss", "block", "bal_acc_va", "ld_corr_va", "sec"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Summary CSV missing required columns: {missing}\nFound columns: {list(df.columns)}")

    all_losses = df["loss"].dropna().astype(str).unique().tolist()

    # Default: ALL present losses, with prefer-first ordering
    losses = choose_losses(all_losses, args.losses, args.prefer, args.k)

    dsel = df[df["loss"].isin(losses)].copy()

    # ---- Plots ----
    plot_scatter_bal_vs_ld(dsel, losses, outdir)
    plot_winner_bar(dsel, losses, outdir)
    plot_box_by_loss(
        dsel,
        losses,
        "bal_acc_va",
        outdir / "box_bal_acc_by_loss.png",
        "Validation balanced accuracy distribution by loss",
        "bal_acc_va",
    )
    plot_box_by_loss(
        dsel,
        losses,
        "ld_corr_va",
        outdir / "box_ld_corr_by_loss.png",
        "Validation LD-correlation distribution by loss",
        "ld_corr_va",
    )
    plot_runtime_vs_perf(dsel, losses, outdir)

    # bookkeeping
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "plots_readme.txt", "w") as f:
        f.write(f"phase1_dir: {phase1_dir}\n")
        f.write(f"summary_csv: {summary_csv}\n")
        f.write(f"losses_plotted: {losses}\n")
        f.write("plots:\n")
        f.write("  - scatter_bal_acc_vs_ld_corr.png\n")
        f.write("  - winner_bar_by_block.png\n")
        f.write("  - box_bal_acc_by_loss.png\n")
        f.write("  - box_ld_corr_by_loss.png\n")
        f.write("  - scatter_runtime_vs_bal_acc.png\n")

    print(f"[done] Saved plots to: {outdir}")
    print(f"[done] Losses plotted: {losses}")


if __name__ == "__main__":
    main()