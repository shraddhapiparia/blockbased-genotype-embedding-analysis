#!/usr/bin/env python3
"""plots_updated.py — Phase 1 summary plots.

Workflow: Core supporting — run after Phase 1; called via CLI directly or by 04_plotting.py (archived).

Phase-1 plotting script (UPDATED with the 6 additional visuals).

Reads:
  <phase1_dir>/vae_summary_ord.csv
  <phase1_dir>/<loss>/logs/<block>.csv   (for KL-over-epochs plot)

Saves PNGs to:
  /Users/shraddh_mac/geno_ld_attention/geno_ld_attention/results/plots

Default: includes ALL losses found in the CSV (including BCE).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_OUTDIR = "/Users/shraddh_mac/geno_ld_attention/geno_ld_attention/results/plots_regions"


# ----------------------------
# Helpers
# ----------------------------
def choose_losses(all_losses: list[str], losses_arg: list[str] | None, prefer: list[str] | None, k: int | None) -> list[str]:
    present = [x for x in all_losses if isinstance(x, str)]

    if losses_arg and len(losses_arg) > 0:
        chosen = [x for x in losses_arg if x in present]
        if not chosen:
            raise ValueError(f"--losses specified {losses_arg} but none were found. Present: {present}")
        return chosen

    prefer = prefer or []
    prefer_present = [x for x in prefer if x in present]
    remainder = [x for x in present if x not in prefer_present]
    ordered = prefer_present + remainder

    if k is not None and k > 0:
        return ordered[:k]
    return ordered


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def cmap_idx(losses: list[str]) -> dict[str, int]:
    return {loss: i for i, loss in enumerate(losses)}


def require_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {where}: {missing}\nFound columns: {list(df.columns)}")


# ----------------------------
# Base plots (from earlier)
# ----------------------------
def plot_scatter_bal_vs_ld(df: pd.DataFrame, losses: list[str], outdir: Path) -> None:
    cm = plt.get_cmap("tab10")
    ci = cmap_idx(losses)

    plt.figure(figsize=(7.5, 5.5))
    for loss in losses:
        d = df[df["loss"] == loss]
        plt.scatter(d["ld_corr_va"], d["bal_acc_va"], label=loss, alpha=0.8, s=35, c=[cm(ci[loss] % 10)])

    plt.xlabel("ld_corr_va (LD structure preservation)")
    plt.ylabel("bal_acc_va (mean per-class recall)")
    plt.title("Validation tradeoff: Balanced accuracy vs LD-correlation")
    plt.legend(frameon=False, ncol=min(3, len(losses)))
    savefig(outdir / "scatter_bal_acc_vs_ld_corr.png")


def plot_winner_bar(df: pd.DataFrame, losses: list[str], outdir: Path) -> None:
    d = df[df["loss"].isin(losses)].copy()

    sort_cols = ["bal_acc_va", "ld_corr_va"]
    ascending = [False, False]
    if "conc_va" in d.columns:
        sort_cols.append("conc_va"); ascending.append(False)
    if "va_loss" in d.columns:
        sort_cols.append("va_loss"); ascending.append(True)

    d = d.sort_values(sort_cols, ascending=ascending)
    winners = d.groupby("block", as_index=False).head(1).copy()
    winners = winners.sort_values("bal_acc_va", ascending=False)

    cm = plt.get_cmap("tab10")
    ci = cmap_idx(losses)
    colors = [cm(ci[x] % 10) for x in winners["loss"].tolist()]

    plt.figure(figsize=(max(10, 0.45 * len(winners)), 5.5))
    plt.bar(winners["block"], winners["bal_acc_va"], color=colors)
    plt.xticks(rotation=75, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Best bal_acc_va")
    plt.title("Per-block winner (highest validation balanced accuracy)")

    handles, labels = [], []
    for loss in losses:
        handles.append(plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, color=cm(ci[loss] % 10)))
        labels.append(loss)
    plt.legend(handles, labels, frameon=False, ncol=min(5, len(losses)))
    savefig(outdir / "winner_bar_by_block.png")


def plot_box_by_loss(df: pd.DataFrame, losses: list[str], value_col: str, outpath: Path, title: str, ylabel: str) -> None:
    d = df[df["loss"].isin(losses)].copy()
    groups = [d.loc[d["loss"] == loss, value_col].dropna().astype(float).values for loss in losses]

    plt.figure(figsize=(max(7.5, 0.9 * len(losses)), 5.5))
    plt.boxplot(groups, labels=losses, showfliers=True)
    plt.ylabel(ylabel)
    plt.title(title)
    savefig(outpath)


def plot_runtime_vs_perf(df: pd.DataFrame, losses: list[str], outdir: Path) -> None:
    cm = plt.get_cmap("tab10")
    ci = cmap_idx(losses)

    plt.figure(figsize=(7.5, 5.5))
    for loss in losses:
        d = df[df["loss"] == loss]
        plt.scatter(d["sec"], d["bal_acc_va"], label=loss, alpha=0.8, s=35, c=[cm(ci[loss] % 10)])

    plt.xlabel("sec (wall-clock per run)")
    plt.ylabel("bal_acc_va")
    plt.title("Runtime vs performance")
    plt.legend(frameon=False, ncol=min(3, len(losses)))
    savefig(outdir / "scatter_runtime_vs_bal_acc.png")


# ----------------------------
# NEW visuals you requested
# ----------------------------

# 1) Per-class recall heatmap faceted by loss
def plot_per_class_heatmaps(df: pd.DataFrame, losses: list[str], outdir: Path) -> None:
    require_cols(df, ["block", "loss", "acc0_va", "acc1_va", "acc2_va"], "per-class heatmap")

    for loss in losses:
        d = df[df["loss"] == loss].copy()
        if d.empty:
            continue

        # Order blocks by balanced accuracy (or conc if bal missing)
        if "bal_acc_va" in d.columns:
            d = d.sort_values("bal_acc_va", ascending=False)
        elif "conc_va" in d.columns:
            d = d.sort_values("conc_va", ascending=False)

        M = d[["acc0_va", "acc1_va", "acc2_va"]].to_numpy(dtype=float)
        blocks = d["block"].astype(str).tolist()

        plt.figure(figsize=(6.0, max(6.0, 0.25 * len(blocks))))
        im = plt.imshow(M, aspect="auto", vmin=0.0, vmax=1.0)
        plt.colorbar(im, fraction=0.03, pad=0.02, label="Recall (TPR)")

        plt.yticks(np.arange(len(blocks)), blocks, fontsize=8)
        plt.xticks([0, 1, 2], ["acc0 (recall0)", "acc1 (recall1)", "acc2 (recall2)"], rotation=30, ha="right")
        plt.title(f"Per-class recall heatmap — {loss}")

        savefig(outdir / f"heatmap_per_class_recall_{loss}.png")


# 2) MAF vs performance scatter
def plot_maf_vs_perf(df: pd.DataFrame, losses: list[str], outdir: Path) -> None:
    require_cols(df, ["loss", "maf_mean", "maf_frac_lt_10pct", "ld_corr_va", "bal_acc_va"], "MAF vs performance")
    cm = plt.get_cmap("tab10")
    ci = cmap_idx(losses)

    # maf_mean vs ld_corr
    plt.figure(figsize=(7.5, 5.5))
    for loss in losses:
        d = df[df["loss"] == loss]
        plt.scatter(d["maf_mean"], d["ld_corr_va"], label=loss, alpha=0.8, s=35, c=[cm(ci[loss] % 10)])
    plt.xlabel("maf_mean")
    plt.ylabel("ld_corr_va")
    plt.title("MAF difficulty vs LD preservation (maf_mean)")
    plt.legend(frameon=False, ncol=min(3, len(losses)))
    savefig(outdir / "scatter_maf_mean_vs_ld_corr.png")

    # maf_frac_lt_10pct vs bal_acc
    plt.figure(figsize=(7.5, 5.5))
    for loss in losses:
        d = df[df["loss"] == loss]
        plt.scatter(d["maf_frac_lt_10pct"], d["bal_acc_va"], label=loss, alpha=0.8, s=35, c=[cm(ci[loss] % 10)])
    plt.xlabel("maf_frac_lt_10pct")
    plt.ylabel("bal_acc_va")
    plt.title("Rare-variant burden vs balanced accuracy (maf_frac_lt_10pct)")
    plt.legend(frameon=False, ncol=min(3, len(losses)))
    savefig(outdir / "scatter_maf_frac_lt_10pct_vs_bal_acc.png")


# 3) Concordance lift over baseline
def plot_conc_lift(df: pd.DataFrame, losses: list[str], outdir: Path) -> None:
    require_cols(df, ["loss", "block", "conc_va", "base_conc_va"], "concordance lift")
    d = df[df["loss"].isin(losses)].copy()
    d["conc_lift_va"] = d["conc_va"].astype(float) - d["base_conc_va"].astype(float)

    # grouped bar (can get wide; dynamic sizing)
    blocks = sorted(d["block"].unique().tolist())
    blocks = [str(b) for b in blocks]
    x = np.arange(len(blocks))

    width = 0.8 / max(1, len(losses))
    cm = plt.get_cmap("tab10")
    ci = cmap_idx(losses)

    plt.figure(figsize=(max(12.0, 0.55 * len(blocks)), 6.0))
    for j, loss in enumerate(losses):
        dd = d[d["loss"] == loss].set_index("block").reindex(blocks)
        y = dd["conc_lift_va"].to_numpy(dtype=float)
        plt.bar(x + (j - (len(losses) - 1) / 2) * width, y, width=width, label=loss, color=cm(ci[loss] % 10))

    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, blocks, rotation=75, ha="right")
    plt.ylabel("conc_va - base_conc_va")
    plt.title("Concordance lift over baseline (validation)")
    plt.legend(frameon=False, ncol=min(5, len(losses)))
    savefig(outdir / "bar_concordance_lift_by_block.png")

    # also add a compact boxplot of lift by loss (easier to read)
    plt.figure(figsize=(max(7.5, 0.9 * len(losses)), 5.5))
    groups = [d.loc[d["loss"] == loss, "conc_lift_va"].dropna().to_numpy(dtype=float) for loss in losses]
    plt.boxplot(groups, labels=losses, showfliers=True)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("conc_va - base_conc_va")
    plt.title("Concordance lift distribution by loss (validation)")
    savefig(outdir / "box_concordance_lift_by_loss.png")


# 4) Train vs val concordance scatter (overfitting)
def plot_conc_train_vs_val(df: pd.DataFrame, losses: list[str], outdir: Path) -> None:
    require_cols(df, ["loss", "conc_tr", "conc_va"], "train vs val concordance")
    cm = plt.get_cmap("tab10")
    ci = cmap_idx(losses)

    plt.figure(figsize=(7.5, 5.5))
    for loss in losses:
        d = df[df["loss"] == loss]
        plt.scatter(d["conc_tr"], d["conc_va"], label=loss, alpha=0.8, s=35, c=[cm(ci[loss] % 10)])

    # diagonal
    lo = float(np.nanmin([df["conc_tr"].min(), df["conc_va"].min()]))
    hi = float(np.nanmax([df["conc_tr"].max(), df["conc_va"].max()]))
    plt.plot([lo, hi], [lo, hi], linewidth=1)

    plt.xlabel("conc_tr")
    plt.ylabel("conc_va")
    plt.title("Train vs validation concordance (overfitting check)")
    plt.legend(frameon=False, ncol=min(3, len(losses)))
    savefig(outdir / "scatter_conc_train_vs_val.png")


# 5) LD correlation by n_snps (binned)
def plot_ld_corr_by_nsnps(df: pd.DataFrame, losses: list[str], outdir: Path) -> None:
    require_cols(df, ["loss", "n_snps", "ld_corr_va"], "LD-corr by n_snps")
    d = df[df["loss"].isin(losses)].copy()

    # bins as you suggested
    bins = [0, 25, 100, 500, np.inf]
    labels = ["<25", "25–100", "100–500", "500+"]

    d["n_bin"] = pd.cut(d["n_snps"].astype(float), bins=bins, labels=labels, right=False)

    # For each bin: boxplot of ld_corr grouped by loss (multiple boxes per bin is messy),
    # so we do one plot per loss: ld_corr across bins.
    for loss in losses:
        dd = d[d["loss"] == loss].copy()
        groups = [dd.loc[dd["n_bin"] == lab, "ld_corr_va"].dropna().astype(float).values for lab in labels]
        plt.figure(figsize=(7.5, 5.5))
        plt.boxplot(groups, labels=labels, showfliers=True)
        plt.ylim(-1.0, 1.0)
        plt.ylabel("ld_corr_va")
        plt.title(f"LD-correlation by block size bin — {loss}")
        savefig(outdir / f"box_ld_corr_by_nsnps_{loss}.png")


# 6) KL divergence over training (representative blocks)
def pick_representative_blocks(df: pd.DataFrame, n: int = 3) -> list[str]:
    """
    Choose representative blocks by size: small/medium/large using quantiles of n_snps.
    We pick blocks from the full df (unique blocks).
    """
    d = df.drop_duplicates(subset=["block"]).copy()
    require_cols(d, ["block", "n_snps"], "representative block selection")

    # quantiles to pick
    qs = [0.1, 0.5, 0.9] if n == 3 else np.linspace(0.1, 0.9, n).tolist()
    targets = [float(d["n_snps"].quantile(q)) for q in qs]

    chosen = []
    for t in targets:
        # nearest block by abs difference in n_snps
        dd = d.assign(diff=(d["n_snps"].astype(float) - t).abs()).sort_values("diff")
        for b in dd["block"].tolist():
            if b not in chosen:
                chosen.append(b)
                break
    return chosen


def load_log_kl(log_csv: Path) -> pd.Series | None:
    """
    Expect your per-epoch CSV to have column 'va_kl' (as in fin['va_kl']).
    If missing or unreadable, return None.
    """
    try:
        df = pd.read_csv(log_csv)
    except Exception:
        return None
    if "va_kl" not in df.columns:
        return None
    return df["va_kl"].astype(float)


def plot_kl_over_epochs(phase1_dir: Path, df_summary: pd.DataFrame, losses: list[str], outdir: Path) -> None:
    require_cols(df_summary, ["block", "n_snps"], "KL-over-epochs")
    reps = pick_representative_blocks(df_summary, n=3)

    # one figure per loss, with up to 3 lines (small/med/large)
    for loss in losses:
        plt.figure(figsize=(7.5, 5.5))
        any_line = False

        for bid in reps:
            log_csv = phase1_dir / loss / "logs" / f"{bid}.csv"
            if not log_csv.exists():
                continue
            kl = load_log_kl(log_csv)
            if kl is None:
                continue
            plt.plot(np.arange(len(kl)), kl.values, label=bid)
            any_line = True

        if not any_line:
            plt.close()
            continue

        plt.xlabel("epoch")
        plt.ylabel("va_kl")
        plt.title(f"Validation KL over epochs — {loss} (rep blocks: {', '.join(reps)})")
        plt.legend(frameon=False, fontsize=8)
        savefig(outdir / f"line_va_kl_over_epochs_{loss}.png")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1_dir", type=str, default="/Users/shraddh_mac/geno_ld_attention/geno_ld_attention/results/output_regions",
                    help="Path to Phase-1 output directory (contains vae_summary_ord.csv and per-loss logs/).")
    ap.add_argument("--summary_csv", type=str, default="vae_summary_ord.csv",
                    help="Summary CSV filename inside phase1_dir.")
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR,
                    help="Directory to save plot PNGs.")
    ap.add_argument("--losses", type=str, nargs="*", default=None,
                    help="Optional explicit list of losses to plot (e.g. --losses ORD MSE MSE_STD CAT BCE).")
    ap.add_argument("--k", type=int, default=None,
                    help="Optional: number of loss types to plot (uses prefer-first ordering). If omitted, uses all.")
    ap.add_argument("--prefer", type=str, nargs="*", default=["ORD", "MSE", "MSE_STD", "CAT", "BCE"],
                    help="Preferred ordering used when selecting all or when --k is set.")
    args = ap.parse_args()

    phase1_dir = Path(args.phase1_dir)
    outdir = Path(args.outdir)
    summary_csv = phase1_dir / args.summary_csv
    if not summary_csv.exists():
        raise FileNotFoundError(f"Could not find summary CSV at: {summary_csv}")

    df = pd.read_csv(summary_csv)

    # base required cols for the core plots
    require_cols(df, ["loss", "block", "bal_acc_va", "ld_corr_va", "sec"], "core plots")

    all_losses = df["loss"].dropna().astype(str).unique().tolist()
    losses = choose_losses(all_losses, args.losses, args.prefer, args.k)
    dsel = df[df["loss"].isin(losses)].copy()

    # ----- Core plots -----
    plot_scatter_bal_vs_ld(dsel, losses, outdir)
    plot_winner_bar(dsel, losses, outdir)
    plot_box_by_loss(dsel, losses, "bal_acc_va", outdir / "box_bal_acc_by_loss.png",
                     "Validation balanced accuracy distribution by loss", "bal_acc_va")
    plot_box_by_loss(dsel, losses, "ld_corr_va", outdir / "box_ld_corr_by_loss.png",
                     "Validation LD-correlation distribution by loss", "ld_corr_va")
    plot_runtime_vs_perf(dsel, losses, outdir)

    # ----- New requested visuals -----
    # 1) per-class recall heatmap per loss
    plot_per_class_heatmaps(dsel, losses, outdir)

    # 2) MAF vs performance scatter
    plot_maf_vs_perf(dsel, losses, outdir)

    # 3) concordance lift (needs conc_va + base_conc_va)
    if "conc_va" in dsel.columns and "base_conc_va" in dsel.columns:
        plot_conc_lift(dsel, losses, outdir)
    else:
        print("[skip] concordance lift: conc_va/base_conc_va not found in summary CSV")

    # 4) train vs val concordance
    if "conc_tr" in dsel.columns and "conc_va" in dsel.columns:
        plot_conc_train_vs_val(dsel, losses, outdir)
    else:
        print("[skip] train vs val concordance: conc_tr/conc_va not found in summary CSV")

    # 5) ld_corr by n_snps bins
    if "n_snps" in dsel.columns:
        plot_ld_corr_by_nsnps(dsel, losses, outdir)
    else:
        print("[skip] ld_corr by n_snps: n_snps not found in summary CSV")

    # 6) KL over epochs (reads per-epoch logs)
    if "va_kl" in dsel.columns or True:
        plot_kl_over_epochs(phase1_dir, dsel, losses, outdir)

    # bookkeeping
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "plots_readme.txt", "w") as f:
        f.write(f"phase1_dir: {phase1_dir}\n")
        f.write(f"summary_csv: {summary_csv}\n")
        f.write(f"losses_plotted: {losses}\n")
        f.write("plots (core):\n")
        f.write("  - scatter_bal_acc_vs_ld_corr.png\n")
        f.write("  - winner_bar_by_block.png\n")
        f.write("  - box_bal_acc_by_loss.png\n")
        f.write("  - box_ld_corr_by_loss.png\n")
        f.write("  - scatter_runtime_vs_bal_acc.png\n")
        f.write("plots (additional):\n")
        f.write("  - heatmap_per_class_recall_<LOSS>.png\n")
        f.write("  - scatter_maf_mean_vs_ld_corr.png\n")
        f.write("  - scatter_maf_frac_lt_10pct_vs_bal_acc.png\n")
        f.write("  - bar_concordance_lift_by_block.png (if conc_va/base_conc_va present)\n")
        f.write("  - box_concordance_lift_by_loss.png (if conc_va/base_conc_va present)\n")
        f.write("  - scatter_conc_train_vs_val.png (if conc_tr/conc_va present)\n")
        f.write("  - box_ld_corr_by_nsnps_<LOSS>.png (if n_snps present)\n")
        f.write("  - line_va_kl_over_epochs_<LOSS>.png (if per-epoch logs have va_kl)\n")

    print(f"[done] Saved plots to: {outdir}")
    print(f"[done] Losses plotted: {losses}")


if __name__ == "__main__":
    main()