import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

LOG_DIR = Path("/Users/shraddh_mac/Documents/GitHub/blockbased-genotype-embedding-analysis/results/output_regions_ord_weighted/ORD/logs")
SUMMARY = pd.read_csv("/Users/shraddh_mac/Documents/GitHub/blockbased-genotype-embedding-analysis/results/output_regions_ord_weighted/vae_summary.csv")
OUT = Path("/Users/shraddh_mac/Documents/GitHub/blockbased-genotype-embedding-analysis/results/output_regions_ord_weighted/training_dynamics")
OUT.mkdir(exist_ok=True)

# ─── Load all logs into one long-form df ───
rows = []
for f in LOG_DIR.glob("*.csv"):
    bid = f.stem
    df = pd.read_csv(f)
    df["block"] = bid
    rows.append(df)
all_logs = pd.concat(rows, ignore_index=True)
print(f"Loaded {len(rows)} blocks, {len(all_logs)} total epoch-rows")

# ─── Priority 1: convergence speed ───
conv_rows = []
for bid, g in all_logs.groupby("block"):
    g = g.sort_values("epoch")
    vmin = g["va_recon"].min()
    vmax = g["va_recon"].iloc[0]
    threshold = vmin + 0.01 * (vmax - vmin)  # within 1% of best
    converged = g[g["va_recon"] <= threshold]
    ep_conv = int(converged["epoch"].iloc[0]) if len(converged) else int(g["epoch"].max())
    conv_rows.append({"block": bid, "epoch_converged": ep_conv,
                      "total_epochs": int(g["epoch"].max()),
                      "min_va_recon": float(vmin)})
conv_df = pd.DataFrame(conv_rows).merge(SUMMARY[["block","n_snps","maf_mean","best_epoch"]],
                                        on="block", how="left")
conv_df.to_csv(OUT/"convergence_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(10,4))
ax.hist(conv_df["epoch_converged"], bins=30)
ax.set_xlabel("Epochs to within 1% of best val_recon")
ax.set_ylabel("Number of blocks")
ax.set_title("Convergence speed distribution (ORD weighted)")
plt.tight_layout(); plt.savefig(OUT/"convergence_hist.png", dpi=150); plt.close()

# Does block size predict convergence speed?
fig, ax = plt.subplots(figsize=(10,4))
ax.scatter(conv_df["n_snps"], conv_df["epoch_converged"], alpha=0.5)
ax.set_xlabel("n_snps"); ax.set_ylabel("epochs_to_converge")
ax.set_title("Convergence speed vs block size")
plt.tight_layout(); plt.savefig(OUT/"convergence_vs_size.png", dpi=150); plt.close()

# ─── Priority 2 & 3: train/val curves + KL for representative blocks ───
REP_BLOCKS = ["region_17q21_core_sb1", "region_11q13_FCER1A",
              "region_2q12_IL1RL1_cluster_sb3", "region_5q21_PDE4D_sb55",
              "region_6p21_HLA_classII_sb1", "region_5q31_type2_cytokine_sb9",
              "region_1q31_TNFSF_cluster_sb4", "control_OCA2_sb10"]

for bid in REP_BLOCKS:
    g = all_logs[all_logs["block"] == bid].sort_values("epoch")
    if g.empty: continue
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(g["epoch"], g["tr_recon"], label="train")
    axes[0].plot(g["epoch"], g["va_recon"], label="val")
    axes[0].axvline(50, ls="--", c="gray", alpha=0.5, label="β warmup end")
    axes[0].set_title(f"{bid} — reconstruction loss")
    axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(g["epoch"], g["tr_kl"], label="train")
    axes[1].plot(g["epoch"], g["va_kl"], label="val")
    axes[1].axvline(50, ls="--", c="gray", alpha=0.5)
    axes[1].set_title(f"{bid} — KL divergence")
    axes[1].set_xlabel("epoch"); axes[1].legend()

    axes[2].plot(g["epoch"], g["tr_loss"], label="train")
    axes[2].plot(g["epoch"], g["va_loss"], label="val")
    axes[2].axvline(50, ls="--", c="gray", alpha=0.5)
    axes[2].set_title(f"{bid} — total loss")
    axes[2].set_xlabel("epoch"); axes[2].legend()

    plt.tight_layout(); plt.savefig(OUT/f"{bid}_curves.png", dpi=150); plt.close()

# ─── Priority 4: best-epoch distribution ───
fig, ax = plt.subplots(figsize=(10,4))
ax.hist(SUMMARY["best_epoch"], bins=30)
ax.axvline(50, ls="--", c="red", label="β warmup ends")
ax.set_xlabel("best_epoch"); ax.set_ylabel("number of blocks")
ax.set_title("Where best checkpoint occurred (ORD weighted)")
ax.legend(); plt.tight_layout()
plt.savefig(OUT/"best_epoch_hist.png", dpi=150); plt.close()

print(f"Wrote analyses to {OUT}/")
print(f"  Convergence: median epoch {conv_df['epoch_converged'].median():.0f}, "
      f"max {conv_df['epoch_converged'].max():.0f}")
print(f"  Best-epoch: median {SUMMARY['best_epoch'].median():.0f}, "
      f"frac before warmup: {(SUMMARY['best_epoch'] < 50).mean() * 100:.1f}%")

# ------------ Cross-block latent geometry ------------
sub = SUMMARY[SUMMARY['loss'] == 'ORD']
print(f"mu_var_median: min={sub['mu_var_median'].min():.3f}, "
      f"median={sub['mu_var_median'].median():.3f}, "
      f"max={sub['mu_var_median'].max():.3f}, "
      f"IQR={sub['mu_var_median'].quantile(0.75) - sub['mu_var_median'].quantile(0.25):.3f}")

# ------------ KL per-block consistency ------------
print(f"kl_per_dim_median: min={sub['kl_per_dim_median'].min():.3f}, "
      f"median={sub['kl_per_dim_median'].median():.3f}, "
      f"max={sub['kl_per_dim_median'].max():.3f}, "
      f"IQR={sub['kl_per_dim_median'].quantile(0.75) - sub['kl_per_dim_median'].quantile(0.25):.3f}")