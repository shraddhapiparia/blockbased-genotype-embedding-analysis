"""
compare_phase1_phase2.py
========================
For each block shared between Phase 1 (per-block .npy files) and
Phase 2 (block_contextual_repr), compute:

  1. Pairwise-distance-matrix (PDM) correlation  (Spearman on upper triangle)
  2. Phenotype association (Spearman r / eta-squared) for both phases
  3. [PLOT]  PDM r per block, sorted, colored by region group
  4. [PLOT]  Individual embedding (Phase 2) vs concatenated Phase 1 PCA
             phenotype association comparison
  5. [PLOT]  PDM r vs recon MSE scatter per block

Outputs
-------
results/phase_comparison/pdm_correlations.csv
results/phase_comparison/pheno_associations_phase1.csv
results/phase_comparison/pheno_associations_phase2.csv
results/phase_comparison/pheno_association_gain.csv
results/phase_comparison/plots/pdm_r_per_block.png
results/phase_comparison/plots/individual_emb_pheno_comparison.png
results/phase_comparison/plots/pdm_r_vs_recon_mse.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# ── paths ──────────────────────────────────────────────────────────────────────
PHASE1_EMB_DIR     = Path("results/output_regions/ORD/embeddings")
PHASE1_SUBJ_CSV    = Path("results/test_ord_mps_run1/subjects.csv")
PHASE2_BLOCK_NPY   = Path("results/output_regions2/ORD/embeddings/block_contextual_repr.npy")
PHASE2_EMB_CSV     = Path("results/output_regions2/ORD/embeddings/individual_embeddings.csv")
PHASE2_BLOCK_ORDER = Path("results/output_regions2/ORD/embeddings/per_block_recon_mse.csv")
PHENO_FILE         = Path("metadata/COS_TRIO_pheno_1165.csv")
OUT_DIR            = Path("results/phase_comparison")
PLOT_DIR           = OUT_DIR / "plots"

CONTINUOUS_PHENOS  = ["G19B", "log10eos", "pctpred_fev1_pre_BD"]
CATEGORICAL_PHENOS = ["G20D"]
ALL_PHENOS         = CONTINUOUS_PHENOS + CATEGORICAL_PHENOS

# ── helpers ────────────────────────────────────────────────────────────────────

def pdm_spearman(X: np.ndarray, Y: np.ndarray) -> float:
    if X.shape[0] < 3 or Y.shape[0] < 3:
        return np.nan
    try:
        r, _ = spearmanr(pdist(X, "euclidean"), pdist(Y, "euclidean"))
        return float(r)
    except Exception:
        return np.nan


def first_pc(X: np.ndarray) -> np.ndarray:
    X = np.squeeze(X)
    if X.ndim == 1:
        return X
    if X.shape[1] == 1:
        return X[:, 0]
    row_ok = np.all(np.isfinite(X), axis=1)
    out = np.full(X.shape[0], np.nan)
    if row_ok.sum() < 2:
        return out
    out[row_ok] = PCA(n_components=1).fit_transform(X[row_ok])[:, 0]
    return out


def spearman_with_pheno(emb: np.ndarray, pheno_vals: np.ndarray) -> float:
    mask = np.isfinite(pheno_vals)
    if mask.sum() < 10:
        return np.nan
    pc1 = first_pc(emb[mask])
    valid = np.isfinite(pc1)
    if valid.sum() < 10:
        return np.nan
    r, _ = spearmanr(pc1[valid], pheno_vals[mask][valid])
    return float(r)


def eta_squared(emb: np.ndarray, groups: np.ndarray) -> float:
    pc1 = first_pc(emb)
    mask = pd.notna(groups) & np.isfinite(pc1)
    if mask.sum() < 10:
        return np.nan
    pc1_m, g_m = pc1[mask], np.array(groups)[mask]
    grand = pc1_m.mean()
    ss_tot = np.sum((pc1_m - grand) ** 2)
    if ss_tot == 0:
        return np.nan
    ss_bet = sum(
        np.sum(g_m == g) * (pc1_m[g_m == g].mean() - grand) ** 2
        for g in np.unique(g_m)
    )
    return float(ss_bet / ss_tot)


def region_group(block_name: str) -> str:
    """Extract coarse region label for coloring (everything before _sb or end)."""
    name = block_name.replace("region_", "").replace("control_", "ctrl_")
    # strip sub-block suffix _sb\d+
    import re
    name = re.sub(r"_sb\d+$", "", name)
    return name


# ── load Phase 2 ───────────────────────────────────────────────────────────────
print("Loading Phase 2 …")
p2_all  = np.load(PHASE2_BLOCK_NPY)           # (N, B, d2)
N_p2, B_p2, d2 = p2_all.shape
print(f"  block_contextual_repr: {p2_all.shape}")

p2_iids = pd.read_csv(PHASE2_EMB_CSV, usecols=["IID"])["IID"].astype(str).values
assert len(p2_iids) == N_p2

p2_ind_emb = pd.read_csv(PHASE2_EMB_CSV).set_index("IID")  # individual embeddings
p2_ind_emb.index = p2_ind_emb.index.astype(str)

block_order_df = pd.read_csv(PHASE2_BLOCK_ORDER)
p2_block_index = {row.block_id: idx for idx, row in block_order_df.iterrows()}
print(f"  {len(p2_block_index)} blocks in Phase 2 order file")

# ── load Phase 1 ───────────────────────────────────────────────────────────────
print("Loading Phase 1 subject IDs …")
p1_subj_df = pd.read_csv(PHASE1_SUBJ_CSV)
p1_iids = (p1_subj_df["IID"] if "IID" in p1_subj_df.columns
           else p1_subj_df.iloc[:, 0]).astype(str).values
print(f"  {len(p1_iids)} subjects")

# ── load phenotypes ────────────────────────────────────────────────────────────
print("Loading phenotypes …")
pheno_df = pd.read_csv(PHENO_FILE, low_memory=False)
pheno_df["S_SUBJECTID"] = pheno_df["S_SUBJECTID"].astype(str)
pheno_lookup = pheno_df.set_index("S_SUBJECTID")[ALL_PHENOS]

# ── common subjects ────────────────────────────────────────────────────────────
common_iids    = np.array([iid for iid in p2_iids if iid in set(p1_iids)])
print(f"  Common subjects: {len(common_iids)}")
p1_pos         = {iid: i for i, iid in enumerate(p1_iids)}
p2_pos         = {iid: i for i, iid in enumerate(p2_iids)}
common_p1_rows = np.array([p1_pos[iid] for iid in common_iids])
common_p2_rows = np.array([p2_pos[iid] for iid in common_iids])
pheno_common   = pheno_lookup.reindex(common_iids).reset_index(drop=True)
print(f"  Non-null G19B: {pheno_common['G19B'].notna().sum()}")

# ── discover Phase 1 block files ──────────────────────────────────────────────
p1_files = sorted([
    f for f in PHASE1_EMB_DIR.glob("*.npy")
    if not f.stem.startswith("all_blocks")
    and f.name not in {"individual_embeddings.npy", "block_contextual_repr.npy"}
])
print(f"Found {len(p1_files)} Phase 1 block files\n")

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── main loop ──────────────────────────────────────────────────────────────────
pdm_rows    = []
pheno1_rows = []
pheno2_rows = []
p1_concat_parts = []   # for analysis 4: collect Phase1 block embeddings

for p1_path in p1_files:
    block_name = p1_path.stem

    if block_name not in p2_block_index:
        print(f"  [skip] {block_name}: not in Phase 2")
        continue
    block_idx = p2_block_index[block_name]

    raw = np.squeeze(np.load(p1_path)).astype(float)
    if raw.ndim == 1:
        raw = raw[:, None]
    assert raw.ndim == 2, f"{p1_path.name}: shape {raw.shape}"

    p1_sub = raw[common_p1_rows]
    p2_sub = p2_all[common_p2_rows, block_idx, :]

    p1_concat_parts.append(p1_sub)

    r_pdm = pdm_spearman(p1_sub, p2_sub)
    pdm_rows.append({"block": block_name, "block_idx": block_idx,
                     "pdm_spearman_r": r_pdm,
                     "region_group": region_group(block_name)})

    row1 = {"block": block_name, "block_idx": block_idx}
    row2 = {"block": block_name, "block_idx": block_idx}
    for col in CONTINUOUS_PHENOS:
        pv = pheno_common[col].values.astype(float)
        row1[col] = spearman_with_pheno(p1_sub, pv)
        row2[col] = spearman_with_pheno(p2_sub, pv)
    for col in CATEGORICAL_PHENOS:
        gv = pheno_common[col].values
        row1[col] = eta_squared(p1_sub, gv)
        row2[col] = eta_squared(p2_sub, gv)

    pheno1_rows.append(row1)
    pheno2_rows.append(row2)

    if len(pdm_rows) % 20 == 0:
        print(f"  processed {len(pdm_rows)}/{len(p1_files)} blocks …")

# ── assemble DataFrames ────────────────────────────────────────────────────────
pdm_df    = pd.DataFrame(pdm_rows).sort_values("pdm_spearman_r")
pheno1_df = pd.DataFrame(pheno1_rows)
pheno2_df = pd.DataFrame(pheno2_rows)

gain_df = pheno2_df[["block", "block_idx"]].copy()
for col in ALL_PHENOS:
    gain_df[col + "_gain"] = pheno2_df[col].values - pheno1_df[col].values

# merge recon MSE onto pdm_df for analysis 5
pdm_df = pdm_df.merge(block_order_df.rename(columns={"block_id": "block"}),
                      on="block", how="left")

pdm_df.to_csv(OUT_DIR    / "pdm_correlations.csv",          index=False)
pheno1_df.to_csv(OUT_DIR / "pheno_associations_phase1.csv", index=False)
pheno2_df.to_csv(OUT_DIR / "pheno_associations_phase2.csv", index=False)
gain_df.to_csv(OUT_DIR   / "pheno_association_gain.csv",    index=False)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — PDM r per block, sorted, colored by region group
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlot 1: PDM r per block …")

groups       = pdm_df["region_group"].values
unique_groups = sorted(set(groups))
cmap         = cm.get_cmap("tab20", len(unique_groups))
group_color  = {g: cmap(i) for i, g in enumerate(unique_groups)}
colors       = [group_color[g] for g in groups]

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(pdm_df)), pdm_df["pdm_spearman_r"].values, color=colors, width=1.0)
ax.axhline(pdm_df["pdm_spearman_r"].mean(), color="black", lw=1.5,
           linestyle="--", label=f"Mean = {pdm_df['pdm_spearman_r'].mean():.3f}")
ax.set_xlabel("Block (sorted by PDM Spearman r)", fontsize=11)
ax.set_ylabel("PDM Spearman r", fontsize=11)
ax.set_title("Phase 1 → Phase 2 geometry preservation per block\n"
             "(Spearman r of pairwise subject distances)", fontsize=12)
ax.set_xlim(-0.5, len(pdm_df) - 0.5)
ax.set_ylim(0, 1)

# legend: one entry per region group
from matplotlib.patches import Patch
legend_handles = [Patch(color=group_color[g], label=g) for g in unique_groups]
ax.legend(handles=legend_handles, fontsize=6.5, ncol=3,
          loc="upper left", framealpha=0.7)
plt.tight_layout()
fig.savefig(PLOT_DIR / "pdm_r_per_block.png", dpi=150)
plt.close(fig)
print(f"  Saved pdm_r_per_block.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Individual embedding (Phase 2) vs concatenated Phase 1 PCA
#          phenotype association comparison (continuous phenotypes only)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 2: individual embedding pheno comparison …")

# Phase 2 individual embedding aligned to common subjects
p2_ind_common = p2_ind_emb.reindex(common_iids).values.astype(float)  # (N, 64)

# Phase 1 concatenated then PCA→64 dims to match
p1_concat = np.hstack(p1_concat_parts)   # (N, sum_of_d_blocks)
print(f"  Phase1 concat shape: {p1_concat.shape}  → PCA to 64")
n_components = min(64, p1_concat.shape[1], p1_concat.shape[0] - 1)
p1_pca = PCA(n_components=n_components).fit_transform(p1_concat)  # (N, 64)

results = []
for col in CONTINUOUS_PHENOS:
    pv = pheno_common[col].values.astype(float)
    mask = np.isfinite(pv)
    if mask.sum() < 20:
        continue
    r_p1, _ = spearmanr(first_pc(p1_pca[mask]), pv[mask])
    r_p2, _ = spearmanr(first_pc(p2_ind_common[mask]), pv[mask])
    results.append({"phenotype": col, "Phase1_concat_PCA": abs(r_p1),
                    "Phase2_individual_emb": abs(r_p2)})
    print(f"  {col}: Phase1 |r|={abs(r_p1):.3f}  Phase2 |r|={abs(r_p2):.3f}")

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / "individual_emb_pheno_comparison.csv", index=False)

x = np.arange(len(res_df))
w = 0.35
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - w/2, res_df["Phase1_concat_PCA"],       w, label="Phase 1 (concat→PCA64)", color="#4C72B0")
ax.bar(x + w/2, res_df["Phase2_individual_emb"],   w, label="Phase 2 (individual emb)", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(res_df["phenotype"], fontsize=10)
ax.set_ylabel("|Spearman r| with PC1", fontsize=11)
ax.set_title("Phenotype association: Phase 1 concat PCA vs Phase 2 individual embedding",
             fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, max(res_df[["Phase1_concat_PCA","Phase2_individual_emb"]].max()) * 1.25)
plt.tight_layout()
fig.savefig(PLOT_DIR / "individual_emb_pheno_comparison.png", dpi=150)
plt.close(fig)
print(f"  Saved individual_emb_pheno_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — PDM r vs recon MSE scatter
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 3: PDM r vs recon MSE …")

plot3_df = pdm_df.dropna(subset=["pdm_spearman_r", "recon_mse"])

r_corr, p_corr = spearmanr(plot3_df["pdm_spearman_r"], plot3_df["recon_mse"])
print(f"  Spearman r(PDM_r, recon_mse) = {r_corr:.3f}  p = {p_corr:.3g}")

groups3       = plot3_df["region_group"].values
colors3       = [group_color.get(g, "grey") for g in groups3]

fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(plot3_df["recon_mse"], plot3_df["pdm_spearman_r"],
                c=colors3, s=40, alpha=0.8, edgecolors="none")
ax.set_xlabel("Reconstruction MSE (Phase 2)", fontsize=11)
ax.set_ylabel("PDM Spearman r (Phase1→Phase2)", fontsize=11)
ax.set_title(f"Geometry change vs reconstruction difficulty\n"
             f"Spearman r = {r_corr:.3f}  (p = {p_corr:.2g})", fontsize=11)

# annotate top-5 most reshaped (lowest PDM r)
top5 = plot3_df.nsmallest(5, "pdm_spearman_r")
for _, row in top5.iterrows():
    ax.annotate(row["block"], (row["recon_mse"], row["pdm_spearman_r"]),
                fontsize=6, xytext=(4, 2), textcoords="offset points")

legend_handles3 = [Patch(color=group_color[g], label=g)
                   for g in unique_groups if g in set(groups3)]
ax.legend(handles=legend_handles3, fontsize=6.5, ncol=2,
          loc="upper right", framealpha=0.7)
plt.tight_layout()
fig.savefig(PLOT_DIR / "pdm_r_vs_recon_mse.png", dpi=150)
plt.close(fig)
print(f"  Saved pdm_r_vs_recon_mse.png")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────────")
print(pdm_df[["block","pdm_spearman_r","recon_mse"]].describe())
print(f"\nMean PDM Spearman r : {pdm_df['pdm_spearman_r'].mean():.4f}")
print(f"PDM r vs recon MSE  : Spearman r = {r_corr:.3f}  p = {p_corr:.3g}")
print("\nMost reshaped blocks (lowest PDM r):")
print(pdm_df[["block","pdm_spearman_r","recon_mse"]].head(10).to_string(index=False))
print("\nMost preserved blocks (highest PDM r):")
print(pdm_df[["block","pdm_spearman_r","recon_mse"]].tail(10).to_string(index=False))
print("\nMean phenotype association gain:")
for col in ALL_PHENOS:
    print(f"  {col}: {gain_df[col+'_gain'].mean():.4f}")
print("\nDone. Outputs in", OUT_DIR)