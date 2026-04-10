#!/usr/bin/env python3
"""
07_block_embedding_phenotype_analysis.py

Tests phenotype and ancestry association of block-level embeddings
(block_contextual_repr.npy) rather than pooling-attention weights.

Per-block feature
-----------------
For each of the 174 blocks we extract the (N, 64) contextual representation
matrix, run a 1-component PCA, and use that single value (block_PC1) as the
block-level summary for every subject.  This captures more signal than a
scalar attention weight.

Ancestry analysis
-----------------
- Pearson r of block_PC1 with genotype PC1 & PC2 → heatmap + top-block table
- UMAP of subject embeddings coloured by genotype PC1 / PC2

Phenotype analysis
------------------
Continuous : BMI, age, G19B, log10eos, log10Ige, pctpred_fev1_pre_BD
Categorical: Hospitalized_Asthma_Last_Yr, smkexp_current  (1=no → 0, 2=yes → 1)
             G20D (binarised: 0 = none, >0 = any)

For each (block × phenotype):
  model_unadj : pheno ~ block_PC1
  model_adj   : pheno ~ block_PC1 + gPC1 + … + gPC10

Outputs  (results/output_regions2/ORD/block_pheno_analysis/)
---------
block_ancestry_correlations.tsv
top_ancestry_associated_blocks.tsv
phenotype_block_associations.tsv
Plots:
  pca_subjects.png
  umap_subjects_by_geno_PC1.png
  umap_subjects_by_geno_PC2.png
  block_ancestry_heatmap.png          (focus blocks highlighted)
  top_ancestry_blocks_bar.png
  pheno_volcano_unadj.png             (per phenotype)
  pheno_volcano_adj.png
  focus_block_pheno_heatmap_unadj.png
  focus_block_pheno_heatmap_adj.png
  scatter_<block>_<pheno>.png         (top hits for focus blocks)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── paths ──────────────────────────────────────────────────────────────────────
EMB_SUBJ_NPY  = "results/output_regions2/ORD/embeddings/individual_embeddings.npy"
EMB_BLOCK_NPY = "results/output_regions2/ORD/embeddings/block_contextual_repr.npy"
ATTN_CSV      = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
BLOCK_ORDER   = "results/output_regions/block_order.csv"
EIGENVEC_FILE = "metadata/ldpruned_997subs.eigenvec"
PHENO_FILE    = "metadata/COS_TRIO_pheno_1165.csv"
OUT_DIR       = "results/output_regions2/ORD/all_blocks_pheno_analysis"

PC_COLS = [f"PC{i}" for i in range(1, 11)]

# Blocks of biological interest
ASTHMA_FOCUS_PATTERNS = ["HLA", "17q21", "IL1RL1", "IL33", "FCER1A", "5q31_type2"]
CONTROL_FOCUS_PATTERNS = ["PDE4D", "NOD2", "CFTR"]
# FOCUS_PATTERNS = ASTHMA_FOCUS_PATTERNS + CONTROL_FOCUS_PATTERNS
# FOCUS_PATTERNS = ["HLA", "17q21", "IL1RL1", "IL33", "FCER1A", "5q31_type2"]
FOCUS_PATTERNS = []

CONTINUOUS_PHENOS = ["G19B", "log10eos", "log10Ige", "pctpred_fev1_pre_BD"]
CATEGORICAL_PHENOS = ["G20D"]
ALL_PHENOS = CONTINUOUS_PHENOS + CATEGORICAL_PHENOS

# ── helpers ────────────────────────────────────────────────────────────────────
def load_eigenvec(path: str) -> pd.DataFrame:
    """Tab-separated; first col may be '#FID', second is IID."""
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lstrip("#")
    df = df.rename(columns={"FID": "FID_ev"})
    df["IID"] = df["IID"].astype(str)
    return df[["IID"] + PC_COLS]


def load_block_names(block_order_path: str, expected_B: int):
    """
    Load block names from block_order.csv and verify length.
    Tries common column names first.
    """
    bo = pd.read_csv(block_order_path)
    possible_cols = ["block", "block_id", "region", "name"]
    name_col = next((c for c in possible_cols if c in bo.columns), None)
    if name_col is None:
        raise ValueError(
            f"No block-name column found in {block_order_path}. "
            f"Available columns: {list(bo.columns)}"
        )
    block_names = bo[name_col].astype(str).tolist()
    if len(block_names) != expected_B:
        raise ValueError(
            f"block_order length {len(block_names)} != embedding blocks {expected_B}"
        )
    return block_names, bo


def add_fdr_within_phenotype(df: pd.DataFrame, p_col: str, out_col: str):
    """
    BH-FDR within phenotype separately.
    """
    df = df.copy()
    df[out_col] = np.nan
    for pheno, idx in df.groupby("phenotype").groups.items():
        pvals = df.loc[idx, p_col].astype(float)
        mask = pvals.notna()
        if mask.sum() == 0:
            continue
        _, qvals, _, _ = multipletests(pvals[mask], method="fdr_bh")
        df.loc[pvals[mask].index, out_col] = qvals
    return df

def get_covars(pheno, available_cols):
    covars = PC_COLS.copy()

    if pheno == "age":
        covars += ["gender_num"]

    elif pheno == "BMI":
        covars += ["gender_num"]

    elif pheno == "G19B":
        covars += ["gender_num"]

    elif pheno == "log10eos":
        covars += ["age", "gender_num"]

    elif pheno == "log10Ige":
        covars += ["age", "gender_num"]

    elif pheno == "pctpred_fev1_pre_BD":
        covars += ["gender_num", "smkexp_current"]

    elif pheno in ["Hospitalized_Asthma_Last_Yr", "G20D"]:
        covars += ["age", "gender_num", "smkexp_current"]

    elif pheno == "smkexp_current":
        covars += ["age", "gender_num"]

    return [c for c in covars if c in available_cols]

def normalize_gender(series: pd.Series) -> pd.Series:
    """
    Convert common gender encodings to numeric 0/1 where possible.
    """
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s = s.astype(str).str.strip().str.lower()
    mapping = {
        "male": 1.0, "m": 1.0, "1": 1.0,
        "female": 0.0, "f": 0.0, "2": 0.0
    }
    return s.map(mapping)


def plot_plain_scatter(X2d: np.ndarray, title: str, out_path: Path,
                       xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X2d[:, 0], X2d[:, 1], s=12, alpha=0.7, color="#4c78a8")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_binary_block_pheno(df: pd.DataFrame, block_col: str, pheno_col: str, out_path: Path):
    sub = df[[block_col, pheno_col]].dropna().copy()
    if len(sub) < 10 or sub[pheno_col].nunique() < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=sub, x=pheno_col, y=block_col, ax=ax, color="white", width=0.5)
    sns.stripplot(data=sub, x=pheno_col, y=block_col, ax=ax, alpha=0.45, size=3)

    means = sub.groupby(pheno_col)[block_col].mean().to_dict()
    n_by_group = sub.groupby(pheno_col).size().to_dict()

    ax.set_xlabel(pheno_col, fontsize=10)
    ax.set_ylabel(f"block_PC1 ({block_col})", fontsize=10)
    ax.set_title(
        f"{block_col}\n{pheno_col}  "
        f"mean0={means.get(0.0, np.nan):.3f}, n0={n_by_group.get(0.0, 0)}; "
        f"mean1={means.get(1.0, np.nan):.3f}, n1={n_by_group.get(1.0, 0)}",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def recode_categorical(pheno: pd.DataFrame) -> pd.DataFrame:
    """
    Recode categorical phenotypes to 0/1, drop rows where values are
    outside the expected set.

    Hospitalized_Asthma_Last_Yr: 1=no→0, 2=yes→1
    smkexp_current             : 1=no→0, 2=yes→1
    G20D                       : 0→0 (none), >0→1 (any)
    """
    df = pheno.copy()

    for col in ["Hospitalized_Asthma_Last_Yr", "smkexp_current"]:
        if col not in df.columns:
            continue
        mask = df[col].isin([1.0, 2.0])
        df.loc[~mask, col] = np.nan
        df[col] = df[col].map({1.0: 0.0, 2.0: 1.0})

    if "G20D" in df.columns:
        df["G20D"] = pd.to_numeric(df["G20D"], errors="coerce")
        df["G20D"] = (df["G20D"] > 0).astype(float)
        df.loc[pheno["G20D"].isna(), "G20D"] = np.nan

    return df

def get_block_group(name: str) -> str:
    lname = name.lower()
    if "control_" in lname:
        return "control"
    return "asthma"

def is_focus_block(name: str) -> bool:
    # return any(p.lower() in name.lower() for p in FOCUS_PATTERNS)
    return True

def block_first_pc(block_repr: np.ndarray, block_idx: int) -> np.ndarray:
    """Return 1st PC of (N, 64) block representation. Shape: (N,)."""
    X = block_repr[:, block_idx, :]        # (N, 64)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1, random_state=42)
    return pca.fit_transform(X).ravel()    # (N,)


def run_ols(formula: str, data: pd.DataFrame):
    try:
        sub = data.dropna()
        if len(sub) < 20:
            return None
        return smf.ols(formula, data=sub).fit()
    except Exception:
        return None


def safe_logit(formula: str, data: pd.DataFrame):
    try:
        sub = data.dropna()
        if sub[formula.split("~")[0].strip()].nunique() < 2:
            return None
        if len(sub) < 20:
            return None
        return smf.logit(formula, data=sub).fit(disp=False, maxiter=200)
    except Exception:
        return None


def volcano_plot(df_assoc: pd.DataFrame, beta_col: str, p_col: str,
                 title: str, focus_col: str, out_path: Path):
    df = df_assoc.dropna(subset=[beta_col, p_col]).copy()
    df["neg_log10_p"] = -np.log10(df[p_col].clip(lower=1e-30))

    fig, ax = plt.subplots(figsize=(9, 6))
    non_focus = df[~df[focus_col]]
    focus     = df[df[focus_col]]

    ax.scatter(non_focus[beta_col], non_focus["neg_log10_p"],
               s=12, alpha=0.35, color="#aaaaaa", label="Other blocks")
    ax.scatter(focus[beta_col], focus["neg_log10_p"],
               s=30, alpha=0.85, color="#d62728", label="Focus blocks", zorder=5)

    thresh = -np.log10(0.05)
    ax.axhline(thresh, color="steelblue", linewidth=0.8, linestyle="--", label="p=0.05")

    for _, row in focus[focus["neg_log10_p"] > thresh].iterrows():
        ax.text(row[beta_col], row["neg_log10_p"] + 0.05,
                row["block"], fontsize=6, ha="center")

    ax.set_xlabel("β (block_PC1 effect)", fontsize=11)
    ax.set_ylabel("-log₁₀(p)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def scatter_block_pheno(df: pd.DataFrame, block_col: str, pheno_col: str,
                        out_path: Path, hue_col: str = None):
    sub = df[[block_col, pheno_col]].dropna()
    if len(sub) < 10:
        return
    r, p = stats.pearsonr(sub[block_col], sub[pheno_col])

    fig, ax = plt.subplots(figsize=(6, 5))
    if hue_col and hue_col in df.columns:
        palette = {0.0: "#4878cf", 1.0: "#d65f5f"}
        for grp, grpdf in df.groupby(hue_col, dropna=False):
            c = palette.get(grp, "#999999")
            lbl = {0.0: f"{hue_col}=0", 1.0: f"{hue_col}=1"}.get(grp, str(grp))
            ax.scatter(grpdf[block_col], grpdf[pheno_col],
                       s=16, alpha=0.45, c=c, label=lbl)
        ax.legend(fontsize=8)
    else:
        ax.scatter(sub[block_col], sub[pheno_col], s=16, alpha=0.45, color="#4878cf")

    ax.set_xlabel(f"block_PC1  ({block_col})", fontsize=10)
    ax.set_ylabel(pheno_col, fontsize=10)
    ax.set_title(f"{block_col}\n{pheno_col}   r={r:.3f}  p={p:.3g}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ── UMAP ───────────────────────────────────────────────────────────────────────
def compute_umap(Z: np.ndarray, n_neighbors: int = 20, min_dist: float = 0.1):
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            metric="cosine", random_state=42)
        return reducer.fit_transform(StandardScaler().fit_transform(Z))
    except ImportError:
        print("  [warn] umap-learn not installed — UMAP skipped")
        return None


def plot_colored_umap(Z2d: np.ndarray, color_vals: np.ndarray,
                      title: str, cbar_label: str, out_path: Path, cmap="plasma"):
    if Z2d is None:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Z2d[:, 0], Z2d[:, 1], c=color_vals, cmap=cmap, s=12, alpha=0.7)
    plt.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-subj",    default=EMB_SUBJ_NPY)
    parser.add_argument("--emb-block",   default=EMB_BLOCK_NPY)
    parser.add_argument("--attn-csv",    default=ATTN_CSV)
    parser.add_argument("--block-order", default=BLOCK_ORDER)
    parser.add_argument("--eigenvec",    default=EIGENVEC_FILE)
    parser.add_argument("--pheno-file",  default=PHENO_FILE)
    parser.add_argument("--out-dir",     default=OUT_DIR)
    parser.add_argument("--umap-neighbors", type=int, default=20)
    parser.add_argument("--umap-min-dist",  type=float, default=0.1)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading embeddings …")
    emb_subj  = np.load(args.emb_subj)       # (N, 64)
    emb_block = np.load(args.emb_block)      # (N, B, 64)
    N, B, D   = emb_block.shape
    print(f"  subject embeddings : {emb_subj.shape}")
    print(f"  block repr         : {emb_block.shape}  ({B} blocks × {D} dims)")

    # Block names from attention CSV header (same order as block_contextual_repr)
    # Subject order still comes from attention CSV / IID file
    attn_df = pd.read_csv(args.attn_csv)
    subj_iids = attn_df["IID"].astype(str).values

    # Block names should come from block_order.csv
    block_names, block_order_df = load_block_names(args.block_order, B)

    # Optional safety check against attention header if same blocks are present there
    attn_block_names = [c for c in attn_df.columns if c != "IID"]
    if len(attn_block_names) == B:
        if attn_block_names != block_names:
            print("  [warn] block_order.csv names do not exactly match attention header order")
            mismatch_n = sum(a != b for a, b in zip(attn_block_names, block_names))
            print(f"         mismatches in first-pass comparison: {mismatch_n}/{B}")
        else:
            print("  block_order.csv matches attention header order")

    print("Loading genotype PCs …")
    pcs_df = load_eigenvec(args.eigenvec)     # IID + PC1..PC10
    
    print("Loading phenotypes …")
    pheno_raw = pd.read_csv(args.pheno_file, low_memory=False)
    pheno_raw = pheno_raw.rename(columns={"S_SUBJECTID": "IID"})
    pheno_raw["IID"] = pheno_raw["IID"].astype(str)
    pheno_raw = recode_categorical(pheno_raw)

    if "gender" in pheno_raw.columns:
        pheno_raw["gender_num"] = normalize_gender(pheno_raw["gender"])

    keep_cols = ["IID"] + [c for c in ALL_PHENOS if c in pheno_raw.columns]
    if "gender_num" in pheno_raw.columns:
        keep_cols.append("gender_num")

    pheno_df = pheno_raw[keep_cols].copy()

    # Master frame: one row per subject, in same order as embeddings
    master = pd.DataFrame({"IID": subj_iids})
    master = master.merge(pcs_df, on="IID", how="left")
    master = master.merge(pheno_df, on="IID", how="left")

    n_pc_ok    = master[PC_COLS].notna().all(axis=1).sum()
    print(f"  subjects with PCs  : {n_pc_ok}/{N}")
    for ph in ALL_PHENOS:
        if ph in master.columns:
            nn = master[ph].notna().sum()
            print(f"    {ph:<35s}  n={nn}")

    # ── 2. Subject-level PCA + UMAP ───────────────────────────────────────────
    print("\nSubject-level PCA …")
    Zs = StandardScaler().fit_transform(emb_subj)

    pca_model = PCA(n_components=10, random_state=42)
    subj_pca = pca_model.fit_transform(Zs)

    pc_var = 100 * pca_model.explained_variance_ratio_
    pc1_lab = f"Embedding PC1 ({pc_var[0]:.1f}% var)"
    pc2_lab = f"Embedding PC2 ({pc_var[1]:.1f}% var)"
    pca_title = (
        f"Subject embeddings (PCA)\n"
        f"PC1={pc_var[0]:.1f}%, PC2={pc_var[1]:.1f}%"
    )

    # plain PCA plot
    plot_plain_scatter(
        subj_pca[:, :2],
        title=pca_title,
        out_path=out / "pca_subjects_plain.png",
        xlabel=pc1_lab,
        ylabel=pc2_lab,
    )
    print("  saved pca_subjects_plain.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, c_vals, c_label, cmap in [
        (axes[0], master["PC1"].values, "Genotype PC1", "plasma"),
        (axes[1], master["PC2"].values, "Genotype PC2", "viridis"),
    ]:
        sc = ax.scatter(
            subj_pca[:, 0], subj_pca[:, 1],
            c=c_vals, cmap=cmap, s=14, alpha=0.65
        )
        plt.colorbar(sc, ax=ax, label=c_label)
        ax.set_xlabel(pc1_lab)
        ax.set_ylabel(pc2_lab)
        ax.set_title(
            f"Subject embeddings (PCA)\n"
            f"coloured by {c_label}"
        )
    plt.tight_layout()
    plt.savefig(out / "pca_subjects.png", dpi=160)
    plt.close()
    print("  saved pca_subjects.png")

    print("Subject-level UMAP …")
    umap_2d = compute_umap(emb_subj, args.umap_neighbors, args.umap_min_dist)
    if umap_2d is not None:
        np.save(out / "umap_2d_subjects.npy", umap_2d)
        plot_plain_scatter(
            umap_2d,
            title="UMAP of subject embeddings",
            out_path=out / "umap_subjects_plain.png",
            xlabel="UMAP-1",
            ylabel="UMAP-2",
        )
        print("  saved umap_subjects_plain.png")
        for pc_col, cmap_name, fname in [
            ("PC1", "plasma",  "umap_subjects_by_geno_PC1.png"),
            ("PC2", "viridis", "umap_subjects_by_geno_PC2.png"),
        ]:
            vals = master[pc_col].values if pc_col in master.columns else None
            if vals is not None:
                plot_colored_umap(
                    umap_2d, vals,
                    title=f"UMAP of subject embeddings\ncoloured by genotype {pc_col}",
                    cbar_label=f"Genotype {pc_col}",
                    out_path=out / fname,
                    cmap=cmap_name,
                )
                print(f"  saved {fname}")

    # ── 3. Per-block PC1 extraction ───────────────────────────────────────────
    print(f"\nExtracting PC1 for all {B} blocks …")
    block_pc1 = np.zeros((N, B), dtype=np.float32)
    for b in range(B):
        block_pc1[:, b] = block_first_pc(emb_block, b)
    print("  done")

    # Build per-block DataFrame (IID + block_PC1 columns)
    bpc_df = pd.DataFrame(block_pc1, columns=block_names)
    bpc_df.insert(0, "IID", subj_iids)

    # ── 4. Ancestry association ───────────────────────────────────────────────
    print("\nAncestry association (block_PC1 ~ genotype PCs) …")
    anc_rows = []
    for b_idx, bname in enumerate(block_names):
        x = block_pc1[:, b_idx]
        row = {
            "block": bname,
            "is_focus": is_focus_block(bname),
            "block_group": get_block_group(bname),
        }
        for pc in PC_COLS:
            gpc = master[pc].values if pc in master.columns else None
            if gpc is None:
                continue
            mask = ~np.isnan(gpc)
            if mask.sum() < 20:
                row[f"r_{pc}"] = np.nan
                row[f"p_{pc}"] = np.nan
            else:
                r, p = stats.pearsonr(x[mask], gpc[mask])
                row[f"r_{pc}"] = round(float(r), 5)
                row[f"p_{pc}"] = round(float(p), 6)
        anc_rows.append(row)

    anc_df = pd.DataFrame(anc_rows)
    anc_df.to_csv(out / "block_ancestry_correlations.tsv", sep="\t", index=False)

    # Top ancestry-associated blocks (by |r| with gPC1)
    top_anc = (
        anc_df[["block", "is_focus", "r_PC1", "p_PC1", "r_PC2", "p_PC2"]]
        .assign(abs_r_PC1=lambda d: d["r_PC1"].abs())
        .sort_values("abs_r_PC1", ascending=False)
        .drop(columns="abs_r_PC1")
        .reset_index(drop=True)
    )
    top_anc.head(50).to_csv(out / "top_ancestry_associated_blocks.tsv", sep="\t", index=False)

    # ── Ancestry heatmap ──────────────────────────────────────────────────────
    # Show all blocks but highlight focus blocks; rows = blocks, cols = PC1..PC10
    r_mat = anc_df.set_index("block")[[f"r_{pc}" for pc in PC_COLS]]
    r_mat.columns = PC_COLS

    focus_mask = anc_df["is_focus"].values
    focus_idx  = np.where(focus_mask)[0]

    fig_h = max(6, B * 0.18)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    sns.heatmap(
        r_mat, annot=False, cmap="coolwarm", center=0,
        vmin=-0.5, vmax=0.5, linewidths=0.0, ax=ax,
        yticklabels=True,
    )
    ax.set_title("Block embedding PC1 × Genotype PC correlation (r)", fontsize=13)
    ax.set_xlabel("Genotype PC"); ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=5)

    # Colour focus block tick labels red
    yticklabels = ax.get_yticklabels()
    for lbl in yticklabels:
        if is_focus_block(lbl.get_text()):
            lbl.set_color("#d62728")
            lbl.set_fontweight("bold")

    red_patch = mpatches.Patch(color="#d62728", label="Focus blocks")
    ax.legend(handles=[red_patch], loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "block_ancestry_heatmap.png", dpi=160)
    plt.close()
    print("  saved block_ancestry_heatmap.png")

    # Top ancestry blocks bar chart (top 30 by |r| with PC1)
    top30 = top_anc.head(30).copy()
    top30 = top30.sort_values("r_PC1")
    colors_bar = ["#d62728" if f else "#4878cf" for f in top30["is_focus"]]
    fig, ax = plt.subplots(figsize=(9, max(5, len(top30) * 0.35)))
    ax.barh(range(len(top30)), top30["r_PC1"], color=colors_bar, alpha=0.85)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30["block"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Pearson r with Genotype PC1")
    ax.set_title("Top 30 blocks by |ancestry PC1 association|", fontsize=12)
    red_patch  = mpatches.Patch(color="#d62728", label="Focus blocks")
    blue_patch = mpatches.Patch(color="#4878cf", label="Other blocks")
    ax.legend(handles=[red_patch, blue_patch], fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "top_ancestry_blocks_bar.png", dpi=160)
    plt.close()
    print("  saved top_ancestry_blocks_bar.png")

    # ── 5. Phenotype association ──────────────────────────────────────────────
    print("\nPhenotype ~ block_PC1 regressions …")
    pheno_cols_present = [p for p in ALL_PHENOS if p in master.columns]

    model_rows = []
    for b_idx, bname in enumerate(block_names):
        safe_b = f"b_{b_idx}"        # formula-safe placeholder
        reg_df = master.copy()
        reg_df[safe_b] = block_pc1[:, b_idx]

        for pheno in pheno_cols_present:
            safe_p = pheno.replace("-", "_")
            reg_df[safe_p] = reg_df[pheno]

            is_cat = pheno in CATEGORICAL_PHENOS
            fit_fn = safe_logit if is_cat else run_ols

            covars = get_covars(pheno, reg_df.columns)
            covar_formula = " + ".join(covars)

            res_u = fit_fn(f"{safe_p} ~ {safe_b}", reg_df)
            res_a = fit_fn(f"{safe_p} ~ {safe_b} + {covar_formula}", reg_df) if covar_formula else res_u

            def extract(res):
                if res is None:
                    return dict(beta=np.nan, pval=np.nan, r2=np.nan, n=np.nan)
                sub = res.model.data.frame
                b = res.params.get(safe_b, np.nan)
                pv = res.pvalues.get(safe_b, np.nan)
                r2 = getattr(res, "rsquared", None)
                if r2 is None:
                    r2 = getattr(res, "prsquared", np.nan)
                return dict(
                    beta=round(float(b), 6),
                    pval=round(float(pv), 6),
                    r2=round(float(r2) if pd.notna(r2) else np.nan, 6),
                    n=len(sub)
                )

            u = extract(res_u)
            a = extract(res_a)
            model_rows.append({
                "block": bname,
                "phenotype": pheno,
                "model_type": "logit" if is_cat else "OLS",
                "is_focus": is_focus_block(bname),
                "block_group": get_block_group(bname),
                "n_unadj": u["n"],
                "beta_unadj": u["beta"],
                "pval_unadj": u["pval"],
                "r2_unadj": u["r2"],
                "n_adj": a["n"],
                "beta_adj": a["beta"],
                "pval_adj": a["pval"],
                "r2_adj": a["r2"],
                "covars_adj": ";".join(covars),
            })

        if (b_idx + 1) % 20 == 0:
            print(f"  block {b_idx+1}/{B} …")

    assoc_df = pd.DataFrame(model_rows)

    # BH-FDR within phenotype
    assoc_df = add_fdr_within_phenotype(assoc_df, "pval_unadj", "FDR_BH_unadj")
    assoc_df = add_fdr_within_phenotype(assoc_df, "pval_adj", "FDR_BH_adj")

    print("\nBlock-group counts:")
    print(assoc_df.groupby("block_group")["block"].nunique())

    assoc_df.to_csv(out / "phenotype_block_associations.tsv", sep="\t", index=False)
    print(f"  saved phenotype_block_associations.tsv  ({len(assoc_df)} rows)")

    # nominal adjusted hits per block
    block_hits = (
        assoc_df.assign(nominal = assoc_df["pval_adj"] < 0.05)
        .groupby(["block", "block_group"])["nominal"]
        .sum()
        .reset_index(name="n_hits")
    )

    summary_multi = (
        block_hits.groupby("block_group")
        .agg(
            n_blocks=("block", "size"),
            mean_hits=("n_hits", "mean"),
            median_hits=("n_hits", "median"),
            blocks_ge1=("n_hits", lambda x: (x >= 1).sum()),
            blocks_ge2=("n_hits", lambda x: (x >= 2).sum()),
            blocks_ge3=("n_hits", lambda x: (x >= 3).sum()),
        )
        .reset_index()
    )

    for k in [1, 2, 3]:
        summary_multi[f"pct_ge{k}"] = (
            100 * summary_multi[f"blocks_ge{k}"] / summary_multi["n_blocks"]
        )

    print("\nMultiple-hit enrichment by block group:")
    print(summary_multi)

    # ── 5b. Per-phenotype asthma vs control summary ─────────────────────────────
    print("\nPer-phenotype asthma vs control summary …")

    alpha = 0.05

    comp_df = assoc_df.copy()
    comp_df = comp_df[comp_df["block_group"].isin(["asthma", "control"])].copy()

    rows = []
    for pheno in sorted(comp_df["phenotype"].dropna().unique()):
        sub = comp_df[comp_df["phenotype"] == pheno].copy()

        asthma = sub[sub["block_group"] == "asthma"].copy()
        control = sub[sub["block_group"] == "control"].copy()

        n_asthma = len(asthma)
        n_control = len(control)

        n_asthma_nom = (asthma["pval_adj"] < alpha).sum()
        n_control_nom = (control["pval_adj"] < alpha).sum()

        frac_asthma = n_asthma_nom / n_asthma if n_asthma > 0 else np.nan
        frac_control = n_control_nom / n_control if n_control > 0 else np.nan

        med_asthma = asthma["pval_adj"].median() if n_asthma > 0 else np.nan
        med_control = control["pval_adj"].median() if n_control > 0 else np.nan

        # top 20 enrichment
        top_n = min(20, len(sub))
        top_hits = sub.sort_values("pval_adj").head(top_n).copy()

        n_asthma_top = (top_hits["block_group"] == "asthma").sum()
        n_control_top = (top_hits["block_group"] == "control").sum()

        asthma_bg = (sub["block_group"] == "asthma").mean() if len(sub) > 0 else np.nan
        control_bg = (sub["block_group"] == "control").mean() if len(sub) > 0 else np.nan

        asthma_expected = top_n * asthma_bg if pd.notna(asthma_bg) else np.nan
        control_expected = top_n * control_bg if pd.notna(control_bg) else np.nan

        asthma_enrich = (n_asthma_top / asthma_expected) if asthma_expected and asthma_expected > 0 else np.nan
        control_enrich = (n_control_top / control_expected) if control_expected and control_expected > 0 else np.nan

        rows.append({
            "phenotype": pheno,
            "asthma_tests": n_asthma,
            "control_tests": n_control,
            "asthma_nominal_n": int(n_asthma_nom),
            "control_nominal_n": int(n_control_nom),
            "asthma_nominal_frac": round(frac_asthma, 4) if pd.notna(frac_asthma) else np.nan,
            "control_nominal_frac": round(frac_control, 4) if pd.notna(frac_control) else np.nan,
            "asthma_median_p_adj": round(float(med_asthma), 4) if pd.notna(med_asthma) else np.nan,
            "control_median_p_adj": round(float(med_control), 4) if pd.notna(med_control) else np.nan,
            "top20_asthma_n": int(n_asthma_top),
            "top20_control_n": int(n_control_top),
            "top20_asthma_enrichment": round(float(asthma_enrich), 3) if pd.notna(asthma_enrich) else np.nan,
            "top20_control_enrichment": round(float(control_enrich), 3) if pd.notna(control_enrich) else np.nan,
        })

    pheno_group_summary = pd.DataFrame(rows)

    # helpful sorting: strongest asthma enrichment first
    pheno_group_summary = pheno_group_summary.sort_values(
        by=["top20_asthma_enrichment", "asthma_nominal_frac"],
        ascending=[False, False]
    )

    pheno_group_summary.to_csv(
        out / "phenotype_asthma_vs_control_summary.tsv",
        sep="\t",
        index=False
    )

    print("  saved phenotype_asthma_vs_control_summary.tsv")
    print("\n  Phenotype-wise asthma vs control summary:")
    for _, row in pheno_group_summary.iterrows():
        print(
            f"    • {row['phenotype']:<28s} "
            f"asthma={row['asthma_nominal_n']}/{row['asthma_tests']} ({100*row['asthma_nominal_frac']:.1f}%)  "
            f"control={row['control_nominal_n']}/{row['control_tests']} ({100*row['control_nominal_frac']:.1f}%)  "
            f"top20 enrich={row['top20_asthma_enrichment']:.2f}x"
        )

    # ── 5c. For each phenotype: nominally associated blocks ─────────────────────
    print("\nNominally associated blocks by phenotype (adjusted p < 0.05) …")

    sig_by_pheno_rows = []

    for pheno in sorted(comp_df["phenotype"].dropna().unique()):
        sub = comp_df[
            (comp_df["phenotype"] == pheno) &
            (comp_df["pval_adj"] < alpha)
        ].copy()

        sub = sub.sort_values("pval_adj")

        # save full table
        sub.to_csv(
            out / f"nominal_blocks_by_phenotype_{pheno}.tsv",
            sep="\t",
            index=False
        )

        print(f"\n  {pheno}: {len(sub)} nominal blocks")
        if len(sub) == 0:
            print("    (none)")
            continue

        for _, row in sub.head(15).iterrows():
            print(
                f"    • {row['block']:<45s} "
                f"group={row['block_group']:<8s} "
                f"beta={row['beta_adj']:+.4f} "
                f"p={row['pval_adj']:.4g}"
            )

            sig_by_pheno_rows.append({
                "phenotype": pheno,
                "block": row["block"],
                "block_group": row["block_group"],
                "beta_adj": row["beta_adj"],
                "pval_adj": row["pval_adj"],
                "FDR_BH_adj": row["FDR_BH_adj"],
            })

    sig_by_pheno_df = pd.DataFrame(sig_by_pheno_rows)
    sig_by_pheno_df.to_csv(
        out / "nominal_blocks_by_phenotype_all.tsv",
        sep="\t",
        index=False
    )

    print("\n  saved nominal_blocks_by_phenotype_*.tsv")

    # ── 5d. Asthma-vs-control enrichment summary ───────────────────────────────
    print("\nAsthma vs control enrichment summary …")

    alpha = 0.05

    # adjusted analysis only
    comp_df = assoc_df[
        assoc_df["block_group"].isin(["asthma", "control"])
    ].copy()

    # per-association summaries
    group_summary = (
        comp_df.groupby("block_group")
        .agg(
            n_tests=("pval_adj", "size"),
            n_nominal=("pval_adj", lambda s: (s < alpha).sum()),
            frac_nominal=("pval_adj", lambda s: (s < alpha).mean()),
            median_p_adj=("pval_adj", "median"),
            median_p_unadj=("pval_unadj", "median"),
        )
        .reset_index()
    )

    group_summary["pct_nominal"] = 100 * group_summary["frac_nominal"]
    group_summary.to_csv(out / "asthma_vs_control_group_summary.tsv", sep="\t", index=False)

    # unique-block summaries: fraction of blocks with at least one nominal hit
    per_block_any = (
        comp_df.assign(is_nominal_adj=comp_df["pval_adj"] < alpha)
        .groupby(["block", "block_group"], as_index=False)["is_nominal_adj"]
        .max()
    )

    block_level_summary = (
        per_block_any.groupby("block_group")
        .agg(
            n_blocks=("block", "nunique"),
            n_blocks_with_nominal_hit=("is_nominal_adj", "sum"),
        )
        .reset_index()
    )

    block_level_summary["frac_blocks_with_nominal_hit"] = (
        block_level_summary["n_blocks_with_nominal_hit"] / block_level_summary["n_blocks"]
    )
    block_level_summary["pct_blocks_with_nominal_hit"] = (
        100 * block_level_summary["frac_blocks_with_nominal_hit"]
    )
    block_level_summary.to_csv(out / "asthma_vs_control_block_level_summary.tsv", sep="\t", index=False)

    # enrichment among top-N adjusted hits
    top_enrichment_rows = []
    for top_n in [20, 50]:
        top_hits = comp_df.sort_values("pval_adj").head(top_n).copy()

        n_asthma_top = (top_hits["block_group"] == "asthma").sum()
        n_control_top = (top_hits["block_group"] == "control").sum()

        asthma_background = (comp_df["block_group"] == "asthma").mean()
        control_background = (comp_df["block_group"] == "control").mean()

        asthma_expected = top_n * asthma_background
        control_expected = top_n * control_background

        top_enrichment_rows.append({
            "top_n": top_n,
            "n_asthma_top": int(n_asthma_top),
            "n_control_top": int(n_control_top),
            "pct_asthma_top": 100 * n_asthma_top / top_n,
            "pct_control_top": 100 * n_control_top / top_n,
            "asthma_background_pct": 100 * asthma_background,
            "control_background_pct": 100 * control_background,
            "asthma_expected_count": asthma_expected,
            "control_expected_count": control_expected,
            "asthma_enrichment_ratio": (n_asthma_top / asthma_expected) if asthma_expected > 0 else np.nan,
            "control_enrichment_ratio": (n_control_top / control_expected) if control_expected > 0 else np.nan,
        })

    top_enrichment_df = pd.DataFrame(top_enrichment_rows)
    top_enrichment_df.to_csv(out / "asthma_vs_control_topN_enrichment.tsv", sep="\t", index=False)

    print("  saved asthma_vs_control_group_summary.tsv")
    print("  saved asthma_vs_control_block_level_summary.tsv")
    print("  saved asthma_vs_control_topN_enrichment.tsv")

    # ── 5d. Top blocks and their associated phenotypes ──────────────────────────
    print("\nTop blocks and the phenotypes they associate with nominally …")

    # choose top blocks by best adjusted p-value across all phenotypes
    top_blocks = (
        comp_df.groupby(["block", "block_group"], as_index=False)
        .agg(best_p_adj=("pval_adj", "min"))
        .sort_values("best_p_adj")
        .head(20)
    )

    top_block_pheno_rows = []

    for _, brow in top_blocks.iterrows():
        bname = brow["block"]
        bgroup = brow["block_group"]

        hits = comp_df[
            (comp_df["block"] == bname) &
            (comp_df["pval_adj"] < alpha)
        ].copy().sort_values("pval_adj")

        print(f"\n  {bname} ({bgroup})")
        if len(hits) == 0:
            print("    (no nominal adjusted hits)")
            continue

        for _, row in hits.iterrows():
            print(
                f"    • {row['phenotype']:<28s} "
                f"beta={row['beta_adj']:+.4f} "
                f"p={row['pval_adj']:.4g}"
            )

            top_block_pheno_rows.append({
                "block": bname,
                "block_group": bgroup,
                "phenotype": row["phenotype"],
                "beta_adj": row["beta_adj"],
                "pval_adj": row["pval_adj"],
                "FDR_BH_adj": row["FDR_BH_adj"],
            })

    top_block_pheno_df = pd.DataFrame(top_block_pheno_rows)
    top_block_pheno_df.to_csv(
        out / "top_blocks_nominal_phenotypes.tsv",
        sep="\t",
        index=False
    )

    print("\n  saved top_blocks_nominal_phenotypes.tsv")

    # ── 6. Volcano plots ──────────────────────────────────────────────────────
    print("\nVolcano plots …")
    for pheno in pheno_cols_present:
        sub = assoc_df[assoc_df["phenotype"] == pheno].copy()
        for adj_flag, beta_col, p_col, fname_tag in [
            ("unadj", "beta_unadj", "pval_unadj", "unadj"),
            ("adj",   "beta_adj",   "pval_adj",   "adj"),
        ]:
            volcano_plot(
                sub,
                beta_col=beta_col,
                p_col=p_col,
                focus_col="is_focus",
                title=f"{pheno} ~ block_PC1 ({adj_flag})",
                out_path=out / f"volcano_{pheno}_{fname_tag}.png",
            )
    print("  saved volcano_*.png")

    # ── 7. Focus-block × phenotype heatmaps ──────────────────────────────────
    print("Focus-block heatmaps …")
    focus_assoc = assoc_df[assoc_df["is_focus"]].copy()

    for adj_flag, p_col, fdr_col, b_col in [
        ("unadj", "pval_unadj", "FDR_BH_unadj", "beta_unadj"),
        ("adj",   "pval_adj",   "FDR_BH_adj",   "beta_adj"),
    ]:
        pivot_b = focus_assoc.pivot(index="block", columns="phenotype", values=b_col)
        pivot_f = focus_assoc.pivot(index="block", columns="phenotype", values=fdr_col)

        annot = pivot_b.copy().astype(str)
        for r in pivot_b.index:
            for c in pivot_b.columns:
                beta_v = pivot_b.loc[r, c]
                fdr_v = pivot_f.loc[r, c]
                if pd.isna(beta_v):
                    annot.loc[r, c] = "NA"
                else:
                    sig = "*" if pd.notna(fdr_v) and fdr_v < 0.10 else ""
                    annot.loc[r, c] = f"{beta_v:.2f}{sig}"

        fig, ax = plt.subplots(
            figsize=(max(8, len(pivot_b.columns) * 1.5),
                     max(5, len(pivot_b) * 0.5))
        )
        sns.heatmap(
            pivot_b,
            annot=annot,
            fmt="s",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(
            f"Focus blocks: phenotype association (β, * FDR<0.10) [{adj_flag}]",
            fontsize=12,
        )
        ax.set_xlabel("Phenotype")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        plt.savefig(out / f"focus_block_pheno_heatmap_{adj_flag}.png", dpi=160)
        plt.close()

    print("  saved focus_block_pheno_heatmap_*.png")

    # ── 8. Scatter plots for top hits in focus blocks ────────────────────────
    print("Plots for top focus-block hits …")
    scatter_dir = out / "scatter_plots"
    scatter_dir.mkdir(exist_ok=True)

    focus_hits = (
        focus_assoc[focus_assoc["pval_unadj"] < 0.1]
        .sort_values("pval_unadj")
        .drop_duplicates(subset=["block", "phenotype"])
    )

    scatter_reg_df = bpc_df.merge(master, on="IID", how="inner")

    for _, hit in focus_hits.iterrows():
        bname = hit["block"]
        pheno = hit["phenotype"]
        if bname not in scatter_reg_df.columns:
            continue

        out_path = scatter_dir / f"plot_{bname}_{pheno}.png"
        if pheno in CATEGORICAL_PHENOS:
            plot_binary_block_pheno(
                scatter_reg_df,
                block_col=bname,
                pheno_col=pheno,
                out_path=out_path,
            )
        else:
            scatter_block_pheno(
                scatter_reg_df,
                block_col=bname,
                pheno_col=pheno,
                out_path=out_path,
                hue_col=None,
            )

    # Always plot a scatter for each focus pattern vs top phenotype
    for pattern in FOCUS_PATTERNS:
        pat_blocks = [b for b in block_names if pattern.lower() in b.lower()]
        if not pat_blocks:
            continue
        # pick the block with smallest unadj p across any phenotype
        pat_hits = focus_assoc[
            focus_assoc["block"].isin(pat_blocks)
        ].sort_values("pval_unadj")
        if len(pat_hits) == 0:
            continue
        best_row = pat_hits.iloc[0]
        bname = best_row["block"]
        pheno = best_row["phenotype"]
        if bname not in scatter_reg_df.columns:
            continue
        fpath = scatter_dir / f"scatter_{bname}_{pheno}_best.png"
        if not fpath.exists():
            scatter_block_pheno(
                scatter_reg_df, block_col=bname, pheno_col=pheno,
                out_path=fpath,
                hue_col=pheno if pheno in CATEGORICAL_PHENOS else None,
            )

    print(f"  saved scatter plots to {scatter_dir}/")

    # ── 9. Summary ────────────────────────────────────────────────────────────
    alpha = 0.05
    nom_unadj = assoc_df[assoc_df["pval_unadj"] < alpha]
    nom_adj   = assoc_df[assoc_df["pval_adj"] < alpha]
    sig_unadj = assoc_df[assoc_df["FDR_BH_unadj"] < alpha]
    sig_adj   = assoc_df[assoc_df["FDR_BH_adj"] < alpha]

    print(f"\n{'═'*70}")
    print("  Summary: nominal and FDR-significant phenotype associations")
    print(f"{'═'*70}")
    print(f"  Nominal unadjusted (p < 0.05): {len(nom_unadj)}")
    print(f"  Nominal adjusted   (p < 0.05): {len(nom_adj)}")
    print(f"  FDR unadjusted     (q < 0.05): {len(sig_unadj)}")
    print(f"  FDR adjusted       (q < 0.05): {len(sig_adj)}")

    if len(sig_adj) > 0:
        print("\n  Adjusted hits:")
        for _, row in sig_adj.sort_values("pval_adj").head(20).iterrows():
            focus_tag = " [FOCUS]" if row["is_focus"] else ""
            print(f"    {row['block']:<50s}  {row['phenotype']:<30s}"
                  f"  β={row['beta_adj']:+.4f}  p={row['pval_adj']:.4g}{focus_tag}")

    print(f"\n  Focus blocks that remain significant after PC adjustment:")

    focus_sig_adj = sig_adj[sig_adj["is_focus"]]

    if len(focus_sig_adj) == 0:
        print("    (none at FDR < 0.05)")
        print("    Showing strongest nominal hits instead (p < 0.05):")

        focus_nom_adj = (
            assoc_df[
                (assoc_df["is_focus"]) &
                (assoc_df["pval_adj"] < 0.05)
            ]
            .sort_values("pval_adj")
        )

        if len(focus_nom_adj) == 0:
            print("    (none at p < 0.05 either)")
        else:
            for _, row in focus_nom_adj.head(15).iterrows():
                print(
                    f"    • {row['block']}  ~  {row['phenotype']}"
                    f"  β={row['beta_adj']:+.4f}"
                    f"  p={row['pval_adj']:.4g}"
                    f"  FDR={row['FDR_BH_adj']:.4g}"
                )
    else:
        for _, row in focus_sig_adj.sort_values("FDR_BH_adj").iterrows():
            print(
                f"    • {row['block']}  ~  {row['phenotype']}"
                f"  β={row['beta_adj']:+.4f}"
                f"  p={row['pval_adj']:.4g}"
                f"  FDR={row['FDR_BH_adj']:.4g}"
            )

    print(f"\n  Top ancestry-associated focus blocks (|r| with geno PC1):")
    focus_anc = top_anc[top_anc["is_focus"]].head(10)
    for _, row in focus_anc.iterrows():
        print(f"    • {row['block']:<50s}  r_PC1={row['r_PC1']:+.4f}  p={row['p_PC1']:.4g}")
    
    print(f"\n  Asthma vs control comparison (adjusted p-values):")

    for _, row in group_summary.iterrows():
        print(
            f"    • {row['block_group']:<8s}  "
            f"tests={int(row['n_tests'])}  "
            f"nominal={int(row['n_nominal'])}  "
            f"frac_nominal={row['frac_nominal']:.3f}  "
            f"median_p_adj={row['median_p_adj']:.4f}"
        )

    print(f"\n  Fraction of unique blocks with ≥1 nominal adjusted hit:")
    for _, row in block_level_summary.iterrows():
        print(
            f"    • {row['block_group']:<8s}  "
            f"blocks={int(row['n_blocks'])}  "
            f"with_hit={int(row['n_blocks_with_nominal_hit'])}  "
            f"frac={row['frac_blocks_with_nominal_hit']:.3f}"
        )

    print(f"\n  Enrichment of asthma blocks among top adjusted hits:")
    for _, row in top_enrichment_df.iterrows():
        print(
            f"    • Top {int(row['top_n'])}: "
            f"asthma={int(row['n_asthma_top'])} "
            f"({row['pct_asthma_top']:.1f}%; bg={row['asthma_background_pct']:.1f}%; "
            f"enrichment={row['asthma_enrichment_ratio']:.2f}x), "
            f"control={int(row['n_control_top'])} "
            f"({row['pct_control_top']:.1f}%; bg={row['control_background_pct']:.1f}%; "
            f"enrichment={row['control_enrichment_ratio']:.2f}x)"
        )

    print(f"\nAll outputs saved to: {out}/")


if __name__ == "__main__":
    main()
