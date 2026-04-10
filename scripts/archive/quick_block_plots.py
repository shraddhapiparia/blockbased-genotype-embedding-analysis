#!/usr/bin/env python3
"""
08_block_embedding_umap_analysis.py

Goal
----
Test whether phenotype-associated blocks form smooth gradients across
subject PCA / UMAP space, and compare those gradients with ancestry signal.

Inputs
------
Uses same files / subject ordering conventions as 07_block_embedding_phenotype_analysis.py

Outputs
-------
gradient_summary.tsv
top_gradient_blocks.tsv
pca_umap_gradient_examples/*.png
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── paths ─────────────────────────────────────────────────────────────
EMB_SUBJ_NPY  = "results/output_regions2/ORD/embeddings/individual_embeddings.npy"
EMB_BLOCK_NPY = "results/output_regions2/ORD/embeddings/block_contextual_repr.npy"
ATTN_CSV      = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
BLOCK_ORDER   = "results/output_regions/block_order.csv"
EIGENVEC_FILE = "metadata/ldpruned_997subs.eigenvec"
PHENO_FILE    = "metadata/COS_TRIO_pheno_1165.csv"
OUT_DIR       = "results/output_regions2/ORD/all_blocks_pheno_analysis"

PC_COLS = [f"PC{i}" for i in range(1, 11)]
CONTINUOUS_PHENOS = ["G19B", "log10eos", "pctpred_fev1_pre_BD"]
CATEGORICAL_PHENOS = ["G20D"]
ALL_PHENOS = CONTINUOUS_PHENOS + CATEGORICAL_PHENOS

# phenotype-prioritized blocks from your slide summary
PRIORITY_PATTERNS = [
    "PDE4D", "HLA", "17q21", "5q31_type2", "TNFSF", "SH2B3"
]

# ── helpers ───────────────────────────────────────────────────────────
def load_eigenvec(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lstrip("#")
    df["IID"] = df["IID"].astype(str)
    return df[["IID"] + PC_COLS]

def load_block_names(block_order_path, expected_B):
    bo = pd.read_csv(block_order_path)
    possible_cols = ["block", "block_id", "region", "name"]
    name_col = next((c for c in possible_cols if c in bo.columns), None)
    if name_col is None:
        raise ValueError(f"No block name column found in {block_order_path}. columns={list(bo.columns)}")
    names = bo[name_col].astype(str).tolist()
    if len(names) != expected_B:
        raise ValueError(f"block_order length {len(names)} != embedding blocks {expected_B}")
    return names

def normalize_gender(series):
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str).str.strip().str.lower()
    mp = {"male": 1.0, "m": 1.0, "1": 1.0,
          "female": 0.0, "f": 0.0, "2": 0.0}
    return s.map(mp)

def recode_categorical(pheno):
    df = pheno.copy()
    if "G20D" in df.columns:
        raw = pd.to_numeric(df["G20D"], errors="coerce")
        df["G20D"] = (raw > 0).astype(float)
        df.loc[raw.isna(), "G20D"] = np.nan
    return df

def block_first_pc(block_repr, block_idx):
    X = block_repr[:, block_idx, :]
    X = StandardScaler().fit_transform(X)
    return PCA(n_components=1, random_state=42).fit_transform(X).ravel()

def compute_umap(Z, n_neighbors=20, min_dist=0.1):
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=42
        )
        return reducer.fit_transform(StandardScaler().fit_transform(Z))
    except ImportError:
        print("[warn] umap-learn not installed; skipping UMAP")
        return None

def corr_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)

def logistic_auc_like_score(x, y):
    # lightweight separation metric for binary outcome
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        return np.nan
    g0 = x[mask][y[mask] == 0]
    g1 = x[mask][y[mask] == 1]
    if len(g0) == 0 or len(g1) == 0:
        return np.nan
    u, _ = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    return float(u / (len(g1) * len(g0)))  # AUC-like

def gradient_r2(score, xy):
    mask = np.isfinite(score) & np.isfinite(xy).all(axis=1)
    if mask.sum() < 20:
        return np.nan
    model = LinearRegression()
    model.fit(xy[mask], score[mask])
    return float(model.score(xy[mask], score[mask]))

def is_priority_block(name):
    lname = name.lower()
    return any(p.lower() in lname for p in PRIORITY_PATTERNS)

def plot_gradient(coords, values, title, xlabel, ylabel, out_path, cmap="plasma"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=14, alpha=0.75)
    plt.colorbar(sc, ax=ax, label="block_PC1")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ── main ──────────────────────────────────────────────────────────────
def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "pca_umap_gradient_examples"
    fig_dir.mkdir(exist_ok=True)

    print("Loading inputs...")
    emb_subj = np.load(EMB_SUBJ_NPY)       # (N, d)
    emb_block = np.load(EMB_BLOCK_NPY)     # (N, B, d)
    N, B, D = emb_block.shape

    attn_df = pd.read_csv(ATTN_CSV)
    subj_iids = attn_df["IID"].astype(str).values
    block_names = load_block_names(BLOCK_ORDER, B)

    pcs_df = load_eigenvec(EIGENVEC_FILE)

    pheno = pd.read_csv(PHENO_FILE, low_memory=False).rename(columns={"S_SUBJECTID": "IID"})
    pheno["IID"] = pheno["IID"].astype(str)
    pheno = recode_categorical(pheno)
    if "gender" in pheno.columns:
        pheno["gender_num"] = normalize_gender(pheno["gender"])

    keep_cols = ["IID"] + [c for c in ALL_PHENOS if c in pheno.columns]
    master = pd.DataFrame({"IID": subj_iids}).merge(pcs_df, on="IID", how="left").merge(
        pheno[keep_cols], on="IID", how="left"
    )

    print("Computing subject PCA/UMAP...")
    Zs = StandardScaler().fit_transform(emb_subj)
    subj_pca = PCA(n_components=10, random_state=42).fit_transform(Zs)
    pca_var = PCA(n_components=10, random_state=42).fit(Zs).explained_variance_ratio_ * 100
    pca2 = subj_pca[:, :2]

    umap2 = compute_umap(emb_subj, n_neighbors=20, min_dist=0.1)
    if umap2 is not None:
        np.save(out / "umap_2d_subjects.npy", umap2)

    print("Extracting per-block PC1...")
    block_pc1 = np.zeros((N, B), dtype=np.float32)
    for b in range(B):
        block_pc1[:, b] = block_first_pc(emb_block, b)

    print("Testing gradients...")
    rows = []
    for b, bname in enumerate(block_names):
        score = block_pc1[:, b]

        # gradient against embedding PCA space
        r_pca1, p_pca1 = corr_safe(score, pca2[:, 0])
        r_pca2, p_pca2 = corr_safe(score, pca2[:, 1])
        pca_r2 = gradient_r2(score, pca2)

        # gradient against UMAP space
        if umap2 is not None:
            r_umap1, p_umap1 = corr_safe(score, umap2[:, 0])
            r_umap2, p_umap2 = corr_safe(score, umap2[:, 1])
            umap_r2 = gradient_r2(score, umap2)
        else:
            r_umap1 = p_umap1 = r_umap2 = p_umap2 = umap_r2 = np.nan

        # ancestry comparison
        r_gpc1, p_gpc1 = corr_safe(score, master["PC1"].values)
        r_gpc2, p_gpc2 = corr_safe(score, master["PC2"].values)

        # phenotype comparison
        row = {
            "block": bname,
            "is_priority": is_priority_block(bname),

            "r_embedPC1": r_pca1,
            "p_embedPC1": p_pca1,
            "r_embedPC2": r_pca2,
            "p_embedPC2": p_pca2,
            "embedPCA_r2": pca_r2,

            "r_umap1": r_umap1,
            "p_umap1": p_umap1,
            "r_umap2": r_umap2,
            "p_umap2": p_umap2,
            "umap_r2": umap_r2,

            "r_genoPC1": r_gpc1,
            "p_genoPC1": p_gpc1,
            "r_genoPC2": r_gpc2,
            "p_genoPC2": p_gpc2,
        }

        for ph in ALL_PHENOS:
            if ph not in master.columns:
                row[f"{ph}_assoc"] = np.nan
                row[f"{ph}_p"] = np.nan
                continue

            y = pd.to_numeric(master[ph], errors="coerce").values
            if ph in CATEGORICAL_PHENOS:
                row[f"{ph}_assoc"] = logistic_auc_like_score(score, y)
                _, p = corr_safe(score, y)
                row[f"{ph}_p"] = p
            else:
                r, p = corr_safe(score, y)
                row[f"{ph}_assoc"] = r
                row[f"{ph}_p"] = p

        rows.append(row)

    grad_df = pd.DataFrame(rows)

    # simple ranking score:
    # high if smooth over embedding space, lower if mainly ancestry-driven
    grad_df["gradient_priority_score"] = (
        grad_df[["embedPCA_r2", "umap_r2"]].fillna(0).mean(axis=1)
        - 0.5 * grad_df["r_genoPC1"].abs().fillna(0)
    )

    grad_df = grad_df.sort_values(
        ["gradient_priority_score", "embedPCA_r2", "umap_r2"],
        ascending=[False, False, False]
    )

    grad_df.to_csv(out / "gradient_summary.tsv", sep="\t", index=False)
    grad_df.head(30).to_csv(out / "top_gradient_blocks.tsv", sep="\t", index=False)

    print("\nTop candidate gradient blocks:")
    show_cols = [
        "block", "gradient_priority_score", "embedPCA_r2", "umap_r2",
        "r_genoPC1", "G19B_assoc", "log10eos_assoc", "pctpred_fev1_pre_BD_assoc", "G20D_assoc"
    ]
    print(grad_df[show_cols].head(15).to_string(index=False))

    # plot examples: top overall + priority phenotype blocks
    to_plot = grad_df.head(6)["block"].tolist()
    priority_hits = grad_df[grad_df["is_priority"]].head(6)["block"].tolist()
    to_plot = list(dict.fromkeys(to_plot + priority_hits))

    name_to_idx = {name: i for i, name in enumerate(block_names)}
    for bname in to_plot:
        idx = name_to_idx[bname]
        score = block_pc1[:, idx]

        plot_gradient(
            pca2,
            score,
            title=f"{bname}\nblock_PC1 on subject PCA",
            xlabel=f"Embedding PC1 ({pca_var[0]:.1f}% var)",
            ylabel=f"Embedding PC2 ({pca_var[1]:.1f}% var)",
            out_path=fig_dir / f"{bname}_on_PCA.png",
        )

        if umap2 is not None:
            plot_gradient(
                umap2,
                score,
                title=f"{bname}\nblock_PC1 on subject UMAP",
                xlabel="UMAP1",
                ylabel="UMAP2",
                out_path=fig_dir / f"{bname}_on_UMAP.png",
            )

    print(f"\nSaved:")
    print(f"  {out / 'gradient_summary.tsv'}")
    print(f"  {out / 'top_gradient_blocks.tsv'}")
    print(f"  {fig_dir}")

if __name__ == "__main__":
    main()