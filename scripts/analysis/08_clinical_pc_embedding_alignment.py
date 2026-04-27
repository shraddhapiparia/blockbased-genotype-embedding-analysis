#!/usr/bin/env python3
"""
08_clinical_pc_embedding_alignment.py

Tests alignment between Phase 2 subject embeddings (ORD loss) and CRA clinical
PCs (2022-07-21_Endotype_CRA.tsv).

Steps
-----
1. Load Phase 2 subject embeddings → compute embedding PCs (PCA).
2. Merge embeddings + CRA clinical PCs + genotype PCs by subject ID (ST-XXXXXXXX).
3. Pearson and Spearman correlations: embedding PCs vs clinical PCs + FDR.
4. OLS regression:
     clinical_PC ~ embedding PCs          (embedding-only)
     clinical_PC ~ embedding PCs + gPC1..gPC10  (adjusted)
5. Ridge regression with 5-fold CV → CV R².
6. Plots: correlation heatmap, top-association scatterplots, UMAP coloured by
   clinical PCs.

Outputs  (results/analysis/clinical_pc_embedding_alignment_cra/)
--------
merged_table.tsv
correlation_matrix.tsv
regression_results.tsv
ridge_regression_results.tsv
correlation_heatmap.png            (Pearson, backward-compat copy)
correlation_heatmap_pearson.png
correlation_heatmap_spearman.png
scatter_top_associations.png
umap_by_clinical_pc.png
"""

import argparse
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import umap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_EMBEDDINGS = "results/output_regions2/ORD/embeddings/individual_embeddings.csv"
DEFAULT_CLINICAL   = "metadata/2022-07-21_Endotype_CRA.tsv"
DEFAULT_EIGENVEC   = "metadata/ldpruned_997subs.eigenvec"
DEFAULT_OUT        = "results/analysis/clinical_pc_embedding_alignment_cra"

CLINICAL_PC_COLS = [
    "PC1_12.626%_zscore",
    "PC2_10.416%_zscore",
    "PC3_7.8591%_zscore",
]
CLINICAL_PC_LABELS = ["clin_PC1", "clin_PC2", "clin_PC3"]
GENO_PC_COLS = [f"PC{i}" for i in range(1, 11)]
N_EMB_PCS = 10   # how many embedding PCs to retain for analysis


# ── loaders ───────────────────────────────────────────────────────────────────
def load_embeddings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "IID" not in df.columns:
        raise ValueError(f"Expected 'IID' column in {path}. Got: {list(df.columns[:5])}")
    df["IID"] = df["IID"].astype(str).str.strip()
    print(f"[embeddings] loaded {len(df)} subjects, {df.shape[1]-1} dims  ({path})")
    return df


def load_clinical(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "S_SUBJECTID" not in df.columns:
        raise ValueError(
            f"Expected 'S_SUBJECTID' in {path}. Got: {list(df.columns)}"
        )
    df = df.rename(columns={"S_SUBJECTID": "IID"})
    df["IID"] = df["IID"].astype(str).str.strip()
    missing_cols = [c for c in CLINICAL_PC_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing clinical PC columns: {missing_cols}")
    n_dup = df["IID"].duplicated().sum()
    if n_dup:
        raise ValueError(f"Duplicate IIDs in clinical file: {n_dup}")
    for old, new in zip(CLINICAL_PC_COLS, CLINICAL_PC_LABELS):
        df = df.rename(columns={old: new})
    df = df[["IID"] + CLINICAL_PC_LABELS]
    for label in CLINICAL_PC_LABELS:
        n_na = df[label].isna().sum()
        if n_na:
            print(f"[clinical]   NA in {label}: {n_na} — will be dropped at merge")
    n_before = len(df)
    df = df.dropna(subset=CLINICAL_PC_LABELS)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[clinical]   dropped {n_dropped} rows with missing clinical PCs")
    print(f"[clinical]   loaded {len(df)} subjects with complete CRA clinical PCs  ({path})")
    return df


def load_eigenvec(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lstrip("#")
    df = df.rename(columns={"FID": "FID_ev"})
    df["IID"] = df["IID"].astype(str).str.strip()
    print(f"[eigenvec]   loaded {len(df)} subjects, {len(GENO_PC_COLS)} geno PCs  ({path})")
    return df[["IID"] + GENO_PC_COLS]


# ── merge ─────────────────────────────────────────────────────────────────────
def build_merged(emb_df: pd.DataFrame, clin_df: pd.DataFrame,
                 geno_df: pd.DataFrame) -> pd.DataFrame:
    emb_cols = [c for c in emb_df.columns if c != "IID"]
    emb_ids  = set(emb_df["IID"])
    clin_ids = set(clin_df["IID"])
    geno_ids = set(geno_df["IID"])
    n_ec  = len(emb_ids & clin_ids)
    n_eg  = len(emb_ids & geno_ids)
    n_cg  = len(clin_ids & geno_ids)
    n_all = len(emb_ids & clin_ids & geno_ids)
    print(f"[merge]      embeddings∩CRA_clinical={n_ec}  "
          f"embeddings∩eigenvec={n_eg}  "
          f"CRA_clinical∩eigenvec={n_cg}  "
          f"all three={n_all}")
    if n_ec == 0:
        print("[merge]  --- DIAGNOSTIC IID SAMPLES ---")
        print(f"  emb  IID sample : {sorted(emb_ids)[:5]}")
        print(f"  clin IID sample : {sorted(clin_ids)[:5]}")
        raise RuntimeError(
            "embeddings ∩ CRA_clinical = 0.\n"
            "  The CRA clinical file and the embedding file share no subject IDs.\n"
            "  Verify that --clinical points to the correct CRA file for this cohort."
        )
    merged = (
        emb_df
        .merge(clin_df, on="IID", how="inner")
        .merge(geno_df, on="IID", how="inner")
    )
    print(f"[merge]      final N (inner join, all three) = {len(merged)}")
    n_dup = merged["IID"].duplicated().sum()
    if n_dup:
        raise RuntimeError(f"Duplicate IIDs in merged table ({n_dup}). Check input files.")
    return merged, emb_cols


# ── embedding PCA ─────────────────────────────────────────────────────────────
def compute_embedding_pcs(merged: pd.DataFrame, emb_cols: list,
                          n_components: int = N_EMB_PCS):
    X = merged[emb_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)
    emb_pc_cols = [f"emb_PC{i+1}" for i in range(n_components)]
    pc_df = pd.DataFrame(scores, columns=emb_pc_cols, index=merged.index)
    print(f"[emb PCA]    top-{n_components} PCs explain "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% of embedding variance")
    return pc_df, emb_pc_cols, pca


# ── correlations ──────────────────────────────────────────────────────────────
def compute_correlations(merged: pd.DataFrame, emb_pc_cols: list) -> pd.DataFrame:
    records = []
    for cpc in CLINICAL_PC_LABELS:
        y = merged[cpc].values
        for epc in emb_pc_cols:
            x = merged[epc].values
            mask = np.isfinite(x) & np.isfinite(y)
            n = mask.sum()
            if n < 10:
                continue
            if x[mask].var() == 0 or y[mask].var() == 0:
                r_p, p_p, r_s, p_s = np.nan, np.nan, np.nan, np.nan
            else:
                r_p, p_p = stats.pearsonr(x[mask], y[mask])
                r_s, p_s = stats.spearmanr(x[mask], y[mask])
            records.append(dict(
                clinical_pc=cpc, emb_pc=epc,
                pearson_r=r_p, pearson_p=p_p,
                spearman_r=r_s, spearman_p=p_s,
                n=n,
            ))
    df = pd.DataFrame(records)

    def _fdr(pvals):
        valid = np.isfinite(pvals)
        q = np.full(len(pvals), np.nan)
        if valid.sum() > 0:
            _, q[valid], _, _ = multipletests(pvals[valid], method="fdr_bh")
        return q

    # pooled FDR across all clinical_PC × emb_PC tests
    df["pearson_q"]  = _fdr(df["pearson_p"].values)
    df["spearman_q"] = _fdr(df["spearman_p"].values)

    # per-clinical-PC FDR (correcting only within each clinical_pc group)
    pq_by = np.full(len(df), np.nan)
    sq_by = np.full(len(df), np.nan)
    for _, grp in df.groupby("clinical_pc"):
        idx = grp.index.values
        pq_by[idx] = _fdr(grp["pearson_p"].values)
        sq_by[idx] = _fdr(grp["spearman_p"].values)
    df["pearson_q_by_clinical_pc"]  = pq_by
    df["spearman_q_by_clinical_pc"] = sq_by

    return df


def _sort_emb_pc_cols(cols):
    return sorted(cols, key=lambda c: int(c.replace("emb_PC", "")))


def pivot_corr_matrix(corr_df: pd.DataFrame, value_col: str = "pearson_r") -> pd.DataFrame:
    pivoted = corr_df.pivot(index="clinical_pc", columns="emb_pc", values=value_col)
    ordered = _sort_emb_pc_cols([c for c in pivoted.columns if c.startswith("emb_PC")])
    return pivoted[ordered]


# ── OLS regression ────────────────────────────────────────────────────────────
def run_ols(merged: pd.DataFrame, emb_pc_cols: list) -> pd.DataFrame:
    records = []
    predictors_geno = " + ".join(GENO_PC_COLS)
    predictors_emb  = " + ".join(emb_pc_cols)
    predictors_full = predictors_emb + " + " + predictors_geno

    for cpc in CLINICAL_PC_LABELS:
        y_col_safe = cpc.replace(" ", "_").replace("%", "pct")
        tmp = merged[[cpc] + emb_pc_cols + GENO_PC_COLS].dropna().copy()
        tmp = tmp.rename(columns={cpc: y_col_safe})

        for label, formula_rhs in [
            ("ancestry_pcs",   predictors_geno),
            ("emb_only",    predictors_emb),
            ("emb+genoPCs", predictors_full),
        ]:
            formula = f"{y_col_safe} ~ {formula_rhs}"
            try:
                res = smf.ols(formula, data=tmp).fit()
                records.append(dict(
                    clinical_pc=cpc,
                    model=label,
                    n=int(res.nobs),
                    r2=res.rsquared,
                    adj_r2=res.rsquared_adj,
                    f_pvalue=res.f_pvalue,
                    aic=res.aic,
                ))
            except Exception as e:
                print(f"  [OLS warning] {cpc} / {label}: {e}")

    return pd.DataFrame(records)


# ── ridge regression ──────────────────────────────────────────────────────────
def run_ridge(merged: pd.DataFrame, emb_pc_cols: list) -> pd.DataFrame:
    records = []
    alphas = np.logspace(-3, 4, 50)

    for cpc in CLINICAL_PC_LABELS:
        tmp = merged[[cpc] + emb_pc_cols + GENO_PC_COLS].dropna()
        y = tmp[cpc].values

        for label, x_cols in [
            ("ancestry_pcs",   GENO_PC_COLS),
            ("emb_only",    emb_pc_cols),
            ("emb+genoPCs", emb_pc_cols + GENO_PC_COLS),
        ]:
            X = tmp[x_cols].values
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge",  RidgeCV(alphas=alphas, scoring="r2", cv=5)),
            ])
            cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
            pipe.fit(X, y)
            best_alpha = pipe.named_steps["ridge"].alpha_
            records.append(dict(
                clinical_pc=cpc,
                model=label,
                n=len(y),
                cv_r2_mean=cv_scores.mean(),
                cv_r2_std=cv_scores.std(),
                best_alpha=best_alpha,
            ))
            print(f"  [Ridge] {cpc} / {label}: CV R²={cv_scores.mean():.3f} "
                  f"(±{cv_scores.std():.3f}), α={best_alpha:.4g}")

    return pd.DataFrame(records)


# ── plots ─────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(corr_df: pd.DataFrame, out_path: Path,
                             method: str = "pearson"):
    r_col = f"{method}_r"
    q_col = f"{method}_q"
    cbar_label  = "Pearson r"   if method == "pearson" else "Spearman ρ"
    method_label = "Pearson r"  if method == "pearson" else "Spearman ρ"

    pivot_r = pivot_corr_matrix(corr_df, r_col)
    pivot_q = pivot_corr_matrix(corr_df, q_col)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot_r.columns) * 0.55), 4))
    sns.heatmap(
        pivot_r, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        linewidths=0.4, ax=ax,
        cbar_kws={"label": cbar_label},
    )
    for i, row in enumerate(pivot_r.index):
        for j, col in enumerate(pivot_r.columns):
            q = pivot_q.loc[row, col] if (row in pivot_q.index and col in pivot_q.columns) else np.nan
            if pd.notna(q):
                if q < 0.05:
                    ax.text(j + 0.5, i + 0.85, "*", ha="center", va="center",
                            fontsize=10, color="black")
                elif q < 0.10:
                    ax.text(j + 0.5, i + 0.85, "†", ha="center", va="center",
                            fontsize=10, color="black")
    ax.set_title(f"Embedding PCs vs CRA Clinical PCs — {method_label}")
    ax.set_xlabel("Embedding PC  (* = pooled FDR < 0.05; † = pooled FDR < 0.10)")
    ax.set_ylabel("Clinical PC")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_top_scatterplots(merged: pd.DataFrame, corr_df: pd.DataFrame,
                          emb_pc_cols: list, out_path: Path, top_n: int = 6):
    top = (
        corr_df
        .assign(abs_r=corr_df["pearson_r"].abs())
        .sort_values("abs_r", ascending=False)
        .head(top_n)
    )
    ncols = min(3, len(top))
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for k, (_, row) in enumerate(top.iterrows()):
        ax = axes_flat[k]
        x = merged[row["emb_pc"]].values
        y = merged[row["clinical_pc"]].values
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], alpha=0.4, s=12, color="steelblue")
        m, b = np.polyfit(x[mask], y[mask], 1)
        xline = np.linspace(x[mask].min(), x[mask].max(), 200)
        ax.plot(xline, m * xline + b, color="firebrick", lw=1.5)
        ax.set_xlabel(row["emb_pc"])
        ax.set_ylabel(row["clinical_pc"])
        ax.set_title(
            f"r={row['pearson_r']:.3f}, q={row['pearson_q']:.3f}\n"
            f"ρ={row['spearman_r']:.3f}, n={row['n']}"
        )

    for ax in axes_flat[len(top):]:
        ax.set_visible(False)

    fig.suptitle("Top embedding-PC / CRA clinical-PC associations", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_umap_by_clinical_pcs(merged: pd.DataFrame, emb_cols: list, out_path: Path):
    X = StandardScaler().fit_transform(merged[emb_cols].values)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(X)

    ncols = len(CLINICAL_PC_LABELS)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), squeeze=False)
    for j, (cpc, label) in enumerate(zip(CLINICAL_PC_LABELS, CLINICAL_PC_LABELS)):
        ax = axes[0, j]
        c = merged[cpc].values
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=c, cmap="RdBu_r", alpha=0.6, s=12)
        plt.colorbar(sc, ax=ax, label=label)
        ax.set_title(f"UMAP coloured by {label}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    fig.suptitle("Subject embedding UMAP — coloured by CRA clinical PCs", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS,
                   help="Phase 2 individual_embeddings.csv")
    p.add_argument("--clinical",   default=DEFAULT_CLINICAL,
                   help="CRA endotype TSV (2022-07-21_Endotype_CRA.tsv)")
    p.add_argument("--eigenvec",   default=DEFAULT_EIGENVEC,
                   help="PLINK .eigenvec file (top 10 genotype PCs)")
    p.add_argument("--out-dir",    default=DEFAULT_OUT,
                   help="Output directory")
    p.add_argument("--n-emb-pcs", type=int, default=N_EMB_PCS,
                   help="Number of embedding PCs to use (default: 10)")
    p.add_argument("--skip-ridge",  action="store_true",
                   help="Skip ridge regression (faster)")
    p.add_argument("--skip-umap",   action="store_true",
                   help="Skip UMAP projection plot")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
    emb_df  = load_embeddings(args.embeddings)
    clin_df = load_clinical(args.clinical)
    geno_df = load_eigenvec(args.eigenvec)

    # ── merge ─────────────────────────────────────────────────────────────────
    merged, emb_cols = build_merged(emb_df, clin_df, geno_df)

    # ── embedding PCA ─────────────────────────────────────────────────────────
    emb_pc_df, emb_pc_cols, pca_model = compute_embedding_pcs(
        merged, emb_cols, n_components=args.n_emb_pcs
    )
    merged = pd.concat([merged.reset_index(drop=True), emb_pc_df], axis=1)

    # ── save merged table ─────────────────────────────────────────────────────
    keep_cols = (["IID"] + CLINICAL_PC_LABELS + GENO_PC_COLS
                 + emb_pc_cols + emb_cols)
    merged[keep_cols].to_csv(out_dir / "merged_table.tsv", sep="\t", index=False)
    print(f"[output] merged_table.tsv  (N={len(merged)})")

    # ── correlations ──────────────────────────────────────────────────────────
    print("\n--- Pearson / Spearman correlations ---")
    corr_df = compute_correlations(merged, emb_pc_cols)
    corr_df["abs_pearson_r"]  = corr_df["pearson_r"].abs()
    corr_df["abs_spearman_r"] = corr_df["spearman_r"].abs()
    corr_df.to_csv(out_dir / "correlation_matrix.tsv", sep="\t", index=False)
    print("Top Pearson associations:")
    print(corr_df.sort_values("abs_pearson_r", ascending=False).head(10).to_string(index=False))
    print("Top Spearman associations:")
    print(corr_df.sort_values("abs_spearman_r", ascending=False).head(10).to_string(index=False))

    # ── OLS regression ────────────────────────────────────────────────────────
    print("\n--- OLS regression ---")
    ols_df = run_ols(merged, emb_pc_cols)
    ols_df.to_csv(out_dir / "regression_results.tsv", sep="\t", index=False)
    print(ols_df.to_string(index=False))

    # ── ridge regression ──────────────────────────────────────────────────────
    if not args.skip_ridge:
        print("\n--- Ridge regression (5-fold CV) ---")
        ridge_df = run_ridge(merged, emb_pc_cols)
        ridge_df.to_csv(out_dir / "ridge_regression_results.tsv", sep="\t", index=False)
        print(ridge_df.to_string(index=False))

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_correlation_heatmap(corr_df, out_dir / "correlation_heatmap_pearson.png",
                             method="pearson")
    plot_correlation_heatmap(corr_df, out_dir / "correlation_heatmap_spearman.png",
                             method="spearman")
    shutil.copy(out_dir / "correlation_heatmap_pearson.png",
                out_dir / "correlation_heatmap.png")
    print(f"[plot] saved {out_dir / 'correlation_heatmap.png'}  (backward-compat copy)")
    plot_top_scatterplots(merged, corr_df, emb_pc_cols,
                          out_dir / "scatter_top_associations.png")
    if not args.skip_umap:
        plot_umap_by_clinical_pcs(merged, emb_cols, out_dir / "umap_by_clinical_pc.png")

    print(f"\nAll outputs written to: {out_dir}/")


if __name__ == "__main__":
    main()
