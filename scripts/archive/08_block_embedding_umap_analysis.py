#!/usr/bin/env python3
"""
08_block_embedding_umap_analysis.py

Subject-level latent-gradient analysis for LD-block-based subject embeddings.

Builds on the output of phase 2 / block embedding analysis and asks:
- Do subject-level latent gradients (UMAP / PCA / spectral / PHATE) associate with phenotypes?
- Do embedding-space clusters show phenotype enrichment?
- Can a multivariate embedding axis explain a phenotype panel?
- Are phenotypes spatially structured on the embedding manifold?
- Can phenotype-specific supervised gradients (PLS1 / ridge score) reveal
  smoother structure than arbitrary UMAP axes?

Main analyses
-------------
1) Regress phenotypes on UMAP1 / UMAP2
2) Plot UMAP coloured by phenotypes
3) Cluster subject embeddings and compare phenotypes across clusters
4) Compute alternative gradients: PCA, spectral embedding, optional PHATE
5) Partial Least Squares (PLS): embeddings -> phenotype panel
5b) Phenotype-specific supervised gradients:
      - phenotype-specific PLS1 score
      - ridge/logistic predicted phenotype score
6) Neighborhood enrichment / Moran-like spatial autocorrelation on manifold

Inputs
------
- individual_embeddings.npy
- pooling_attention_weights.csv   (for subject IID order)
- ldpruned_997subs.eigenvec
- COS_TRIO_pheno_1165.csv

Outputs
-------
results/output_regions2/ORD/umap_analysis/
  subject_embedding_pca.tsv
  subject_embedding_umap.tsv
  subject_embedding_spectral.tsv
  subject_embedding_phate.tsv                 (if available)
  phenotype_umap_regressions.tsv
  phenotype_gradient_regressions.tsv
  phenotype_specific_supervised_scores.tsv
  phenotype_specific_supervised_summary.tsv
  phenotype_supervised_gradient_regressions.tsv
  cluster_assignments.tsv
  cluster_phenotype_associations.tsv
  pls_subject_scores.tsv
  pls_feature_loadings.tsv
  pls_phenotype_weights.tsv
  neighborhood_enrichment.tsv
  figures/
    subject_pca_plain.png
    subject_umap_plain.png
    subject_spectral_plain.png
    umap_by_<phenotype>.png
    pca_by_<phenotype>.png
    spectral_by_<phenotype>.png
    cluster_umap_kmeans_k<N>.png
    cluster_umap_hdbscan.png                  (if available)
    pls_component1_by_<phenotype>.png
    umap_by_PLS1_<phenotype>.png
    umap_by_RIDGE_<phenotype>.png
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency, kruskal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Defaults / paths
# ──────────────────────────────────────────────────────────────────────────────
EMB_SUBJ_NPY  = "results/output_regions2/ORD/embeddings/individual_embeddings.npy"
ATTN_CSV      = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
EIGENVEC_FILE = "metadata/ldpruned_997subs.eigenvec"
PHENO_FILE    = "metadata/COS_TRIO_pheno_1165.csv"
OUT_DIR       = "results/output_regions2/ORD/umap_analysis"

PC_COLS = [f"PC{i}" for i in range(1, 11)]

CONTINUOUS_PHENOS = [
    "BMI",
    "age",
    "G19B",
    "log10eos",
    "pctpred_fev1_pre_BD",
]
CATEGORICAL_PHENOS = [
    "Hospitalized_Asthma_Last_Yr",
    "smkexp_current",
    "G20D",
]
ALL_PHENOS = CONTINUOUS_PHENOS + CATEGORICAL_PHENOS

SUPERVISED_GRADIENT_PHENOS = [
    "G19B",
    "G20D",
    "log10eos",
    "pctpred_fev1_pre_BD",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


def load_eigenvec(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lstrip("#")
    df = df.rename(columns={"FID": "FID_ev"})
    df["IID"] = df["IID"].astype(str)
    return df[["IID"] + PC_COLS]


def normalize_gender(series: pd.Series) -> pd.Series:
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str).str.strip().str.lower()
    mapping = {
        "male": 1.0, "m": 1.0, "1": 1.0,
        "female": 0.0, "f": 0.0, "2": 0.0,
    }
    return s.map(mapping)


def recode_categorical(pheno: pd.DataFrame) -> pd.DataFrame:
    df = pheno.copy()

    for col in ["Hospitalized_Asthma_Last_Yr", "smkexp_current"]:
        if col not in df.columns:
            continue
        mask = df[col].isin([1.0, 2.0])
        df.loc[~mask, col] = np.nan
        df[col] = df[col].map({1.0: 0.0, 2.0: 1.0})

    if "G20D" in df.columns:
        orig = pd.to_numeric(df["G20D"], errors="coerce")
        df["G20D"] = (orig > 0).astype(float)
        df.loc[orig.isna(), "G20D"] = np.nan

    return df


def get_covars(pheno: str, available_cols):
    covars = PC_COLS.copy()

    if pheno == "age":
        covars += ["gender_num"]
    elif pheno == "BMI":
        covars += ["gender_num"]
    elif pheno == "G19B":
        covars += ["gender_num"]
    elif pheno == "log10eos":
        covars += ["age", "gender_num"]
    elif pheno == "pctpred_fev1_pre_BD":
        covars += ["gender_num", "smkexp_current"]
    elif pheno in ["Hospitalized_Asthma_Last_Yr", "G20D"]:
        covars += ["age", "gender_num", "smkexp_current"]
    elif pheno == "smkexp_current":
        covars += ["age", "gender_num"]

    return [c for c in covars if c in available_cols]


def run_ols(formula: str, data: pd.DataFrame):
    try:
        sub = data.dropna()
        if len(sub) < 20:
            return None
        return smf.ols(formula, data=sub).fit()
    except Exception:
        return None


def run_logit(formula: str, data: pd.DataFrame):
    try:
        sub = data.dropna()
        y_name = formula.split("~")[0].strip()
        if len(sub) < 20:
            return None
        if sub[y_name].nunique() < 2:
            return None
        return smf.logit(formula, data=sub).fit(disp=False, maxiter=200)
    except Exception:
        return None


def extract_model_stat(res, term_name: str):
    if res is None:
        return {
            "beta": np.nan,
            "pval": np.nan,
            "r2": np.nan,
            "n": np.nan,
        }

    r2 = getattr(res, "rsquared", None)
    if r2 is None:
        r2 = getattr(res, "prsquared", np.nan)

    return {
        "beta": float(res.params.get(term_name, np.nan)),
        "pval": float(res.pvalues.get(term_name, np.nan)),
        "r2": float(r2) if pd.notna(r2) else np.nan,
        "n": int(len(res.model.data.frame)),
    }


def compute_umap(Z: np.ndarray, n_neighbors: int = 20, min_dist: float = 0.1):
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(Z)
    except Exception:
        print("  [warn] UMAP unavailable; skipping UMAP.")
        return None


def compute_phate(Z: np.ndarray):
    try:
        import phate
        op = phate.PHATE(random_state=42, n_jobs=1)
        return op.fit_transform(Z)
    except Exception:
        print("  [warn] PHATE unavailable; skipping PHATE.")
        return None


def compute_spectral(Z: np.ndarray, n_components: int = 2, n_neighbors: int = 20):
    try:
        emb = SpectralEmbedding(
            n_components=n_components,
            n_neighbors=n_neighbors,
            affinity="nearest_neighbors",
            random_state=42,
        )
        return emb.fit_transform(Z)
    except Exception:
        print("  [warn] Spectral embedding failed; skipping.")
        return None


def save_coords(iids, coords, colnames, out_path: Path):
    if coords is None:
        return
    df = pd.DataFrame(coords, columns=colnames)
    df.insert(0, "IID", iids)
    df.to_csv(out_path, sep="\t", index=False)


def plot_plain_scatter(X2d, title, out_path, xlabel, ylabel):
    if X2d is None:
        return
    if X2d.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X2d[:, 0], X2d[:, 1], s=14, alpha=0.7, color="#4c78a8")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_continuous_embedding(coords, vals, title, out_path, xlabel, ylabel, cbar_label):
    if coords is None or coords.shape[1] < 2:
        return
    sub = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "val": vals,
    }).dropna()
    if len(sub) < 20:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(sub["x"], sub["y"], c=sub["val"], cmap="viridis", s=16, alpha=0.8)
    plt.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_binary_embedding(coords, vals, title, out_path, xlabel, ylabel, label):
    if coords is None or coords.shape[1] < 2:
        return
    sub = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "val": vals,
    }).dropna()
    if len(sub) < 20 or sub["val"].nunique() < 2:
        return

    palette = {0.0: "#4c78a8", 1.0: "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for grp, gdf in sub.groupby("val"):
        ax.scatter(
            gdf["x"], gdf["y"],
            s=18, alpha=0.75,
            color=palette.get(grp, "#999999"),
            label=f"{label}={int(grp)}",
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_cluster_embedding(coords, clusters, title, out_path, xlabel, ylabel):
    if coords is None or coords.shape[1] < 2:
        return
    sub = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": clusters,
    }).dropna()
    if len(sub) < 20:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    uniq = sorted(sub["cluster"].unique())
    palette = sns.color_palette("tab10", n_colors=max(len(uniq), 3))
    cmap = {c: palette[i % len(palette)] for i, c in enumerate(uniq)}

    for c, gdf in sub.groupby("cluster"):
        ax.scatter(
            gdf["x"], gdf["y"],
            s=18, alpha=0.8,
            color=cmap[c],
            label=f"cluster {c}",
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def add_bh_correction(df: pd.DataFrame, pval_col: str, qval_col: str) -> pd.DataFrame:
    df = df.copy()
    valid = df[pval_col].notna()
    q = np.full(len(df), np.nan)
    if valid.sum() > 0:
        _, q_vals, _, _ = multipletests(df.loc[valid, pval_col].values, method="fdr_bh")
        q[valid.values] = q_vals
    df[qval_col] = q
    return df


def fit_gradient_models(df: pd.DataFrame, grad_cols, pheno_cols_present):
    rows = []

    for pheno in pheno_cols_present:
        safe_p = safe_name(pheno)
        reg_df = df.copy()
        reg_df[safe_p] = reg_df[pheno]

        covars = get_covars(pheno, reg_df.columns)
        is_cat = pheno in CATEGORICAL_PHENOS
        fit_fn = run_logit if is_cat else run_ols

        for g in grad_cols:
            safe_g = safe_name(g)
            reg_df[safe_g] = reg_df[g]

            res_u = fit_fn(f"{safe_p} ~ {safe_g}", reg_df)

            covar_formula = " + ".join(safe_name(c) for c in covars)
            if covar_formula:
                for c in covars:
                    sc = safe_name(c)
                    if sc not in reg_df.columns:
                        reg_df[sc] = reg_df[c]
                res_a = fit_fn(f"{safe_p} ~ {safe_g} + {covar_formula}", reg_df)
            else:
                res_a = res_u

            u = extract_model_stat(res_u, safe_g)
            a = extract_model_stat(res_a, safe_g)

            rows.append({
                "phenotype": pheno,
                "gradient": g,
                "model_type": "logit" if is_cat else "OLS",
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

    result = pd.DataFrame(rows)
    result = add_bh_correction(result, "pval_unadj", "qval_unadj")
    result = add_bh_correction(result, "pval_adj", "qval_adj")
    return result


def choose_best_kmeans(Z: np.ndarray, k_values=(2, 3, 4, 5, 6)):
    best = None
    X = StandardScaler().fit_transform(Z)

    for k in k_values:
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10, init="k-means++")
            labels = km.fit_predict(X)
            if len(np.unique(labels)) < 2:
                continue
            sil = silhouette_score(X, labels)
            row = {"k": k, "silhouette": sil, "labels": labels}
            if best is None or sil > best["silhouette"]:
                best = row
        except Exception:
            continue
    return best


def run_hdbscan(Z: np.ndarray):
    try:
        import hdbscan
        X = StandardScaler().fit_transform(Z)
        model = hdbscan.HDBSCAN(min_cluster_size=30)
        labels = model.fit_predict(X)
        return labels
    except Exception:
        print("  [warn] HDBSCAN unavailable; skipping.")
        return None


def cluster_pheno_association(df: pd.DataFrame, cluster_col: str, pheno_cols_present):
    rows = []

    if cluster_col not in df.columns:
        print(f"  [warn] cluster column '{cluster_col}' missing from master; skipping.")
        return pd.DataFrame()

    for pheno in pheno_cols_present:
        sub = df[[cluster_col, pheno]].dropna().copy()
        if len(sub) < 20 or sub[cluster_col].nunique() < 2:
            continue

        if pheno in CONTINUOUS_PHENOS or pheno == "gender_num":
            groups = [g[pheno].values for _, g in sub.groupby(cluster_col)]
            if any(len(x) < 5 for x in groups):
                continue
            try:
                stat, pval = kruskal(*groups)
            except Exception:
                stat, pval = np.nan, np.nan
            means = sub.groupby(cluster_col)[pheno].mean().to_dict()
            ns = sub.groupby(cluster_col).size().to_dict()
            rows.append({
                "cluster_method": cluster_col,
                "phenotype": pheno,
                "test": "kruskal",
                "stat": stat,
                "pval": pval,
                "cluster_means": str(means),
                "cluster_ns": str(ns),
            })
        else:
            tab = pd.crosstab(sub[cluster_col], sub[pheno])
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                continue
            try:
                chi2, pval, _, _ = chi2_contingency(tab)
            except Exception:
                chi2, pval = np.nan, np.nan
            prop1 = sub.groupby(cluster_col)[pheno].mean().to_dict()
            ns = sub.groupby(cluster_col).size().to_dict()
            rows.append({
                "cluster_method": cluster_col,
                "phenotype": pheno,
                "test": "chi2",
                "stat": chi2,
                "pval": pval,
                "cluster_means": str(prop1),
                "cluster_ns": str(ns),
            })

    return pd.DataFrame(rows)


def make_pheno_matrix(master: pd.DataFrame, pheno_cols_present):
    cols = []
    tmp = master[["IID"]].copy()

    for ph in pheno_cols_present:
        if ph not in master.columns:
            continue
        tmp[ph] = pd.to_numeric(master[ph], errors="coerce")
        cols.append(ph)

    sub = tmp.dropna(subset=cols)
    if len(sub) < 50 or len(cols) < 2:
        return None, None, None

    X = sub[cols].copy()
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.shape[1] < 2:
        return None, None, None

    return sub["IID"].values, X.values, list(X.columns)


def run_pls_embedding_to_phenos(emb_subj: np.ndarray, master: pd.DataFrame, pheno_cols_present):
    iid_keep, Y, y_cols = make_pheno_matrix(master, pheno_cols_present)
    if iid_keep is None:
        return None, None, None

    iid_to_idx = {iid: i for i, iid in enumerate(master["IID"].values)}
    row_idx = [iid_to_idx[iid] for iid in iid_keep if iid in iid_to_idx]

    if len(row_idx) != len(iid_keep):
        print("  [warn] Some IIDs in phenotype matrix not found in master; dropping.")
        mask = np.array([iid in iid_to_idx for iid in iid_keep])
        iid_keep = iid_keep[mask]
        Y = Y[mask]
        row_idx = [iid_to_idx[iid] for iid in iid_keep]

    X = emb_subj[row_idx]

    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)

    n_comp = min(2, Ys.shape[1], Xs.shape[1])
    pls = PLSRegression(n_components=n_comp)
    pls.fit(Xs, Ys)

    x_scores = pls.x_scores_
    x_weights = pls.x_weights_

    scores_df = pd.DataFrame(x_scores, columns=[f"PLS{i+1}" for i in range(n_comp)])
    scores_df.insert(0, "IID", iid_keep)

    loadings_df = pd.DataFrame(
        x_weights,
        columns=[f"PLS{i+1}_weight" for i in range(n_comp)]
    )
    loadings_df.insert(0, "embedding_dim", [f"dim_{i}" for i in range(x_weights.shape[0])])

    yload_df = pd.DataFrame({
        "phenotype": y_cols,
        **{f"PLS{i+1}_y_weight": pls.y_weights_[:, i] for i in range(n_comp)}
    })

    return scores_df, loadings_df, yload_df


def build_supervised_gradient_scores(emb_subj: np.ndarray, master: pd.DataFrame, phenotypes):
    rows = []
    summary_rows = []

    X_all = emb_subj.astype(float)

    for ph in phenotypes:
        if ph not in master.columns:
            continue

        y_all = pd.to_numeric(master[ph], errors="coerce").values
        keep = ~np.isnan(y_all)
        if keep.sum() < 20:
            print(f"[gradient] skipping {ph}: only {keep.sum()} non-missing samples")
            continue

        X = X_all[keep]
        y = y_all[keep]

        # decide phenotype type
        uniq = np.unique(y[~np.isnan(y)])
        is_binary = len(uniq) <= 2 and set(np.round(uniq).astype(int)).issubset({0, 1})

        # standardize embeddings
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # ------------------------------------------------------------
        # 1. PLS1 score
        # ------------------------------------------------------------
        try:
            pls = PLSRegression(n_components=1)
            pls.fit(Xs, y)

            pls_score = cross_val_predict(pls, Xs, y, cv=5).ravel()

            if is_binary:
                pls_assoc = roc_auc_score(y, pls_score)
                pls_metric = "AUC"
            else:
                pls_assoc = np.corrcoef(pls_score, y)[0, 1]
                pls_metric = "r"

            print(f"[gradient] {ph} PLS1 {pls_metric}={pls_assoc:.3f}")

        except Exception as e:
            print(f"[gradient] {ph} PLS1 failed: {e}")
            pls_score = np.full(len(y), np.nan)
            pls_assoc = np.nan
            pls_metric = "r"

        # ------------------------------------------------------------
        # 2. Ridge predicted score
        # ------------------------------------------------------------
        try:
            if is_binary:
                model = LogisticRegressionCV(
                    Cs=10,
                    cv=5,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    scoring="roc_auc",
                )
                model.fit(Xs, y.astype(int))
                ridge_score = cross_val_predict(model, Xs, y.astype(int), cv=5, method="predict_proba")[:, 1]
                ridge_assoc = roc_auc_score(y, ridge_score)
                ridge_metric = "AUC"

            else:
                model = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5)
                model.fit(Xs, y)
                ridge_score = cross_val_predict(model, Xs, y, cv=5)
                ridge_assoc = np.corrcoef(ridge_score, y)[0, 1]
                ridge_metric = "r"

            print(f"[gradient] {ph} Ridge {ridge_metric}={ridge_assoc:.3f}")

        except Exception as e:
            print(f"[gradient] {ph} Ridge failed: {e}")
            ridge_score = np.full(len(y), np.nan)
            ridge_assoc = np.nan
            ridge_metric = "r"

        # ------------------------------------------------------------
        # save subject-level rows
        # ------------------------------------------------------------
        keep_idx = np.where(keep)[0]
        tmp = master.iloc[keep_idx][["subject_id"]].copy()
        tmp["phenotype"] = ph
        tmp["value"] = y
        tmp["pls1_score"] = pls_score
        tmp["ridge_score"] = ridge_score

        rows.append(tmp)

        # ------------------------------------------------------------
        # summary table
        # ------------------------------------------------------------
        summary_rows.append({
            "phenotype": ph,
            "n": keep.sum(),
            "type": "binary" if is_binary else "continuous",
            "pls_metric": pls_metric,
            "pls_assoc": pls_assoc,
            "ridge_metric": ridge_metric,
            "ridge_assoc": ridge_assoc,
        })

    grad_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    return grad_df, summary_df


def main():
    out_dir = Path(OUT_DIR)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("[load] embeddings")
    emb_subj = np.load(EMB_SUBJ_NPY)

    # if embeddings are N x B x D, flatten to N x (B*D)
    if emb_subj.ndim == 3:
        emb_subj = emb_subj.reshape(emb_subj.shape[0], -1)

    print(f"[load] emb shape = {emb_subj.shape}")

    print("[load] subject order")
    attn = pd.read_csv(ATTN_CSV)
    iid_col = "IID" if "IID" in attn.columns else attn.columns[0]
    iids = attn[iid_col].astype(str).values

    print("[load] phenotype + PCs")
    pheno = pd.read_csv(PHENO_FILE)
    pheno = pheno.rename(columns={"S_SUBJECTID": "IID"})
    pheno["IID"] = pheno["IID"].astype(str)

    eigen = load_eigenvec(EIGENVEC_FILE)

    master = pd.merge(pheno, eigen, on="IID", how="left")
    master = recode_categorical(master)

    master = master.set_index("IID").reindex(iids).reset_index()

    if "gender" in master.columns:
        master["gender_num"] = normalize_gender(master["gender"])

    master["subject_id"] = master["IID"]

    # ------------------------------------------------------------
    # PCA / UMAP / spectral
    # ------------------------------------------------------------
    Xs = StandardScaler().fit_transform(emb_subj)

    pca = PCA(n_components=10, random_state=42)
    pca_coords = pca.fit_transform(Xs)

    umap_coords = compute_umap(Xs)
    spectral_coords = compute_spectral(Xs)

    save_coords(iids, pca_coords[:, :2], ["PC1", "PC2"], out_dir / "subject_embedding_pca.tsv")
    save_coords(iids, umap_coords, ["UMAP1", "UMAP2"], out_dir / "subject_embedding_umap.tsv")
    save_coords(iids, spectral_coords, ["SPEC1", "SPEC2"], out_dir / "subject_embedding_spectral.tsv")

    # ------------------------------------------------------------
    # Supervised phenotype-specific gradients
    # ------------------------------------------------------------
    grad_df, grad_summary = build_supervised_gradient_scores(
        emb_subj=emb_subj,
        master=master,
        phenotypes=SUPERVISED_GRADIENT_PHENOS,
    )

    grad_df.to_csv(
        out_dir / "phenotype_specific_supervised_scores.tsv",
        sep="\t",
        index=False,
    )

    grad_summary.to_csv(
        out_dir / "phenotype_specific_supervised_summary.tsv",
        sep="\t",
        index=False,
    )

    print("\n[summary]")
    print(grad_summary)

    # ------------------------------------------------------------
    # Plot UMAP colored by supervised scores
    # ------------------------------------------------------------
    if umap_coords is not None and not grad_df.empty:
        for ph in grad_df["phenotype"].unique():
            sub = grad_df[grad_df["phenotype"] == ph].copy()

            idx = master["IID"].isin(sub["subject_id"])
            coords = umap_coords[idx.values]

            plot_continuous_embedding(
                coords,
                sub["pls1_score"].values,
                title=f"UMAP colored by PLS1 score: {ph}",
                out_path=fig_dir / f"umap_by_PLS1_{safe_name(ph)}.png",
                xlabel="UMAP1",
                ylabel="UMAP2",
                cbar_label="PLS1 score",
            )

            plot_continuous_embedding(
                coords,
                sub["ridge_score"].values,
                title=f"UMAP colored by Ridge score: {ph}",
                out_path=fig_dir / f"umap_by_RIDGE_{safe_name(ph)}.png",
                xlabel="UMAP1",
                ylabel="UMAP2",
                cbar_label="Predicted score",
            )

    print(f"\n[done] results written to: {out_dir}")


if __name__ == "__main__":
    main()