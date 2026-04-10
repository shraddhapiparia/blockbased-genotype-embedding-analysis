#!/usr/bin/env python3
"""
12_phenotype_cluster_analysis.py

Goal
----
Test whether subject position in the learned embedding space predicts clinical
phenotype using two parallel approaches:

  A) Cluster-based  — phenotype ~ cluster (0/1/2, KMeans k=3 from script 10)
  B) Continuous     — phenotype ~ embedPC1 / embedPC2 / embedPC3

Phenotypes tested
-----------------
  Continuous  : BMI, G19B, log10eos, pctpred_fev1_pre_BD
  Binary      : smkexp_current, Hospitalized_Asthma_Last_Yr, G20D

Covariates (per phenotype, always include genotype PC1-10)
----------------------------------------------------------
  BMI                        : age, gender_num
  G19B                       : gender_num
  log10eos                   : age, gender_num
  pctpred_fev1_pre_BD        : gender_num, smkexp_current
  Hospitalized_Asthma_Last_Yr: age, gender_num, smkexp_current
  smkexp_current             : age, gender_num
  G20D                       : age, gender_num, smkexp_current

Outputs
-------
results/.../phenotype_analysis/
  phenotype_cluster_results.tsv      — cluster-based tests (A)
  phenotype_continuous_results.tsv   — continuous embedding PC tests (B)
  phenotype_combined_summary.tsv     — one row per phenotype, both approaches
  figures/
    <pheno>_by_cluster.png           — boxplot/bar per phenotype
    <pheno>_vs_embedPC1.png          — scatter + regression line
    volcano_eta2_r2.png              — effect size overview both approaches
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
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────
PHENO_FILE    = "metadata/COS_TRIO_pheno_1165.csv"
EMB_SUBJ_NPY  = "results/output_regions2/ORD/embeddings/individual_embeddings.npy"
ATTN_CSV      = "results/output_regions2/ORD/embeddings/pooling_attention_weights.csv"
EIGENVEC_FILE = "metadata/ldpruned_997subs.eigenvec"
OUT_DIR       = "results/output_regions2/ORD/phenotype_analysis"

PC_COLS = [f"PC{i}" for i in range(1, 11)]

CONTINUOUS_PHENOS = ["BMI", "G19B", "log10eos", "pctpred_fev1_pre_BD", "log10Ige"]
BINARY_PHENOS     = ["smkexp_current", "Hospitalized_Asthma_Last_Yr", "G20D"]
ALL_PHENOS        = CONTINUOUS_PHENOS + BINARY_PHENOS

# Phenotypes for which joint embedPC1+2+3 R² is reported in supplementary
JOINT_PHENOS = ["log10Ige", "log10eos", "G19B"]

EMBED_PCS = ["embedPC1", "embedPC2", "embedPC3"]


# ── covariate map ─────────────────────────────────────────────────────
def get_covars(pheno: str, available_cols: list) -> list:
    base  = PC_COLS.copy()
    extra = {
        "BMI":                         ["age", "gender_num"],
        "G19B":                        ["gender_num"],
        "log10eos":                    ["age", "gender_num"],
        "log10Ige":                    ["age", "gender_num"],
        "pctpred_fev1_pre_BD":         ["gender_num", "smkexp_current"],
        "Hospitalized_Asthma_Last_Yr": ["age", "gender_num", "smkexp_current"],
        "smkexp_current":              ["age", "gender_num"],
        "G20D":                        ["age", "gender_num", "smkexp_current"],
    }
    covars = base + extra.get(pheno, [])
    return [c for c in covars if c in available_cols and c != pheno]


# ── phenotype loading helpers ─────────────────────────────────────────
def normalize_gender(series: pd.Series) -> pd.Series:
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str).str.strip().str.lower()
    return s.map({"male": 1.0, "m": 1.0, "1": 1.0,
                  "female": 0.0, "f": 0.0, "2": 0.0})


def recode_categorical(pheno: pd.DataFrame) -> pd.DataFrame:
    df = pheno.copy()
    for col in ["Hospitalized_Asthma_Last_Yr", "smkexp_current"]:
        if col not in df.columns:
            continue
        mask = df[col].isin([1.0, 2.0])
        df.loc[~mask, col] = np.nan
        df[col] = df[col].map({1.0: 0.0, 2.0: 1.0})
    if "G20D" in df.columns:
        df["G20D"] = pd.to_numeric(df["G20D"], errors="coerce")
        orig_na    = pheno["G20D"].isna()
        df["G20D"] = (df["G20D"] > 0).astype(float)
        df.loc[orig_na, "G20D"] = np.nan
    return df


def load_eigenvec(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lstrip("#")
    df = df.rename(columns={"FID": "FID_ev"})
    df["IID"] = df["IID"].astype(str)
    return df[["IID"] + PC_COLS]


# ── statistical helpers ───────────────────────────────────────────────
def eta_squared(groups):
    groups    = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    all_vals  = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total   = np.sum((all_vals - grand_mean) ** 2)
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def cramers_v(chi2, n, r, k):
    denom = n * max(1, min(r - 1, k - 1))
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan


def _empty_cluster_result(col_type):
    return {
        "col_type": col_type, "n": np.nan,
        "anova_F": np.nan, "anova_p": np.nan,
        "kruskal_H": np.nan, "kruskal_p": np.nan,
        "chi2": np.nan, "chi2_p": np.nan,
        "eta2": np.nan, "cramers_v": np.nan,
        "cluster0_mean": np.nan, "cluster1_mean": np.nan, "cluster2_mean": np.nan,
        "cluster0_median": np.nan, "cluster1_median": np.nan, "cluster2_median": np.nan,
        "cluster0_n": np.nan, "cluster1_n": np.nan, "cluster2_n": np.nan,
    }


def test_cluster(values: np.ndarray, cluster: np.ndarray, col_type: str,
                 covariates: np.ndarray = None) -> dict:
    """
    Test phenotype ~ cluster.
    If covariates supplied, residualise phenotype first then test residuals.
    """
    base_df = pd.DataFrame({"v": values, "c": cluster})
    base_df = base_df.dropna()

    if covariates is not None:
        idx   = base_df.index
        cov   = covariates[idx]
        valid = np.isfinite(cov).all(axis=1)
        base_df = base_df[valid].copy()
        cov     = cov[valid]
        if len(base_df) < 20:
            return _empty_cluster_result(col_type)
        lr = LinearRegression().fit(cov, base_df["v"].values)
        base_df["v"] = base_df["v"].values - lr.predict(cov)

    groups = [g["v"].values for _, g in base_df.groupby("c")]
    result = _empty_cluster_result(col_type)
    result["n"] = len(base_df)

    if len(groups) < 2 or any(len(g) < 5 for g in groups):
        return result

    H, pk = stats.kruskal(*groups)
    result.update({"kruskal_H": float(H), "kruskal_p": float(pk),
                   "eta2": eta_squared(groups)})

    if col_type == "continuous":
        F, p = stats.f_oneway(*groups)
        result.update({"anova_F": float(F), "anova_p": float(p)})
    else:
        try:
            tab = pd.crosstab(base_df["c"], base_df["v"])
            if tab.shape[0] >= 2 and tab.shape[1] >= 2:
                chi2, p, _, _ = stats.chi2_contingency(tab)
                cv = cramers_v(chi2, tab.values.sum(), *tab.shape)
                result.update({"chi2": float(chi2), "chi2_p": float(p),
                                "cramers_v": float(cv)})
        except Exception:
            pass

    for k, g in base_df.groupby("c"):
        result[f"cluster{k}_mean"]   = float(g["v"].mean())
        result[f"cluster{k}_median"] = float(g["v"].median())
        result[f"cluster{k}_n"]      = int(len(g))

    return result


def test_continuous_predictor(y: np.ndarray, x: np.ndarray,
                               covariates: np.ndarray = None) -> dict:
    """
    Partial regression: regress y ~ x after removing shared covariate variance.
    Returns beta, se, t, p, partial_r2, spearman_r.
    """
    mask = np.isfinite(y) & np.isfinite(x)
    if covariates is not None:
        mask &= np.isfinite(covariates).all(axis=1)

    empty = {"n": int(mask.sum()), "beta": np.nan, "se": np.nan,
             "t": np.nan, "p": np.nan, "partial_r2": np.nan,
             "spearman_r": np.nan, "spearman_p": np.nan}

    if mask.sum() < 20:
        return empty

    y_ = y[mask]
    x_ = x[mask]

    sr, sp = stats.spearmanr(x_, y_)
    empty["spearman_r"] = float(sr)
    empty["spearman_p"] = float(sp)

    if covariates is not None:
        cov_ = covariates[mask]
        lr_y = LinearRegression().fit(cov_, y_)
        lr_x = LinearRegression().fit(cov_, x_)
        y_resid = y_ - lr_y.predict(cov_)
        x_resid = x_ - lr_x.predict(cov_)
    else:
        y_resid = y_ - y_.mean()
        x_resid = x_ - x_.mean()

    ss_x = np.sum(x_resid ** 2)
    if ss_x == 0:
        return empty

    beta  = np.sum(x_resid * y_resid) / ss_x
    y_hat = beta * x_resid
    ss_res = np.sum((y_resid - y_hat) ** 2)
    ss_tot = np.sum(y_resid ** 2)

    partial_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    n   = len(y_resid)
    se  = np.sqrt(ss_res / max(n - 2, 1) / ss_x)
    t   = float(beta / se) if se > 0 else np.nan
    p   = float(2 * stats.t.sf(abs(t), df=n - 2)) if np.isfinite(t) else np.nan

    return {"n": n, "beta": float(beta), "se": float(se),
            "t": t, "p": p, "partial_r2": partial_r2,
            "spearman_r": float(sr), "spearman_p": float(sp)}


# ── plotting helpers ──────────────────────────────────────────────────
def plot_by_cluster(df, pheno, col_type, out_path):
    sub = df[["cluster", pheno]].dropna()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    if col_type == "continuous":
        sns.boxplot(data=sub, x="cluster", y=pheno, ax=ax,
                    color="white", width=0.55)
        sns.stripplot(data=sub, x="cluster", y=pheno, ax=ax,
                      alpha=0.4, size=3)
    else:
        prop = (sub.groupby("cluster")[pheno]
                   .value_counts(normalize=True)
                   .unstack(fill_value=0))
        prop.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
        ax.set_ylabel("Proportion")
        ax.legend(title=pheno, bbox_to_anchor=(1.02, 1), loc="upper left")

    means  = sub.groupby("cluster")[pheno].mean().round(3).to_dict()
    counts = sub.groupby("cluster")[pheno].count().to_dict()
    label  = " | ".join(
        [f"C{k}: mean={means.get(k, np.nan):.3f}, n={counts.get(k, 0)}"
         for k in sorted(sub["cluster"].dropna().unique())]
    )
    ax.text(0.02, 0.02, label, transform=ax.transAxes, fontsize=7.5, va="bottom")
    ax.set_title(f"{pheno} by subject cluster", fontsize=10)
    ax.set_xlabel("Cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_scatter_embedpc(df, pheno, embed_col, out_path):
    sub = df[[embed_col, pheno]].dropna()
    if len(sub) < 10:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(sub[embed_col], sub[pheno], alpha=0.5, s=14, color="steelblue")
    x_ = sub[embed_col].values
    y_ = sub[pheno].values
    m, b, r, p, _ = stats.linregress(x_, y_)
    xr = np.linspace(x_.min(), x_.max(), 100)
    ax.plot(xr, m * xr + b, color="red", linewidth=1.5,
            label=f"r={r:.3f}, p={p:.3g}")
    ax.set_xlabel(embed_col)
    ax.set_ylabel(pheno)
    ax.set_title(f"{pheno} vs {embed_col}", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_volcano(summary_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for _, row in summary_df.iterrows():
        eta2 = row.get("cluster_eta2", np.nan)
        kp   = row.get("cluster_kruskal_p", np.nan)
        if not (np.isfinite(eta2) and np.isfinite(kp)):
            continue
        ax.scatter(eta2, -np.log10(kp + 1e-300), s=65, alpha=0.85)
        ax.annotate(row["phenotype"], (eta2, -np.log10(kp + 1e-300)),
                    fontsize=8, ha="left", va="bottom")
    ax.axhline(-np.log10(0.05), color="red", linestyle="--",
               linewidth=0.8, label="p=0.05")
    ax.set_xlabel("η² (cluster, covariate-adjusted)")
    ax.set_ylabel("-log10(Kruskal p)")
    ax.set_title("Cluster-based effect sizes")
    ax.legend(fontsize=8)

    ax = axes[1]
    for _, row in summary_df.iterrows():
        r2 = row.get("embedPC1_partial_r2", np.nan)
        p  = row.get("embedPC1_p", np.nan)
        if not (np.isfinite(r2) and np.isfinite(p)):
            continue
        ax.scatter(r2, -np.log10(p + 1e-300), s=65, alpha=0.85)
        ax.annotate(row["phenotype"], (r2, -np.log10(p + 1e-300)),
                    fontsize=8, ha="left", va="bottom")
    ax.axhline(-np.log10(0.05), color="red", linestyle="--",
               linewidth=0.8, label="p=0.05")
    ax.set_xlabel("Partial R² (embedPC1, covariate-adjusted)")
    ax.set_ylabel("-log10(p)")
    ax.set_title("Continuous embedPC1 effect sizes")
    ax.legend(fontsize=8)

    plt.suptitle("Phenotype ~ embedding: effect size overview", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ── joint embedding R² helper ────────────────────────────────────────
def test_joint_embed(y: np.ndarray, embed_matrix: np.ndarray,
                     covariates: np.ndarray = None) -> dict:
    """
    Regress y ~ embedPC1 + embedPC2 + embedPC3 (+ covariates).
    Reports:
      joint_r2        — R² of full model (covariates + embed PCs)
      covar_r2        — R² of covariate-only model
      incremental_r2  — gain from adding embed PCs (the number to report)
      f_p             — F-test p for incremental fit
    """
    mask = np.isfinite(y) & np.isfinite(embed_matrix).all(axis=1)
    if covariates is not None:
        mask &= np.isfinite(covariates).all(axis=1)

    empty = {"n": int(mask.sum()), "joint_r2": np.nan, "covar_r2": np.nan,
             "incremental_r2": np.nan, "f_stat": np.nan, "f_p": np.nan}
    if mask.sum() < 30:
        return empty

    y_       = y[mask]
    embed_   = embed_matrix[mask]

    if covariates is not None:
        cov_ = covariates[mask]
        X_base = cov_
        X_full = np.hstack([cov_, embed_])
    else:
        X_base = np.ones((len(y_), 1))
        X_full = embed_

    # covariate-only model
    lr_base = LinearRegression().fit(X_base, y_)
    ss_res_base = np.sum((y_ - lr_base.predict(X_base)) ** 2)
    ss_tot      = np.sum((y_ - y_.mean()) ** 2)
    r2_base     = float(1 - ss_res_base / ss_tot) if ss_tot > 0 else np.nan

    # full model
    lr_full = LinearRegression().fit(X_full, y_)
    ss_res_full = np.sum((y_ - lr_full.predict(X_full)) ** 2)
    r2_full     = float(1 - ss_res_full / ss_tot) if ss_tot > 0 else np.nan

    incr_r2 = float(r2_full - r2_base) if (np.isfinite(r2_full) and
                                             np.isfinite(r2_base)) else np.nan

    # partial F-test: (ΔSS/q) / (SS_res_full / df_res)
    n      = len(y_)
    q      = embed_.shape[1]           # 3 embed PCs added
    p_full = X_full.shape[1]
    df_res = n - p_full - 1
    if df_res > 0 and ss_res_full > 0:
        f_stat = ((ss_res_base - ss_res_full) / q) / (ss_res_full / df_res)
        f_p    = float(stats.f.sf(f_stat, dfn=q, dfd=df_res))
    else:
        f_stat, f_p = np.nan, np.nan

    return {"n": n, "joint_r2": r2_full, "covar_r2": r2_base,
            "incremental_r2": incr_r2,
            "f_stat": float(f_stat) if np.isfinite(f_stat) else np.nan,
            "f_p": f_p}


def plot_joint_r2(joint_df: pd.DataFrame, out_path: Path):
    """Bar chart of incremental R² with F-test p annotation."""
    sub = joint_df.dropna(subset=["incremental_r2"]).copy()
    sub = sub.sort_values("incremental_r2", ascending=False)

    fig, ax = plt.subplots(figsize=(max(5, len(sub) * 1.1), 4.5))
    colors = ["#d62728" if row["f_p"] < 0.05 else "#aec7e8"
              for _, row in sub.iterrows()]
    bars = ax.bar(sub["phenotype"], sub["incremental_r2"] * 100,
                  color=colors, edgecolor="white", width=0.6)

    for bar, (_, row) in zip(bars, sub.iterrows()):
        p_label = f"p={row['f_p']:.3f}" if np.isfinite(row["f_p"]) else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                p_label, ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Incremental R² (%) from embedPC1+2+3")
    ax.set_xlabel("Phenotype")
    ax.set_title("Joint embedding PCs → phenotype\n"
                 "(covariate-adjusted incremental R²; red = F-test p<0.05)",
                 fontsize=10)
    ax.axhline(0, color="black", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ── main ──────────────────────────────────────────────────────────────
def main():
    out     = Path(OUT_DIR)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    # ── load embeddings & build subject PCA + clusters ────────────────
    print("Loading embeddings...")
    emb_subj  = np.load(EMB_SUBJ_NPY)
    attn_df   = pd.read_csv(ATTN_CSV)
    subj_iids = attn_df["IID"].astype(str).values

    Zs        = StandardScaler().fit_transform(emb_subj)
    pca_model = PCA(n_components=10, random_state=42)
    subj_pca  = pca_model.fit_transform(Zs)

    km           = KMeans(n_clusters=3, random_state=42, n_init=20)
    clusters_raw = km.fit_predict(subj_pca[:, :10])

    # relabel by PC1 mean — consistent with script 10
    tmp   = pd.DataFrame({"c": clusters_raw, "pc1": subj_pca[:, 0]})
    order = tmp.groupby("c")["pc1"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    clusters = np.array([remap[c] for c in clusters_raw], dtype=int)

    subject_df = pd.DataFrame({
        "IID":      subj_iids,
        "cluster":  clusters,
        "embedPC1": subj_pca[:, 0],
        "embedPC2": subj_pca[:, 1],
        "embedPC3": subj_pca[:, 2],
    })

    # ── load genotype PCs ─────────────────────────────────────────────
    print("Loading genotype PCs...")
    pcs_df     = load_eigenvec(EIGENVEC_FILE)
    subject_df = subject_df.merge(pcs_df, on="IID", how="left")

    # ── load phenotypes ───────────────────────────────────────────────
    print("Loading phenotypes...")
    pheno_raw = pd.read_csv(PHENO_FILE, low_memory=False)
    pheno_raw = pheno_raw.rename(columns={"S_SUBJECTID": "IID"})
    pheno_raw["IID"] = pheno_raw["IID"].astype(str)

    pheno_raw = recode_categorical(pheno_raw)
    if "gender" in pheno_raw.columns:
        pheno_raw["gender_num"] = normalize_gender(pheno_raw["gender"])

    # keep phenotypes + covariate columns
    keep_cols = ["IID"]
    for c in ALL_PHENOS + ["age", "gender_num", "smkexp_current"]:
        if c in pheno_raw.columns and c not in keep_cols:
            keep_cols.append(c)

    pheno_df = pheno_raw[keep_cols].copy()

    missing = [p for p in ALL_PHENOS if p not in pheno_df.columns]
    if missing:
        print(f"  WARNING: phenotypes not found in file — {missing}")

    # ── merge embeddings + genotype PCs + phenotypes ──────────────────
    df = subject_df.merge(pheno_df, on="IID", how="inner")
    print(f"  Subjects after merge: {len(df)}")

    all_cols = df.columns.tolist()

    # ── run tests ─────────────────────────────────────────────────────
    cluster_rows  = []
    cont_rows     = []
    combined_rows = []

    phenos_to_test = [p for p in ALL_PHENOS if p in df.columns]

    for pheno in phenos_to_test:
        col_type = "continuous" if pheno in CONTINUOUS_PHENOS else "binary"
        print(f"\n{'─'*55}")
        print(f"Testing: {pheno}  [{col_type}]")

        covars     = get_covars(pheno, all_cols)
        cov_matrix = (df[covars]
                      .apply(pd.to_numeric, errors="coerce")
                      .values
                      if covars else None)

        y = pd.to_numeric(df[pheno], errors="coerce").values

        # ── A) cluster-based ─────────────────────────────────────────
        cluster_res            = test_cluster(y, df["cluster"].values,
                                              col_type, covariates=cov_matrix)
        cluster_res["phenotype"] = pheno
        cluster_rows.append(cluster_res)

        print(f"  Cluster   η²={cluster_res['eta2']:.4f}  "
              f"kruskal_p={cluster_res['kruskal_p']:.4g}  "
              f"n={cluster_res['n']}")
        for k in [0, 1, 2]:
            mn = cluster_res.get(f"cluster{k}_mean", np.nan)
            n  = cluster_res.get(f"cluster{k}_n",    np.nan)
            print(f"           C{k}: mean={mn:.4f}  n={n}")

        plot_by_cluster(df, pheno, col_type,
                        fig_dir / f"{pheno}_by_cluster.png")

        # ── B) continuous embedding PCs ───────────────────────────────
        for epc in EMBED_PCS:
            x       = df[epc].values
            cont_res = test_continuous_predictor(y, x, covariates=cov_matrix)
            cont_res["phenotype"] = pheno
            cont_res["embed_pc"]  = epc
            cont_res["col_type"]  = col_type
            cont_rows.append(cont_res)

            print(f"  {epc}  beta={cont_res['beta']:.4f}  "
                  f"p={cont_res['p']:.4g}  "
                  f"partial_r2={cont_res['partial_r2']:.4f}  "
                  f"spearman_r={cont_res['spearman_r']:.4f}")

        plot_scatter_embedpc(df, pheno, "embedPC1",
                             fig_dir / f"{pheno}_vs_embedPC1.png")

        # ── combined summary row ──────────────────────────────────────
        def _get(pc):
            return next((r for r in cont_rows
                         if r["phenotype"] == pheno and r["embed_pc"] == pc), {})

        pc1 = _get("embedPC1")
        pc2 = _get("embedPC2")
        combined_rows.append({
            "phenotype":           pheno,
            "col_type":            col_type,
            "n_cluster":           cluster_res["n"],
            "cluster_eta2":        cluster_res["eta2"],
            "cluster_kruskal_p":   cluster_res["kruskal_p"],
            "cluster_anova_p":     cluster_res.get("anova_p", np.nan),
            "cluster0_mean":       cluster_res.get("cluster0_mean", np.nan),
            "cluster1_mean":       cluster_res.get("cluster1_mean", np.nan),
            "cluster2_mean":       cluster_res.get("cluster2_mean", np.nan),
            "embedPC1_beta":       pc1.get("beta",       np.nan),
            "embedPC1_p":          pc1.get("p",          np.nan),
            "embedPC1_partial_r2": pc1.get("partial_r2", np.nan),
            "embedPC1_spearman_r": pc1.get("spearman_r", np.nan),
            "embedPC1_spearman_p": pc1.get("spearman_p", np.nan),
            "embedPC2_beta":       pc2.get("beta",       np.nan),
            "embedPC2_p":          pc2.get("p",          np.nan),
            "embedPC2_partial_r2": pc2.get("partial_r2", np.nan),
            "embedPC2_spearman_r": pc2.get("spearman_r", np.nan),
            "embedPC2_spearman_p": pc2.get("spearman_p", np.nan),
        })

    # ── FDR correction ────────────────────────────────────────────────
    cluster_df  = pd.DataFrame(cluster_rows)
    cont_df     = pd.DataFrame(cont_rows)
    combined_df = pd.DataFrame(combined_rows)

    def add_fdr(df_res, pcol, qcol):
        if pcol not in df_res.columns:
            return
        pvals = df_res[pcol].values.astype(float)
        mask  = np.isfinite(pvals)
        qvals = np.full(len(df_res), np.nan)
        if mask.sum() > 1:
            _, qtmp, _, _ = multipletests(pvals[mask], method="fdr_bh")
            qvals[mask]   = qtmp
        df_res[qcol] = qvals

    add_fdr(cluster_df,  "kruskal_p",          "kruskal_fdr_bh")
    add_fdr(cluster_df,  "anova_p",             "anova_fdr_bh")
    add_fdr(cluster_df,  "chi2_p",              "chi2_fdr_bh")
    add_fdr(cont_df,     "p",                   "p_fdr_bh")
    add_fdr(cont_df,     "spearman_p",          "spearman_fdr_bh")
    add_fdr(combined_df, "cluster_kruskal_p",   "cluster_kruskal_fdr")
    add_fdr(combined_df, "embedPC1_p",          "embedPC1_fdr")
    add_fdr(combined_df, "embedPC1_spearman_p", "embedPC1_spearman_fdr")

    # ── save ──────────────────────────────────────────────────────────
    cluster_df.to_csv( out / "phenotype_cluster_results.tsv",   sep="\t", index=False)
    cont_df.to_csv(    out / "phenotype_continuous_results.tsv", sep="\t", index=False)
    combined_df.to_csv(out / "phenotype_combined_summary.tsv",   sep="\t", index=False)

    plot_volcano(combined_df, fig_dir / "volcano_eta2_r2.png")

    # ── joint embedPC1+2+3 R² (supplementary analysis) ───────────────
    print("\n" + "=" * 70)
    print("JOINT EMBEDDING R² (embedPC1 + embedPC2 + embedPC3)")
    print("=" * 70)

    embed_matrix = df[EMBED_PCS].values.astype(float)
    joint_rows   = []

    joint_targets = [p for p in JOINT_PHENOS if p in df.columns]
    if not joint_targets:
        print("  No JOINT_PHENOS found in data — skipping.")
    else:
        for pheno in joint_targets:
            col_type = "continuous" if pheno in CONTINUOUS_PHENOS else "binary"
            covars   = get_covars(pheno, all_cols)
            cov_mat  = (df[covars].apply(pd.to_numeric, errors="coerce").values
                        if covars else None)
            y        = pd.to_numeric(df[pheno], errors="coerce").values

            res = test_joint_embed(y, embed_matrix, covariates=cov_mat)
            res["phenotype"] = pheno
            res["col_type"]  = col_type
            joint_rows.append(res)

            print(f"  {pheno:30s}  n={res['n']}  "
                  f"covar_r2={res['covar_r2']:.4f}  "
                  f"joint_r2={res['joint_r2']:.4f}  "
                  f"incremental_r2={res['incremental_r2']:.4f}  "
                  f"F_p={res['f_p']:.4g}")

        joint_df = pd.DataFrame(joint_rows)
        joint_df.to_csv(out / "phenotype_joint_embed_r2.tsv", sep="\t", index=False)
        plot_joint_r2(joint_df, fig_dir / "joint_embed_r2_barplot.png")
        print(f"\n  Saved: phenotype_joint_embed_r2.tsv")
        print(f"  Saved: figures/joint_embed_r2_barplot.png")

    # ── final summary table ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    show = [
        "phenotype", "col_type",
        "cluster_eta2", "cluster_kruskal_p", "cluster_kruskal_fdr",
        "embedPC1_partial_r2", "embedPC1_p", "embedPC1_fdr",
        "embedPC1_spearman_r", "embedPC1_spearman_p",
    ]
    show = [c for c in show if c in combined_df.columns]
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(combined_df[show].to_string(index=False))
    print(f"\nOutputs saved to: {out}")


if __name__ == "__main__":
    main()