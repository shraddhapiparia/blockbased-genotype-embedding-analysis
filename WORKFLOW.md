# Analysis Workflow

This document describes the execution order and role of each script in the
active pipeline. See `README.md` for scientific framing and data overview.

---

## Directory layout

```
scripts/
  core/       Core pipeline steps — run in order
  analysis/   Post-hoc analysis scripts — run after core pipeline
  archive/    Obsolete, exploratory, or merged scripts — not part of active workflow
```

---

## Phase 1 — Per-block VAE training

**Script:** `scripts/core/VAE_phase1.py`

```bash
python scripts/core/VAE_phase1.py --config configs/config_phase1.yaml
python scripts/core/VAE_phase1.py --config configs/config_phase1.yaml --tune   # hyperparameter search
python scripts/core/VAE_phase1.py --config configs/config_phase1.yaml --dry-run
```

Trains a per-block β-VAE on LD-block genotype data. Each block gets its own
VAE; the latent embedding for every subject is saved. Outputs feed Phase 2.

| Input | Output |
|---|---|
| `data/region_blocks/<block>.npy` | `results/output_regions/block_order.csv` |
| `data/block_plan/manifest.tsv` | `results/output_regions/subjects.csv` |
| `configs/config_phase1.yaml` | `results/output_regions/vae_summary.csv` |
| | `results/output_regions/<loss>/embeddings/all_blocks.npy` |

---

## Phase 2 — Cross-block attention aggregation

**Script:** `scripts/core/attention_phase2.py`

```bash
python scripts/core/attention_phase2.py --config configs/config_phase2.yaml
python scripts/core/attention_phase2.py --config configs/config_phase2.yaml --dry-run
```

Consumes Phase 1 embeddings. Trains a Transformer to aggregate across blocks
and produce subject-level embeddings, per-block attention weights, and
contextual block representations. Also runs clustering on subject embeddings.

| Input | Output |
|---|---|
| Phase 1 output dir | `results/output_regions2/phase2_summary.csv` |
| `configs/config_phase2.yaml` | `results/output_regions2/<loss>/embeddings/` |
| | `results/output_regions2/<loss>/clustering/cluster_labels.csv` |

---

## Core supporting scripts

These are called after Phase 2 to produce block-level summaries and plots.
They do not require rerunning the full training pipeline.

| Script | Purpose |
|---|---|
| `core/analyze_phase2_blocks.py` | Rank blocks by attention weight; compare asthma vs control block groups |
| `core/plots_updated.py` | Phase 1 summary plots (VAE loss curves, accuracy, KL over epochs) |

---

## Active analysis scripts

Run after Phase 2 to produce interpretation, validation, and figures.
Intended execution order follows the numbering where present.

| Script | Purpose | Key outputs |
|---|---|---|
| `05_umap_hla_interpretation.py` | UMAP embedding + HLA cluster interpretation | UMAP plots, cluster labels |
| `06_attention_confounder_analysis.py` | OLS: attention ~ asthma + PC1–10 | Confounder tables, heatmaps |
| `07_block_embedding_phenotype_analysis.py` | Block-level PC1 phenotype associations | `phenotype_block_associations.tsv` |
| `09_unsupervised_subject_cluster_analysis.py` | UMAP + HDBSCAN subject clustering | Cluster assignments, plots |
| `10_hla_cluster_analysis.py` | HLA block_PC1 by cluster; tests ancestry confounding | HLA cluster summary |
| `11_leave_hla_out.py` | Leave-HLA-out anti-circularity validation | Re-clustered subject assignments |
| `12_cluster_stability.py` | KMeans seed / algorithm / k stability | ARI tables, elbow plots |
| `13_phenotype_cluster_analysis.py` | Phenotype ~ cluster + embedding PCs | Phenotype analysis tables |
| `17q21_genotype_embedding_validation.py` | 17q21 subblock validation (4 figures + correlation tables) | fig1–fig4 PNGs, SNP correlation TSVs |
| `17q21_baseline_comparison.py` | Embedding PC vs raw genotype PC at 17q21 | Incremental R² comparison |
| `compare_phase1_phase2_blocks.py` | PDM correlation + phenotype association Phase 1 vs Phase 2 | `pdm_correlations.csv`, comparison plots |

---

## Archive

`scripts/archive/` contains scripts that are **not part of the active workflow**:
wrappers that have been merged into core files, obsolete iterations, exploratory
one-offs, and debugging scripts. See `scripts/archive/README.md` for per-file
reasons.
