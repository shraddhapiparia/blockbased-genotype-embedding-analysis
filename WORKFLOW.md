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
Scripts are numbered in intended execution order.

| Script | Purpose | Key outputs |
|---|---|---|
| `01_block_embedding_phenotype_analysis.py` | Phenotype and ancestry association of block-level contextual representations (`block_contextual_repr.npy`) | `phenotype_block_associations.tsv` |
| `02_subject_cluster_analysis.py` | Three-stage subject analysis: (A) HDBSCAN clustering + PCA/UMAP; (B) UMAP coloured by attention weights, top-block identification; (C) KMeans(k=3) + HLA block_PC1 vs genotype-PC per cluster | Cluster assignments, UMAP plots, HLA cluster summary |
| `03_leave_hla_out_analysis.py` | Leave-HLA-out anti-circularity validation; re-clusters on Phase 1 embeddings with HLA blocks masked | Re-clustered subject assignments |
| `04_cluster_stability_analysis.py` | KMeans seed / algorithm / k stability | ARI tables, elbow plots |
| `05_attention_confounder_analysis.py` | OLS: attention ~ asthma; attention ~ asthma + PC1–10; Pearson r with each PC | Confounder tables, heatmaps |
| `06_phase1_phase2_block_comparison.py` | Pairwise-distance-matrix (PDM) Spearman correlation and phenotype association comparing Phase 1 and Phase 2 block representations | `pdm_correlations.csv`, comparison plots |
| `07_17q21_validation.py` | Two-stage 17q21 validation: (A) genotype↔embedding alignment, SNP-level correlations, 4 publication figures; (B) OLS baseline comparison — embedding PC vs raw genotype PC ~ FEV1, incremental R² | fig1–fig4 PNGs, SNP correlation TSVs |

---

## Attribution scripts (08–10)

Run after Phase 2 and the core analysis scripts. Scripts 08–10 form a
hierarchical attribution pipeline: embedding alignment → block attribution →
SNP attribution.

| Script | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `08_clinical_pc_embedding_alignment.py` | Align learned Phase 2 embedding PCs with clinical phenotype PCs; Pearson + Spearman heatmaps; OLS and Ridge baselines including ancestry-PC-only model | Phase 2 `individual_embeddings.csv`, `ldpruned_997subs.eigenvec`, phenotype CSV | `correlation_heatmap_pearson.png`, `correlation_heatmap_spearman.png`, `embedding_pc_correlations.tsv`, R² comparison tables |
| `09_phase2_block_attribution.py` | Leave-one-block-out (LOBO) attribution: for each of the 174 Phase 1 blocks, mean-mask it in the Phase 2 input and measure change in embedding PC1/PC2 and IgE ridge score | Phase 1 `all_blocks.npy`, Phase 2 checkpoint + `individual_embeddings.npy`, `block_order.csv`, `subjects.csv` | `phase2_PC1_leave_one_block_out.csv`, `phase2_PC2_leave_one_block_out.csv`, `phase2_log10Ige_leave_one_block_out.csv`, top-20 bar plots |
| `10_phase1_snp_attribution_within_blocks.py` | For user-selected blocks from script 09: mean-mask each SNP in the Phase 1 input, re-encode through the frozen VAE, re-run Phase 2, project onto fixed PCA axes, and rank SNPs by mean absolute delta | Phase 1 block checkpoints (`.pt`), per-block `.raw` genotype files, Phase 2 checkpoint + embeddings, script 09 LOBO CSVs (optional, enables block-weighted priority) | Per-block and combined `*_snp_attribution.csv`, bar plots |

```bash
# Run 08 (requires Phase 2 outputs at default paths)
python scripts/analysis/08_clinical_pc_embedding_alignment.py

# Run 09
python scripts/analysis/09_phase2_block_attribution.py

# Run 10 (requires --selected-blocks from 09 results)
python scripts/analysis/10_phase1_snp_attribution_within_blocks.py \
    --selected-blocks region_9p24_IL33,region_6p21_HLA_classII_sb15 \
    --lobo-pc1-csv results/analysis/phase2_block_attribution/phase2_PC1_leave_one_block_out.csv \
    --lobo-pc2-csv results/analysis/phase2_block_attribution/phase2_PC2_leave_one_block_out.csv
```

---

## Archive

`scripts/archive/` contains scripts that are **not part of the active workflow**:
wrappers that have been merged into core files, obsolete iterations, exploratory
one-offs, and debugging scripts. See `scripts/archive/README.md` for per-file
reasons.
