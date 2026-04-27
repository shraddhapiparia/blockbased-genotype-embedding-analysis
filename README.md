# Block-Based Genotype Embedding Analysis

[![CI](https://github.com/shraddhapiparia/blockbased-genotype-embedding-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/shraddhapiparia/blockbased-genotype-embedding-analysis/actions/workflows/ci.yml)

Status: ongoing research. README reflects current findings; documentation, CI, and tests are being added incrementally.

Unsupervised learning of subject-level genomic representations from LD-aware blocks,
with application to asthma-relevant loci and downstream phenotype association.

![Pipeline overview](docs/images/pipeline_overview.png)
*Phase 1 learns a compact β-VAE embedding per LD block. Phase 2 aggregates block
embeddings across the genome via a Transformer with cross-block attention, producing
subject-level embeddings and interpretable block-importance weights.*

---

## Key Idea

Standard genotype analysis treats each SNP independently or applies global LD pruning,
losing local genomic context. This project takes a hierarchical approach:

1. **Phase 1 — Per-block β-VAE.** Each LD block is encoded independently into a
   low-dimensional latent vector that captures local haplotype structure.
2. **Phase 2 — Cross-block Transformer.** A Transformer aggregates all block embeddings
   into a single subject-level representation. Learned attention weights identify which
   blocks are most informative for organizing genetic variation across individuals.

The result is an embedding that is biologically interpretable (attention scores), within same ancestry, and structured for downstream analysis (clustering, phenotype association, leave-one-block-out validation).

---

## Key Results

- **HLA class II dominates the learned space.** Transformer attention weights
  consistently rank HLA class II blocks highest; these blocks drive the primary
  axis of subject-level variation.
- **HLA block embeddings outperform ancestry PCs.** HLA block_PC1 explains subject
  cluster structure beyond what genotype PC1–10 can account for, confirming genuine
  biological signal rather than ancestry confounding. HLA sb15 η² = 0.767 vs η² = 0.051
  for genotype PC3 (script `02_subject_cluster_analysis.py`, step C;
  `results/output_regions2/ORD/clustering/hla_block_cluster_association.tsv` and
  `results/output_regions2/ORD/clustering/genotype_pc_cluster_association.tsv`).
- **PDE4D emerges after masking HLA.** A leave-HLA-out re-clustering experiment
  (script `03_leave_hla_out_analysis.py`) reveals PDE4D as the next most structurally
  informative block — consistent with its established role in asthma and β-agonist
  pharmacogenomics.
- **Phenotype signal is real but subtle; IgE is the strongest.** Continuous phenotypes
  (blood eosinophil count, IgE, lung function spirometry test, exacerbation) show the
  most consistent association with block-level PC features across subjects
  (script `01_block_embedding_phenotype_analysis.py`;
  `results/output_regions2/ORD/all_blocks_pheno_analysis/phenotype_block_associations.tsv`).
- **Biology recovered without phenotype labels.** The model was trained unsupervised on
  genotype data only. The emergence of HLA class II and PDE4D in post-hoc analysis
  validates that the learned geometry reflects known asthma biology.

---

## What Phase 2 Adds Beyond PCA

Conventional PCA on raw genotype data primarily captures population structure — the top
components reflect ancestry rather than disease-relevant biology. Phase 1 VAE embeddings
preserve local LD-block haplotype structure that SNP-level PCA discards. Phase 2 adds
cross-block context: the Transformer learns which blocks co-vary meaningfully across
subjects, reorganizing rather than destroying the Phase 1 geometry (Phase 1 vs Phase 2
median pairwise-distance Spearman r ≈ 0.68 across blocks; script
`06_phase1_phase2_block_comparison.py`,
`results/phase_comparison/pdm_correlations.csv`). The result is a subject-level space
where HLA class II dominates the primary axis, PDE4D emerges as the next structurally
informative block after HLA removal, and IgE shows stronger phenotype association —
biological signal that ancestry-adjusted PCA does not recover.

---

## Figures

### 1 — Pipeline architecture


Schematic of the two-phase architecture: per-block VAE (Phase 1) feeding into the
cross-block Transformer (Phase 2) to produce subject embeddings is shown above.


---


### 2 — Subject embedding PCA reveals stable genomic structure

![Subject PCA colored by cluster](docs/images/subject_pca_clusters.png)

PCA of Phase 2 subject embeddings reveals three reproducible strata (`k=3`; mean
pairwise ARI = 0.999 across 50 random seeds). The weak silhouette score (`0.139`)
indicates the learned space is structured as a continuous gradient rather than sharply
separated clinical subtypes.

*Source: ARI from `scripts/analysis/04_cluster_stability_analysis.py`
(`results/output_regions2/ORD/cluster_stability/ari_kmeans_seeds.tsv`); silhouette from
`scripts/analysis/02_subject_cluster_analysis.py` step A
(`results/output_regions2/ORD/clustering/clustering_metrics.csv`).
Filename: `docs/images/subject_pca_clusters.png`*

---

### 3 — HLA class II dominance

![HLA class II dominates the learned embedding space](docs/images/hla_embedding_dominance.png)

HLA class II subblocks strongly organize the Phase 2 embedding space. HLA sb15 explains
far more cluster variance than ancestry PCs (η² = 0.767 vs 0.051 for genotype PC1;
script `02_subject_cluster_analysis.py` step C,
`results/output_regions2/ORD/clustering/hla_block_cluster_association.tsv`). HLA sb15
block_PC1 also shows strong negative Spearman correlation with the main embedding axis;
the specific value (r ≈ −0.88) was observed in exploratory analysis and is not yet
tracked in a dedicated output table.
*Source: `scripts/analysis/02_subject_cluster_analysis.py`.*

---

### 4 — Three-finding summary

![Summary of main findings](docs/images/findings_summary.png)

Slide-style summary panel: (1) HLA class II anchors the embedding space, (2) PDE4D
is the next signal after HLA removal, (3) phenotype associations are present and
biologically coherent without supervised training.
*Source: conclusions/summary slide. Suggested filename:
`docs/images/findings_summary.png`*

---

## Limitations

- **Internal validation only.** All results are from a single cohort (COS/TRIO).
  Clustering structure, attention rankings, and phenotype associations have not been
  validated in an independent dataset.
- **No causal inference.** Attention weights and LOBO/perturbation attribution scores
  identify blocks that are statistically influential for the learned embedding geometry.
  They are heuristic measures, not evidence of mechanistic importance or causality.
- **Post-hoc phenotype associations.** Associations between embedding features and
  clinical phenotypes (IgE, eosinophils, FEV1, exacerbation) were tested after
  unsupervised training. They should be treated as hypothesis-generating and require
  external validation.
- **Ancestry and generalizability.** The embedding was trained within a single ancestry
  stratum. Generalization to other ancestries or multi-ancestry cohorts has not been
  assessed.
- **Data availability.** Input genotype and phenotype data are access-restricted and not
  version-controlled. Full reproduction from this repository requires obtaining restricted
  data access separately. The synthetic smoke test (`bash test_run.sh`) validates pipeline
  wiring only.
- **Embedding geometry is not clinical classification.** The learned subject-level
  embedding space reflects genomic variation structure, not disease phenotype. Cluster
  membership does not imply clinical disease subtypes or therapeutic relevance.

---

## Repository Structure

```
scripts/
  core/       Core pipeline — Phase 1 VAE, Phase 2 Transformer, block analysis, plotting
  analysis/   Numbered post-hoc scripts:
                01–07  phenotype association, clustering, HLA validation,
                       confounder analysis, 17q21 validation
                08     clinical PC / embedding alignment (Pearson + Spearman, OLS, Ridge)
                09     Phase 2 block attribution — leave-one-block-out (LOBO)
                10     Phase 1 SNP attribution within selected blocks
  archive/    Superseded wrappers, exploratory one-offs, debug scripts
configs/      YAML configs for Phase 1, Phase 2, no-HLA variant, and synthetic smoke test
data/         Genotype block files and block manifest (access-restricted, not tracked)
              data/synthetic/  — generated synthetic data for smoke testing (gitignored)
results/      Pipeline outputs (access-restricted, not tracked)
metadata/     Phenotype table, eigenvec file (access-restricted, not tracked)
docs/         Method notes and figures
tests/        Lightweight unit tests (pytest); no real data required
environment.yml
pytest.ini
WORKFLOW.md   Step-by-step execution guide with CLI examples
run_pipeline.sh  Single entry point — runs full pipeline or --dry-run input check
test_run.sh   Smoke test — Phase 1 → Phase 2 on synthetic data (no restricted data needed)
CLAUDE.md     AI assistance constraints and workflow summary
```

See [WORKFLOW.md](WORKFLOW.md) for full CLI instructions, expected inputs/outputs per
step, and execution order. See [docs/data_contract.md](docs/data_contract.md) for the
Phase 1 → Phase 2 data format contract, subject/block ordering conventions, and common
failure modes.

---

## Quick Start

```bash
conda env create -f environment.yml
conda activate genotype-embedding-env

# Run full pipeline (requires restricted data in data/ and metadata/)
./run_pipeline.sh

# Validate inputs only — no training
./run_pipeline.sh --dry-run

# Or run phases individually
python scripts/core/VAE_phase1.py --config configs/config_phase1.yaml
python scripts/core/attention_phase2.py --config configs/config_phase2.yaml

# Post-hoc analysis (example)
python scripts/analysis/03_leave_hla_out_analysis.py

# Attribution pipeline (scripts 08–10; requires Phase 2 outputs)
python scripts/analysis/08_clinical_pc_embedding_alignment.py
python scripts/analysis/09_phase2_block_attribution.py
python scripts/analysis/10_phase1_snp_attribution_within_blocks.py \
    --selected-blocks region_9p24_IL33
```

Use `--dry-run` on `run_pipeline.sh` or on either phase script to validate inputs without
running training. Full details in [WORKFLOW.md](WORKFLOW.md).

---

## Synthetic Smoke Test

To verify the Phase 1 → Phase 2 pipeline wiring without restricted data access:

```bash
./test_run.sh
```

Expected runtime: under 5 minutes on CPU. The script generates 30 fake subjects across
4 synthetic LD blocks (10 SNPs each, random integers in {0, 1, 2}), runs Phase 1 and
Phase 2 with minimal settings (3 epochs), and confirms that required output files are
written to `results/synthetic_test/` and `results/synthetic_test2/`.

> These outputs validate pipeline wiring only. Synthetic data has no biological meaning
> and should not be interpreted scientifically.

---

## Data

Raw genotype data and phenotype tables are not version-controlled (access-restricted).
The repository preserves analysis logic, configuration, derived summaries, and
documentation sufficient for rerunning with appropriate input access.

Core inputs: per-block `.npy` genotype matrices, block manifest TSV,
subject phenotype CSV, ancestry eigenvec file.

---

## Environment

Python 3.10+. Key dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`,
`matplotlib`, `umap-learn`, `hdbscan`, `statsmodels`, `scipy`, `seaborn`, `yaml`.

```bash
conda env create -f environment.yml
```
