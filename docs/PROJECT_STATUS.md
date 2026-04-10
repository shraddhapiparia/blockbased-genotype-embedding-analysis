# PROJECT_STATUS.md

## Project Overview

This repository contains an evolving framework for learning subject-level genotype embeddings from asthma-relevant linkage disequilibrium (LD) blocks and relating those embeddings to clinical phenotypes. The overall idea is to move beyond SNP-level matrices by representing local genomic regions as compact latent embeddings and combining them into a subject-level representation.

The project currently includes exploratory and partially validated work spanning:

- LD block definition and organization
- Learned block-level and subject-level embeddings
- PCA and UMAP exploration
- Clustering of subject embeddings
- Association of embeddings or genomic blocks with asthma-related phenotypes
- Preliminary interpretation of influential blocks

The repository is currently in transition from exploratory analysis toward a more reproducible and structured codebase.

---

# Current Development State

The project is not yet fully organized into a single reproducible pipeline. Some code is stable and reflects the current scientific direction, while other files are exploratory, duplicated, partially debugged, or superseded by newer approaches.

The immediate goal is not to redesign the scientific approach, but rather to:

1. Identify the canonical workflow
2. Separate reusable code from exploratory code
3. Standardize inputs and outputs
4. Improve reproducibility and documentation
5. Build the next analytical steps on top of a stable structure

---

# What Is Considered Stable

The following concepts and outputs are considered scientifically meaningful and should generally be preserved:

- LD-aware block-based representation of genotype data
- Learned subject-level embeddings
- PCA comparison between embeddings and raw SNP matrices
- Evidence that embedding PCA captures more structure than SNP PCA
- Use of asthma-relevant loci such as:
  - IL1RL1 / IL18R1 / IL18RAP
  - IL33
  - FCER1A
  - STAT6
  - 17q21 region (ORMDL3 / GSDMB / ZPBP2)
- Use of phenotype association testing between blocks and asthma-related traits
- Subject identifier alignment using a common IID field or equivalent

The following outputs should be treated as reference outputs unless explicitly replaced:

- Existing PCA plots of embeddings
- Existing UMAP plots and cluster assignments
- Existing phenotype association result tables
- Existing clustering metrics and summaries
- Existing block metadata and block ordering files

---

# Files Likely To Exist

The repository may contain some or all of the following files. Names may differ slightly across versions.

## Core Input Files

- phenotype table
  - Example: `COS_TRIO_pheno_1165.csv`
- subject-level embedding matrix
  - Example: `individual_embeddings.npy`
- block metadata or order file
  - Example: `block_order.csv`
- attention or pooling weights
  - Example: `pooling_attention_weights.csv`
- cluster labels
  - Example: `cluster_labels.csv`
- clustering summary metrics
  - Example: `clustering_metrics.csv`
- ancestry principal components
  - Example: `ldpruned_997subs.eigenvec`

## Frequently Used Columns

Common subject identifier columns include:

- `IID`
- `iid`
- `S_SUBJECTID`
- `subject_id`

Phenotype table commonly uses:

- `S_SUBJECTID`

The following phenotype columns have been used repeatedly:

- `BMI`
- `age`
- `smkexp_current`
- `pctpred_fev1_pre_BD`
- `Hospitalized_Asthma_Last_Yr`
- `Oral_Steroids_Last_Year`
- `G19B`
- `G20D`

Binary outcomes may require remapping before modeling.

Example mapping previously used:

```python
BIN_MAP = {1: 0, 2: 1}
```

## Known Issues and Current Pain Points

### Identifier Mismatches

One of the largest recurring issues is mismatching subject identifiers between phenotype tables, embeddings, cluster labels, and auxiliary files.

Common problems include:

- Embedding matrix missing subject identifiers
- Different identifier column names across files
- Phenotype table having more rows than embeddings
- Merges silently dropping large numbers of subjects

Expected behavior:

- Embedding rows and phenotype rows should be merged explicitly on a common subject ID
- Scripts should report:
  - Number of phenotype rows
  - Number of embedding rows
  - Number of merged rows
  - Number of missing subjects

No merge should proceed silently.

### Exploratory Code Mixed With Final Code

The repository currently contains a mixture of:

- Temporary debugging code
- Exploratory notebooks
- Partially working scripts
- Duplicated analysis logic
- More stable analysis code

These should eventually be separated into:

- `src/` for reusable functions
- `scripts/` for reproducible pipeline steps
- `notebooks/` for exploratory work
- `archive/` for superseded material

### Lack of Canonical Execution Order

There is not yet a single agreed-upon path from raw inputs to final outputs.

One of the highest-priority tasks is to identify:

- Which script generates embeddings
- Which script loads and merges phenotype data
- Which script performs PCA / UMAP / clustering
- Which script performs phenotype association
- Which outputs are considered final

---

## Recommended Canonical Pipeline

The intended long-term workflow is:

1. Load phenotype and genotype-derived inputs
2. Validate subject identifiers
3. Construct or load block-level embeddings
4. Aggregate block embeddings into subject-level embeddings
5. Run PCA and UMAP
6. Run clustering
7. Evaluate clustering metrics
8. Merge embeddings with phenotype data
9. Perform phenotype association analyses
10. Generate figures and summary tables

---

## Current Priority Tasks

The following tasks should be done before adding major new analyses.

### High Priority

- [ ] Inventory all existing scripts and notebooks
- [ ] Identify which files are canonical vs exploratory
- [ ] Create a standard input file specification
- [ ] Standardize subject identifier handling
- [ ] Move reusable logic into `src/`
- [ ] Standardize output directories under `results/`
- [ ] Add consistent logging and merge diagnostics
- [ ] Create a documented execution order

### Medium Priority

- [ ] Add configuration file for paths and phenotype definitions
- [ ] Add covariate handling and ancestry PCs consistently
- [ ] Add automatic plotting utilities
- [ ] Add multiple-testing correction consistently
- [ ] Add summary report generation

### Lower Priority / Future

- [ ] Add attention-based interpretation of influential blocks
- [ ] Add attribution methods for block importance
- [ ] Compare embeddings across alternate model architectures
- [ ] Integrate eQTL or external biological annotation resources
- [ ] Expand to multimodal integration