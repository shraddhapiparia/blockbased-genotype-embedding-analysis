# CLAUDE_HANDOFF.md

## Repository Purpose

This repository develops genotype representation learning for asthma-relevant genomic regions using LD-aware block embeddings. The goal is to learn subject-level genomic embeddings that capture meaningful biological structure and support downstream visualization, clustering, and phenotype association.

The repository evolved through multiple iterations. Earlier work was performed at the gene level, while the current and preferred direction is region-based analysis. The region-based workflow should be treated as the canonical direction for this project.

---

## Current Scientific Direction

The preferred scientific direction is:

- region-based LD block analysis
- block-level embeddings aggregated into subject-level embeddings
- frozen block embeddings as the primary analysis path
- downstream PCA, UMAP, clustering, and phenotype association
- interpretation of influential genomic regions

Please preserve and prioritize the region-based approach when inspecting or reorganizing the repository.

---

## Evolution of the Project

The project developed in two major phases.

### Earlier Iteration: Gene-Based Block VAEs

The initial version of the project used genes as the primary unit of analysis. This version included:

- gene-level block VAEs
- comparison of multiple loss functions
- attention-based aggregation of block embeddings
- clustering and metric comparisons
- ranking blocks by attention weights

These earlier materials remain useful because they contain:

- architecture ideas
- loss comparisons
- training metrics
- clustering evaluation logic
- code that may still be reusable

However, these gene-based experiments should now be treated as historical or exploratory work.

If there is a conflict between older gene-based logic and newer region-based logic, always prefer the region-based version.

---

## Canonical Direction: Region-Based Analysis

The project has now shifted from gene-based units to region-based LD blocks.

Important guidance:

- region-based analysis is the preferred representation
- current outputs in `results/` reflect region-based analysis
- future development should build on region-based analysis
- any remaining gene-based materials should be preserved only as historical context or moved into `archive/`

The repository should eventually make the distinction between these two phases very clear.

---

## Current Trusted Outputs

The following should be treated as trusted and preserved:

- current region-based outputs in `results/`
- current subject-level embedding results
- PCA and UMAP outputs from region-based embeddings
- cluster assignments and clustering metrics
- phenotype association tables
- block metadata and region ordering files

Do not overwrite these outputs unless writing new results into a separate directory.

Suggested approach:

- keep trusted outputs in `results/final/`
- write new experiments to `results/experimental/`

---

## Important Notebook-Only Findings

Some important results currently exist only in Jupyter notebooks and presentation slides.

One especially important finding is:

- a notebook-based UMAP analysis suggested that HLA-related regions strongly contribute to separation in embedding space

This observation is considered scientifically important and should be preserved in project documentation.

The current evidence for this finding exists in notebook code and slide material. The notebook implementation is not yet fully script-clean, but it should eventually be converted into reusable code or a reproducible script.

Important notes:

- HLA-related regions appear to be driving important structure in the learned embeddings
- this should be treated as an interpretation priority
- preserve any notebook, figure, or slide associated with this observation

---

## Uploaded Context and Supporting Materials

The repository includes additional supporting materials that provide important historical and scientific context.

### Iteration 1 Presentation

The iteration-1 presentation documents the earlier gene-based VAE phase of the project and includes:

- block VAE architecture
- loss function comparisons
- clustering metrics
- attention-based aggregation
- evaluation framework
- early selection of candidate blocks

This material should be preserved as background context but not treated as the final project direction.

### Attention / Regional Embedding Presentation

The more recent presentation captures the current region-based approach and includes:

- region-based embeddings
- attention-based aggregation
- 997 subjects with 64-dimensional genomic embeddings
- HLA-related block observations
- ideas for future modeling and interpretation

Please treat this material as more representative of the current scientific direction.

---

## Current Repository State

Current facts about the repository:

- the `scripts/` directory currently contains 5 scripts
- some analyses are implemented in notebooks
- some scripts are exploratory or duplicated
- the project lacks a fully documented execution order
- the repository needs cleanup and better separation between stable and exploratory work

The repository is not yet fully organized into a single reproducible pipeline.

---

## Immediate Goal

Before making major changes, start in audit mode.

Please:

1. Inspect the repository structure
2. Inventory all scripts, notebooks, and result files
3. Identify which components belong to the main region-based workflow
4. Identify which components are exploratory or superseded
5. Identify notebook logic that should eventually become reusable code
6. Propose a clearer, git-friendly repository structure
7. Explain the likely canonical execution order

Do not immediately perform a large refactor. First summarize your understanding and propose a plan.

---

## Desired Repository Structure

Please move the project toward a structure like:

```text
data/
  raw/
  processed/
  metadata/

src/
  io/
  embedding/
  clustering/
  association/
  plotting/
  interpretation/

scripts/
  01_validate_inputs.py
  02_region_embeddings.py
  03_dimensionality_reduction.py
  04_clustering.py
  05_association_analysis.py
  06_interpret_regions.py

notebooks/
results/
  final/
  experimental/
docs/
archive/

The goal is not necessarily to implement this immediately, but to move gradually toward it.

---

## Primary Analysis Path To Preserve

The primary workflow that should be preserved is:

1. Load region-based genotype inputs
2. Load phenotype data
3. Validate subject identifier alignment
4. Load or generate block embeddings
5. Freeze block embeddings
6. Aggregate block embeddings into subject-level embeddings
7. Perform PCA, UMAP, and clustering
8. Merge with phenotype data
9. Perform phenotype association analysis
10. Interpret important regions

This frozen-block-embedding path is the primary and trusted analysis workflow.

---

## Secondary Modeling Goal

In addition to the primary workflow, I want to develop a secondary end-to-end model.

Important guidance:

- The end-to-end model is a secondary analysis
- It should not replace the frozen embedding workflow
- It should live alongside the main workflow for comparison
- Results from both approaches should be easy to compare

The repository should eventually support:

- Frozen block embedding workflow
- End-to-end workflow
- Side-by-side comparison between the two

Potential future organization:

```text
src/
  frozen_embedding_pipeline/
  end_to_end_pipeline/
```

## Interpretation Priorities

Interpretation should focus on biologically meaningful genomic regions.

Current interpretation priorities include:

- HLA-related regions
- Region importance ranking
- Connection between embedding structure and known asthma loci
- Later integration of attention weights or attribution scores

Important asthma-relevant regions include:

- IL1RL1 / IL18R1 / IL18RAP
- IL33
- FCER1A
- STAT6
- 17q21 region
- HLA-related regions

Please preserve region names clearly and avoid replacing them with anonymous latent feature identifiers when possible.

---

## Guidance On Cleanup

When cleaning the repository:

- Preserve working code even if imperfect
- Avoid deleting historical materials immediately
- Move older or superseded material into `archive/`
- Isolate notebook-only logic from canonical scripts
- Avoid rewriting scientific assumptions unless necessary
- Improve naming consistency
- Reduce duplicated logic
- Make the project easier for a collaborator to understand

Please prioritize clarity and reproducibility over aggressive refactoring.

---

## Expected Deliverables From Audit Mode

The first response should include:

- Repository inventory
- Canonical workflow proposal
- List of stable vs exploratory components
- List of duplicated or superseded scripts
- Proposed folder structure
- Recommendation for which notebook analyses should be migrated
- Recommendation for the next 3–5 highest-value implementation steps

Only after that should implementation begin.