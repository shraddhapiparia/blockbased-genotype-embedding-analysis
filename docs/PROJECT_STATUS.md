# PROJECT_STATUS.md

## Current State (post analysis-refactor-cleanup)

The repository has been reorganized into a reproducible, documented pipeline. The
exploratory phase is complete; the codebase is now structured for clean execution
and publication.

---

## Repository Structure

```
scripts/
  core/       Canonical pipeline — Phase 1 VAE, Phase 2 Transformer, block analysis, plots
  analysis/   Numbered post-hoc scripts (01–07) covering phenotype association,
              clustering, HLA validation, confounder analysis, and 17q21 validation
  archive/    Superseded and exploratory scripts — not part of active workflow
configs/      YAML configs for Phase 1, Phase 2, and no-HLA variant
docs/         Project documentation and figures
environment.yml
WORKFLOW.md   Step-by-step execution guide with CLI examples
run_pipeline.sh  Single entry point to run the full pipeline
CLAUDE.md     AI assistance constraints and workflow summary
```

---

## Canonical Workflow

1. `scripts/core/VAE_phase1.py` — per-block β-VAE training
2. `scripts/core/attention_phase2.py` — cross-block Transformer, subject embeddings
3. `scripts/core/analyze_phase2_blocks.py` — block attention ranking
4. `scripts/core/plots_updated.py` — Phase 1 summary plots
5. `scripts/analysis/01_block_embedding_phenotype_analysis.py`
6. `scripts/analysis/02_subject_cluster_analysis.py`
7. `scripts/analysis/03_leave_hla_out_analysis.py`
8. `scripts/analysis/04_cluster_stability_analysis.py`
9. `scripts/analysis/05_attention_confounder_analysis.py`
10. `scripts/analysis/06_phase1_phase2_block_comparison.py`
11. `scripts/analysis/07_17q21_validation.py`

Use `./run_pipeline.sh` to run all steps or `--dry-run` to validate inputs only.

---

## Scientific Direction

- Region-based LD block analysis is the canonical approach (gene-based iteration is
  archived).
- Frozen block embeddings are the primary analysis path.
- Key findings: HLA class II dominance, PDE4D emergence after HLA removal, IgE as
  strongest phenotype association. See README.md for details.

---

## Data

Raw inputs (genotype block `.npy` files, phenotype CSV, eigenvec) are access-restricted
and not version-controlled. `data/`, `metadata/`, and `results/` are gitignored.

---

## Known Gaps

- No synthetic or toy dataset exists for testing without restricted data access.
- `notebooks/umap_hla_interpretation.ipynb` contains exploratory HLA analysis that
  has since been superseded by `scripts/analysis/02_subject_cluster_analysis.py`
  and `03_leave_hla_out_analysis.py`, but is preserved as historical reference.
