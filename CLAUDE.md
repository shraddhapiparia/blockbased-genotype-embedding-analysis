# CLAUDE.md

## Repository Purpose

Block-based genotype embedding analysis for asthma-relevant genomic loci.
Phase 1 trains a per-block β-VAE on LD-block genotype data. Phase 2 aggregates block
embeddings via a cross-block Transformer to produce subject-level representations.
Downstream analysis covers clustering, phenotype association, and block interpretation.

## Restricted Data

Raw genotype and phenotype inputs are access-restricted and not version-controlled.
`data/`, `metadata/`, and `results/` are all gitignored. Do not commit files from those
directories. Never hardcode subject identifiers, phenotype values, or file paths that
expose cohort structure.

## Workflow Order

```
Phase 1  →  Phase 2  →  Core supporting scripts  →  Analysis scripts (01–07)
```

Full CLI details are in [WORKFLOW.md](WORKFLOW.md). A convenience wrapper is in
[run_pipeline.sh](run_pipeline.sh). Use `--dry-run` on Phase 1 or Phase 2 to validate
inputs without running training.

## Constraints for AI Assistance

- **Do not modify scientific logic, model architecture, loss functions, or hyperparameters
  unless explicitly asked.** This includes `scripts/core/` and `scripts/analysis/`.
- **Do not change configs** (`configs/`) unless explicitly asked.
- **Do not delete or overwrite anything in `results/`**; new experiment outputs belong in
  a separate subdirectory (e.g., `results/experimental/`).
- Prefer editing existing files over creating new ones.
- Keep changes minimal and scoped to the task at hand.
- If a task touches scientific decisions, stop and ask before proceeding.

## Environment

```bash
conda env create -f environment.yml
conda activate genotype-embedding-env
```

Python 3.10. Key dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`,
`statsmodels`, `seaborn`, `matplotlib`, `umap-learn`, `hdbscan`, `requests`.
