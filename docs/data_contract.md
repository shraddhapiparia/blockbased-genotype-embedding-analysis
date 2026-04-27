# Phase 1 → Phase 2 Data Contract

This document defines the format, shape, and ordering conventions that Phase 2 expects
from Phase 1 outputs. Violating these contracts produces silent errors (wrong geometry)
or loud errors (shape mismatches) that can be hard to debug. Read this before modifying
either pipeline stage.

---

## Phase 1 Inputs

### Per-block genotype files

- **Format:** PLINK additive-dosage `.raw` files, one per LD block.
- **Location:** `data/region_blocks/<block_id>.raw` (path from `data.raw_dir` in config).
- **Schema:**

  ```
  FID  IID  PAT  MAT  SEX  PHENOTYPE  <SNP1>_<allele>  <SNP2>_<allele>  ...
  ```

  Columns 0–5 are PLINK subject metadata; the pipeline loader skips them.
  SNP dosage columns contain integers in {0, 1, 2}.

- **Subject ordering:** All block files must contain **the same set of subject IIDs in
  the same row order.** The loader reads subject IDs from the first block and asserts
  that all subsequent blocks match. A mismatch raises a `ValueError` before training.

### Block manifest

- **Location:** `data/block_plan/manifest.tsv` (path from `data.block_def` in config).
- **Required columns:** `block_id`, and optionally `chr`, `start`, `end`, `n_snps`.
  The `block_id` column must match the filenames in `data.raw_dir` exactly
  (i.e. `<block_id>.raw` must exist).
- **Row order:** Determines the canonical block order used throughout the pipeline.
  Do not change this order between Phase 1 and Phase 2 runs on the same cohort.

---

## Phase 1 Outputs

All outputs are written to `data.output_dir` (default `results/output_regions/`).

### `subjects.csv`

| Column | Description |
|---|---|
| `IID` | Subject identifiers, one per row |

Row order matches the row order of the input `.raw` files. Phase 2 reads this file to
establish the subject-level index; it must not be modified between runs.

### `block_order.csv`

| Column | Description |
|---|---|
| `block_id` | Block identifier |
| `block_index` | 0-based integer index (determines position in `all_blocks.npy`) |
| `n_snps` | Number of SNPs in the block |

Row order defines the canonical block dimension in `all_blocks.npy`. Phase 2 uses this
to map attention weights back to block names.

### `all_blocks.npy`

- **Shape:** `(N, B, d_max)` where:
  - `N` = number of subjects (matching `subjects.csv` row count)
  - `B` = number of blocks (matching `block_order.csv` row count)
  - `d_max` = maximum latent dimension across all blocks
- **dtype:** `float32`
- **Axis 0 (N):** Subject order matches `subjects.csv`.
- **Axis 1 (B):** Block order matches `block_order.csv`.
- **Axis 2 (d_max):** Per-block latent vector; blocks with fewer latent dimensions are
  right-padded with zeros to `d_max`.
- **Location:** `<output_dir>/<loss_type>/embeddings/all_blocks.npy` — one file per
  loss type (MSE, BCE, ORD, etc.).

### `train_idx.npy` / `val_idx.npy`

- **Shape:** `(n_train,)` and `(n_val,)` — integer indices into the subject axis.
- Phase 2 respects this split to avoid data leakage.

---

## Phase 2 Input Expectations

Phase 2 reads Phase 1 outputs from `phase1_dir` (configured in `configs/config_phase2.yaml`).

| File | Expectation |
|---|---|
| `subjects.csv` | Must exist; IID column defines the subject index |
| `block_order.csv` | Must exist; defines block names and their axis-1 order in embeddings |
| `train_idx.npy` | Must exist; used to partition training vs validation |
| `val_idx.npy` | Must exist |
| `<loss>/embeddings/all_blocks.npy` | Must exist for each loss in `loss_functions` |

**Critical:** Phase 2 does not re-validate that `all_blocks.npy` was generated from the
same subjects as `subjects.csv`. If you rerun Phase 1 with a different cohort or subject
filter without also rerunning Phase 2, subjects and embeddings will be silently
misaligned.

---

## Phase 2 Outputs

All outputs are written to `output_dir` (default `results/output_regions2/`).

### `<loss>/embeddings/individual_embeddings.npy`

- **Shape:** `(N, d_model)`
- **Axis 0:** Same subject order as Phase 1 `subjects.csv`.
- **dtype:** `float32`

### `<loss>/embeddings/individual_embeddings.csv`

Same data as the `.npy`, with an `IID` column prepended for human-readable inspection.

### `<loss>/embeddings/pooling_attention_weights.csv`

- **Columns:** `IID` + one column per block (named by `block_id` from `block_order.csv`)
- **Values:** Softmax attention weights summing to 1.0 across blocks per subject.
- **Shape:** `(N, 1 + B)` rows × columns

### `<loss>/clustering/cluster_labels.csv`

- Per-subject cluster assignments for each clustering method and `k` tried.
- Produced by `scripts/core/attention_phase2.py`.

### `phase2_summary.csv`

One row per loss type; training statistics (epochs, validation loss, silhouette, etc.).

---

## Common Failure Modes

| Symptom | Likely cause | Remedy |
|---|---|---|
| `ValueError: IID set mismatch` or mismatched row counts | Phase 1 was re-run with a different subject subset or filter | Re-run Phase 2 after re-running Phase 1; never mix outputs from different cohort runs |
| `ValueError: block order mismatch` or wrong attention weight labels | `block_order.csv` was manually edited or manifest changed between runs | Re-run Phase 1 to regenerate `block_order.csv`, then re-run Phase 2 |
| NaN embeddings in `all_blocks.npy` | Missing or all-zero SNP columns in a block `.raw` file; NaN propagates through the VAE encoder | Inspect the block `.raw` file for missing dosages; check VAE reconstruction loss |
| Wrong `all_blocks.npy` shape | Stale output from a previous run with different config (different number of subjects or blocks) | Delete the output directory and re-run Phase 1 |
| Phase 2 clustering silhouette = NaN | All subjects in one cluster; likely upstream embedding collapse | Check Phase 2 training loss and embedding variance |
| `individual_embeddings.npy` and `pooling_attention_weights.csv` subject counts differ | Partial Phase 2 run was interrupted and outputs are from different epochs | Re-run Phase 2 from scratch |

---

## Synthetic Smoke Test Data Contract

The synthetic data at `data/synthetic/` satisfies the same contract:

| Property | Value |
|---|---|
| N subjects | 30 |
| B blocks | 4 |
| d_max (Phase 1 MSE) | 4 |
| d_model (Phase 2 ORD) | 16 |
| Expected `all_blocks.npy` shape | (30, 4, 4) |
| Expected `individual_embeddings.npy` shape | (30, 16) |

These values are verified by `tests/test_smoke_outputs.py`.
