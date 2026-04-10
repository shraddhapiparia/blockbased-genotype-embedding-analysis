# IMPLEMENTATION_DECISIONS.md

## Project Direction
- Moved from gene-based blocks to region-based LD blocks because region-based analysis is now canonical and trusted results are region-level.
- Region-based LD block selection is based on asthma-relevant loci and should be the default in all pipeline paths.
- Frozen block embeddings are primary because they allow stable, reproducible subject embedding aggregation and interpretation (Phase 1 generates block-level encodings; Phase 2 pools with attention).
- End-to-end model is secondary (comparison only) to avoid disrupting the validated frozen embedding path.

## Loss Functions
- Tested losses: MSE, MSE_STD, ORD, BCE, CAT.
- ORD performed better than MSE/MSE_STD in existing experiments.
- ORD is preferred for reproducing earlier block embedding workflow and base subject embeddings.

## Embedding Dimensionality
- Phase 1 numeric latent dimension: controlled by `latent_dim_by_snps` in `VAE_phase1.py`.
- Example: small blocks 4, medium 8, large 12 etc; final subject-level embedding in Phase 2 is 64 dimensions by default (attention aggregator output `d_model=64`).
- 64-dim subject embeddings chosen as a balanced embedding capacity for 997-subject dataset and downstream UMAP/clustering.

## Region Selection
- Region set is defined by `data/block_plan/manifest_regions.tsv`.
- Retain LD blocks with status OK; block-level quality metrics filtered in Phase 1/2.
- Region subsets in reporting are top attention blocks and those with highest association with PC/UMAP gradients.

## Filtering Thresholds
- No explicit global threshold in default script; per-block recon metrics and attention scores are used for post hoc block ranking.
- Candidate regions may be pruned if they fail to converge due to low SNP count or ID mismatch.

## Subject Inclusion
- Phase 1 intersects subjects across all selected blocks; missing subjects are excluded from block-level training.
- Final analysis uses 997 subjects (the intersection of available subject IDs after block alignment and phenotype merge).
- Keep an explicit row count log in each script regarding trimmed subjects.

## ID Alignment
- Required identifier columns include `IID` (in .raw embeddings and attention CSV), `S_SUBJECTID` or `subject_id` in phenotype data.
- Merge is done with `inner` join after harmonizing ID strings.
- Scripts must report numbers: embedding rows, phenotype rows, merge hits, and missing subjects.
- If mismatch is >5% by default, raise a warning.

## Missing Subjects Handling
- Missing subjects are dropped in merged datasets; preserve a CSV listing missing IDs for audit.
- If missing subjects appear in `results/output_regions/` but not phenotype, they are excluded from phenotype association.

## Clustering Methods
- Tested: KMeans (primary), silhouette, ARI, NMI, optionally HDBSCAN in notebook.
- Preferred: KMeans with silhouette-based model selection.

## PCA / UMAP Settings
- phase2 uses PCA for elbow and UMAP for visualization.
- notebook HLA script uses UMAP `n_neighbors=20`, `min_dist=0.1`, `metric='cosine'`, `random_state=42`.
- Significant separation in UMAP emerges with these settings; larger `n_neighbors`/`min_dist` can blur cluster boundaries.

## HLA Finding
- HLA-related regions appear to drive separation in embedding space (UMAP/attention analysis), as discovered in `notebooks/umap_hla_interpretation.ipynb`.
- Keep this in interpretation pipeline and report as supporting biological insight.

## Attention Aggregation Settings
- `attention_phase2.py` uses a transformer-style block aggregator + pooling attention weights.
- Attention weights are normalized at the softmax pooling stage; per-cluster and global means are computed in downstream analysis.
- Block ranking uses mean attention, coefficient of variation, and per-block reconstruction MAE/MSE from Phase 1.

## Trusted Outputs
- `results/output_regions/` and `results/output_regions2/` are trusted. Do not overwrite in place.
- New results should go to `results/final/` or `results/experimental/`.

## Exploratory vs. Canonical Code
- Exploratory: `notebooks/umap_hla_interpretation.ipynb`, `plots.py`.
- Canonical: `scripts/VAE_phase1.py`, `scripts/attention_phase2.py`, `scripts/analyze_phase2_blocks.py`, `scripts/plots_updated.py`.
- Notebook to migrate: `umap_hla_interpretation.ipynb` → `scripts/05_umap_hla_interpretation.py`.

## Important Architecture Details to Preserve
- Phase 1 supports variable SNP counts per block and uses padding to align block embeddings to shape (N, B, max_d) through `all_blocks.npy`.
- This padding is critical to ensure transformer compatibility and equal block interface in Phase 2.
- Phase 2 applies an attention aggregator to those block-level feature stacks.

## Implementation Risks / Propagation Checks
- Inconsistent block order across files can silently misalign attention scores and embeddings. Validate order in each stage.
- Sample ordering mismatches between embeddings and phenotype data can produce wrong association statistics. Validate with explicit index assertions.
- Shape mismatches in stacked arrays are prevented by block padding; re-check after adding new blocks.
- Output path confusion between `output_regions` and `output_regions2` can cause stale overwrites; use clear production paths.
- Config/CLI mismatch can skip key settings; always print effective config values and do not silently ignore overrides.
- Notebook hidden state may not translate to script; capture explicit data inputs and reproducible random seeds.

## Open Questions / Unresolved Choices
1. Should we enforce a strict block-level inclusion threshold for minimum N SNPs or minimum validation LD score?
2. Should we keep all loss types in production or run only ORD by default?
3. If we add end-to-end model, which architecture should be used (transformer-based, MLP, or dedicated feed-forward)?
4. Should subject embedding dimension remain fixed at 64 or become config-driven with automatic selection?
5. What exact HLA block list should be hard-coded for interpretation versus data-driven identification?
