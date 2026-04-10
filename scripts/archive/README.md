# Archive

These scripts are **not part of the active workflow**. They are preserved for
reproducibility reference and historical context, but should not be run as part
of a fresh analysis.

## Why files are here

| File | Reason |
|---|---|
| `01_phase1_block_embedding.py` | Merged — CLI wrapper whose logic is now in `core/VAE_phase1.py` |
| `02_phase2_attention_aggregation.py` | Merged — CLI wrapper whose logic is now in `core/attention_phase2.py` |
| `03_block_analysis.py` | Wrapper — thin CLI over `core/analyze_phase2_blocks.py`; no added logic |
| `04_plotting.py` | Wrapper — thin CLI over `core/plots_updated.py`; no added logic |
| `08_block_embedding_umap_analysis.py` | Obsolete — superseded; not part of the final analysis pipeline |
| `14_block_ftest_multipc.py` | Exploratory — joint F-test multi-PC sensitivity; no outputs saved |
| `15_block_scalar_norm_multipc.py` | Exploratory — L2-norm multi-PC comparison; no outputs saved |
| `16_quickstats.py` | Debug — inline Wilcoxon test snippet; no function structure |
| `plots.py` | Superseded — earlier version of `core/plots_updated.py`; hardcoded absolute paths to old machine layout |
| `quick_block_plots.py` | Exploratory — earlier gradient analysis draft; internally titled `08_block_embedding_umap_analysis.py` |
| `check.py` | Debug — prints file paths and CSV head; one-off inspection script |
| `05_umap_hla_interpretation.py` | Merged — Step B of `analysis/02_subject_cluster_analysis.py` |
| `09_unsupervised_subject_cluster_analysis.py` | Merged — Step A of `analysis/02_subject_cluster_analysis.py` |
| `10_hla_cluster_analysis.py` | Merged — Step C of `analysis/02_subject_cluster_analysis.py` |
| `13_phenotype_cluster_analysis.py` | Archived — phenotype~cluster analysis; superseded by `analysis/01_` and `analysis/02_` |
| `17q21_baseline_comparison.py` | Merged — Step B of `analysis/07_17q21_validation.py` |
| `17q21_genotype_embedding_validation.py` | Merged — Step A of `analysis/07_17q21_validation.py` |

## Import note

The wrapper files (`01_`–`04_`) contain `from scripts.<module>` imports that now
point to stale locations (the flat `scripts/` layout before reorganization). They
will not run from this archive directory without import path updates. This is
intentional — use the active `core/` and `analysis/` scripts instead.
