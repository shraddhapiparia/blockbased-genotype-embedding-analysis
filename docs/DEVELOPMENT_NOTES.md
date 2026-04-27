# Development Notes

Design decisions and project evolution context not captured in code or commit history.

---

## Project Evolution

The project developed in two phases.

**Earlier phase — gene-based block VAEs.** The initial version used individual genes as
the unit of analysis. It included gene-level VAEs, loss-function comparisons,
attention-based aggregation, and clustering evaluation. This work is preserved in
`scripts/archive/` as historical context.

**Current phase — region-based LD block analysis.** The preferred and canonical
direction. LD blocks are defined regionally rather than by gene boundaries. All
active scripts, configs, and results files reflect region-based analysis. If there is
a conflict between archived gene-based logic and current region-based logic, the
region-based version takes precedence.

---

## Asthma-Relevant Regions of Interest

Interpretation of attention weights and block attributions prioritises these loci:

- **HLA class II** (6p21) — dominates the learned embedding space; highest attention
  weights consistently; confirmed with leave-HLA-out re-clustering
- **17q21** (ORMDL3/GSDMB) — major childhood asthma locus; validated with dedicated
  script 07
- **PDE4D** (5q21) — emerges as next most informative block after HLA removal;
  established asthma pharmacogenomics role
- **IL33 / IL1RL1 / IL18R1 / IL18RAP** (9p24 / 2q12) — type-2 inflammation pathway
- **FCER1A** (11q13) — IgE receptor; consistent with IgE being the strongest
  phenotype signal
- **STAT6** (12q13) — IL-4/IL-13 signalling
- **5q31 type-2 cytokine cluster** — IL4/IL5/IL13 region

Keep region IDs (e.g. `region_6p21_HLA_classII_sb15`) intact in output files and
figures so results remain interpretable without consulting block_order.csv.

---

## HLA Observation

Early UMAP analysis (notebook-based, later formalized in
`scripts/analysis/02_subject_cluster_analysis.py` step B) showed that HLA-related
regions strongly drive separation in the Phase 2 embedding space. This was treated
as the primary interpretability result and is now formally validated:

- η² = 0.767 for HLA sb15 block_PC1 predicting subject cluster (script 02, step C)
- Leave-HLA-out re-clustering reveals PDE4D as next informative block (script 03)
- Confounder analysis shows HLA signal is not explained by ancestry PCs (script 05)

---

## Output Directory Conventions

- `results/output_regions/` — Phase 1 outputs (canonical real-data run)
- `results/output_regions2/` — Phase 2 outputs (canonical real-data run)
- `results/phase_comparison/` — Phase 1 vs Phase 2 comparison (script 06)
- `results/synthetic_test/` / `results/synthetic_test2/` — smoke-test outputs only;
  no biological meaning
- `results/logs/` — per-run timestamped log directories from `run_pipeline.sh`

Experimental or exploratory runs should go to a separate directory
(e.g. `results/exp_<descriptor>/`) to avoid overwriting canonical outputs.

---

## Secondary Modeling Goal (Future)

A secondary end-to-end model (jointly trained VAE + Transformer) is a future goal.
It should live alongside the frozen-embedding pipeline, not replace it. Results from
both approaches should be easy to compare. This is not part of the active workflow.
