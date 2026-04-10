# Analysis Scripts

Post-hoc analysis scripts run after `scripts/core/` completes Phase 1 and Phase 2.
Scripts are numbered to suggest execution order. Most are independent and can be
run in any order after Phase 2 outputs exist.

See [WORKFLOW.md](../../WORKFLOW.md) for full CLI examples and I/O paths.

---

## Script inventory

### `01_block_embedding_phenotype_analysis.py`

**Purpose:** Per-block phenotype association using block-level PCA (PC1).
For each of 174 blocks × phenotype: runs unadjusted and ancestry-adjusted OLS/logistic
regression; produces the main phenotype association table used downstream.

| | |
|---|---|
| **Inputs** | `block_contextual_repr.npy`, `pooling_attention_weights.csv`, `block_order.csv`, eigenvec, phenotype CSV |
| **Outputs** | `all_blocks_pheno_analysis/phenotype_block_associations.tsv`, per-phenotype heatmaps |
| **Feeds** | `06_`, `07_` (uses `phenotype_block_associations.tsv`) |

---

### `02_subject_cluster_analysis.py`

**Purpose:** Three-step subject-level clustering and HLA analysis.
- **Step A** (`--step clustering`): HDBSCAN subject clustering, PCA/UMAP, block-driver heatmap.
- **Step B** (`--step umap_hla`): UMAP colored by attention weights; top-block identification.
- **Step C** (`--step hla_analysis`, default): KMeans(k=3); HLA block_PC1 and genotype PC
  association per cluster; raw vs residual η² comparison.

| | |
|---|---|
| **Inputs** | `individual_embeddings.npy`, `block_contextual_repr.npy`, `pooling_attention_weights.csv`, `block_order.csv`, eigenvec |
| **Outputs** | `subject_cluster_assignments.csv`, PCA/UMAP plots, `hla_block_cluster_association.tsv`, `genotype_pc_cluster_association.tsv`, eta² residual table |
| **Feeds** | `03_` (leave-HLA-out), `04_` (cluster stability) |

```bash
python scripts/analysis/02_subject_cluster_analysis.py --step hla_analysis
python scripts/analysis/02_subject_cluster_analysis.py --step clustering \
    --subject-embeddings results/.../individual_embeddings.npy \
    --outdir results/.../subject_cluster_analysis/
```

---

### `03_leave_hla_out_analysis.py`

**Purpose:** Leave-HLA-out anti-circularity validation. Zero-masks HLA block embeddings,
re-runs Phase 2 attention without HLA, clusters subjects on new embeddings, then tests
whether the *original* HLA block_PC1 still separates the new clusters.

| | |
|---|---|
| **Inputs** | Phase 1 `all_blocks.npy`, `block_order.csv`, `configs/config_phase2_noHLA.yaml` |
| **Outputs** | Re-clustered subject assignments, HLA block_PC1 separation plots |
| **Key result** | HLA structure persists after removal → not circular |

---

### `04_cluster_stability_analysis.py`

**Purpose:** Validates that KMeans k=3 clusters are stable.
Three checks: (1) KMeans seed stability across 50 seeds (ARI), (2) algorithm robustness
(KMeans vs GMM vs HDBSCAN), (3) elbow + silhouette for k=2..6.

| | |
|---|---|
| **Inputs** | `individual_embeddings.npy` |
| **Outputs** | `cluster_stability/{ari_*.tsv, elbow_silhouette.tsv, subject_cluster_stability.tsv, figures/}` |

---

### `05_attention_confounder_analysis.py`

**Purpose:** Tests whether attention weights survive ancestry PC adjustment.
Per candidate block: OLS attention ~ asthma, OLS attention ~ asthma + PC1..PC10,
Pearson r with each PC.

| | |
|---|---|
| **Inputs** | `pooling_attention_weights.csv`, eigenvec, phenotype CSV |
| **Outputs** | `confounder_analysis/{block_asthma_pc_models.csv, block_pc_correlations.csv}`, heatmaps |

---

### `06_phase1_phase2_block_comparison.py`

**Purpose:** Quantifies embedding improvement from Phase 1 → Phase 2.
For each shared block: pairwise-distance-matrix (PDM) Spearman correlation, phenotype
association comparison, PDM r vs reconstruction MSE scatter.

| | |
|---|---|
| **Inputs** | Phase 1 per-block `.npy` embeddings, Phase 2 `block_contextual_repr.npy`, `pooling_attention_weights.csv` |
| **Outputs** | `phase_comparison/{pdm_correlations.csv, pheno_associations_phase*.csv, pheno_association_gain.csv, plots/}` |

---

### `07_17q21_validation.py`

**Purpose:** Two-stage validation for the 17q21 locus (sb1–sb5).
- **Step A** (`--mode validation`): genotype ↔ embedding alignment, SNP-level correlations,
  rsID lookup via Ensembl, 4 publication figures, per-subblock correlation tables.
- **Step B** (`--mode baseline`): OLS comparison of geno_PC1/norm3 vs emb_PC1/norm3
  for predicted FEV1; incremental R² tables (terminal output only).

| | |
|---|---|
| **Inputs** | `data/region_blocks/region_17q21_core_sb*.raw`, `block_contextual_repr.npy`, eigenvec, phenotype CSV, (Step A) `phenotype_block_associations.tsv` |
| **Outputs** | `17q21_validation/{figures/fig1–fig4.png, tables/snp_correlations_*.tsv}` |

```bash
python scripts/analysis/07_17q21_validation.py --mode all        # default
python scripts/analysis/07_17q21_validation.py --mode validation  # figures + tables only
python scripts/analysis/07_17q21_validation.py --mode baseline    # terminal comparison only
```
