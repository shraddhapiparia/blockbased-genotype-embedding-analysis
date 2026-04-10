# ANALYSIS_DECISIONS.md

## Preferred Analysis Unit

The preferred analysis unit is region-based LD blocks.

Earlier iterations used genes as the primary unit of analysis, but the project has now moved to genomic regions and should continue in that direction.

If there is a conflict between gene-based and region-based logic, use the region-based version.

---

## Primary Workflow

The primary workflow is:

- load or generate block embeddings
- freeze the block embeddings
- aggregate block embeddings into subject-level embeddings
- perform PCA, UMAP, clustering, and phenotype association
- interpret the resulting regions

This workflow is the current trusted path.

---

## Secondary Workflow

A secondary end-to-end workflow should also be developed.

This end-to-end workflow should:

- be implemented separately from the frozen embedding pipeline
- allow side-by-side comparison
- not replace the primary workflow

The frozen embedding pipeline remains the canonical direction.

---

## Current Trusted Results

The current trusted results are the region-based outputs already present in `results/`.

These include:

- region-based subject embeddings
- PCA outputs
- UMAP outputs
- cluster labels
- clustering metrics
- phenotype association tables

These should not be overwritten.

Preferred organization:

```text
results/
  final/
  experimental/
```

## Historical Materials

Older iteration-1 materials are still useful because they include:

- Block VAE architectures
- Loss comparisons
- Training metrics
- Evaluation strategies
- Clustering logic

However, these should now be treated as historical context rather than the current final direction.

Suggested location:

archive/
└── iteration1_gene_based/

---

## Important Notebook Findings

A notebook-based UMAP analysis suggested that HLA-related regions play an important role in separating subjects in the learned embedding space.

This result is scientifically important and should be reflected in:

- Project documentation
- Interpretation notes
- Later reusable scripts

The current notebook implementation may remain exploratory for now, but the finding should not be lost.

---

## Metrics To Preserve

The repository should preserve and compare:

- Reconstruction loss
- Validation loss
- KL divergence
- Clustering metrics
- Silhouette score
- Attention-based block ranking
- Phenotype association statistics

Metrics from older gene-based experiments may still be useful for comparison, but the main reported metrics should come from region-based analysis.

---

## Region Selection Guidance

The project originally experimented with gene-level blocks and many candidate genes.

The preferred approach now is:

- Select a biologically meaningful subset of genomic regions
- Report metrics only for selected region-based blocks
- Avoid reverting to the earlier gene-based organization

The region-based outputs currently in `results/` should define which blocks are canonical.

---

## Future Interpretation Goals

Future interpretation should include:

- Ranking important regions
- Understanding why HLA-related regions drive separation
- Connecting important regions to asthma biology
- Integrating attention weights or attribution scores
- Comparing frozen-embedding and end-to-end pipelines