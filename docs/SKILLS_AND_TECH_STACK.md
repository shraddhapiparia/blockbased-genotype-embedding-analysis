
And a second file that may help Claude Code understand what technologies are already intentional in the project:

```md
# SKILLS_AND_TECH_STACK.md

## Scientific Domain

- Statistical genetics
- Asthma genetics
- Complex disease genomics
- LD-aware genomic region analysis
- Phenotype association analysis
- Representation learning for genotype data

## Programming Languages

- Python
- R
- Bash

## Core Python Libraries

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy
- statsmodels

## Dimensionality Reduction and Clustering

- PCA
- UMAP
- KMeans
- silhouette score
- adjusted Rand index
- normalized mutual information

Optional or experimental:

- HDBSCAN

## Modeling Approaches Already Used

- Autoencoders
- Block-level genotype embeddings
- Attention-based aggregation
- Logistic regression
- Linear regression
- Multiple-testing correction

## Frequently Used File Types

- `.csv`
- `.npy`
- `.tsv`
- `.eigenvec`
- `.png`
- `.pdf`

## Common Input Objects

- phenotype table
- embedding matrix
- cluster labels
- block metadata
- ancestry PCs
- attention weights

## Important Phenotypes Already Used

- BMI
- Smoking exposure
- Lung function
- Asthma severity
- Oral steroid use
- Hospitalization history

## Expected Project Capabilities

The final project should be able to:

- Load phenotype and embedding data
- Validate subject identifiers
- Merge data consistently
- Run PCA and UMAP
- Cluster subject embeddings
- Associate embeddings or blocks with phenotypes
- Generate figures and tables automatically
- Preserve reproducibility and logging

## Important Constraints

- Sample size is small relative to SNP dimensionality
- Subject ID mismatches are common
- Existing outputs should not be overwritten
- Some scripts are exploratory and should not be treated as final
- Reproducibility is more important than aggressive refactoring

## Desired Future Additions

- Config-driven execution
- Biological annotation of important blocks
- eQTL integration
- Attribution methods
- Transformer-based aggregation
- Cleaner command-line workflow
- Better automated report generation