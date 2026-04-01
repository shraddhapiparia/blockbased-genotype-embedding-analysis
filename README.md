# Genotype Embedding Analysis for Asthma-Relevant LD Blocks

This repository contains an evolving analysis framework for learning and evaluating subject-level genotype embeddings derived from linkage disequilibrium (LD)-aware genomic blocks, with a focus on asthma-relevant loci and downstream phenotype association. The project was developed to move beyond conventional SNP-level analysis in settings where the number of variants is much larger than the number of samples, making direct modeling difficult and often biologically hard to interpret. Instead of treating all SNPs independently, this work groups variants into biologically meaningful LD blocks, learns compact representations for each block, and aggregates those representations into subject-level embeddings that can be used for visualization, clustering, and statistical association with clinical phenotypes.

The overall motivation is that SNP-level genotype matrices are extremely high dimensional, sparse in biological meaning at the individual feature level, and often poorly suited for standard machine learning workflows when sample size is limited. In contrast, LD-aware block embeddings offer a more structured representation of the genome. They reduce dimensionality while preserving local genomic context, create a more manageable input for downstream models, and may capture disease-relevant variation in a way that is easier to interpret than raw genotype matrices. This repository reflects that goal and documents both the analysis completed so far and the next steps needed to make the project more reproducible, interpretable, and extensible.

## Project Goal

The main goal of this project is to learn compact genotype embeddings from asthma-relevant LD blocks and evaluate whether these embeddings reveal meaningful structure across individuals. Specifically, the project asks whether block-based embeddings can recover stronger subject-level variation than standard genotype PCA, whether the resulting subject embeddings cluster into biologically or clinically meaningful groups, and whether individual blocks or latent features are associated with phenotypes relevant to asthma severity, treatment response, or clinical heterogeneity.

## Current Status

The repository currently contains code and analysis outputs for multiple stages of the workflow, including data preparation, embedding construction, dimensionality reduction, clustering, and preliminary phenotype association. Some steps are more mature than others. The central analysis concept has been established, and the core representation-learning direction is working. However, the repository is still in an active development state and needs organization, cleanup, reproducibility improvements, and a clearer execution flow before it is ready to serve as a polished research or portfolio project.

Completed work includes the construction and analysis of subject-level embeddings derived from LD-aware genotype blocks, exploratory visualization using PCA and UMAP, clustering of learned embeddings, comparison with conventional genotype PCA, and downstream association of blocks or embeddings with clinical phenotypes. Intermediate files, exploratory notebooks, and analysis scripts may coexist in the repository. Some are final or close to final, while others represent iterations, debugging, or exploratory work and should not yet be treated as canonical pipeline components.

## Scientific Framing

Traditional genotype analyses often begin with SNP-level matrices or LD-pruned genotype sets. While useful, those approaches may fail to capture richer local structure, especially when many correlated SNPs act together within genomic regions. This project reframes genotype representation as a hierarchical problem. Instead of analyzing individual SNPs in isolation, it represents each genomic block as a structured input, learns a latent encoding for that block, and then combines those block encodings into a subject-level representation. This makes it possible to study both local and global genomic structure. The block level preserves regional context, while the subject level supports clustering, phenotype modeling, and comparison across individuals.

This framing is particularly relevant in asthma, where disease risk and heterogeneity are shaped by multiple loci of modest effect rather than a single deterministic mutation. Asthma-associated variation often lies in complex immune-related regions, including loci such as IL1RL1, IL33, FCER1A, STAT6, and the 17q21 region. A block-based embedding strategy may therefore be better aligned with the biology than a flat SNP-level input matrix.

## Data Overview

The analyses in this repository are based on genotype-derived block representations and subject-level phenotype data. The phenotype table includes demographic, clinical, and asthma-relevant outcome variables. The project also uses learned subject embeddings, block metadata, clustering outputs, and statistical association tables generated during downstream analysis. In several places, subject identifiers are aligned through a common IID field or equivalent subject identifier column. Because the project has gone through multiple iterations, exact file names and paths may vary across scripts, but the same core data objects reappear repeatedly: phenotype data, embedding matrices, block order metadata, attention or pooling outputs, clustering labels, and result summaries.

The repository may contain data placeholders or scripts that assume files exist outside version control. Raw data should not be committed if restricted, large, or sensitive. Instead, this repository is intended to preserve the analysis logic, expected file formats, derived summaries where appropriate, and documentation sufficient for rerunning the workflow with access-controlled inputs.

## Analysis Workflow

The workflow can be thought of as six conceptual stages. First, genotype data are organized into LD-aware genomic blocks chosen to reflect asthma-relevant loci or other biologically meaningful regions. Second, each block is converted into a learned latent representation, such as an embedding generated by an autoencoder or related dimensionality reduction model. Third, block-level embeddings are aggregated across the genome to create a subject-level embedding that summarizes an individual's genotype structure across all selected regions. Fourth, the subject embeddings are explored using dimensionality reduction methods such as PCA and UMAP, and clustering methods such as KMeans are used to identify structure among subjects. Fifth, clustering outputs and embedding dimensions are compared against known phenotypes to determine whether learned genomic structure maps onto clinical variables. Sixth, block-level contributions, association statistics, and interpretation steps are used to identify genomic regions that may be driving observed variation.

This repository may implement parts of these stages in notebooks, scripts, or utility modules. Over time, the goal is to convert the workflow into a more modular and reproducible pipeline with clearly named entry points and documented dependencies.

## What Has Been Built So Far

The work completed so far supports the central idea that block-based genotype embeddings can reveal stronger structure than conventional PCA on high-dimensional genotype matrices. In exploratory analyses, PCA on learned embeddings captured substantially more variance in leading components than PCA performed on LD-pruned SNP matrices, suggesting that the learned latent representation compresses relevant structure more effectively. Subject-level embeddings have been visualized with PCA and UMAP, and clustering methods have been applied to assess whether individuals partition into distinct groups. Downstream analyses have also connected selected blocks or embedding-derived features to phenotypes such as BMI, smoking exposure, lung function measures, and asthma-related outcomes.

The repository also contains evidence of iterative troubleshooting around subject identifier matching, phenotype merges, clustering label alignment, and export of summary tables. These are important parts of the project's real development history and should be treated as part of the working research process rather than as noise. At the same time, one of the next priorities is to separate reusable pipeline code from one-off debugging code.

## Why This Approach Matters

This project is motivated by a broader challenge in statistical genetics and scientific machine learning: how to represent genomic variation in a way that is computationally tractable, biologically meaningful, and useful for downstream inference. Flat genotype matrices often force a tradeoff between dimensionality reduction and interpretability. Block-based embeddings provide an alternative by compressing local genomic variation into units that still correspond to real genomic regions. They also open the door to hierarchical modeling, including attribution or attention mechanisms that can help identify influential blocks. In practical terms, this means the project is not only about better clustering but also about creating a more interpretable genome representation that can support future work in disease subtype discovery, treatment response modeling, and region-level prioritization.

## Repository Organization

The repository is being structured so that core code, exploratory analysis, results, and documentation are easier to separate and understand. The `data/` directory should hold raw and processed input tables or references to external inputs. The `scripts/` directory should contain command-line or sequential analysis scripts that perform individual stages of the workflow. The `src/` directory should eventually hold reusable Python modules for loading data, validating identifiers, computing dimensionality reduction, clustering embeddings, running association tests, and generating summary outputs. The `notebooks/` directory can retain exploratory or interpretive analyses that are useful for development but are not the primary execution path. The `results/` directory should store tables, figures, and model artifacts generated from reproducible runs. The `docs/` directory can hold project notes, method explanations, and figure descriptions. The `archive/` directory can be used for outdated or superseded exploratory files that are kept for reference but are no longer part of the main analysis path.

## Expected Inputs

The workflow assumes the existence of several core input types. These may include a phenotype table with one row per subject, a subject embedding matrix or block-level embedding outputs, metadata describing block identities and ordering, and optional clustering or attention-weight files from prior runs. The phenotype table should include a unique subject identifier column that can be aligned with embeddings. The embedding table or matrix should either already include subject identifiers or be accompanied by a mapping file. Block metadata should describe which latent features correspond to which genomic regions. Additional files such as ancestry PCs, cluster label tables, and phenotype covariates may be required for certain downstream models.

Because file names may differ across development stages, one of the next cleanup tasks is to define a canonical input specification and ensure all scripts read from it consistently.

## Expected Outputs

Expected outputs include subject-level embedding tables, PCA and UMAP coordinates, clustering labels, clustering evaluation summaries, block-level or embedding-level phenotype association tables, and publication-quality figures. Figures may include PCA scatterplots, UMAP projections, cluster overlays, explained variance plots, heatmaps of block contributions, and phenotype comparisons across clusters. Result tables may include regression coefficients, standard errors, p-values, odds ratios for binary outcomes, and multiple-testing-adjusted significance values. As the repository matures, these outputs should be written into stable and clearly named subdirectories under `results/`.

## Known Gaps

Although the scientific direction is established, the repository still has several gaps. First, the project needs a clearer end-to-end execution order so that a new collaborator or code assistant can understand how raw or intermediate inputs become final results. Second, some code currently reflects exploratory analysis and debugging rather than polished pipeline modules. Third, identifier harmonization and merge validation need to be formalized so that mismatches between phenotype and embedding tables are caught consistently. Fourth, the analysis would benefit from clearer documentation of assumptions, phenotype coding, and covariate handling. Fifth, the repository should explicitly distinguish validated outputs from exploratory or draft outputs. Sixth, environment setup and dependency pinning should be improved so the project can be rerun more easily.

## Near-Term Next Steps

The highest-priority next steps are repository organization, reproducibility cleanup, and formalization of the analysis pipeline. The first step is to identify the true entry points of the current workflow and create a documented sequence of scripts or notebooks that represent the main analysis path. The second step is to move reusable logic out of exploratory notebooks and into `src/` or well-named scripts. The third step is to create a configuration-driven way to define input files, identifier columns, phenotype columns, and output paths. The fourth step is to standardize result writing so every run produces outputs in the same locations with consistent names. The fifth step is to document phenotype processing decisions, binary encodings, and covariate use. The sixth step is to improve interpretation by linking influential embedding features or attention weights back to block metadata and biological loci.

## Longer-Term Directions

This repository also serves as a foundation for future scientific and technical extensions. Methodologically, the block embeddings could be improved with more formal latent models, attribution methods, or transformer-based aggregation across blocks. Analytically, the subject embeddings could be used for subgroup discovery, phenotype prediction, ancestry-aware adjustment, or multimodal integration with transcriptomic, proteomic, or clinical features. Biologically, the most influential blocks could be compared against known asthma loci, eQTL resources, regulatory annotations, or pathway-level mechanisms. From a software perspective, the project could be converted into a reproducible CLI-driven workflow or pipeline that supports multiple cohorts and alternate block definitions.

## Guidance for Code Assistants and Collaborators

If you are inspecting this repository as a code assistant or collaborator, start by identifying the files that implement the main embedding analysis workflow, the files that define or load subject embeddings, the files that perform phenotype merges, and the files that generate PCA, UMAP, clustering, or association outputs. Do not assume every script represents a finalized component. Some files are exploratory and may duplicate logic developed elsewhere. Before making changes, first produce an inventory of what exists, what appears complete, what appears partial, and which files seem to define the current canonical path. Prefer preserving working logic while improving structure around it. Do not overwrite validated results unless explicitly requested. Prioritize reproducibility, clear input/output boundaries, and documentation of assumptions.

## Reproducibility

Environment setup is still being consolidated. The intended approach is to provide either a conda environment file or a pip requirements file under `environment/`, along with notes on Python version and major dependencies such as NumPy, pandas, scikit-learn, matplotlib, and any additional libraries used for dimensionality reduction or modeling. As the repository matures, all core scripts should run from a documented environment and write outputs deterministically to versioned result directories. Random seeds should be fixed where appropriate for clustering and embedding comparisons.

## Versioning Philosophy

This repository reflects an active scientific workflow. Not every script or output was produced under final, locked assumptions. Some analyses were developed iteratively, and parts of the repository may contain alternative versions of the same idea. Rather than deleting that history immediately, the preferred strategy is to preserve useful development artifacts while making the main workflow explicit. Older or superseded materials can be moved into `archive/` once the canonical execution path is defined.

## What This Repository Demonstrates

At its core, this repository demonstrates an attempt to rethink genotype representation for downstream scientific analysis. It combines statistical genetics intuition with machine learning representation learning, organizes variation at the LD-block level rather than the single-variant level, and uses the resulting embeddings for subject-level structure discovery and phenotype association. It also demonstrates practical research engineering: dealing with identifier mismatches, high-dimensional data constraints, exploratory visualization, clustering evaluation, and the transition from ad hoc analysis to a more reproducible project structure.

## Immediate TODO

The immediate to-do list is to identify canonical scripts, define a stable input contract, create a single end-to-end execution path, document all expected files and columns, standardize outputs under `results/`, and move one-off analysis code into either `src/` or `archive/`. Once that cleanup is complete, the next analytical priorities are to strengthen embedding interpretation, improve phenotype association summaries, and better connect influential blocks back to biological knowledge.

## Contact and Notes

This repository is under active development and should be interpreted as a working research codebase rather than a finished software package. The project is intended both as a scientific analysis framework and as a reproducible record of method development. As the structure becomes clearer, this README will be updated to point to specific scripts, result folders, and recommended execution commands.
