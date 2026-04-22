# CLAUDE.md — Block-Based Genotype Embedding Project

## Goal
Ensure reliable development and analysis of LD block-based genotype embeddings
using VAE (Phase 1) and transformer (Phase 2) pipelines.

---

## Core Rules

- Do not hallucinate schema → verify before writing SQL or joins
- Do not assume files exist → check paths and outputs
- Do not claim results without execution or validation
- If something can be verified by running code, prefer execution over reasoning
- Do not fake domain expertise → state limits and suggest validation methods
- Push back on incorrect assumptions instead of building on them
- State uncertainty explicitly when unsure
- Verify API syntax against correct library versions (do not rely on memory)
- Update `.memory/` after every successful pipeline run
- Avoid em dashes; use simple and clear sentence structures

---

## Pipeline Overview

Phase 1:
- Input: genotype data (LD blocks)
- Output: block-level embeddings

Phase 2:
- Input: Phase 1 embeddings
- Output:
  - subject embeddings (N × 64)
  - contextual block embeddings (N × B × 64)
  - attention weights

Analysis:
- clustering (KMeans, PCA)
- block-PC1 associations
- phenotype associations
- HLA masking experiments

---

## Active Code Structure

- `scripts/core/VAE_phase1.py` → block embedding model
- `scripts/core/attention_phase2.py` → transformer aggregation
- `scripts/analysis/` → all downstream analysis
- `scripts/archive/` → legacy (ignore)
- `configs/` → model + pipeline configs

Never use archive scripts for new analysis.

---

## Key Scientific Constraints

- HLA region dominates embedding structure
- Multiple independent HLA sub-block directions exist
- PDE4D emerges as secondary axis after HLA masking
- Phenotype signal is weak but enriched in asthma-related regions

Claude must:
- avoid over-interpreting weak signals
- separate biological signal vs technical artifact
- always check if HLA is driving results

---

## Common Failure Modes

- Incorrect joins between embeddings and phenotype
- Confusion between genotype PC and embedding PC
- Silent row drops during merges
- Using wrong phenotype encoding
- Treating exploratory results as final

---

## Validation Checklist

For every analysis:

- Are subject counts consistent across steps?
- Do embeddings align with block order?
- Are correlations computed correctly (Pearson vs Spearman)?
- Are covariates applied where required?
- Do known signals (HLA) behave as expected?
- Are results reproducible?

---

## Memory System

Maintain `.memory/` with:

- failed experiments (e.g., leakage in PLS gradients)
- fixes (e.g., cross_val_predict)
- key insights (e.g., HLA dominance, PDM corr = 0.68)

---

## Example Prompts

Design:
"Given Phase 1 and Phase 2 pipeline, list all assumptions that could break biological interpretation."

Analysis:
"Compare Phase 1 vs Phase 2 embeddings. What metrics are valid given different dimensions?"

Debugging:
"Cluster separation looks too strong. Could this be driven by HLA? How to test?"

---

## Final Principle

The model is useful only if:
- it recovers known biology (HLA, PDE4D)
- results are reproducible
- interpretations are cautious and verified

Do not optimize for impressive results.
Optimize for correct results.
