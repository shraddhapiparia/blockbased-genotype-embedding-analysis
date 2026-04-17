#!/usr/bin/env bash
# Full pipeline: Phase 1 VAE → Phase 2 Attention → Post-hoc analysis
#
# Usage:
#   ./run_pipeline.sh              # run everything
#   ./run_pipeline.sh --dry-run    # validate inputs only (no training)
#
# Prerequisites:
#   conda activate genotype-embedding-env
#   Data must be present at paths specified in configs/config_phase1.yaml and
#   configs/config_phase2.yaml

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

# --- Input checks ---------------------------------------------------------
# Fail with a clear message before starting any training if restricted inputs
# are absent. These files are gitignored; obtain access separately.
check_inputs() {
  local missing=0
  local checks=(
    "data/region_blocks:genotype block directory"
    "data/block_plan/manifest.tsv:block manifest"
    "metadata/COS_TRIO_pheno_1165.csv:phenotype table"
    "metadata/ldpruned_997subs.eigenvec:ancestry eigenvectors"
  )
  for entry in "${checks[@]}"; do
    path="${entry%%:*}"
    label="${entry##*:}"
    if [[ ! -e "$path" ]]; then
      echo "ERROR: missing restricted input — $label ($path)" >&2
      missing=1
    fi
  done
  if [[ $missing -eq 1 ]]; then
    echo "Obtain access to restricted data before running the pipeline." >&2
    exit 1
  fi
}

check_inputs

if $DRY_RUN; then
  echo "[dry-run] Validating Phase 1 inputs..."
  python scripts/core/VAE_phase1.py --config configs/config_phase1.yaml --dry-run
  echo "[dry-run] Validating Phase 2 inputs..."
  python scripts/core/attention_phase2.py --config configs/config_phase2.yaml --dry-run
  echo "[dry-run] Input validation passed."
  exit 0
fi

echo "=== Phase 1: Per-block VAE ==="
python scripts/core/VAE_phase1.py --config configs/config_phase1.yaml

echo "=== Phase 2: Cross-block Attention ==="
python scripts/core/attention_phase2.py --config configs/config_phase2.yaml

echo "=== Core: Block analysis and summary plots ==="
python scripts/core/analyze_phase2_blocks.py
python scripts/core/plots_updated.py

echo "=== Analysis 01: Block embedding phenotype associations ==="
python scripts/analysis/01_block_embedding_phenotype_analysis.py

echo "=== Analysis 02: Subject cluster analysis ==="
python scripts/analysis/02_subject_cluster_analysis.py

echo "=== Analysis 03: Leave-HLA-out validation ==="
python scripts/analysis/03_leave_hla_out_analysis.py

echo "=== Analysis 04: Cluster stability ==="
python scripts/analysis/04_cluster_stability_analysis.py

echo "=== Analysis 05: Attention confounder analysis ==="
python scripts/analysis/05_attention_confounder_analysis.py

echo "=== Analysis 06: Phase 1 vs Phase 2 block comparison ==="
python scripts/analysis/06_phase1_phase2_block_comparison.py

echo "=== Analysis 07: 17q21 validation ==="
python scripts/analysis/07_17q21_validation.py

echo "=== Pipeline complete ==="
