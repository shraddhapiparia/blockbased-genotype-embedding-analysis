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
#
# Logs: each step tees stdout+stderr to results/logs/<timestamp>/<step>.log.
#       The timestamped directory is created once at startup; all steps in the
#       same invocation share it.  With set -euo pipefail, a non-zero exit from
#       any step halts the pipeline; the failing step name and log path remain
#       visible in the terminal.

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

# --- Logging setup -----------------------------------------------------------
# Each run gets its own timestamped subdirectory so previous runs are preserved.
LOG_DIR="results/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

run_step() {
  local step_name="$1"
  shift
  local log_file="$LOG_DIR/${step_name}.log"
  echo ""
  echo "=== ${step_name} ==="
  echo "[log] ${log_file}"
  # With pipefail set, the pipe's exit code reflects the left-side command,
  # so a failing python invocation propagates and halts the script.
  "$@" 2>&1 | tee "$log_file"
}

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
  run_step "dryrun_phase1_validate" \
    python scripts/core/VAE_phase1.py --config configs/config_phase1.yaml --dry-run
  run_step "dryrun_phase2_validate" \
    python scripts/core/attention_phase2.py --config configs/config_phase2.yaml --dry-run
  echo ""
  echo "[dry-run] Input validation passed."
  echo "[logs] dry-run logs written to: ${LOG_DIR}/"
  exit 0
fi

run_step "phase1_vae" \
  python scripts/core/VAE_phase1.py --config configs/config_phase1.yaml

run_step "phase2_attention" \
  python scripts/core/attention_phase2.py --config configs/config_phase2.yaml

run_step "core_block_analysis" \
  python scripts/core/analyze_phase2_blocks.py

run_step "core_plots" \
  python scripts/core/plots_updated.py

run_step "analysis_01_phenotype_assoc" \
  python scripts/analysis/01_block_embedding_phenotype_analysis.py

run_step "analysis_02_subject_clusters" \
  python scripts/analysis/02_subject_cluster_analysis.py

run_step "analysis_03_leave_hla_out" \
  python scripts/analysis/03_leave_hla_out_analysis.py

run_step "analysis_04_cluster_stability" \
  python scripts/analysis/04_cluster_stability_analysis.py

run_step "analysis_05_attention_confounders" \
  python scripts/analysis/05_attention_confounder_analysis.py

run_step "analysis_06_phase1_phase2_comparison" \
  python scripts/analysis/06_phase1_phase2_block_comparison.py

run_step "analysis_07_17q21_validation" \
  python scripts/analysis/07_17q21_validation.py

# ── Attribution pipeline (requires full Phase 2 outputs) ──────────────────────
# Scripts 08–10 use default result paths from Phase 1/2 above.
# Script 10 additionally requires per-block .raw files and --selected-blocks;
# set SNP_ATTRIBUTION_BLOCKS to a comma-separated list of block IDs to enable it.

run_step "analysis_08_clinical_pc_alignment" \
  python scripts/analysis/08_clinical_pc_embedding_alignment.py

run_step "analysis_09_block_attribution_lobo" \
  python scripts/analysis/09_phase2_block_attribution.py

echo ""
echo "=== Analysis 10: Phase 1 SNP attribution within selected blocks ==="
if [[ -n "${SNP_ATTRIBUTION_BLOCKS:-}" ]]; then
  run_step "analysis_10_snp_attribution" \
    python scripts/analysis/10_phase1_snp_attribution_within_blocks.py \
      --selected-blocks "$SNP_ATTRIBUTION_BLOCKS"
else
  echo "[skip] SNP_ATTRIBUTION_BLOCKS not set — skipping per-SNP attribution."
  echo "       To run: export SNP_ATTRIBUTION_BLOCKS=region_9p24_IL33,region_6p21_HLA_classII_sb15"
  echo "       then re-run this script."
fi

echo ""
echo "=== Pipeline complete ==="
echo "[logs] all step logs written to: ${LOG_DIR}/"
