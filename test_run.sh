#!/usr/bin/env bash
# Smoke test: Phase 1 → Phase 2 on synthetic data.
# No restricted genotype data required.
#
# Usage:
#   ./test_run.sh
#
# Expected runtime: < 5 minutes on CPU.
# Outputs written to results/synthetic_test/ and results/synthetic_test2/
# — for pipeline validation only, not biological interpretation.

set -euo pipefail

P1_OUT="results/synthetic_test"
P2_OUT="results/synthetic_test2"

echo "=== Synthetic smoke test ==="
echo

# Step 1: generate synthetic data files
echo "--- Step 1: generating synthetic data ---"
python data/synthetic/create_synthetic_data.py

# Step 2: verify generated inputs exist
echo "--- Step 2: verifying synthetic inputs ---"
for f in \
  "data/synthetic/block_plan/manifest.tsv" \
  "data/synthetic/region_blocks/synth_block_A.raw" \
  "data/synthetic/region_blocks/synth_block_B.raw" \
  "data/synthetic/region_blocks/synth_block_C.raw" \
  "data/synthetic/region_blocks/synth_block_D.raw"
do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: expected synthetic file not found: $f" >&2
    exit 1
  fi
done
echo "Inputs OK"
echo

# Step 3: Phase 1 VAE on synthetic data
echo "--- Step 3: Phase 1 VAE (synthetic) ---"
python scripts/core/VAE_phase1.py --config configs/config_phase1_synthetic.yaml
echo

# Step 4: Phase 2 Attention on synthetic data
echo "--- Step 4: Phase 2 Attention (synthetic) ---"
python scripts/core/attention_phase2.py --config configs/config_phase2_synthetic.yaml
echo

# Step 5: verify key outputs were written
echo "--- Step 5: verifying outputs ---"
for f in \
  "$P1_OUT/block_order.csv" \
  "$P1_OUT/subjects.csv" \
  "$P1_OUT/vae_summary.csv" \
  "$P1_OUT/MSE/embeddings/all_blocks.npy" \
  "$P2_OUT/phase2_summary.csv" \
  "$P2_OUT/ORD/embeddings/individual_embeddings.npy" \
  "$P2_OUT/ORD/embeddings/pooling_attention_weights.csv"
do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: expected output missing: $f" >&2
    exit 1
  fi
done

echo
echo "=== Smoke test passed ==="
echo "Phase 1 outputs : $P1_OUT"
echo "Phase 2 outputs : $P2_OUT"
echo "These are pipeline-validation outputs only — not biological results."
