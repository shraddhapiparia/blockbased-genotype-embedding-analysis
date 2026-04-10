#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

# Ensure local package import works when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.attention_phase2 import load_config, run_phase2


def validate_cfg(cfg):
    phase1_dir = Path(cfg.get("phase1_dir", ""))
    out_dir = Path(cfg.get("output_dir", ""))

    if not phase1_dir.exists():
        raise FileNotFoundError(f"phase1_dir missing: {phase1_dir}")

    required_files = ["subjects.csv", "train_idx.npy", "val_idx.npy", "block_order.csv"]
    for f in required_files:
        fp = phase1_dir / f
        if not fp.exists():
            raise FileNotFoundError(f"Required Phase 1 file missing: {fp}")

    # Check per-loss embeddings
    loss_functions = cfg.get("loss_functions", [])
    for lt in loss_functions:
        emb_fp = phase1_dir / lt / "embeddings" / "all_blocks.npy"
        if not emb_fp.exists():
            raise FileNotFoundError(f"Missing embeddings for loss {lt}: {emb_fp}")

    out_dir.mkdir(parents=True, exist_ok=True)

    return phase1_dir, out_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 2 attention aggregation wrapper")
    parser.add_argument("--config", type=str, default="configs/config_phase2.yaml", help="Path to Phase2 config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Display configuration without running")
    args = parser.parse_args()

    config_path = args.config or "configs/config_phase2.yaml"
    print(f"[phase2] using config: {config_path}")
    cfg = load_config(config_path)
    phase1_dir, out_dir = validate_cfg(cfg)

    print(f"[phase2] phase1_dir={phase1_dir}")
    print(f"[phase2] output_dir={out_dir}")
    print(f"[phase2] device={cfg['attention'].get('device', 'auto')}")
    print(f"[phase2] loss_functions={cfg.get('loss_functions', [])}")
    print(f"[phase2] all required Phase 1 artifacts present: yes")

    if args.dry_run:
        print("[phase2] dry-run complete; no pipeline executed.")
        sys.exit(0)

    run_phase2(cfg)

    # Check expected outputs
    expected = [out_dir / "phase2_summary.csv"]
    loss_functions = cfg.get("loss_functions", [])
    for lt in loss_functions:
        expected.append(out_dir / lt / "clustering" / "cluster_labels.csv")
    for p in expected:
        if not p.exists():
            raise FileNotFoundError(f"Expected output missing after Phase2: {p}")

    print(f"[phase2] complete (took {time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
