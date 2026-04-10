#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

# Ensure local package import works when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.VAE_phase1 import load_config, run_phase1, run_tuning


def validate_cfg(cfg):
    raw_dir = Path(cfg["data"]["raw_dir"])
    block_def = Path(cfg["data"]["block_def"])
    out_dir = Path(cfg["data"]["output_dir"])

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir missing: {raw_dir}")
    if not block_def.exists():
        raise FileNotFoundError(f"block_def missing: {block_def}")

    out_dir.mkdir(parents=True, exist_ok=True)

    return raw_dir, block_def, out_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 1 block embedding wrapper")
    parser.add_argument("--config", type=str, default=None, help="Path to Phase1 config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Display configuration without running")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning mode")
    args = parser.parse_args()

    print(f"CLI --config: {args.config}")
    print(f"Config file exists: {os.path.exists(args.config) if args.config else 'N/A (using default)'}")

    cfg = load_config(args.config)
    resolved_config = "configs/config_phase1.yaml" if args.config is None else args.config
    print(f"Resolved config path: {resolved_config}")
    print(f"Top-level cfg keys: {list(cfg.keys())}")
    print(f"Loaded cfg['tuning']: {cfg.get('tuning', 'MISSING')}")
    print(f"Loaded cfg['representative']: {cfg.get('representative', 'MISSING')}")
    start = time.time()

    print(f"[phase1] using config: {args.config or 'DEFAULT'}")
    raw_dir, block_def, out_dir = validate_cfg(cfg)

    print(f"[phase1] raw_dir={raw_dir}")
    print(f"[phase1] block_def={block_def}")
    print(f"[phase1] output_dir={out_dir}")

    if args.dry_run:
        print("[phase1] dry-run complete; no pipeline executed.")
        sys.exit(0)

    if args.tune:
        run_tuning(cfg)
    else:
        run_phase1(cfg)

    # post-run validation
    if args.tune:
        tuning_dir = out_dir / "tuning"
        expected = [
            tuning_dir / "best_params.yaml",
            tuning_dir / "tuning_results.csv",
            tuning_dir / "tuning_summary.csv",
        ]
        for p in expected:
            if not p.exists():
                raise FileNotFoundError(f"Expected tuning output missing: {p}")
        print(f"[validation] tuning complete: {tuning_dir}")
    else:
        expected = [
            out_dir / "block_order.csv",
            out_dir / "subjects.csv",
            out_dir / "vae_summary.csv",
        ]
        for p in expected:
            if not p.exists():
                raise FileNotFoundError(f"Expected Phase1 output missing: {p}")

        for lt in cfg["loss_functions"]:
            emb_dir = out_dir / lt / "embeddings"
            if emb_dir.exists() and any(emb_dir.glob("*.npy")):
                print(f"[validation] embeddings found for {lt}")
            else:
                print(f"[validation] WARNING: no embeddings for {lt}")

        print(f"[validation] phase1 complete: {out_dir}")

    print(f"[phase1] complete (took {time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
