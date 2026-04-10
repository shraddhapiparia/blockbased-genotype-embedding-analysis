#!/usr/bin/env python3
import argparse
import time
import sys
from pathlib import Path

# Ensure local package import works when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analyze_phase2_blocks import main as phase2_analysis_main


def main():
    parser = argparse.ArgumentParser(description="Phase 2 block analysis wrapper")
    parser.add_argument("--loss-dir", required=True, help="Phase2 loss output dir (e.g. results/output_regions2/ORD)")
    parser.add_argument("--phase1-dir", default=None, help="Optional Phase1 directory containing block_order.csv")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"[block_analysis] loss_dir={args.loss_dir}")
    print(f"[block_analysis] phase1_dir={args.phase1_dir}")
    if args.dry_run:
        print("[block_analysis] dry-run complete; no pipeline executed.")
        return

    start = time.time()
    sys_args = ["--loss-dir", args.loss_dir]
    if args.phase1_dir:
        sys_args += ["--phase1-dir", args.phase1_dir]
    sys_args += ["--topk", str(args.topk)]

    # Reuse analyze_phase2_blocks argparse semantics
    import sys as _sys
    _sys.argv = ["block_analysis.py"] + sys_args
    phase2_analysis_main()

    print(f"[block_analysis] complete (took {time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
