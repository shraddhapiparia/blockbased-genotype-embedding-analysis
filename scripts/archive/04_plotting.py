#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import sys

# Ensure local package import works when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.plots_updated import main as plots_main


def main():
    parser = argparse.ArgumentParser(description="Phase 1 plotting wrapper")
    parser.add_argument("--phase1-dir", type=str, default=None, help="Path to Phase1 output dir")
    parser.add_argument("--summary-csv", type=str, default=None, help="Summary CSV filename (vae_summary_ord.csv)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"[plotting] phase1_dir={args.phase1_dir}")
    if args.dry_run:
        print("[plotting] dry-run complete; no pipeline executed.")
        return

    # Set args for existing main() function from plots_updated
    import sys as _sys
    cli = ["plots.py"]
    if args.phase1_dir:
        cli += ["--phase1_dir", args.phase1_dir]
    if args.summary_csv:
        cli += ["--summary_csv", args.summary_csv]
    if args.outdir:
        cli += ["--outdir", args.outdir]
    _sys.argv = cli

    start = time.time()
    plots_main()
    print(f"[plotting] complete (took {time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
