#!/usr/bin/env python3
"""
11_top_snp_ld_check.py

Compute pairwise LD (r²) among the top N attributed SNPs from a Phase 1 SNP
attribution run (output of 10_phase1_snp_attribution_within_blocks.py).

Run as:
    python scripts/analysis/11_top_snp_ld_check.py \\
        --attrib-csv results/analysis/snp_attribution/all_selected_blocks_snp_attribution.csv \\
        --block-id region_9p24_IL33 \\
        --target emb_PC1 \\
        --plink /path/to/plink \\
        --top-n 20

What this does
--------------
1. Reads the SNP attribution CSV (per-block or combined output of script 10).
2. Filters to --block-id and --target.
3. Optionally filters to a single --run (full or noHLA); otherwise deduplicates
   on snp_locus, keeping the highest mean_abs_delta_score per SNP.
4. Sorts by mean_abs_delta_score descending and takes the top --top-n SNPs.
5. Reads the block's .raw header to map each snp_locus to its PLINK .raw column
   name. Fails loudly if any top SNP cannot be mapped.
6. Strips the trailing allele suffix to obtain exact PLINK variant IDs.
7. Writes a variant-ID file for PLINK --extract.
8. Verifies .bed/.bim/.fam exist, then runs:
       <plink> --bfile <bfile> --extract <ids.txt> --r2 square --out <prefix>
   via subprocess.run(..., check=True).
9. Loads the resulting .ld matrix and saves a tidy long-format LD summary CSV.
10. Prints all commands and output paths.

Outputs  (--out-dir, default results/analysis/snp_ld_check/)
-------
<prefix>_snp_loci.txt       snp_locus IDs for the top N SNPs (human-readable)
<prefix>_plink_ids.txt      PLINK variant IDs for --extract
<prefix>_top_snps.csv       top-N SNP attribution rows
<prefix>.ld                 raw PLINK r² square matrix
<prefix>_ld_summary.csv     tidy long-format r² pairs with attribution metadata

Limitations
-----------
- LD is estimated from the per-block cohort genotypes; sample size equals the
  full cohort, not a case-stratified subset.
- High r² between two top-ranked SNPs suggests redundant signal, not biological
  co-regulation. Interpret alongside annotation and fine-mapping results.
- Do not interpret LD structure as evidence of causality.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUT = "results/analysis/snp_ld_check"
RAW_DIR     = "data/raw_plink"
BFILE_DIR   = "data/region_blocks"


# ── SNP column name parsing ───────────────────────────────────────────────────

def parse_snp_locus(col_name: str) -> str:
    """Strip trailing PLINK allele suffix: '12:111742373:A:G_A' -> '12:111742373:A:G'."""
    parts = col_name.rsplit("_", 1)
    if len(parts) == 2 and 1 <= len(parts[1]) <= 2 and parts[1].isalpha():
        return parts[0]
    return col_name


# ── .raw header reader ────────────────────────────────────────────────────────

def read_raw_header(raw_path: Path) -> list:
    """Return SNP column names (cols 6+) from the first line of a .raw file."""
    with open(raw_path, "r") as fh:
        header_line = fh.readline()
    cols = header_line.strip().split()
    if len(cols) < 7:
        raise ValueError(
            f".raw header has only {len(cols)} columns (expected ≥7): {raw_path}"
        )
    return cols[6:]  # skip FID IID PAT MAT SEX PHENOTYPE


# ── locus → PLINK variant ID mapping ─────────────────────────────────────────

def map_loci_to_plink_ids(snp_loci: list, raw_snp_cols: list) -> dict:
    """
    Build {snp_locus: plink_variant_id} from .raw header column names.

    plink_variant_id = parse_snp_locus(raw_col_name) — i.e. the variant ID
    without the allele suffix, matching the .bim 2nd column.

    Raises ValueError if any locus is absent from the .raw header.
    """
    locus_to_col = {}
    for col in raw_snp_cols:
        locus = parse_snp_locus(col)
        locus_to_col[locus] = col

    result = {}
    missing = []
    for locus in snp_loci:
        if locus in locus_to_col:
            result[locus] = parse_snp_locus(locus_to_col[locus])
        else:
            missing.append(locus)

    if missing:
        raise ValueError(
            f"Cannot map {len(missing)} SNP locus ID(s) to .raw header column names:\n"
            + "\n".join(f"  {m}" for m in missing[:10])
            + ("\n  ..." if len(missing) > 10 else "")
        )
    return result


# ── PLINK helpers ─────────────────────────────────────────────────────────────

def check_bfile(bfile_prefix: str) -> None:
    """Raise FileNotFoundError if any of .bed/.bim/.fam are absent."""
    missing = [
        bfile_prefix + ext
        for ext in (".bed", ".bim", ".fam")
        if not Path(bfile_prefix + ext).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "PLINK binary file(s) not found:\n"
            + "\n".join(f"  {m}" for m in missing)
        )


def run_plink_ld(plink_exe: str, bfile: str,
                 extract_file: str, out_prefix: str) -> Path:
    """Run PLINK r² square matrix computation; return path to .ld output file."""
    cmd = [
        plink_exe,
        "--bfile",   bfile,
        "--extract", extract_file,
        "--r2",      "square",
        "--out",     out_prefix,
    ]
    print("\n[plink]      command:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    ld_path = Path(out_prefix + ".ld")
    if not ld_path.exists():
        raise RuntimeError(
            f"PLINK did not produce expected .ld output: {ld_path}\n"
            "Check the PLINK log for errors."
        )
    return ld_path


# ── LD matrix loader ──────────────────────────────────────────────────────────

def load_ld_matrix(ld_path: Path, snp_ids: list) -> pd.DataFrame:
    """
    Load a whitespace-delimited r² square matrix produced by PLINK --r2 square.
    Return a tidy long-format DataFrame with one row per unique SNP pair (i < j).
    """
    mat = np.loadtxt(str(ld_path))
    n = len(snp_ids)
    if mat.ndim == 1 and n == 1:
        mat = mat.reshape(1, 1)
    if mat.shape != (n, n):
        raise RuntimeError(
            f"LD matrix shape {mat.shape} does not match {n} extracted SNPs. "
            "Verify that all variant IDs are present in the .bim file."
        )
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(dict(
                snp_a=snp_ids[i],
                snp_b=snp_ids[j],
                r2=float(mat[i, j]),
            ))
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["snp_a", "snp_b", "r2"]
    )


# ── Argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attrib-csv", required=True,
                   help="SNP attribution CSV from script 10 "
                        "(per-block or all_selected_blocks_snp_attribution.csv)")
    p.add_argument("--raw-dir", default=RAW_DIR,
                   help="Directory containing per-block .raw PLINK dosage files "
                        "(default: %(default)s)")
    p.add_argument("--bfile-dir", default=BFILE_DIR,
                   help="Directory containing per-block PLINK binary files "
                        "(.bed/.bim/.fam). Expects <bfile-dir>/<block-id> prefix. "
                        "Default: %(default)s")
    p.add_argument("--block-id", required=True,
                   help="Block ID to compute LD for, e.g. region_9p24_IL33")
    p.add_argument("--target", required=True,
                   choices=["emb_PC1", "emb_PC2", "log10Ige_ridge_pred"],
                   help="Attribution target to rank SNPs by")
    p.add_argument("--run", default=None, choices=["full", "noHLA"],
                   help="Filter to a specific model run. "
                        "Default None = use all runs, keep best score per SNP locus.")
    p.add_argument("--top-n", type=int, default=20,
                   help="Number of top SNPs by mean_abs_delta_score (default 20)")
    p.add_argument("--plink", required=True,
                   help="Full path to the PLINK executable. Not assumed to be on PATH.")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    block_id = args.block_id
    target   = args.target
    top_n    = args.top_n

    safe_tgt = target.lower().replace(" ", "_").replace("/", "_")
    prefix   = f"{block_id}_{safe_tgt}_top{top_n}"

    # ── 1. Read attribution CSV ───────────────────────────────────────────────
    print(f"[load]       attribution CSV: {args.attrib_csv}")
    df = pd.read_csv(args.attrib_csv)

    required_cols = {"block_id", "snp_locus", "target", "mean_abs_delta_score"}
    missing_cols  = required_cols - set(df.columns)
    if missing_cols:
        sys.exit(
            f"ERROR: Attribution CSV missing required columns: {sorted(missing_cols)}\n"
            f"Available: {sorted(df.columns.tolist())}"
        )

    # ── 2. Filter to block_id and target ─────────────────────────────────────
    df_f = df[(df["block_id"] == block_id) & (df["target"] == target)].copy()
    if df_f.empty:
        sys.exit(
            f"ERROR: No rows for block_id='{block_id}' and target='{target}' "
            f"in {args.attrib_csv}.\n"
            f"Available block IDs: {sorted(df['block_id'].unique().tolist())}\n"
            f"Available targets:   {sorted(df['target'].unique().tolist())}"
        )
    print(f"             {len(df_f)} rows matched "
          f"(block='{block_id}', target='{target}')")

    # ── 3. Optional run filter ────────────────────────────────────────────────
    if args.run is not None:
        if "run" not in df_f.columns:
            sys.exit(
                "ERROR: Attribution CSV has no 'run' column; cannot filter by --run.\n"
                "Remove --run to use all available rows."
            )
        df_f = df_f[df_f["run"] == args.run].copy()
        if df_f.empty:
            sys.exit(
                f"ERROR: No rows remain after filtering to run='{args.run}'.\n"
                f"Available run values: "
                f"{sorted(df['run'].dropna().unique().tolist())}"
            )
        print(f"             {len(df_f)} rows after --run='{args.run}' filter")

    # ── 4. Sort, deduplicate on snp_locus (keep best score), take top N ───────
    df_f = df_f.sort_values("mean_abs_delta_score", ascending=False)
    df_f = df_f.drop_duplicates(subset="snp_locus", keep="first")
    top_df = df_f.head(top_n).reset_index(drop=True)

    actual_n = len(top_df)
    if actual_n < top_n:
        print(f"[warn]       only {actual_n} unique SNPs available; "
              f"requested {top_n}")
    print(f"             {actual_n} SNP(s) selected for LD computation")

    # ── 5. Read .raw header → validate and map loci to PLINK variant IDs ──────
    raw_path = Path(args.raw_dir) / f"{block_id}.raw"
    if not raw_path.exists():
        raise FileNotFoundError(f".raw file not found: {raw_path}")
    print(f"[raw]        reading header: {raw_path}")
    raw_snp_cols = read_raw_header(raw_path)
    print(f"             {len(raw_snp_cols)} SNP columns in .raw header")

    snp_loci  = top_df["snp_locus"].tolist()
    locus_map = map_loci_to_plink_ids(snp_loci, raw_snp_cols)
    plink_ids = [locus_map[locus] for locus in snp_loci]
    print(f"             all {len(plink_ids)} loci mapped to PLINK variant IDs")

    # ── 6. Write snp_loci.txt and plink_ids.txt ───────────────────────────────
    loci_file = out_dir / f"{prefix}_snp_loci.txt"
    ids_file  = out_dir / f"{prefix}_plink_ids.txt"

    loci_file.write_text("\n".join(snp_loci) + "\n")
    ids_file.write_text("\n".join(plink_ids) + "\n")
    print(f"[output]     {loci_file.name}  ({len(snp_loci)} loci)")
    print(f"[output]     {ids_file.name}  ({len(plink_ids)} PLINK IDs)")

    # ── 7. Verify PLINK executable and bfile ──────────────────────────────────
    plink_exe = args.plink
    if not Path(plink_exe).exists():
        raise FileNotFoundError(f"PLINK executable not found: {plink_exe}")

    bfile = str(Path(args.bfile_dir) / block_id)
    check_bfile(bfile)
    print(f"[plink]      bfile prefix: {bfile}")

    # ── 8. Run PLINK LD ───────────────────────────────────────────────────────
    ld_out_prefix = str(out_dir / prefix)
    ld_path = run_plink_ld(plink_exe, bfile, str(ids_file), ld_out_prefix)
    print(f"[plink]      LD matrix: {ld_path}")

    # ── 9. Load r² matrix → tidy long-format CSV ──────────────────────────────
    ld_long = load_ld_matrix(ld_path, plink_ids)

    if not ld_long.empty:
        score_map = dict(zip(plink_ids, top_df["mean_abs_delta_score"].values))
        ld_long["mean_abs_delta_score_a"] = ld_long["snp_a"].map(score_map)
        ld_long["mean_abs_delta_score_b"] = ld_long["snp_b"].map(score_map)
        ld_long = ld_long.sort_values("r2", ascending=False).reset_index(drop=True)

    ld_csv = out_dir / f"{prefix}_ld_summary.csv"
    ld_long.to_csv(ld_csv, index=False)
    print(f"[output]     {ld_csv.name}  ({len(ld_long)} pairs)")

    # ── 10. Save top-SNP attribution rows ─────────────────────────────────────
    top_csv = out_dir / f"{prefix}_top_snps.csv"
    top_df.to_csv(top_csv, index=False)
    print(f"[output]     {top_csv.name}  ({len(top_df)} SNPs)")

    # ── 11. Summary ───────────────────────────────────────────────────────────
    print(f"\n══ Top {actual_n} SNPs — {target} — {block_id} ══")
    disp_cols = ["snp_locus", "mean_abs_delta_score"]
    if "rank_within_block" in top_df.columns:
        disp_cols = ["rank_within_block"] + disp_cols
    if "run" in top_df.columns:
        disp_cols = disp_cols + ["run"]
    print(top_df[disp_cols].to_string(index=False))

    if not ld_long.empty:
        high_ld = ld_long[ld_long["r2"] >= 0.8]
        if not high_ld.empty:
            print(f"\n[LD]         {len(high_ld)} SNP pair(s) with r² ≥ 0.80:")
            print(high_ld[["snp_a", "snp_b", "r2"]].head(10).to_string(index=False))
        else:
            print("\n[LD]         No SNP pairs with r² ≥ 0.80 among top SNPs.")

    print(f"\n[done]       All outputs written to: {out_dir}/")


if __name__ == "__main__":
    main()
