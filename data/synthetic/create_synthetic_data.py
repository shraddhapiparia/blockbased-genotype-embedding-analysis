#!/usr/bin/env python3
"""
Generate minimal synthetic genotype data for pipeline smoke-testing.

Creates:
  data/synthetic/region_blocks/<block_id>.raw  -- 4 blocks, 30 subjects, 10 SNPs
  data/synthetic/block_plan/manifest.tsv       -- block manifest

Run before test_run.sh (or let test_run.sh call it automatically):
    python data/synthetic/create_synthetic_data.py

Values are seeded random integers in {0, 1, 2} — no biological meaning.
"""
import os
import numpy as np

SEED = 42
N_SUBJECTS = 30
N_SNPS = 10
BLOCKS = ["synth_block_A", "synth_block_B", "synth_block_C", "synth_block_D"]


def generate():
    rng = np.random.default_rng(SEED)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    blocks_dir = os.path.join(base_dir, "region_blocks")
    plan_dir = os.path.join(base_dir, "block_plan")
    os.makedirs(blocks_dir, exist_ok=True)
    os.makedirs(plan_dir, exist_ok=True)

    subject_ids = [f"SUB{i + 1:03d}" for i in range(N_SUBJECTS)]

    manifest_path = os.path.join(plan_dir, "manifest.tsv")
    with open(manifest_path, "w") as f:
        f.write(
            "block_id\tgene\tclass\tsubblock\tchr\tfrom_bp\tto_bp"
            "\tsnp_count_original\tout_prefix\tstatus\n"
        )
        for idx, bid in enumerate(BLOCKS):
            c = idx + 1
            f.write(
                f"{bid}\tGENE_{c}\tsynthetic\t1\t{c}"
                f"\t{c * 100000}\t{c * 100000 + 10000}"
                f"\t{N_SNPS}\t{bid}\tok\n"
            )
    print(f"  wrote {manifest_path}")

    snp_names = [f"SNP{j + 1:02d}_A" for j in range(N_SNPS)]
    header = "FID IID PAT MAT SEX PHENOTYPE " + " ".join(snp_names)

    for bid in BLOCKS:
        geno = rng.integers(0, 3, size=(N_SUBJECTS, N_SNPS))
        # Guarantee no column is constant (required by ld_corr_score)
        for j in range(N_SNPS):
            while len(np.unique(geno[:, j])) < 2:
                geno[:, j] = rng.integers(0, 3, size=N_SUBJECTS)

        raw_path = os.path.join(blocks_dir, f"{bid}.raw")
        with open(raw_path, "w") as f:
            f.write(header + "\n")
            for i, sid in enumerate(subject_ids):
                vals = " ".join(str(int(v)) for v in geno[i])
                f.write(f"1 {sid} 0 0 1 -9 {vals}\n")
        print(f"  wrote {raw_path}")

    print(
        f"\nSynthetic data ready: {N_SUBJECTS} subjects × "
        f"{len(BLOCKS)} blocks × {N_SNPS} SNPs"
    )


if __name__ == "__main__":
    generate()
