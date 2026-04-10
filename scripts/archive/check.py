import numpy as np, pandas as pd
from pathlib import Path

# What order did Phase 2 training use for blocks?
# Check if there's a block name list saved anywhere
for p in Path("results/output_regions2/ORD").rglob("*block*name*"):
    print(p)
for p in Path("results/output_regions2/ORD").rglob("*block*order*"):
    print(p)
for p in Path("results/output_regions2/ORD").rglob("*block*list*"):
    print(p)

# Also check the per_block_recon_mse.csv — it should have block names in Phase2 order
mse = pd.read_csv("results/output_regions2/ORD/embeddings/per_block_recon_mse.csv")
print(mse.head(10))
print(mse.columns.tolist())
print("Shape:", mse.shape)