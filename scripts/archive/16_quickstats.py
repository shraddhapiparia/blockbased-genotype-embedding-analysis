import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("results/output_regions2/ORD/all_blocks_pheno_analysis/phenotype_block_associations.tsv", sep="\t")

# min adjusted p-value per block across all phenotypes
min_p = (df.groupby(["block", "block_group"])["pval_adj"]
           .min()
           .reset_index())

asthma_p = min_p[min_p["block_group"] == "asthma"]["pval_adj"].values
control_p = min_p[min_p["block_group"] == "control"]["pval_adj"].values

stat, p = stats.mannwhitneyu(asthma_p, control_p, alternative="less")

print(f"Asthma n={len(asthma_p)}, median min-p={np.median(asthma_p):.3f}")
print(f"Control n={len(control_p)}, median min-p={np.median(control_p):.3f}")
print(f"Wilcoxon U={stat:.0f}, p={p:.4f}")