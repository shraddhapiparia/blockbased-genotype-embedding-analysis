#!/usr/bin/env python3
"""VAE_phase1.py — Phase 1: per-block β-VAE training and embedding.

Purpose : Train an independent β-VAE per LD block; save subject-level latent
          embeddings for every block.
Inputs  : data/region_blocks/<block>.npy, data/block_plan/manifest.tsv,
          configs/config_phase1.yaml
Outputs : results/output_regions/{block_order,subjects,vae_summary}.csv,
          results/output_regions/<loss>/embeddings/all_blocks.npy
Workflow: Step 1 of 2 — must run before attention_phase2.py.
"""
import os, sys, time, yaml, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import json
import platform
import socket
import sklearn
from itertools import product
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# ──────────────────────────────────────────────────────────────────
# 1.  DEFAULT CONFIGURATION
# ──────────────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "data": {
        "raw_dir":    "data/region_blocks",       
        "block_def":  "data/block_plan/manifest.tsv",
        # "block_def_ctrl":  "data/block_plan/manifest_blocks_ctrl.tsv",   
        "output_dir": "results/output_regions",
        # "raw_dir":    "data/region_blocks",       
        # "block_def":  "data/block_plan_test.tsv",
        # # "block_def_ctrl":  "data/block_plan/manifest_blocks_ctrl.tsv",   
        # "output_dir": "results/test_ord_mps_run1",
    },
    "runtime": {
        "device": "cpu",
    },

    "vae": {
        "latent_dim":   8,
        # buckets: [min_snps, max_snps, latent_dim]
        "latent_dim_by_snps": [
            [8,   49,   4],
            [50,  149,  8],
            [150, 299, 12],
            [300, 799, 16],
            [800, 10**9, 16],
        ],

        "dropout":      0.30,
        "lr":           1e-3,
        "batch_size":   64,
        "epochs":       200,
        "beta_max":     0.50,   # max KL weight
        "beta_warmup":  50,     # linear anneal over this many epochs
        "patience":     20,     # early-stopping patience
        "grad_clip":    1.0,
        "val_frac":     0.20,
        "seed":         42,
        "cat_weight_clip": 10.0,
    },
    # "loss_functions": ["MSE", "BCE", "MSE_STD","CAT","ORD"],
    "loss_functions": ["MSE"],
    "tuning": {
        "enabled": False,
        "loss": "MSE",
        "metric": "bal_acc_va",
        "blocks": [],
        "params": {
            "dropout": [0.3],
            "lr": [0.001],
            "beta_max": [0.5]
        }
    },
    "representative": {
        "blocks": [],
        "metric": "bal_acc_va",
        "top_k": 2,
        "bottom_k": 2
    }
}


def load_config(path):
    if path is None:
        path = "configs/config_phase1.yaml"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = copy.deepcopy(DEFAULT_CFG)
    if os.path.exists(path):
        with open(path) as fh:
            usr = yaml.safe_load(fh)
        for sec in cfg:
            if sec in usr and isinstance(cfg[sec], dict):
                cfg[sec].update(usr[sec])
            elif sec in usr:
                cfg[sec] = usr[sec]
        # Add any new top-level keys from usr not in DEFAULT_CFG
        for sec in usr:
            if sec not in cfg:
                cfg[sec] = usr[sec]
    return cfg

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def get_device(requested="auto"):
    if requested and requested != "auto":
        print(f"[device] forced {requested}")
        return torch.device(requested)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.zeros(2, device="mps")
            print("[device] Apple MPS")
            return torch.device("mps")
        except Exception:
            pass
    if torch.cuda.is_available():
        print("[device] CUDA")
        return torch.device("cuda")
    print("[device] CPU")
    return torch.device("cpu")

# def get_device():
#     if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#         try:
#             torch.zeros(2, device="mps")          # quick smoke-test
#             print("[device] Apple MPS")
#             return torch.device("mps")
#         except Exception:
#             pass
#     if torch.cuda.is_available():
#         print("[device] CUDA")
#         return torch.device("cuda")
#     print("[device] CPU")
#     return torch.device("cpu")

def latent_dim_for_p(p: int, cfg: dict) -> int:
    v = cfg.get("vae", {})
    sched = v.get("latent_dim_by_snps", None)
    if not sched:
        return int(v.get("latent_dim", 8))

    for lo, hi, d in sched:
        if p >= int(lo) and p <= int(hi):
            return int(d)

    return int(v.get("latent_dim", 8))

# ──────────────────────────────────────────────────────────────────
# 2.  DATA  LOADING
# ──────────────────────────────────────────────────────────────────

# Your TSV columns (has header row)
BLOCK_COLS = [
    "block_id", "gene", "class", "subblock", "chr",
    "from_bp", "to_bp", "snp_count_original", "out_prefix", "status"
]

def load_block_defs(tsv: str) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t", header=0)
    # keep only OK blocks
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("ok")].copy()
    print(f"[blocks] {len(df)} OK entries from {tsv}")
    return df

def read_raw(path: str):
    """Return (subject_ids, snp_names, geno_matrix float32)."""
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    sids = df["IID"].astype(str).values
    snp_cols = list(df.columns[6:])  # after FID IID PAT MAT SEX PHENOTYPE
    G = df[snp_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    if np.isnan(G).any():
        bad = np.isnan(G).sum()
        raise ValueError(f"{path}: genotype matrix contains {bad} NaNs after parsing (non-numeric values).")

    gmin, gmax = float(G.min()), float(G.max())
    if gmin < -1e-6 or gmax > 2 + 1e-6:
        raise ValueError(f"{path}: genotype values out of expected range [0,2]. min={gmin:.3f}, max={gmax:.3f}")

    # Optional: strict check that values are (close to) {0,1,2}
    vals = np.unique(G)
    if len(vals) <= 10 and not np.all(np.isin(np.round(vals), [0, 1, 2])):
        raise ValueError(f"{path}: unexpected genotype levels found: {vals[:10]}")

    return sids, snp_cols, G

def load_all_blocks(block_df: pd.DataFrame, raw_dir: str, fail_on_missing: bool = True):
    """
    Loads .raw files from ONE directory: raw_dir/<block_id>.raw
    Aligns subjects to the intersection across all loaded blocks.
    Preserves TSV order for downstream phases.
    """
    blocks = {}
    common = None
    raw_dir = str(raw_dir)

    # preserve TSV order
    ordered_block_ids = block_df["block_id"].astype(str).tolist()

    for bid in ordered_block_ids:
        fp = os.path.join(raw_dir, f"{bid}.raw")
        if not os.path.isfile(fp):
            msg = f"[missing] {fp} not found"
            if fail_on_missing:
                raise FileNotFoundError(msg)
            print("  " + msg + " (skipping)")
            continue

        row = block_df.loc[block_df["block_id"].astype(str) == bid].iloc[0]
        sids, snps, G = read_raw(fp)

        blocks[bid] = {
            "sids": sids,
            "snps": snps,
            "geno": G,
            "gene": str(row.get("gene", "")),
            "class": str(row.get("class", "")),
            "chr": row.get("chr", np.nan),
            "from_bp": row.get("from_bp", np.nan),
            "to_bp": row.get("to_bp", np.nan),
        }

        common = set(sids) if common is None else (common & set(sids))
        print(f"  {bid:30s}  {G.shape[1]:>4d} SNPs   {G.shape[0]} subj")

    if not blocks:
        raise RuntimeError("No blocks loaded. Check raw_dir and block_ids.")

    common = sorted(common)
    if len(common) == 0:
        raise RuntimeError("Subject intersection across blocks is empty. Check IID consistency.")
    if len(common) < 10:
        raise RuntimeError(
            f"Subject intersection across blocks is too small (n={len(common)}). "
            "Likely mismatched IID sets across .raw files."
        )


    print(f"[data] {len(common)} subjects × {len(blocks)} blocks")

    # align each block to the common subject order (fast mapping)
    common_arr = np.array(common, dtype=str)
    for bid, d in blocks.items():
        sid2i = {sid: i for i, sid in enumerate(d["sids"])}
        idx = [sid2i[sid] for sid in common]  # common is guaranteed subset
        d["geno"] = d["geno"][idx]
        d["sids"] = common_arr
        d["n_snps"] = d["geno"].shape[1]

    return blocks, common_arr, [bid for bid in ordered_block_ids if bid in blocks]

def geno_class_counts(G012: np.ndarray):
    """
    G012: (N,P) with values 0/1/2 (or float but exactly those)
    Returns (n0, n1, n2) as Python ints.
    """
    y = np.clip(np.round(G012), 0, 2).astype(np.int64).reshape(-1)
    c = np.bincount(y, minlength=3)
    return int(c[0]), int(c[1]), int(c[2])

# ──────────────────────────────────────────────────────────────────
# 3.  VAE  MODEL
# ──────────────────────────────────────────────────────────────────
class BlockVAE(nn.Module):
    def __init__(self, p, d=16, drop=0.3, loss_type="MSE", class_weights=None):
        super().__init__()
        self.p, self.d, self.loss_type = p, d, loss_type
        h = [64, 32] if p < 100 else [128, 64]

        # ---- encoder ----
        layers, inp = [], p
        for w in h:
            layers += [nn.Linear(inp, w), nn.BatchNorm1d(w),
                       nn.GELU(), nn.Dropout(drop)]
            inp = w
        self.enc  = nn.Sequential(*layers)
        self.mu   = nn.Linear(h[-1], d)
        self.lv   = nn.Linear(h[-1], d)

        # ---- decoder (mirror) ----
        layers, inp = [], d
        for w in reversed(h):
            layers += [nn.Linear(inp, w), nn.BatchNorm1d(w),
                       nn.GELU(), nn.Dropout(drop)]
            inp = w

        if self.loss_type == "CAT":
            layers.append(nn.Linear(h[0], p * 3))   # logits for 3 classes per SNP
        elif self.loss_type == "ORD":
            layers.append(nn.Linear(h[0], p * 2)) # teo thresholds per SNP
        else:
            layers.append(nn.Linear(h[0], p))       # old behavior

        self.dec = nn.Sequential(*layers)

        if class_weights is not None:
            self.register_buffer("ce_w", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.ce_w = None

    def encode(self, x):
        h  = self.enc(x)
        return self.mu(h), self.lv(h).clamp(-20, 2)

    def reparam(self, mu, lv):
        if self.training:
            return mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return mu

    def decode(self, z):
        out = self.dec(z)
        if self.loss_type == "CAT":
            B = out.size(0)
            return out.view(B, self.p, 3)
        elif self.loss_type == "ORD":
            return out.view(out.size(0), self.p, 2)
        return out

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        return self.decode(z), mu, lv

    def compute_loss(self, x_in, recon, mu, lv, beta, y=None):
        """
        x_in: float input to encoder, shape (B,P)
        recon:
          - CAT: logits shape (B,P,3)
          - else: shape (B,P)
        y: required for CAT, long targets shape (B,P) with values {0,1,2}
        """
        B = x_in.size(0)

        if self.loss_type == "CAT":
            if y is None:
                raise ValueError("For loss_type='CAT', provide y (LongTensor {0,1,2}) to compute_loss().")
            # Flatten: (B*P,3) vs (B*P,)
            logits = recon.reshape(-1, 3)
            targets = y.reshape(-1)

            ce = F.cross_entropy(logits, targets, reduction="sum",
                                weight=self.ce_w) / B
            rl = ce
        elif self.loss_type == "ORD":
            if y is None:
                raise ValueError("ORD loss requires y targets (LongTensor 0/1/2).")
            rl = ordinal_loss(recon, y)  # see corrected ordinal_loss below

        elif self.loss_type == "BCE":
            rl = F.binary_cross_entropy_with_logits(recon, x_in, reduction="sum") / B

        else:
            rl = F.mse_loss(recon, x_in, reduction="sum") / B

        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / B
        return rl + beta * kl, rl, kl


# ──────────────────────────────────────────────────────────────────
# 4.  DATA  TRANSFORMS  (per loss type)
# ──────────────────────────────────────────────────────────────────
def prepare_data(geno, loss_type, tr_ix, va_ix):
    tr, va = geno[tr_ix].copy(), geno[va_ix].copy()
    stats = {}
    if loss_type == "CAT":  # categorical
        # tr/va are expected to be {0,1,2} ints (or floats that are exactly 0/1/2)
        tr_x = torch.tensor(tr, dtype=torch.float32)   # encoder input
        va_x = torch.tensor(va, dtype=torch.float32)

        tr_y = torch.tensor(tr, dtype=torch.long)      # CE targets
        va_y = torch.tensor(va, dtype=torch.long)

        return (tr_x, va_x, tr_y, va_y, stats)
    
    if loss_type == "ORD":
        tr_x = torch.tensor(tr, dtype=torch.float32)
        va_x = torch.tensor(va, dtype=torch.float32)
        tr_y = torch.tensor(tr, dtype=torch.long)
        va_y = torch.tensor(va, dtype=torch.long)
        return (tr_x, va_x, tr_y, va_y, stats)

    
    if loss_type == "BCE":
        tr, va = tr / 2.0, va / 2.0               # → {0, 0.5, 1}

    elif loss_type == "MSE_STD":
        m  = tr.mean(0, keepdims=True)
        s  = tr.std(0, keepdims=True); s[s < 1e-8] = 1.0
        stats = {"mean": m, "std": s}
        tr, va = (tr - m) / s, (va - m) / s
    
    return (torch.tensor(tr, dtype=torch.float32),
            torch.tensor(va, dtype=torch.float32), stats)

def compute_class_weights(tr_geno_block, eps=1e-6):
    # tr_geno_block: numpy array (N,P) with values 0/1/2 (ints or exact floats)
    y = tr_geno_block.reshape(-1).astype(np.int64)
    counts = np.bincount(y, minlength=3).astype(np.float64)
    freq = counts / (counts.sum() + eps)
    w = 1.0 / (freq + eps)
    w = w / w.mean()  # normalize so average weight ~1
    return w.tolist(), counts.tolist()

def ordinal_loss(logits2: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits2: (B,P,2)
      logits2[...,0] = t0
      logits2[...,1] = raw_delta  (we enforce t1 = t0 + softplus(delta) so t1>=t0)
    targets: (B,P) in {0,1,2}
    Returns: sum-over-features / mean-over-batch
    """
    B = targets.size(0)
    t0 = logits2[..., 0]
    t1 = t0 + F.softplus(logits2[..., 1])   # enforce ordering

    p0 = torch.sigmoid(t0)  # P(Y<=0)
    p1 = torch.sigmoid(t1)  # P(Y<=1)

    p_cls = torch.stack([p0, p1 - p0, 1 - p1], dim=-1).clamp(1e-6, 1.0)
    log_p = p_cls.log().gather(-1, targets.clamp(0, 2).unsqueeze(-1)).squeeze(-1)
    return -log_p.sum() / B


# ──────────────────────────────────────────────────────────────────
# 5.  TRAINING  LOOP  (single block)
# ──────────────────────────────────────────────────────────────────
def train_block_vae(model, tr_t, va_t, cfg, device, log_csv, tr_y=None, va_y=None):
    v = cfg["vae"]

    if tr_y is None:
        tr_ds = TensorDataset(tr_t)
        va_ds = TensorDataset(va_t)
    else:
        tr_ds = TensorDataset(tr_t, tr_y)
        va_ds = TensorDataset(va_t, va_y)

    g = torch.Generator()
    g.manual_seed(v["seed"])

    tr_dl = DataLoader(
        tr_ds,
        batch_size=v["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=g,
    )
    va_dl = DataLoader(
        va_ds,
        batch_size=v["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=v["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=10, factor=0.5, min_lr=1e-6
    )

    best_val, wait, best_sd, log = float("inf"), 0, None, []
    best_epoch = 0
    best_metrics = {}

    for ep in range(1, v["epochs"] + 1):
        beta = min(1.0, ep / v["beta_warmup"]) * v["beta_max"]

        # Train
        model.train()
        total_tr_loss, total_tr_recon, total_tr_kl = 0, 0, 0
        for batch in tr_dl:
            if tr_y is not None:
                x, y = batch
            else:
                x = batch[0]
                y = None
            x = x.to(device)
            if y is not None:
                y = y.to(device)
            opt.zero_grad()
            recon, mu, lv = model(x)
            loss, recon_loss, kl_loss = model.compute_loss(x, recon, mu, lv, beta, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), v["grad_clip"])
            opt.step()
            total_tr_loss += loss.item() * x.size(0)
            total_tr_recon += recon_loss.item() * x.size(0)
            total_tr_kl += kl_loss.item() * x.size(0)
        tL = total_tr_loss / len(tr_dl.dataset)
        tR = total_tr_recon / len(tr_dl.dataset)
        tK = total_tr_kl / len(tr_dl.dataset)

        # Val
        model.eval()
        total_va_loss, total_va_recon, total_va_kl = 0, 0, 0
        with torch.no_grad():
            for batch in va_dl:
                if va_y is not None:
                    x, y = batch
                else:
                    x = batch[0]
                    y = None
                x = x.to(device)
                if y is not None:
                    y = y.to(device)
                recon, mu, lv = model(x)
                loss, recon_loss, kl_loss = model.compute_loss(x, recon, mu, lv, beta, y)
                total_va_loss += loss.item() * x.size(0)
                total_va_recon += recon_loss.item() * x.size(0)
                total_va_kl += kl_loss.item() * x.size(0)
        vL = total_va_loss / len(va_dl.dataset)
        vR = total_va_recon / len(va_dl.dataset)
        vK = total_va_kl / len(va_dl.dataset)

        sched.step(vL)
        log.append({
            "epoch": ep,
            "tr_loss": tL,
            "va_loss": vL,
            "tr_recon": tR,
            "tr_kl": tK,
            "va_recon": vR,
            "va_kl": vK,
        })

        if vL < best_val:
            best_val, wait = vL, 0
            best_epoch = ep
            best_metrics = {
                "best_val_loss": vL,
                "best_val_recon": vR,
                "best_val_kl": vK,
                "best_tr_loss": tL,
                "best_tr_recon": tR,
                "best_tr_kl": tK,
            }
            best_sd = {k: w.cpu().clone() for k, w in model.state_dict().items()}
        else:
            wait += 1

        if wait > v["patience"]:
            print(f"      early stop at epoch {ep}")
            break

    model.load_state_dict(best_sd)
    model.to("cpu")
    if log_csv is not None:
        pd.DataFrame(log).to_csv(log_csv, index=False)
    return log, best_epoch, best_metrics


# ──────────────────────────────────────────────────────────────────
# 6.  CONCORDANCE  &  EMBEDDING  HELPERS
# ──────────────────────────────────────────────────────────────────
def concordance(model, x_t, loss_type, stats):
    model.eval()
    with torch.no_grad():
        rec = model(x_t)[0].cpu().numpy()

    if loss_type == "CAT":
        pred = rec.argmax(axis=2)  # (N,P)
        truth = x_t.cpu().numpy()
        truth = np.clip(np.round(truth), 0, 2)
        return float(np.mean(pred == truth))
    
    if loss_type == "ORD":
        t0 = rec[..., 0]
        delta = rec[..., 1]
        t1 = t0 + np.log1p(np.exp(delta))  # numpy softplus: log(1 + exp(x))

        p0 = 1 / (1 + np.exp(-t0))
        p1 = 1 / (1 + np.exp(-t1))
        p_cls = np.stack([p0, p1 - p0, 1 - p1], axis=-1)
        pred = p_cls.argmax(axis=2)
        truth = np.clip(np.round(x_t.cpu().numpy()), 0, 2)
        return float(np.mean(pred == truth))

    if loss_type == "BCE":
        pred = torch.sigmoid(torch.tensor(rec)).numpy() * 2.0
    elif loss_type == "MSE_STD":
        pred = rec * stats["std"] + stats["mean"]
    else:
        pred = rec

    pred = np.clip(np.round(pred), 0, 2)

    truth = x_t.cpu().numpy()
    if loss_type == "BCE":
        truth = truth * 2.0
    elif loss_type == "MSE_STD":
        truth = truth * stats["std"] + stats["mean"]
    truth = np.clip(np.round(truth), 0, 2)

    return float(np.mean(pred == truth))


def extract_emb(model, geno_raw, loss_type, stats):
    """Return (N, d) posterior-mean array for ALL subjects."""
    if loss_type == "BCE":
        x = geno_raw / 2.0
    elif loss_type == "MSE_STD":
        x = (geno_raw - stats["mean"]) / stats["std"]
    else:
        x = geno_raw.copy()
    x = torch.tensor(x, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x)
    return mu.numpy()

def ld_corr_score(truth012: np.ndarray, pred012: np.ndarray, max_snps: int = 200, seed: int = 0) -> float:
    """
    Compare SNP-SNP correlation matrices (Pearson) between truth and prediction.
    Subsamples SNPs to max_snps for speed.
    Filters constant SNPs because correlation is undefined when std=0.
    Returns correlation between upper triangles.
    """
    rng = np.random.default_rng(seed)
    P = truth012.shape[1]
    idx = np.arange(P) if P <= max_snps else rng.choice(P, size=max_snps, replace=False)

    T = truth012[:, idx].astype(np.float32)
    Pm = pred012[:, idx].astype(np.float32)

    # remove SNPs that are constant in either truth or prediction
    keep = (T.std(axis=0) > 0) & (Pm.std(axis=0) > 0)
    T = T[:, keep]
    Pm = Pm[:, keep]

    # need at least 2 SNPs to form a correlation matrix
    if T.shape[1] < 2:
        return float("nan")

    Ct = np.corrcoef(T, rowvar=False)
    Cp = np.corrcoef(Pm, rowvar=False)

    iu = np.triu_indices(T.shape[1], k=1)
    a, b = Ct[iu], Cp[iu]
    ok = np.isfinite(a) & np.isfinite(b)

    if ok.sum() < 10:
        return float("nan")

    return float(np.corrcoef(a[ok], b[ok])[0, 1])

def block_maf_stats(G012: np.ndarray) -> dict:
    """
    G012: (N, P) raw dosages in {0,1,2}
    Returns block-level MAF summary.
    """
    p_hat = G012.mean(axis=0) / 2.0                       # allele freq estimate
    maf = np.minimum(p_hat, 1.0 - p_hat)                  # MAF per SNP
    return {
        "maf_mean": float(np.mean(maf)),
        "maf_median": float(np.median(maf)),
        "maf_min": float(np.min(maf)),
        "maf_max": float(np.max(maf)),
        "maf_frac_lt_10pct": float(np.mean(maf < 0.10)),
        "maf_frac_lt_20pct": float(np.mean(maf < 0.20)),
    }

def eval_genotype_metrics(model, x_t, loss_type, stats):
    """
    Returns:
      - conc: exact match fraction (your current metric)
      - base_conc: baseline concordance (always predict majority class per SNP)
      - acc0/acc1/acc2: per-genotype recall (accuracy within each true class)
      - bal_acc: mean(acc0, acc1, acc2) over classes that appear
      - r2: dosage R^2 on continuous scale before rounding (post-inverse-transform)
    """
    model.eval()
    with torch.no_grad():
        rec = model(x_t)[0].cpu().numpy()  # raw decoder output / logits

    if loss_type == "CAT":
        pred = rec.argmax(axis=2).astype(np.int8)
        truth = x_t.cpu().numpy().astype(np.int8)
        pred_cont = pred.astype(np.float32)
        truth_cont = truth.astype(np.float32)
    elif loss_type == "ORD":
        # rec: (N,P,2)
        t0 = rec[..., 0]
        delta = rec[..., 1]
        t1 = t0 + np.log1p(np.exp(delta))
        p0 = 1/(1+np.exp(-t0))
        p1 = 1/(1+np.exp(-t1))
        p_cls = np.stack([p0, p1 - p0, 1 - p1], axis=-1)
        p_cls = np.clip(p_cls, 1e-6, 1.0)
        pred = p_cls.argmax(axis=2).astype(np.int8)
        truth = x_t.cpu().numpy().astype(np.int8)
        pred_cont = pred.astype(np.float32)
        truth_cont = truth.astype(np.float32)
    elif loss_type == "BCE":
        pred_cont = torch.sigmoid(torch.tensor(rec)).numpy() * 2.0
        truth_cont = x_t.cpu().numpy() * 2.0
    elif loss_type == "MSE_STD":
        pred_cont = rec * stats["std"] + stats["mean"]
        truth_cont = x_t.cpu().numpy() * stats["std"] + stats["mean"]
    else:
        pred_cont = rec
        truth_cont = x_t.cpu().numpy()

    pred = np.clip(np.round(pred_cont), 0, 2).astype(np.int8)
    truth = np.clip(np.round(truth_cont), 0, 2).astype(np.int8)
    # ld = ld_corr_score(truth.round().clip(0,2).astype(np.float32), pred.clip(0, None), max_snps=200, seed=0)
    ld = ld_corr_score(truth, pred, max_snps=200, seed=0)
    conc = float(np.mean(pred == truth))

    # baseline: per-SNP majority-class predictor
    # (compute mode of truth for each SNP column, then compare)
    # truth: (N, P). mode per column:
    # counts shape (P, 3)
    counts = np.stack([(truth == g).sum(axis=0) for g in (0, 1, 2)], axis=1)  # (P,3)
    mode = counts.argmax(axis=1).astype(np.int8)  # (P,)
    base_pred = np.broadcast_to(mode, truth.shape)  # (N,P)
    base_conc = float(np.mean(base_pred == truth))

    # per-genotype accuracy (recall per true class)
    accs = {}
    for g in (0, 1, 2):
        m = (truth == g)
        accs[g] = float(np.mean(pred[m] == g)) if m.any() else np.nan

    present = [a for a in accs.values() if not np.isnan(a)]
    bal_acc = float(np.mean(present)) if present else np.nan

    # dosage R^2 (flatten)
    y = truth_cont.reshape(-1)
    yhat = pred_cont.reshape(-1)
    var = np.var(y)
    r2 = float(1.0 - np.mean((y - yhat) ** 2) / (var + 1e-12))

    cm = None
    if loss_type in ["CAT", "ORD"]:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(truth.reshape(-1), pred.reshape(-1))

    return {
        "conc": conc,
        "base_conc": base_conc,
        "ld_corr":ld,
        "acc0": accs[0],
        "acc1": accs[1],
        "acc2": accs[2],
        "bal_acc": bal_acc,
        "r2": r2,
        "confusion_matrix": cm,
    }


# ──────────────────────────────────────────────────────────────────
# 7.  MAIN  PHASE-1  PIPELINE
# ──────────────────────────────────────────────────────────────────
def run_tuning(cfg):
    vc = cfg["vae"]
    tc = cfg.get("tuning", {})

    print("\n══════ Tuning Mode ══════")
    print(f"Tuning config - loss: {tc.get('loss', 'N/A')}")
    print(f"Tuning config - metric: {tc.get('metric', 'N/A')}")
    print(f"Tuning config - blocks from config: {tc.get('blocks', [])}")
    tuning_blocks = tc.get("blocks", [])
    print(f"Tuning blocks - is empty after config load: {len(tuning_blocks) == 0}")
    set_seed(vc["seed"])
    dev = get_device(cfg.get("runtime", {}).get("device", "auto"))

    out = Path(cfg["data"]["output_dir"])
    tuning_dir = out / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    bdf = load_block_defs(cfg["data"]["block_def"])
    blocks, subjects, loaded_block_ids = load_all_blocks(bdf, cfg["data"]["raw_dir"])

    # Select tuning blocks
    tuning_blocks = tc.get("blocks", [])
    if tuning_blocks:
        # Manual list provided
        requested = set(tuning_blocks)
        found = [b for b in requested if b in loaded_block_ids]
        missing = [b for b in requested if b not in loaded_block_ids]
        print(f"Tuning blocks - Requested: {sorted(requested)}")
        print(f"Tuning blocks - Found: {sorted(found)}")
        if missing:
            print(f"Tuning blocks - Missing: {sorted(missing)}")
            print("Warning: Some requested tuning blocks not found. Continuing with found blocks.")
        if not found:
            print("Error: None of the requested tuning blocks were found.")
            print(f"Available blocks (first 10): {sorted(loaded_block_ids)[:10]}")
            raise ValueError("No valid tuning blocks found.")
        tuning_blocks = found
    else:
        # Auto-select
        ranges = [(8, 49), (50, 149), (150, float('inf'))]
        selected = []
        for lo, hi in ranges:
            candidates = [bid for bid in loaded_block_ids if lo <= blocks[bid]["n_snps"] <= hi]
            if candidates:
                selected.append(candidates[0])  # first one
        tuning_blocks = selected
    print(f"Tuning blocks - Final selected: {tuning_blocks}")

    # Params
    params = tc.get("params", {})
    combos = list(product(
        params.get("dropout", [0.3]),
        params.get("lr", [0.001]),
        params.get("beta_max", [0.5])
    ))

    # Loss to tune
    lt = tc.get("loss", cfg["loss_functions"][0] if cfg["loss_functions"] else "MSE")

    # Split
    N = len(subjects); n_val = int(N * vc["val_frac"])
    perm = np.random.permutation(N)
    va_ix, tr_ix = perm[:n_val], perm[n_val:]

    # Results
    results = []
    for bid in tuning_blocks:
        if bid not in blocks:
            continue
        G = blocks[bid]["geno"]
        p = G.shape[1]
        d_block = latent_dim_for_p(p, cfg)
        if lt in ["ORD", "CAT"]:
            tr_t, va_t, tr_y, va_y, stats = prepare_data(G, lt, tr_ix, va_ix)
        else:
            tr_t, va_t, stats = prepare_data(G, lt, tr_ix, va_ix)
            tr_y = va_y = None

        for drop, lr, beta_max in combos:
            # Reset seed for comparability
            set_seed(vc["seed"])
            
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy["vae"]["dropout"] = drop
            cfg_copy["vae"]["lr"] = lr
            cfg_copy["vae"]["beta_max"] = beta_max

            Ntr = tr_t.shape[0] if hasattr(tr_t, 'shape') else 'NA'
            Nva = va_t.shape[0] if hasattr(va_t, 'shape') else 'NA'
            P = tr_t.shape[1] if hasattr(tr_t, 'shape') and len(tr_t.shape) > 1 else 'NA'
            print(f"[tuning] {lt} | {bid} | Ntr={Ntr} Nva={Nva} P={P} | dropout={drop} lr={lr} beta_max={beta_max}")

            if lt in ["ORD", "CAT"] and (tr_y is None or va_y is None):
                print(f"[tuning][WARN] expected targets for {lt} but tr_y/va_y missing (tr_y={tr_y}, va_y={va_y})")

            model = BlockVAE(p, d_block, drop, loss_type=lt)
            t0 = time.time()
            log, best_epoch, best_metrics = train_block_vae(model, tr_t, va_t, cfg_copy, dev, None, tr_y=tr_y, va_y=va_y)
            dt = time.time() - t0
            m_va = eval_genotype_metrics(model, va_t, lt, stats)

            results.append({
                "loss": lt,
                "block": bid,
                "dropout": drop,
                "lr": lr,
                "beta_max": beta_max,
                "best_epoch": best_epoch,
                "best_val_loss": best_metrics["best_val_loss"],
                "bal_acc_va": m_va["bal_acc"],
                "ld_corr_va": m_va["ld_corr"],
                "conc_va": m_va["conc"],
                "runtime_sec": dt
            })

    # Save results
    rdf = pd.DataFrame(results)
    rdf.to_csv(tuning_dir / "tuning_results.csv", index=False)

    # Aggregate with std
    agg = rdf.groupby(["dropout", "lr", "beta_max"]).agg({
        "best_val_loss": ["mean", "std"],
        "bal_acc_va": ["mean", "std"],
        "ld_corr_va": ["mean", "std"],
        "conc_va": ["mean", "std"],
        "block": "count"
    }).reset_index()
    agg.columns = ["dropout", "lr", "beta_max", "mean_best_val_loss", "std_best_val_loss", "mean_bal_acc_va", "std_bal_acc_va", "mean_ld_corr_va", "std_ld_corr_va", "mean_conc_va", "std_conc_va", "n_blocks"]
    agg.to_csv(tuning_dir / "tuning_summary.csv", index=False)

    # Select best
    metric = tc.get("metric", "bal_acc_va")
    mean_col = f"mean_{metric}"
    if metric in ["bal_acc_va", "ld_corr_va", "conc_va"]:
        best_row = agg.loc[agg[mean_col].idxmax()]
    elif metric == "best_val_loss":
        best_row = agg.loc[agg[mean_col].idxmin()]
    else:
        best_row = agg.iloc[0]

    best_params = {
        "loss": str(lt),
        "dropout": float(best_row["dropout"]),
        "lr": float(best_row["lr"]),
        "beta_max": float(best_row["beta_max"]),
        "metric": str(metric),
        "value": float(best_row[mean_col]),
    }
    with open(tuning_dir / "best_params.yaml", "w") as f:
        yaml.safe_dump(best_params, f, sort_keys=False)

    print(f"Tuning complete. Best params saved to {tuning_dir / 'best_params.yaml'}")
    print("[tuning] selected best params:")
    print(best_params)


def select_representative_blocks(sdf, cfg, blocks):
    rep_cfg = cfg.get("representative", {})
    manual = rep_cfg.get("blocks", [])
    if manual:
        valid = [b for b in manual if b in blocks]
        invalid = [b for b in manual if b not in blocks]
        if invalid:
            print(f"Warning: Representative blocks not found: {invalid}")
        return list(set(valid))  # remove duplicates
    
    # Automatic selection
    metric = rep_cfg.get("metric", "bal_acc_va")
    top_k = rep_cfg.get("top_k", 2)
    bottom_k = rep_cfg.get("bottom_k", 2)
    selected = set()
    
    # Biological importance
    known = ["region_17q21_core_sb1", "11q13_FCER1A_sb1", "region_2q12_IL1RL1_cluster_sb3", "region_5q21_PDE4D_sb55", "region_6p21_HLA_classII_sb1", "region_5q31_type2_cytokine_sb1"]
    for b in known:
        if b in blocks:
            selected.add(b)
    
    # Size diversity
    sizes = [(8, 49, "small"), (50, 149, "medium"), (150, float('inf'), "large")]
    for lo, hi, label in sizes:
        candidates = sdf[(sdf["n_snps"] >= lo) & (sdf["n_snps"] <= hi)]["block"].unique()
        if len(candidates) > 0:
            selected.add(candidates[0])
    
    # Performance diversity
    for loss in sdf["loss"].unique():
        sub = sdf[sdf["loss"] == loss].sort_values(metric, ascending=False)
        top = sub.head(top_k)["block"].tolist()
        bottom = sub.tail(bottom_k)["block"].tolist()
        selected.update(top + bottom)
    
    # Limit to ~8
    return list(selected)[:8]


def plot_training_curve(log_df, bid, out_dir):
    if not HAS_PLT:
        return
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(log_df['epoch'], log_df['tr_loss'], label='Train Loss')
    ax.plot(log_df['epoch'], log_df['va_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{bid} Training Curve')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{bid}_training.png", dpi=150)
    plt.close()


def plot_confusion_matrix(cm, bid, loss, out_dir):
    if not HAS_PLT or cm is None:
        return
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_title(f'{bid} {loss} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / f"{bid}_{loss}_confusion.png", dpi=150)
    plt.close()


def plot_cross_loss_metrics(sdf, bid, out_dir):
    if not HAS_PLT:
        return
    sub = sdf[sdf["block"] == bid]
    metrics = ["conc_va", "bal_acc_va", "r2_va", "ld_corr_va", "concordance_gain_va"]
    for met in metrics:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.bar(sub["loss"], sub[met])
        ax.set_title(f'{bid} {met} across Losses')
        ax.set_ylabel(met)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / f"{bid}_{met}_cross_loss.png", dpi=150)
        plt.close()


def plot_aggregate_boxes(sdf, out_dir):
    if not HAS_PLT:
        return
    metrics = ["conc_va", "bal_acc_va", "r2_va", "ld_corr_va", "concordance_gain_va"]
    for met in metrics:
        fig, ax = plt.subplots(figsize=(10,6))
        data = [sdf[sdf["loss"] == loss][met].dropna() for loss in sdf["loss"].unique()]
        ax.boxplot(data, labels=sdf["loss"].unique())
        ax.set_title(f'Distribution of {met} across Losses')
        ax.set_ylabel(met)
        plt.tight_layout()
        plt.savefig(out_dir / f"aggregate_{met}_boxplot.png", dpi=150)
        plt.close()


def plot_scatter_metric_vs(sdf, x_col, out_dir):
    if not HAS_PLT:
        return
    metrics = ["conc_va", "bal_acc_va", "r2_va", "ld_corr_va", "concordance_gain_va"]
    for met in metrics:
        fig, ax = plt.subplots(figsize=(10,6))
        for loss in sdf["loss"].unique():
            sub = sdf[sdf["loss"] == loss]
            ax.scatter(sub[x_col], sub[met], label=loss, alpha=0.6)
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(met.replace('_', ' ').title())
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_{met}_vs_{x_col}.png", dpi=150)
        plt.close()


def run_phase1(cfg):
    vc  = cfg["vae"]
    set_seed(vc["seed"])
    # dev = get_device()
    dev = get_device(cfg.get("runtime", {}).get("device", "auto"))

    out = Path(cfg["data"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # Auto-load best tuning params if available
    best_params_path = out / "tuning" / "best_params.yaml"
    if best_params_path.exists():
        with open(best_params_path) as f:
            best = yaml.safe_load(f) or {}
        vc["dropout"] = float(best.get("dropout", vc["dropout"]))
        vc["lr"] = float(best.get("lr", vc["lr"]))
        vc["beta_max"] = float(best.get("beta_max", vc["beta_max"]))
        print(f"[tuning] loaded best params from {best_params_path}: "
            f"dropout={vc['dropout']}, lr={vc['lr']}, beta_max={vc['beta_max']}")

    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "device": str(dev),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "cuda_available": torch.cuda.is_available(),
        "seed": vc["seed"],
        "deterministic_algorithms_requested": True,
        "raw_dir": cfg["data"]["raw_dir"],
        "block_def": cfg["data"]["block_def"],
        "output_dir": cfg["data"]["output_dir"],
        "config_path": args.config if 'args' in locals() else "default",
    }
    with open(out / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # persist config used
    with open(out / "config_phase1.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ---- load ----
    print("\n══════ Loading data ══════")
    bdf = load_block_defs(cfg["data"]["block_def"])
    blocks, subjects, loaded_block_ids = load_all_blocks(bdf, cfg["data"]["raw_dir"])
    # bdf_ctrl = load_block_defs(cfg["data"]["block_def_ctrl"])
    # merged_bdf = pd.concat([bdf, bdf_ctrl], ignore_index=True)
    # blocks, subjects, loaded_block_ids = load_all_blocks(merged_bdf, cfg["data"]["raw_dir"])
    block_ids = loaded_block_ids
    pd.DataFrame({"IID": subjects}).to_csv(out/"subjects.csv", index=False)

    # ---- split (same for every loss type) ----
    N = len(subjects); n_val = int(N * vc["val_frac"])
    perm = np.random.permutation(N)
    va_ix, tr_ix = perm[:n_val], perm[n_val:]
    np.save(out/"train_idx.npy", tr_ix)
    np.save(out/"val_idx.npy",   va_ix)
    print(f"[split] train {len(tr_ix)} / val {len(va_ix)}")
    pd.DataFrame({"IID": subjects[tr_ix]}).to_csv(out / "train_subjects.csv", index=False)
    pd.DataFrame({"IID": subjects[va_ix]}).to_csv(out / "val_subjects.csv", index=False)
    
    # Representative blocks
    rep_cfg = cfg.get("representative", {})
    manual_rep = rep_cfg.get("blocks", [])
    rep_blocks_set = set()
    if manual_rep:
        requested = set(manual_rep)
        found = [b for b in requested if b in loaded_block_ids]
        missing = [b for b in requested if b not in loaded_block_ids]
        print(f"Representative blocks - Requested: {sorted(requested)}")
        print(f"Representative blocks - Found: {sorted(found)}")
        if missing:
            print(f"Representative blocks - Missing: {sorted(missing)}")
            print("Warning: Some requested representative blocks not found. Skipping missing ones.")
        rep_blocks_set = set(found)
    # If no manual or manual empty, auto-select later
    out_rep = out / "representative_blocks"
    out_rep.mkdir(exist_ok=True)
    
    rows = []  # summary collector

    for lt in cfg["loss_functions"]:
        print(f"\n{'═'*55}\n  Loss: {lt}\n{'═'*55}")
        ld = out / lt
        for sub in ("logs", "models", "embeddings"):
            (ld / sub).mkdir(parents=True, exist_ok=True)

        emb_dict = {}  # block_id → (N, d)

        for bid in block_ids:
            G     = blocks[bid]["geno"]
            n0_tr, n1_tr, n2_tr = geno_class_counts(G[tr_ix])
            n0_va, n1_va, n2_va = geno_class_counts(G[va_ix])
            p     = G.shape[1]
            maf_stats = block_maf_stats(G)
            print(f"\n  ── {bid} ({p} SNPs) ──")

            d_block = latent_dim_for_p(p, cfg)

            if lt in ("CAT", "ORD"):
                tr_t, va_t, tr_y, va_y, stats = prepare_data(G, lt, tr_ix, va_ix)
            else:
                tr_t, va_t, stats = prepare_data(G, lt, tr_ix, va_ix)
                tr_y = va_y = None

            if lt == "CAT":
                w, counts = compute_class_weights(blocks[bid]["geno"][tr_ix])
                w = np.clip(np.array(w, dtype=np.float32), 0.25, vc.get("cat_weight_clip", 10.0)).tolist()
                model = BlockVAE(p, d_block, vc["dropout"], loss_type="CAT", class_weights=w)
                print(f"      latent_dim {d_block} | CAT weights {w}  counts {counts}")
            else:
                model = BlockVAE(p, d_block, vc["dropout"], loss_type=lt)
                print(f"      latent_dim {d_block}")

            npar  = sum(q.numel() for q in model.parameters())
            print(f"      params {npar:,}")

            t0  = time.time()
            log, best_epoch, best_metrics = train_block_vae(model, tr_t, va_t, cfg, dev,ld/"logs"/f"{bid}.csv",tr_y=tr_y, va_y=va_y)
            dt  = time.time() - t0

            m_tr = eval_genotype_metrics(model, tr_t, lt, stats)
            m_va = eval_genotype_metrics(model, va_t, lt, stats)
            print(
                f"      conc tr {m_tr['conc']:.4f}  va {m_va['conc']:.4f} | "
                f"base va {m_va['base_conc']:.4f} | bal va {m_va['bal_acc']:.4f} | "
                f"r2 va {m_va['r2']:.4f} | ld_corr va {m_va['ld_corr']:.4f}  ({dt:.1f}s)"
            )

            if bid in rep_blocks_set and "confusion_matrix" in m_va and m_va["confusion_matrix"] is not None:
                rep_dir = out_rep / bid
                rep_dir.mkdir(exist_ok=True)
                np.save(rep_dir / f"{bid}_{lt}_confusion.npy", m_va["confusion_matrix"])
                plot_confusion_matrix(m_va["confusion_matrix"], bid, lt, rep_dir)

            emb = extract_emb(model, G, lt, stats)
            emb_dict[bid] = emb
            np.save(ld/"embeddings"/f"{bid}.npy", emb)
            torch.save(model.state_dict(), ld/"models"/f"{bid}.pt")

            fin = log[-1]
            rows.append(dict(
                loss=lt, block=bid, gene=blocks[bid]["gene"], latent_dim=d_block,
                n_snps=p, **maf_stats, params=npar, epochs=len(log),
                best_epoch=best_epoch,
                n0_tr=n0_tr, n1_tr=n1_tr, n2_tr=n2_tr,
                n0_va=n0_va, n1_va=n1_va, n2_va=n2_va,
                **best_metrics,
                conc_tr=round(m_tr["conc"],4),
                conc_va=round(m_va["conc"],4),
                base_conc_va=round(m_va["base_conc"],4),
                concordance_gain_va=round(m_va["conc"] - m_va["base_conc"],4),
                bal_acc_va=round(m_va["bal_acc"],4),
                acc0_va=round(m_va["acc0"],4) if not np.isnan(m_va["acc0"]) else np.nan,
                acc1_va=round(m_va["acc1"],4) if not np.isnan(m_va["acc1"]) else np.nan,
                acc2_va=round(m_va["acc2"],4) if not np.isnan(m_va["acc2"]) else np.nan,
                r2_va=round(m_va["r2"],4),
                ld_corr_va=round(m_va["ld_corr"], 4) if np.isfinite(m_va["ld_corr"]) else np.nan,
                sec=round(dt,1)
            ))

        # stack padded (N, B, d) for Phase 2
        dims = np.array([emb_dict[b].shape[1] for b in block_ids], dtype=np.int32)
        max_d = int(dims.max())

        N = emb_dict[block_ids[0]].shape[0]
        B = len(block_ids)
        stack = np.zeros((N, B, max_d), dtype=np.float32)

        for j, b in enumerate(block_ids):
            e = emb_dict[b].astype(np.float32)          # (N, d_j)
            stack[:, j, :e.shape[1]] = e                # pad right with zeros

        np.save(ld/"embeddings"/"all_blocks.npy", stack)
        np.save(ld/"embeddings"/"all_blocks_latent_dims.npy", dims)
        print(f"\n  stacked embeddings (padded) {stack.shape} | max_d={max_d} | dims={dict(pd.Series(dims).value_counts().sort_index())}")
        print(f"\n  stacked embeddings  {stack.shape}")

    # ---- save block-order metadata (Phase 2 needs it) ----
    meta = [dict(pos=i, block_id=b, gene=blocks[b]["gene"],
                 n_snps=blocks[b]["n_snps"]) for i, b in enumerate(block_ids)]
    pd.DataFrame(meta).to_csv(out/"block_order.csv", index=False)

    # ---- summary table ----
    sdf = pd.DataFrame(rows)
    sdf.to_csv(out/"vae_summary.csv", index=False)
    print(f"\n{'═'*55}\n  Phase 1 complete — summary\n{'═'*55}")
    print(sdf.to_string(index=False, max_cols=None))

    # ---- representative blocks selection ----
    if not manual_rep:
        rep_blocks = select_representative_blocks(sdf, cfg, block_ids)
        rep_blocks_set = set(rep_blocks)
    
    # ---- output organization ----
    out_summary = out / "summary"
    out_agg_plots = out / "aggregate_plots"
    out_rep = out / "representative_blocks"
    out_summary.mkdir(exist_ok=True)
    out_agg_plots.mkdir(exist_ok=True)
    out_rep.mkdir(exist_ok=True)
    
    # Move summary
    sdf.to_csv(out_summary / "vae_summary.csv", index=False)
    
    # Aggregate plots
    plot_aggregate_boxes(sdf, out_agg_plots)
    plot_scatter_metric_vs(sdf, "n_snps", out_agg_plots)
    plot_scatter_metric_vs(sdf, "maf_mean", out_agg_plots)
    
    # Top/bottom CSVs
    metric = rep_cfg.get("metric", "bal_acc_va")
    for loss in sdf["loss"].unique():
        sub = sdf[sdf["loss"] == loss].sort_values(metric, ascending=False)
        top = sub.head(10)
        bottom = sub.tail(10)
        top.to_csv(out_summary / f"top10_{loss}_{metric}.csv", index=False)
        bottom.to_csv(out_summary / f"bottom10_{loss}_{metric}.csv", index=False)
    
    # Detailed plots for representative blocks
    for bid in rep_blocks_set:
        rep_dir = out_rep / bid
        rep_dir.mkdir(exist_ok=True)
        
        # Training curves for each loss
        for lt in cfg["loss_functions"]:
            log_path = out / lt / "logs" / f"{bid}.csv"
            if log_path.exists():
                log_df = pd.read_csv(log_path)
                plot_training_curve(log_df, f"{bid}_{lt}", rep_dir)
        
        # Cross-loss metrics
        plot_cross_loss_metrics(sdf, bid, rep_dir)

    return sdf


# ──────────────────────────────────────────────────────────────────
# 8.  CLI  (validate_cfg merged from 01_phase1_block_embedding.py)
# ──────────────────────────────────────────────────────────────────
def validate_cfg(cfg):
    """Pre-flight checks: verify required paths exist and create output dir."""
    raw_dir  = Path(cfg["data"]["raw_dir"])
    block_def = Path(cfg["data"]["block_def"])
    out_dir  = Path(cfg["data"]["output_dir"])

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir missing: {raw_dir}")
    if not block_def.exists():
        raise FileNotFoundError(f"block_def missing: {block_def}")

    out_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, block_def, out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Phase 1 · Block VAE")
    ap.add_argument("--config", default=None, help="Path to Phase1 config YAML (default: configs/config_phase1.yaml)")
    ap.add_argument("--tune", action="store_true", help="Run hyperparameter tuning mode")
    ap.add_argument("--dry-run", action="store_true", help="Display configuration without running")
    ap.add_argument("--save-config", action="store_true",
                    help="write default YAML and exit")
    args = ap.parse_args()

    if args.save_config:
        with open("config_phase1_default.yaml", "w") as f:
            yaml.dump(DEFAULT_CFG, f, default_flow_style=False)
        print("wrote config_phase1_default.yaml"); sys.exit(0)

    resolved_config = args.config or "configs/config_phase1.yaml"
    print(f"CLI --config: {args.config}")
    print(f"Config file exists: {os.path.exists(args.config) if args.config else 'N/A (using default)'}")
    print(f"Resolved config path: {resolved_config}")

    cfg = load_config(args.config)
    print(f"Top-level cfg keys: {list(cfg.keys())}")
    print(f"Loaded cfg['tuning']: {cfg.get('tuning', 'MISSING')}")
    print(f"Loaded cfg['representative']: {cfg.get('representative', 'MISSING')}")

    raw_dir, block_def, out_dir = validate_cfg(cfg)
    print(f"[phase1] using config: {resolved_config}")
    print(f"[phase1] raw_dir={raw_dir}")
    print(f"[phase1] block_def={block_def}")
    print(f"[phase1] output_dir={out_dir}")

    if args.dry_run:
        print("[phase1] dry-run complete; no pipeline executed.")
        sys.exit(0)

    t0 = time.time()
    if args.tune:
        run_tuning(cfg)
    else:
        run_phase1(cfg)

    # Post-run output validation
    if args.tune:
        tuning_dir = out_dir / "tuning"
        for p in [tuning_dir / "best_params.yaml",
                  tuning_dir / "tuning_results.csv",
                  tuning_dir / "tuning_summary.csv"]:
            if not p.exists():
                raise FileNotFoundError(f"Expected tuning output missing: {p}")
        print(f"[validation] tuning complete: {tuning_dir}")
    else:
        for p in [out_dir / "block_order.csv",
                  out_dir / "subjects.csv",
                  out_dir / "vae_summary.csv"]:
            if not p.exists():
                raise FileNotFoundError(f"Expected Phase1 output missing: {p}")
        for lt in cfg["loss_functions"]:
            emb_dir = out_dir / lt / "embeddings"
            if emb_dir.exists() and any(emb_dir.glob("*.npy")):
                print(f"[validation] embeddings found for {lt}")
            else:
                print(f"[validation] WARNING: no embeddings for {lt}")
        print(f"[validation] phase1 complete: {out_dir}")

    print(f"\n[phase1] complete (took {time.time() - t0:.1f}s)")