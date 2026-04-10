#!/usr/bin/env python3
"""attention_phase2.py — Phase 2: cross-block Transformer attention aggregation.

Purpose : Consume Phase 1 per-block embeddings; train a Transformer to aggregate
          across blocks; produce subject-level embeddings, per-block attention
          weights, and contextual block representations; cluster subjects.
Inputs  : Phase 1 output dir (block_order.csv, subjects.csv, all_blocks.npy per
          loss), configs/config_phase2.yaml
Outputs : results/output_regions2/{phase2_summary.csv,
          <loss>/embeddings/*, <loss>/clustering/cluster_labels.csv}
Workflow: Step 2 of 2 — requires Phase 1 output.
"""
import os, sys, time, math, yaml, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------------------------------------
# Optional dependencies
# ------------------------------------------------------------

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        silhouette_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
    )
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# Uncomment if installed
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import hdbscan as _hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


# ============================================================
# 1. DEFAULT CONFIG
# ============================================================
DEFAULT_CFG = {
    "phase1_dir": "results/output_regions",
    "output_dir": "results/output_regions2",
    "attention": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 128,
        "dropout": 0.10,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "epochs": 300,
        "patience": 30,
        "grad_clip": 1.0,
        "seed": 42,

        # New: self-attention extraction controls
        "extract_self_attn": True,
        "save_full_self_attn": False,   # can be very large: (N, H, B, B) per layer
    },
    "clustering": {
        "k_range": [2, 3, 4, 5, 6, 8, 10],
        "use_hdbscan": True,
        "hdbscan_min_size": 10,
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "umap_seed": 42,
    },
    "loss_functions": ["ORD", "MSE", "MSE_STD"],
}


def load_config(path=None):
    if path is None:
        path = "configs/config_phase2.yaml"
    cfg = {
        k: (v.copy() if isinstance(v, dict) else v[:] if isinstance(v, list) else v)
        for k, v in DEFAULT_CFG.items()
    }
    if os.path.exists(path):
        with open(path) as fh:
            usr = yaml.safe_load(fh) or {}
        for sec in cfg:
            if sec in usr and isinstance(cfg[sec], dict):
                cfg[sec].update(usr[sec])
            elif sec in usr:
                cfg[sec] = usr[sec]
    return cfg


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_cfg):
    if device_cfg == "cpu":
        print("[device] CPU (as configured)")
        return torch.device("cpu")
    if device_cfg == "cuda" and torch.cuda.is_available():
        print(f"[device] CUDA — {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if device_cfg == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.zeros(2, device="mps")
            print("[device] Apple MPS")
            return torch.device("mps")
        except Exception:
            pass
    print("[device] CPU (fallback)")
    return torch.device("cpu")


# ============================================================
# 2. LOAD FROZEN PHASE-1 OUTPUTS
# ============================================================
def load_phase1(p1_dir: str, loss_functions: list):
    """
    Returns
    -------
    subjects   : ndarray of IID strings  (N,)
    tr_ix      : ndarray  (n_train,)
    va_ix      : ndarray  (n_val,)
    block_meta : DataFrame [pos, block_id, gene, n_snps]
    embeddings : dict {loss_type: ndarray (N, B, d)}
    """
    p1 = Path(p1_dir)
    required = ["subjects.csv", "train_idx.npy", "val_idx.npy", "block_order.csv"]
    for f in required:
        fp = p1 / f
        if not fp.exists():
            raise FileNotFoundError(f"Phase 1 output missing: {fp}")

    subjects = pd.read_csv(p1 / "subjects.csv")["IID"].astype(str).values
    tr_ix = np.load(p1 / "train_idx.npy")
    va_ix = np.load(p1 / "val_idx.npy")
    block_meta = pd.read_csv(p1 / "block_order.csv")

    embeddings = {}
    N, B, d_in = None, None, None

    for lt in loss_functions:
        fp = p1 / lt / "embeddings" / "all_blocks.npy"
        if not fp.exists():
            raise FileNotFoundError(f"Missing stacked embeddings: {fp}")
        emb = np.load(fp)  # (N, B, d)
        if N is None:
            N, B, d_in = emb.shape
        elif emb.shape != (N, B, d_in):
            raise ValueError(f"Embedding shape mismatch for {lt}: expected {(N, B, d_in)}, got {emb.shape}")
        embeddings[lt] = emb
        print(f"  [{lt:8s}] loaded embeddings {emb.shape}")

    return subjects, tr_ix, va_ix, block_meta, embeddings


# ============================================================
# 3. EXPLICIT TRANSFORMER LAYER WITH ATTN RETURN
# ============================================================
class CustomEncoderLayer(nn.Module):
    """
    Equivalent in spirit to nn.TransformerEncoderLayer with:
      - batch_first=True
      - norm_first=True
      - GELU activation
    but explicitly returns per-head self-attention maps.
    """

    def __init__(self, d_model, n_heads, d_ff=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x, return_attn=False):
        """
        x: (batch, B, d_model)

        Returns
        -------
        x_out : (batch, B, d_model)
        attn_weights : (batch, n_heads, B, B) if return_attn else None
        """
        # Pre-LN self-attention
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            x_norm, x_norm, x_norm,
            need_weights=return_attn,
            average_attn_weights=False
        )
        x = x + self.dropout1(attn_out)

        # Pre-LN feedforward
        x_norm = self.norm2(x)
        ff = self.linear2(self.ff_dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(ff)

        return x, attn_weights if return_attn else None


class AttentionAggregator(nn.Module):
    """
    Transformer-style model:
      (B, d_in) frozen block embeddings
        -> projected block tokens
        -> contextualized by self-attention
        -> pooled to one subject embedding
        -> decoded back to (B, d_in)

    Outputs:
      - pooled subject embedding
      - pooling attention weights (subject embedding importance over blocks)
      - contextualized block representations
      - optional self-attention maps per layer/head (true block->block attention)
    """

    def __init__(
        self,
        n_blocks: int,
        d_in: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.pos_emb = nn.Parameter(torch.randn(1, n_blocks, d_model) * 0.02)

        self.transformer_layers = nn.ModuleList([
            CustomEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.post_norm = nn.LayerNorm(d_model)

        # learned pooling query
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self._scale = math.sqrt(d_model)

        self.embed_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, n_blocks * d_in),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x, return_self_attn=False):
        """
        Parameters
        ----------
        x : (batch, B, d_in)

        Returns
        -------
        embedding : (batch, d_model)
        pool_attn : (batch, B)
        h_blocks  : (batch, B, d_model)
        self_attn_maps : list[n_layers] of (batch, n_heads, B, B), optional
        """
        batch_size = x.size(0)
        h = self.input_proj(x) + self.pos_emb

        self_attn_maps = [] if return_self_attn else None
        for layer in self.transformer_layers:
            h, w = layer(h, return_attn=return_self_attn)
            if return_self_attn:
                self_attn_maps.append(w)

        h = self.post_norm(h)

        # pooling attention (NOT block->block; this is block->subject embedding weight)
        q = self.pool_query.expand(batch_size, -1, -1)             # (batch, 1, d_model)
        scores = torch.bmm(q, h.transpose(1, 2)) / self._scale     # (batch, 1, B)
        pool_attn = F.softmax(scores, dim=-1)                      # (batch, 1, B)
        pooled = torch.bmm(pool_attn, h).squeeze(1)                # (batch, d_model)

        embedding = self.embed_head(pooled)

        if return_self_attn:
            return embedding, pool_attn.squeeze(1), h, self_attn_maps
        return embedding, pool_attn.squeeze(1), h

    def decode(self, z):
        return self.decoder(z).view(-1, self.n_blocks, self.d_in)

    def forward(self, x, return_self_attn=False):
        if return_self_attn:
            emb, pool_attn, h_blocks, self_attn_maps = self.encode(
                x, return_self_attn=True
            )
            recon = self.decode(emb)
            return recon, emb, pool_attn, h_blocks, self_attn_maps
        else:
            emb, pool_attn, _ = self.encode(x, return_self_attn=False)
            recon = self.decode(emb)
            return recon, emb, pool_attn


# ============================================================
# 4. TRAINING
# ============================================================
def train_attention_model(model, tr_t, va_t, cfg, device, log_csv):
    ac = cfg["attention"]

    tr_dl = DataLoader(
        TensorDataset(tr_t),
        batch_size=ac["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    va_dl = DataLoader(
        TensorDataset(va_t),
        batch_size=ac["batch_size"],
        shuffle=False,
    )

    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=ac["lr"],
        weight_decay=ac["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=15, factor=0.5, min_lr=1e-6
    )

    best_val = float("inf")
    best_sd = None
    best_epoch = 0
    wait = 0
    logs = []

    for ep in range(1, ac["epochs"] + 1):
        ep_t0 = time.time()

        # ---- training pass ----
        model.train()
        tr_loss_acc = 0.0
        tr_n = 0
        for (xb,) in tr_dl:
            xb = xb.to(device)
            opt.zero_grad()
            recon, _, _ = model(xb, return_self_attn=False)
            loss = F.mse_loss(recon, xb)
            loss.backward()
            if ac.get("grad_clip", 0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), ac["grad_clip"])
            opt.step()
            tr_loss_acc += loss.item() * xb.size(0)
            tr_n += xb.size(0)
        tr_loss = tr_loss_acc / tr_n

        # ---- validation pass ----
        model.eval()
        va_loss_acc = 0.0
        va_n = 0
        with torch.no_grad():
            for (xb,) in va_dl:
                xb = xb.to(device)
                recon, _, _ = model(xb, return_self_attn=False)
                loss = F.mse_loss(recon, xb)
                va_loss_acc += loss.item() * xb.size(0)
                va_n += xb.size(0)
        va_loss = va_loss_acc / va_n

        sched.step(va_loss)
        current_lr = opt.param_groups[0]["lr"]
        epoch_sec = time.time() - ep_t0

        logs.append({
            "epoch": ep,
            "tr_loss": round(tr_loss, 6),
            "va_loss": round(va_loss, 6),
            "lr": current_lr,
            "epoch_sec": round(epoch_sec, 3),
        })

        if ep % 10 == 0 or ep == 1:
            print(
                f"    ep {ep:4d}  tr={tr_loss:.5f}  va={va_loss:.5f}"
                f"  lr={current_lr:.2e}  {epoch_sec:.1f}s"
            )

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = ep
            wait = 0
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if wait >= ac["patience"]:
            print(f"    → early stop at epoch {ep}")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    model.to("cpu")

    if log_csv is not None:
        pd.DataFrame(logs).to_csv(log_csv, index=False)
    return logs, best_epoch, best_val


# ============================================================
# 5. EXTRACTION HELPERS
# ============================================================
@torch.no_grad()
def extract_all(
    model,
    data_np,
    batch_size=256,
    return_self_attn=False,
    save_full_self_attn=False,
):
    """
    Run frozen model over all subjects.

    Returns
    -------
    embeddings      : (N, d_model)
    pool_attn       : (N, B)
    reconstructions : (N, B, d_in)
    block_reprs     : (N, B, d_model)
    self_attn_mean  : list[n_layers] of (n_heads, B, B) or None
    self_attn_full  : list[n_layers] of (N, n_heads, B, B) or None
    """
    model.eval()

    dl = DataLoader(
        TensorDataset(torch.tensor(data_np, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    embs, pool_attns, recons, block_reprs = [], [], [], []

    self_attn_sums = None
    self_attn_full = None
    total_seen = 0

    for (batch,) in dl:
        if return_self_attn:
            emb, pool_attn, h_blocks, attn_maps = model.encode(
                batch, return_self_attn=True
            )
        else:
            emb, pool_attn, h_blocks = model.encode(
                batch, return_self_attn=False
            )
            attn_maps = None

        rec = model.decode(emb)

        embs.append(emb.cpu().numpy())
        pool_attns.append(pool_attn.cpu().numpy())
        recons.append(rec.cpu().numpy())
        block_reprs.append(h_blocks.cpu().numpy())

        if return_self_attn:
            if self_attn_sums is None:
                self_attn_sums = [
                    np.zeros((w.shape[1], w.shape[2], w.shape[3]), dtype=np.float64)
                    for w in attn_maps
                ]
                if save_full_self_attn:
                    self_attn_full = [[] for _ in attn_maps]

            for li, w in enumerate(attn_maps):
                w_np = w.cpu().numpy()  # (batch, n_heads, B, B)
                self_attn_sums[li] += w_np.sum(axis=0)
                if save_full_self_attn:
                    self_attn_full[li].append(w_np)

            total_seen += batch.size(0)

    embs = np.concatenate(embs, axis=0)
    pool_attns = np.concatenate(pool_attns, axis=0)
    recons = np.concatenate(recons, axis=0)
    block_reprs = np.concatenate(block_reprs, axis=0)

    if return_self_attn:
        self_attn_mean = [x / total_seen for x in self_attn_sums]  # (H, B, B)
        if save_full_self_attn:
            self_attn_full = [np.concatenate(x, axis=0) for x in self_attn_full]
    else:
        self_attn_mean = None
        self_attn_full = None

    return embs, pool_attns, recons, block_reprs, self_attn_mean, self_attn_full


def per_block_mse(recon, truth):
    """(N, B, d) -> (B,) mean SE per block."""
    return np.mean((recon - truth) ** 2, axis=(0, 2))


def summarize_block_to_block_attention(attn_mean, block_names):
    """
    attn_mean: (n_heads, B, B) averaged across subjects
    Returns:
      head_df      : one row per (head, src_block, dst_block)
      overall_df   : one row per (src_block, dst_block), averaged across heads
    """
    n_heads, B, _ = attn_mean.shape
    rows_head = []
    rows_overall = []

    overall = attn_mean.mean(axis=0)  # (B, B)

    for h in range(n_heads):
        for i in range(B):
            for j in range(B):
                rows_head.append({
                    "head": h,
                    "src_block": block_names[i],
                    "dst_block": block_names[j],
                    "attention": float(attn_mean[h, i, j]),
                })

    for i in range(B):
        for j in range(B):
            rows_overall.append({
                "src_block": block_names[i],
                "dst_block": block_names[j],
                "attention": float(overall[i, j]),
            })

    head_df = pd.DataFrame(rows_head)
    overall_df = pd.DataFrame(rows_overall).sort_values(
        "attention", ascending=False
    ).reset_index(drop=True)

    return head_df, overall_df


# ============================================================
# 6. CLUSTERING
# ============================================================
def run_clustering(emb, cc, out_dir):
    """
    K-Means sweep + optional HDBSCAN.

    Returns
    -------
    labels  : dict {method_string: ndarray(N,)}
    metrics : DataFrame
    """
    if not HAS_SKLEARN:
        warnings.warn("scikit-learn not installed — skipping clustering")
        return {}, pd.DataFrame()

    Z = StandardScaler().fit_transform(emb)
    labels, rows = {}, []

    # ---- KMeans ----
    for k in cc["k_range"]:
        km = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=cc.get("umap_seed", 42),
        )
        lab = km.fit_predict(Z)
        sil = silhouette_score(Z, lab) if k > 1 else 0.0

        key = f"kmeans_k{k}"
        labels[key] = lab
        rows.append({
            "method": "KMeans",
            "k": k,
            "silhouette": round(sil, 4),
            "n_clusters": k,
            "n_noise": 0,
        })
        print(f"    KMeans k={k:2d}  silhouette={sil:.4f}")

    # ---- HDBSCAN ----
    if cc.get("use_hdbscan", False):
        if not HAS_HDBSCAN:
            warnings.warn("HDBSCAN requested but package not installed — skipping")
        else:
            try:
                import hdbscan as hdbscan_lib
                ms = cc.get("hdbscan_min_size", 10)
                cl = hdbscan_lib.HDBSCAN(min_cluster_size=ms)
                lab = cl.fit_predict(Z)

                nc = len(set(lab) - {-1})
                nn_ = int((lab == -1).sum())
                mask = lab >= 0
                sil = silhouette_score(Z[mask], lab[mask]) if (mask.sum() > 1 and nc > 1) else 0.0

                labels["hdbscan"] = lab
                rows.append({
                    "method": "HDBSCAN",
                    "k": nc,
                    "silhouette": round(sil, 4),
                    "n_clusters": nc,
                    "n_noise": nn_,
                })
                print(f"    HDBSCAN  k={nc}  noise={nn_}  silhouette={sil:.4f}")
            except ImportError:
                warnings.warn("HDBSCAN import failed — skipping")

    mdf = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    mdf.to_csv(out_dir / "clustering_metrics.csv", index=False)
    if labels:
        pd.DataFrame(labels).to_csv(out_dir / "cluster_labels.csv", index=False)

    return labels, mdf


# ============================================================
# 7. VISUALIZATION
# ============================================================
def compute_umap(emb, cc):
    if not HAS_UMAP:
        return None
    # return umap.UMAP(
    #     n_neighbors=cc["umap_n_neighbors"],
    #     min_dist=cc["umap_min_dist"],
    #     random_state=cc.get("umap_seed", 42),
    # ).fit_transform(emb)


def _best_kmeans_key(labels):
    for k in sorted(labels):
        if k.startswith("kmeans"):
            return k
    return None


def plot_umap_clusters(Z2d, labels, loss_type, out):
    if not HAS_PLT or Z2d is None or not labels:
        return

    key = _best_kmeans_key(labels)
    if key is None:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Z2d[:, 0], Z2d[:, 1], c=labels[key], cmap="tab10", s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Cluster")
    ax.set_title(f"{loss_type} — UMAP ({key})", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out / f"umap_{loss_type}.png", dpi=150)
    plt.close()


def plot_attention_summary(pool_attn, block_meta, loss_type, out):
    """
    pool_attn is the learned pooling attention:
    one subject-level weight per block, not block->block self-attention.
    """
    if not HAS_PLT:
        return

    bnames = block_meta["block_id"].values
    B = len(bnames)

    fig, axes = plt.subplots(
        1, 3, figsize=(22, max(6, B * 0.35)),
        gridspec_kw={"width_ratios": [3, 2, 2]}
    )

    # (a) heatmap — random subjects
    rng = np.random.RandomState(42)  # for reproducible sampling
    n_show = min(100, pool_attn.shape[0])
    idx = rng.choice(pool_attn.shape[0], n_show, replace=False)

    ax = axes[0]
    im = ax.imshow(pool_attn[idx], aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(B))
    ax.set_xticklabels(bnames, rotation=90, fontsize=7)
    ax.set_ylabel("Individuals (random sample)")
    ax.set_title(f"{loss_type} — pooling attention per individual")
    plt.colorbar(im, ax=ax, shrink=0.6)

    # (b) box plot
    ax = axes[1]
    ax.boxplot(pool_attn, vert=False, labels=bnames)
    ax.set_xlabel("Pooling attention weight")
    ax.set_title("Distribution per block")

    # (c) mean ± std bar
    ax = axes[2]
    mu = pool_attn.mean(0)
    sd = pool_attn.std(0)
    y = np.arange(B)
    ax.barh(y, mu, xerr=sd, color="steelblue", alpha=0.8, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(bnames, fontsize=7)
    ax.set_xlabel("Mean pooling attention")
    ax.set_title("Mean ± SD")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out / f"attention_summary_{loss_type}.png", dpi=150)
    plt.close()


def plot_reconstruction_per_block(block_mse, block_meta, loss_type, out):
    if not HAS_PLT:
        return
    bnames = block_meta["block_id"].values
    fig, ax = plt.subplots(figsize=(10, max(4, len(bnames) * 0.35)))
    y = np.arange(len(bnames))
    ax.barh(y, block_mse, color="salmon", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(bnames, fontsize=7)
    ax.set_xlabel("Mean Squared Error")
    ax.set_title(f"{loss_type} — Reconstruction MSE per block")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out / f"recon_per_block_{loss_type}.png", dpi=150)
    plt.close()


def plot_self_attention_heatmaps(self_attn_mean, block_meta, loss_type, out):
    """
    self_attn_mean: list[n_layers] of (H, B, B), already averaged over subjects
    """
    if not HAS_PLT or self_attn_mean is None:
        return

    block_names = block_meta["block_id"].values

    for li, layer_map in enumerate(self_attn_mean):
        H, B, _ = layer_map.shape

        # average over heads
        mean_over_heads = layer_map.mean(axis=0)

        fig, ax = plt.subplots(figsize=(max(7, B * 0.5), max(6, B * 0.5)))
        im = ax.imshow(mean_over_heads, cmap="viridis", aspect="auto")
        ax.set_xticks(range(B))
        ax.set_xticklabels(block_names, rotation=90, fontsize=7)
        ax.set_yticks(range(B))
        ax.set_yticklabels(block_names, fontsize=7)
        ax.set_xlabel("Key block (attended to)")
        ax.set_ylabel("Query block (updated from)")
        ax.set_title(f"{loss_type} — self-attention layer {li} (mean over heads)")
        plt.colorbar(im, ax=ax, shrink=0.7)
        plt.tight_layout()
        plt.savefig(out / f"self_attention_layer{li}_mean.png", dpi=150)
        plt.close()


# ============================================================
# 8. CROSS-LOSS COMPARISON
# ============================================================
def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XtY = np.linalg.norm(X.T @ Y, "fro") ** 2
    XtX = np.linalg.norm(X.T @ X, "fro") ** 2
    YtY = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return float(XtY / (np.sqrt(XtX * YtY) + 1e-12))


def compare_across_losses(all_res, out_dir):
    """
    Pairwise:
      - Linear CKA of subject embeddings
      - ARI / NMI of matching clustering outputs
      - Pearson r of mean pooling attention profiles
    """
    lts = list(all_res.keys())
    if len(lts) < 2:
        print("  [skip] need >= 2 loss functions for comparison")
        return pd.DataFrame()

    if not HAS_SKLEARN:
        warnings.warn("scikit-learn missing — skipping comparison metrics")
        return pd.DataFrame()

    print(f"\n{'═' * 55}")
    print("  Cross-loss comparison")
    print(f"{'═' * 55}")

    rows = []

    for la, lb in combinations(lts, 2):
        Ea = all_res[la]["embeddings"]
        Eb = all_res[lb]["embeddings"]
        cka = linear_cka(Ea, Eb)

        pa = all_res[la]["pool_attn"].mean(0)
        pb = all_res[lb]["pool_attn"].mean(0)
        attn_r = float(np.corrcoef(pa, pb)[0, 1])

        for key in sorted(all_res[la]["cluster_labels"]):
            if key not in all_res[lb]["cluster_labels"]:
                continue

            ya = all_res[la]["cluster_labels"][key]
            yb = all_res[lb]["cluster_labels"][key]
            mask = (ya >= 0) & (yb >= 0)
            if mask.sum() < 10:
                continue

            ari = adjusted_rand_score(ya[mask], yb[mask])
            nmi = normalized_mutual_info_score(ya[mask], yb[mask])

            rows.append({
                "loss_a": la,
                "loss_b": lb,
                "clust_method": key,
                "linear_CKA": round(cka, 4),
                "ARI": round(ari, 4),
                "NMI": round(nmi, 4),
                "pool_attn_pearson_r": round(attn_r, 4),
            })

            print(
                f"  {la} vs {lb} [{key}] "
                f"CKA={cka:.4f}  ARI={ari:.4f}  NMI={nmi:.4f}  pool_attn_r={attn_r:.4f}"
            )

    cdf = pd.DataFrame(rows)
    cdf.to_csv(out_dir / "cross_loss_comparison.csv", index=False)

    if HAS_UMAP and HAS_PLT:
        _plot_joint_umap(all_res, out_dir)
        _plot_attention_comparison(all_res, out_dir)

    return cdf


def _plot_joint_umap(all_res, out_dir):
    lts = list(all_res.keys())
    n = len(lts)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for i, lt in enumerate(lts):
        ax = axes[i]
        Z2d = all_res[lt].get("umap_2d")
        if Z2d is None:
            ax.set_title(f"{lt} (no UMAP)")
            continue

        key = _best_kmeans_key(all_res[lt]["cluster_labels"])
        c = all_res[lt]["cluster_labels"].get(key)
        if c is not None:
            ax.scatter(Z2d[:, 0], Z2d[:, 1], c=c, cmap="tab10", s=6, alpha=0.6)
        else:
            ax.scatter(Z2d[:, 0], Z2d[:, 1], s=6, alpha=0.6)

        ax.set_title(lt, fontsize=14)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    plt.tight_layout()
    plt.savefig(out_dir / "joint_umap_comparison.png", dpi=150)
    plt.close()
    print("  saved joint_umap_comparison.png")


def _plot_attention_comparison(all_res, out_dir):
    lts = list(all_res.keys())
    n = len(lts)
    block_names = all_res[lts[0]]["block_names"]
    B = len(block_names)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, max(4, B * 0.3)), sharey=True)
    if n == 1:
        axes = [axes]

    y = np.arange(B)
    for i, lt in enumerate(lts):
        mu = all_res[lt]["pool_attn"].mean(0)
        axes[i].barh(y, mu, color="steelblue", alpha=0.85)
        axes[i].set_yticks(y)
        axes[i].set_yticklabels(block_names, fontsize=7)
        axes[i].set_xlabel("Mean pooling attention")
        axes[i].set_title(lt)
        axes[i].invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_dir / "attention_comparison.png", dpi=150)
    plt.close()
    print("  saved attention_comparison.png")


# ============================================================
# 9. MAIN PHASE-2 PIPELINE
# ============================================================
def run_phase2(cfg):
    ac = cfg["attention"]
    cc = cfg["clustering"]

    set_seed(ac["seed"])
    dev = get_device(ac.get("device", "cpu"))

    # Set deterministic behavior
    if dev.type == "cpu":
        torch.use_deterministic_algorithms(True)
    elif dev.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "config_phase2.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("\n══════ Step 8 · Loading Phase 1 frozen embeddings ══════")
    subjects, tr_ix, va_ix, block_meta, embeddings = load_phase1(
        cfg["phase1_dir"], cfg["loss_functions"]
    )

    N, B, d_in = list(embeddings.values())[0].shape
    block_names = block_meta["block_id"].values
    print(f"  N={N}  B={B}  d_in={d_in}")

    all_results = {}
    summary_rows = []

    for lt in cfg["loss_functions"]:
        print(f"\n{'═' * 55}")
        print(f"  Loss function: {lt}")
        print(f"{'═' * 55}")

        lt_dir = out / lt
        for sub in ("logs", "models", "embeddings", "clustering", "plots", "attention_maps"):
            (lt_dir / sub).mkdir(parents=True, exist_ok=True)

        emb_block = embeddings[lt]  # (N, B, d_in)

        # ------------------ Step 9: train ------------------
        print("\n  Step 9 · Training attention aggregator ...")
        tr_t = torch.tensor(emb_block[tr_ix], dtype=torch.float32)
        va_t = torch.tensor(emb_block[va_ix], dtype=torch.float32)

        model = AttentionAggregator(
            n_blocks=B,
            d_in=d_in,
            d_model=ac["d_model"],
            n_heads=ac["n_heads"],
            n_layers=ac["n_layers"],
            d_ff=ac["d_ff"],
            dropout=ac["dropout"],
        )

        npar = sum(p.numel() for p in model.parameters())
        print(
            f"    architecture: {B}x{d_in} -> d_model={ac['d_model']}  "
            f"heads={ac['n_heads']}  layers={ac['n_layers']}  params={npar:,}"
        )

        t0 = time.time()
        log, best_epoch, best_val_loss = train_attention_model(
            model, tr_t, va_t, cfg, dev,
            lt_dir / "logs" / "attention_training.csv"
        )
        dt = time.time() - t0
        print(f"    done in {dt:.1f}s  ({len(log)} epochs, best at {best_epoch})")

        torch.save(model.state_dict(), lt_dir / "models" / "attention_aggregator.pt")

        # ------------------ Step 10: extract ------------------
        print("\n  Step 10 · Extracting embeddings, pooling attention, and self-attention ...")
        final_emb, pool_attn, recon, block_repr, self_attn_mean, self_attn_full = extract_all(
            model,
            emb_block,
            batch_size=256,
            return_self_attn=ac.get("extract_self_attn", True),
            save_full_self_attn=ac.get("save_full_self_attn", False),
        )

        global_mse = float(np.mean((recon - emb_block) ** 2))
        blk_mse = per_block_mse(recon, emb_block)

        print(f"    individual embedding : {final_emb.shape}")
        print(f"    pooling attention    : {pool_attn.shape}")
        print(f"    reconstruction MSE   : {global_mse:.6f}")
        print(f"    worst block MSE      : {block_names[blk_mse.argmax()]} ({blk_mse.max():.6f})")

        # ---- save arrays ----
        np.save(lt_dir / "embeddings" / "individual_embeddings.npy", final_emb)
        np.save(lt_dir / "embeddings" / "pooling_attention_weights.npy", pool_attn)
        np.save(lt_dir / "embeddings" / "reconstructions.npy", recon)
        np.save(lt_dir / "embeddings" / "block_contextual_repr.npy", block_repr)

        # ---- human-readable CSVs ----
        emb_df = pd.DataFrame(final_emb, columns=[f"emb_{i}" for i in range(final_emb.shape[1])])
        emb_df.insert(0, "IID", subjects)
        emb_df.to_csv(lt_dir / "embeddings" / "individual_embeddings.csv", index=False)

        pool_df = pd.DataFrame(pool_attn, columns=block_names)
        pool_df.insert(0, "IID", subjects)
        pool_df.to_csv(lt_dir / "embeddings" / "pooling_attention_weights.csv", index=False)

        blk_mse_df = pd.DataFrame({
            "block_id": block_names,
            "recon_mse": blk_mse.round(6),
        })
        blk_mse_df.to_csv(lt_dir / "embeddings" / "per_block_recon_mse.csv", index=False)

        # ---- self-attention outputs ----
        if self_attn_mean is not None:
            for li, arr in enumerate(self_attn_mean):
                np.save(lt_dir / "attention_maps" / f"self_attention_layer{li}_mean.npy", arr)

            if self_attn_full is not None:
                for li, arr in enumerate(self_attn_full):
                    np.save(lt_dir / "attention_maps" / f"self_attention_layer{li}_full.npy", arr)

            for li, arr in enumerate(self_attn_mean):
                head_df, overall_df = summarize_block_to_block_attention(arr, block_names)
                head_df.to_csv(
                    lt_dir / "attention_maps" / f"self_attention_layer{li}_by_head.csv",
                    index=False
                )
                overall_df.to_csv(
                    lt_dir / "attention_maps" / f"self_attention_layer{li}_overall.csv",
                    index=False
                )

                # top non-diagonal block->block pairs
                top_pairs = overall_df[overall_df["src_block"] != overall_df["dst_block"]].head(50)
                top_pairs.to_csv(
                    lt_dir / "attention_maps" / f"self_attention_layer{li}_top_pairs.csv",
                    index=False
                )

        # ------------------ Step 11: clustering ------------------
        print(f"\n  Step 11 · Clustering ({len(cc['k_range'])} K-Means + HDBSCAN) ...")
        cluster_labels, cluster_metrics = run_clustering(
            final_emb, cc, lt_dir / "clustering"
        )

        print("    computing UMAP ...")
        umap_2d = compute_umap(final_emb, cc)
        if umap_2d is not None:
            np.save(lt_dir / "embeddings" / "umap_2d.npy", umap_2d)

        # ------------------ plots ------------------
        plot_umap_clusters(umap_2d, cluster_labels, lt, lt_dir / "plots")
        plot_attention_summary(pool_attn, block_meta, lt, lt_dir / "plots")
        plot_reconstruction_per_block(blk_mse, block_meta, lt, lt_dir / "plots")
        plot_self_attention_heatmaps(self_attn_mean, block_meta, lt, lt_dir / "plots")

        # ------------------ stash for comparison ------------------
        best_sil = float(cluster_metrics["silhouette"].max()) if len(cluster_metrics) > 0 else 0.0

        all_results[lt] = {
            "embeddings": final_emb,
            "pool_attn": pool_attn,
            "cluster_labels": cluster_labels,
            "cluster_metrics": cluster_metrics,
            "umap_2d": umap_2d,
            "recon_mse": global_mse,
            "block_names": block_names,
        }

        summary_rows.append({
            "loss": lt,
            "params": npar,
            "epochs": len(log),
            "best_epoch": best_epoch,
            "final_tr_loss": log[-1]["tr_loss"],
            "final_va_loss": log[-1]["va_loss"],
            "best_va_loss": round(best_val_loss, 6),
            "recon_mse": round(global_mse, 6),
            "best_silhouette": round(best_sil, 4),
            "seconds": round(dt, 1),
            "seed": ac["seed"],
            "device": str(dev),
            "n_subjects": N,
            "n_blocks": B,
            "d_in": d_in,
            "loss_functions": str(cfg["loss_functions"]),
        })

    # ------------------ Step 12: compare losses ------------------
    cdf = compare_across_losses(all_results, out)

    # ------------------ Step 13: summary ------------------
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(out / "phase2_summary.csv", index=False)

    print(f"\n{'═' * 55}")
    print("  Phase 2 complete — summary")
    print(f"{'═' * 55}")
    print(sdf.to_string(index=False))

    if len(cdf) > 0:
        print(f"\n  Cross-loss comparison ({len(cdf)} rows):")
        print(cdf.to_string(index=False))

    return all_results


# ============================================================
# 10. CLI  (validate_cfg merged from 02_phase2_attention_aggregation.py)
# ============================================================
def validate_cfg(cfg):
    """Pre-flight checks: verify Phase 1 artifacts exist and create output dir."""
    phase1_dir = Path(cfg.get("phase1_dir", ""))
    out_dir    = Path(cfg.get("output_dir", ""))

    if not phase1_dir.exists():
        raise FileNotFoundError(f"phase1_dir missing: {phase1_dir}")

    for f in ["subjects.csv", "train_idx.npy", "val_idx.npy", "block_order.csv"]:
        fp = phase1_dir / f
        if not fp.exists():
            raise FileNotFoundError(f"Required Phase 1 file missing: {fp}")

    for lt in cfg.get("loss_functions", []):
        emb_fp = phase1_dir / lt / "embeddings" / "all_blocks.npy"
        if not emb_fp.exists():
            raise FileNotFoundError(f"Missing embeddings for loss {lt}: {emb_fp}")

    out_dir.mkdir(parents=True, exist_ok=True)
    return phase1_dir, out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Phase 2 · Attention Aggregation")
    ap.add_argument("--config", default="configs/config_phase2.yaml", help="YAML config (overrides defaults)")
    ap.add_argument("--dry-run", action="store_true", help="Display configuration without running")
    ap.add_argument("--save-config", action="store_true", help="write default YAML and exit")
    args = ap.parse_args()

    if args.save_config:
        with open("config_phase2_default.yaml", "w") as f:
            yaml.dump(DEFAULT_CFG, f, default_flow_style=False)
        print("wrote config_phase2_default.yaml")
        sys.exit(0)

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

    t0 = time.time()
    run_phase2(cfg)

    # Post-run output validation
    expected = [out_dir / "phase2_summary.csv"]
    for lt in cfg.get("loss_functions", []):
        expected.append(out_dir / lt / "clustering" / "cluster_labels.csv")
    for p in expected:
        if not p.exists():
            raise FileNotFoundError(f"Expected output missing after Phase2: {p}")

    print(f"\n[phase2] complete (took {time.time() - t0:.1f}s)")