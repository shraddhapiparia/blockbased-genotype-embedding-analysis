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

# ── reproducibility metadata helper (same directory) ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_metadata import write_run_metadata as _write_run_metadata

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
        # Multi-token pooling: 1 = original single-query pooling (default, backward compat)
        "n_pool_tokens": 1,
    },
    "clustering": {
        "k_range": [2, 3, 4, 5, 6, 8, 10],
        "use_hdbscan": True,
        "hdbscan_min_size": 10,
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "umap_seed": 42,
    },
    "diagnostics": {
        "enabled": True,
        "save_initial_block_repr": True,
        "compute_contextualization_change": True,
        "compute_phase1_phase2_join": True,
    },
    "baselines": {
        "enabled": True,
        "run_pca": True,
        "run_mean_pool": True,
        "pca_n_components": None,   # None → use attention.d_model with cap
        "pca_sweep": [64, 128, 256, 512] 
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
        print("[device] CPU")
        return torch.device("cpu")
    if device_cfg == "cuda":
        if torch.cuda.is_available():
            print(f"[device] CUDA — {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        print("[device] CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")
    if device_cfg == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.zeros(2, device="mps")
                print("[device] Apple MPS")
                return torch.device("mps")
            except Exception:
                pass
        print("[device] MPS requested but not available — falling back to CPU")
        return torch.device("cpu")
    # "auto" or anything else: try MPS → CUDA → CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.zeros(2, device="mps")
            print("[device] Apple MPS (auto)")
            return torch.device("mps")
        except Exception:
            pass
    if torch.cuda.is_available():
        print(f"[device] CUDA (auto) — {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("[device] CPU (auto)")
    return torch.device("cpu")


# ============================================================
# 2. LOAD FROZEN PHASE-1 OUTPUTS
# ============================================================
def load_phase1(p1_dir: str, loss_functions: list):
    """
    Returns
    -------
    subjects          : ndarray of IID strings  (N,)
    tr_ix             : ndarray  (n_train,)
    va_ix             : ndarray  (n_val,)
    te_ix             : ndarray  (n_test,)
    block_meta        : DataFrame [pos, block_id, gene, n_snps]
    embeddings        : dict {loss_type: ndarray (N, B, d)}
    latent_dims_per_loss : dict {loss_type: list[int] | None}
    """
    p1 = Path(p1_dir)
    required = ["subjects.csv", "train_idx.npy", "val_idx.npy", "test_idx.npy", "block_order.csv"]
    for f in required:
        fp = p1 / f
        if not fp.exists():
            raise FileNotFoundError(f"Phase 1 output missing: {fp}")

    subjects = pd.read_csv(p1 / "subjects.csv")["IID"].astype(str).values
    tr_ix = np.load(p1 / "train_idx.npy")
    va_ix = np.load(p1 / "val_idx.npy")
    te_ix = np.load(p1 / "test_idx.npy")
    block_meta = pd.read_csv(p1 / "block_order.csv")

    n_subjects = len(subjects)
    n_blocks_meta = len(block_meta)
    has_latent_dim = "latent_dim" in block_meta.columns

    embeddings = {}
    latent_dims_per_loss = {}  # per-loss actual dims, None means fall back to block_meta
    N, B, d_in = None, None, None

    for lt in loss_functions:
        fp = p1 / lt / "embeddings" / "all_blocks.npy"
        if not fp.exists():
            raise FileNotFoundError(f"Missing stacked embeddings: {fp}")
        emb = np.load(fp)  # (N, B, d)

        # sanity: subject count
        if emb.shape[0] != n_subjects:
            raise ValueError(
                f"[{lt}] all_blocks.npy subject count {emb.shape[0]} "
                f"!= subjects.csv count {n_subjects}"
            )
        # sanity: block count
        if emb.shape[1] != n_blocks_meta:
            raise ValueError(
                f"[{lt}] all_blocks.npy block count {emb.shape[1]} "
                f"!= block_order.csv count {n_blocks_meta}"
            )

        # Prefer all_blocks_latent_dims.npy (records actual post-clamping dims) over
        # block_order.csv["latent_dim"] (records the requested dims) for the sanity
        # check and for building the per-loss latent mask in run_phase2().
        dims_fp = p1 / lt / "embeddings" / "all_blocks_latent_dims.npy"
        if dims_fp.exists():
            lt_dims = np.load(dims_fp).astype(int)
            if len(lt_dims) != n_blocks_meta:
                raise ValueError(
                    f"[{lt}] all_blocks_latent_dims.npy has {len(lt_dims)} entries "
                    f"!= block_order.csv count {n_blocks_meta}"
                )
            expected_max_d = int(lt_dims.max())
            latent_dims_per_loss[lt] = lt_dims.tolist()
            dim_source = "all_blocks_latent_dims.npy"
        elif has_latent_dim:
            expected_max_d = int(block_meta["latent_dim"].max())
            latent_dims_per_loss[lt] = None  # run_phase2 falls back to block_meta
            dim_source = "block_order.csv"
        else:
            expected_max_d = None
            latent_dims_per_loss[lt] = None
            dim_source = None

        if expected_max_d is not None and emb.shape[2] != expected_max_d:
            raise ValueError(
                f"[{lt}] all_blocks.npy dim {emb.shape[2]} "
                f"!= max_d {expected_max_d} (from {dim_source})"
            )

        if N is None:
            N, B, d_in = emb.shape
        elif emb.shape != (N, B, d_in):
            raise ValueError(f"Embedding shape mismatch for {lt}: expected {(N, B, d_in)}, got {emb.shape}")
        embeddings[lt] = emb
        print(f"  [{lt:8s}] loaded embeddings {emb.shape}  dim_source={dim_source}")

    return subjects, tr_ix, va_ix, te_ix, block_meta, embeddings, latent_dims_per_loss


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

        return x, (attn_weights if return_attn else None)


class BlockProjector(nn.Module):
    """Per-block input projection when blocks have heterogeneous latent dims.

    Accepts a padded (batch, B, max_d) tensor.  For block i, slices the first
    block_dims[i] columns and projects them independently to d_model.
    Drop-in replacement for a shared nn.Linear(d_in, d_model) when all
    block_dims are equal.
    """

    def __init__(self, block_dims: list, d_model: int):
        super().__init__()
        self.block_dims = list(block_dims)
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d_i, d_model), nn.LayerNorm(d_model), nn.GELU())
            for d_i in block_dims
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, B, max_d) → (batch, B, d_model)"""
        tokens = [
            self.projectors[i](x[:, i, :self.block_dims[i]])
            for i in range(len(self.block_dims))
        ]
        return torch.stack(tokens, dim=1)


class AttentionAggregator(nn.Module):
    """
    Transformer-style model:
      (B, d_in) frozen block embeddings
        -> projected block tokens  (per-block via BlockProjector, or shared Linear)
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
        block_dims: list = None,
        n_pool_tokens: int = 1,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_pool_tokens = n_pool_tokens
        self.emb_dim = n_pool_tokens * d_model

        if block_dims is not None:
            if len(block_dims) != n_blocks:
                raise ValueError(
                    f"block_dims length {len(block_dims)} != n_blocks {n_blocks}"
                )
            self.input_proj = BlockProjector(block_dims, d_model)
        else:
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

        # learned pooling queries: (1, K, d_model) where K = n_pool_tokens
        K = n_pool_tokens
        self.pool_queries = nn.Parameter(torch.randn(1, K, d_model) * 0.02)
        self._scale = math.sqrt(d_model)

        # embed_head and decoder operate on K*d_model; for K=1 this is identical to before
        self.embed_head = nn.Sequential(
            nn.Linear(K * d_model, K * d_model),
            nn.LayerNorm(K * d_model),
            nn.GELU(),
        )
        # For K>1, avoid immediately compressing K*d_model back to a small d_ff.
        # This makes multi-token pooling a true wider-bottleneck test.
        decoder_hidden = max(d_ff, K * d_model)
        self.decoder_hidden = decoder_hidden

        self.decoder = nn.Sequential(
            nn.Linear(K * d_model, decoder_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, n_blocks * d_in),
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
        embedding : (batch, K*d_model)  , where K = model.n_pool_tokens
        pool_attn : (batch, B)          , mean over K tokens (backward-compat)
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
        # pool_queries: (1, K, d_model) → expand to (batch, K, d_model)
        q = self.pool_queries.expand(batch_size, -1, -1)           # (batch, K, d_model)
        scores = torch.bmm(q, h.transpose(1, 2)) / self._scale     # (batch, K, B)
        pool_attn_full = F.softmax(scores, dim=-1)                  # (batch, K, B)
        pooled = torch.bmm(pool_attn_full, h)                       # (batch, K, d_model)
        pooled_flat = pooled.reshape(batch_size, self.emb_dim)      # (batch, K*d_model)

        embedding = self.embed_head(pooled_flat)                    # (batch, K*d_model)

        # Return mean over K for backward compatibility: (batch, B)
        pool_attn = pool_attn_full.mean(dim=1)                      # (batch, B)

        if return_self_attn:
            return embedding, pool_attn, h, self_attn_maps
        return embedding, pool_attn, h

    def decode(self, z):
        return self.decoder(z).view(-1, self.n_blocks, self.d_in)

    def get_initial_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected block tokens before self-attention: (batch, B, d_model)."""
        return self.input_proj(x) + self.pos_emb

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
def _masked_mse(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE restricted to real latent dimensions.

    mask : (B, max_d) float — 1 for real dims, 0 for padding.
    Broadcast over the batch dimension; divide by the number of real elements.
    """
    sq = (recon - target) ** 2            # (batch, B, max_d)
    return (sq * mask).sum() / mask.sum() / recon.size(0)


def train_attention_model(model, tr_t, va_t, cfg, device, log_csv, latent_mask_t=None):
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

    mk = latent_mask_t.to(device) if latent_mask_t is not None else None

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
            loss = _masked_mse(recon, xb, mk) if mk is not None else F.mse_loss(recon, xb)
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
                loss = _masked_mse(recon, xb, mk) if mk is not None else F.mse_loss(recon, xb)
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


def per_block_mse(recon, truth, block_dims=None):
    """(N, B, d) -> (B,) mean SE per block.

    When block_dims is given, MSE is computed only over the first d_i columns
    for block i, ignoring zero-padding in the stacked array.
    """
    if block_dims is not None:
        result = np.empty(recon.shape[1], dtype=np.float64)
        for i, d_i in enumerate(block_dims):
            result[i] = np.mean((recon[:, i, :d_i] - truth[:, i, :d_i]) ** 2)
        return result
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
    return umap.UMAP(
        n_neighbors=cc["umap_n_neighbors"],
        min_dist=cc["umap_min_dist"],
        random_state=cc.get("umap_seed", 42),
    ).fit_transform(emb)


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
    ax.boxplot(pool_attn, vert=False, tick_labels=bnames)
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
# 9. PHASE 2 DIAGNOSTICS AND BASELINES
# ============================================================

@torch.no_grad()
def _extract_initial_tokens(model, data_np, batch_size=256):
    """Return projected block tokens before self-attention: (N, B, d_model)."""
    model.eval()
    dl = DataLoader(
        TensorDataset(torch.tensor(data_np, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    out = []
    for (xb,) in dl:
        out.append(model.get_initial_tokens(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def _extract_pool_attn_by_token(model, data_np, batch_size=256):
    """Return per-token pooling attention (N, K, B). Only meaningful when K > 1.

    For K=1 this is (N, 1, B), which equals pool_attn[:, np.newaxis, :].
    """
    model.eval()
    dl = DataLoader(
        TensorDataset(torch.tensor(data_np, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    out = []
    for (xb,) in dl:
        h = model.input_proj(xb) + model.pos_emb
        for layer in model.transformer_layers:
            h, _ = layer(h, return_attn=False)
        h = model.post_norm(h)
        q = model.pool_queries.expand(xb.size(0), -1, -1)
        scores = torch.bmm(q, h.transpose(1, 2)) / model._scale
        pool_attn_full = F.softmax(scores, dim=-1)              # (batch, K, B)
        out.append(pool_attn_full.cpu().numpy())
    return np.concatenate(out, axis=0)                          # (N, K, B)


def _pool_attn_entropy(pool_attn: np.ndarray) -> float:
    """Mean Shannon entropy of per-subject pooling attention weights (nats)."""
    eps = 1e-10
    return float(-(pool_attn * np.log(pool_attn + eps)).sum(axis=-1).mean())


def _pool_attn_topk_mass(pool_attn: np.ndarray, k: int) -> float:
    """Mean total attention mass in the top-k blocks per subject."""
    k = max(1, min(k, pool_attn.shape[-1]))
    desc = np.sort(pool_attn, axis=-1)[:, ::-1]
    return float(desc[:, :k].sum(axis=-1).mean())

def _pool_attn_token_entropy(pool_attn_by_token: np.ndarray) -> float:
    """Mean entropy across subjects and pooling tokens. Shape: (N, K, B)."""
    eps = 1e-10
    ent = -(pool_attn_by_token * np.log(pool_attn_by_token + eps)).sum(axis=-1)
    return float(ent.mean())


def _pool_attn_token_pairwise_corr(pool_attn_by_token: np.ndarray) -> float:
    """Mean pairwise correlation among pooling-token attention profiles.

    Uses token-level mean attention profiles averaged across subjects.
    Shape input: (N, K, B).
    High value near 1 means tokens are redundant.
    Lower value means tokens specialize differently.
    """
    K = pool_attn_by_token.shape[1]
    if K < 2:
        return float("nan")

    token_profiles = pool_attn_by_token.mean(axis=0)  # (K, B)
    cors = []
    for i in range(K):
        for j in range(i + 1, K):
            a = token_profiles[i]
            b = token_profiles[j]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            cors.append(float(np.corrcoef(a, b)[0, 1]))

    return float(np.mean(cors)) if cors else float("nan")


def _pool_attn_token_diversity(pool_attn_by_token: np.ndarray) -> float:
    """Simple diversity score = 1 - mean pairwise token correlation."""
    c = _pool_attn_token_pairwise_corr(pool_attn_by_token)
    return float(1.0 - c) if np.isfinite(c) else float("nan")

def compute_contextualization_change(
    initial_np: np.ndarray,
    contextual_np: np.ndarray,
    block_names,
    block_meta: pd.DataFrame = None,
) -> tuple:
    """Compare initial projected tokens to post-Transformer contextual tokens.

    These diagnostics identify which blocks are most modified by cross-block
    context — not which blocks are causal drivers.

    Returns (per_block_df, per_subject_df).
    """
    delta = contextual_np - initial_np                           # (N, B, d_model)
    l2    = np.linalg.norm(delta, axis=-1)                      # (N, B)

    eps    = 1e-8
    init_n = initial_np    / (np.linalg.norm(initial_np,    axis=-1, keepdims=True) + eps)
    ctx_n  = contextual_np / (np.linalg.norm(contextual_np, axis=-1, keepdims=True) + eps)
    cos_dist = 1.0 - (init_n * ctx_n).sum(axis=-1)             # (N, B)

    mean_l2 = l2.mean(axis=0)
    std_l2  = l2.std(axis=0)
    mean_cd = cos_dist.mean(axis=0)
    std_cd  = cos_dist.std(axis=0)
    ranks   = (-mean_l2).argsort().argsort() + 1               # 1 = most changed

    row = {"block_id": block_names}
    if block_meta is not None:
        for col in ("n_snps", "latent_dim"):
            if col in block_meta.columns:
                row[col] = block_meta[col].values
    row.update({
        "mean_context_delta_l2":     np.round(mean_l2, 6),
        "std_context_delta_l2":      np.round(std_l2,  6),
        "mean_context_delta_cosine": np.round(mean_cd, 6),
        "std_context_delta_cosine":  np.round(std_cd,  6),
        "context_change_rank":       ranks,
    })
    per_block_df = pd.DataFrame(row)

    per_subj_df = pd.DataFrame({
        "mean_context_delta_l2_per_subject":     np.round(l2.mean(axis=1),       6),
        "mean_context_delta_cosine_per_subject": np.round(cos_dist.mean(axis=1), 6),
    })
    return per_block_df, per_subj_df


def _save_attention_correlation_summary(df: pd.DataFrame, diag_dir: Path):
    """Pearson and Spearman correlations between key block-level columns."""
    try:
        from scipy.stats import pearsonr, spearmanr
        _has_sp = True
    except ImportError:
        _has_sp = False

    pairs = [
        ("mean_pool_attn",  "n_snps"),
        ("mean_pool_attn",  "n_active_latents"),
        ("mean_pool_attn",  "frac_dims_collapsed"),
        ("mean_pool_attn",  "ld_corr_va"),
        ("mean_pool_attn",  "phase2_recon_mse"),
        ("mean_pool_attn",  "mean_context_delta_l2"),
        ("phase2_recon_mse", "mean_context_delta_l2"),
    ]
    rows = []
    for ca, cb in pairs:
        base = {
            "col_a": ca, "col_b": cb,
            "pearson_r": float("nan"), "pearson_p": float("nan"),
            "spearman_r": float("nan"), "spearman_p": float("nan"),
            "n": 0,
        }
        if ca not in df.columns or cb not in df.columns:
            rows.append(base)
            continue
        mask = df[ca].notna() & df[cb].notna()
        n = int(mask.sum())
        base["n"] = n
        if n < 3 or not _has_sp:
            rows.append(base)
            continue
        a = df.loc[mask, ca].values.astype(float)
        b = df.loc[mask, cb].values.astype(float)
        pr, pp = pearsonr(a, b)
        sr, sp = spearmanr(a, b)
        rows.append({**base,
                     "pearson_r":  round(float(pr), 4),
                     "pearson_p":  round(float(pp), 4),
                     "spearman_r": round(float(sr), 4),
                     "spearman_p": round(float(sp), 4),
                     "n": n})
    pd.DataFrame(rows).to_csv(
        diag_dir / "phase2_attention_diagnostic_summary.csv", index=False
    )


def run_phase2_block_diagnostics(
    block_names,
    pool_attn: np.ndarray,
    blk_mse: np.ndarray,
    ctx_per_block_df,
    p1_dir: str,
    lt: str,
    lt_dir: Path,
) -> pd.DataFrame:
    """Join Phase 2 block-level stats with Phase 1 vae_summary.csv (if available).

    Diagnostics identify blocks the aggregator relies on or modifies most —
    not causal driver blocks.
    """
    diag_dir = lt_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    mean_pool = pool_attn.mean(axis=0)
    std_pool  = pool_attn.std(axis=0)
    pool_rank = (-mean_pool).argsort().argsort() + 1

    p2_df = pd.DataFrame({
        "block_id":         block_names,
        "mean_pool_attn":   np.round(mean_pool, 6),
        "std_pool_attn":    np.round(std_pool,  6),
        "pool_attn_rank":   pool_rank,
        "phase2_recon_mse": np.round(blk_mse,   6),
    })

    if ctx_per_block_df is not None:
        ctx_cols = [c for c in [
            "block_id", "mean_context_delta_l2", "std_context_delta_l2",
            "mean_context_delta_cosine", "std_context_delta_cosine",
            "context_change_rank",
        ] if c in ctx_per_block_df.columns]
        p2_df = p2_df.merge(ctx_per_block_df[ctx_cols], on="block_id", how="left")

    # Phase 1 join (graceful: missing file or columns are silently skipped)
    p1_summary = Path(p1_dir) / lt / "vae_summary.csv"
    p1_want = [
        "block_id", "n_snps", "latent_dim", "n_active_latents",
        "frac_dims_collapsed", "latent_underused", "conc_va",
        "bal_acc_va", "ld_corr_va", "mean_r2_va", "r2_va",
    ]
    if p1_summary.exists():
        try:
            p1_df  = pd.read_csv(p1_summary)
            avail  = ["block_id"] + [c for c in p1_want[1:] if c in p1_df.columns]
            p2_df  = p2_df.merge(p1_df[avail], on="block_id", how="left")
        except Exception:
            pass

    p2_df.to_csv(diag_dir / "phase2_block_diagnostics.csv", index=False)
    _save_attention_correlation_summary(p2_df, diag_dir)
    return p2_df

def run_pca_baseline_sweep(
    emb_block, tr_ix, va_ix, te_ix, block_dims, latent_mask,
    n_components_list, out_dir,
):
    """Run PCA baseline at multiple k values; return DataFrame and best (smallest k)."""
    rows = []
    for k in n_components_list:
        try:
            n_used, va_mse, te_mse = run_pca_baseline(
                emb_block, tr_ix, va_ix, te_ix, block_dims, latent_mask,
                k, out_dir / f"pca_k{k}",
            )
            rows.append({
                "n_components_req": k,
                "n_components_used": n_used,
                "val_recon_mse": va_mse,
                "test_recon_mse": te_mse,
            })
        except Exception as e:
            warnings.warn(f"PCA sweep k={k} failed: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pca_baseline_sweep.csv", index=False)
    return df

def run_pca_baseline(
    emb_block: np.ndarray,
    tr_ix: np.ndarray,
    va_ix: np.ndarray,
    te_ix: np.ndarray,
    block_dims,
    latent_mask,
    n_components_req: int,
    out_dir: Path,
) -> tuple:
    """Concat + PCA subject-embedding baseline.  Fits PCA on training subjects only.

    n_components_req : target number of components (from config or d_model default).
                       Always capped at min(n_train - 1, flat_dim, n_components_req).
    Returns (n_components, val_recon_mse, test_recon_mse).
    This is a Phase-2-level PCA over flattened Phase-1 block embeddings —
    distinct from any PCA loss function used in Phase 1.
    """
    if not HAS_SKLEARN:
        return float("nan"), float("nan"), float("nan")
    from sklearn.decomposition import PCA

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N, B, max_d = emb_block.shape

    # Flatten: concatenate only real latent dims per block
    if block_dims is not None:
        flat = np.concatenate(
            [emb_block[:, i, :int(block_dims[i])] for i in range(B)], axis=1
        )
    else:
        flat = emb_block.reshape(N, -1)
    flat_dim = flat.shape[1]

    n_train = len(tr_ix)
    n_comp  = max(1, min(n_train - 1, flat_dim, int(n_components_req)))

    pca = PCA(n_components=n_comp)
    pca.fit(flat[tr_ix])              # fit on TRAIN ONLY

    pca_all   = pca.transform(flat)              # (N, n_comp)
    recon_all = pca.inverse_transform(pca_all)   # (N, flat_dim)

    def _mse(idx):
        if len(idx) == 0:
            return float("nan")
        return float(np.mean((recon_all[idx] - flat[idx]) ** 2))

    tr_mse = _mse(tr_ix)
    va_mse = _mse(va_ix)
    te_mse = _mse(te_ix)

    np.save(out_dir / "pca_subject_embeddings.npy", pca_all.astype(np.float32))
    pd.DataFrame(
        pca_all, columns=[f"pc_{i}" for i in range(n_comp)]
    ).to_csv(out_dir / "pca_subject_embeddings.csv", index=False)

    if recon_all.nbytes < 50 * 1024 * 1024:
        np.save(
            out_dir / "pca_reconstructions_flat.npy", recon_all.astype(np.float32)
        )

    pd.DataFrame({
        "requested_n_components":   [int(n_components_req)],
        "n_components":             [n_comp],
        "flat_dim":                 [flat_dim],
        "pca_train_recon_loss":     [round(tr_mse, 6)],
        "pca_val_recon_loss":       [round(va_mse, 6)],
        "pca_test_recon_loss":      [round(te_mse, 6) if np.isfinite(te_mse) else float("nan")],
        "explained_variance_ratio": [round(float(pca.explained_variance_ratio_.sum()), 4)],
    }).to_csv(out_dir / "pca_baseline_summary.csv", index=False)

    return n_comp, va_mse, te_mse


@torch.no_grad()
def run_mean_pool_baseline(
    model,
    emb_block: np.ndarray,
    tr_ix: np.ndarray,
    va_ix: np.ndarray,
    te_ix: np.ndarray,
    latent_mask,
    latent_mask_t,
    out_dir: Path,
) -> tuple:
    """Two mean-pool baselines sharing the same projected tokens.

    raw      : mean_token → decoder
               (input_proj + pos_emb, mean across blocks, decode — no attn, no pooling)
    embedhead: mean_token → embed_head → decoder
               (adds the trained embed_head MLP before decoding)

    Both use the trained model's weights; neither trains new parameters.
    Reconstruction loss is directly comparable to the Transformer model's MSE.

    Returns (raw_val_mse, raw_test_mse, embedhead_val_mse, embedhead_test_mse).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    dl = DataLoader(
        TensorDataset(torch.tensor(emb_block, dtype=torch.float32)),
        batch_size=256,
        shuffle=False,
    )

    K = model.n_pool_tokens
    mean_embs, recons_raw, recons_eh = [], [], []
    for (xb,) in dl:
        tokens  = model.get_initial_tokens(xb)   # (batch, B, d_model)
        z_raw   = tokens.mean(dim=1)              # (batch, d_model)
        # Expand to K*d_model so decoder input size matches (for K=1 this is a no-op)
        z_raw_rep = z_raw.unsqueeze(1).expand(-1, K, -1).reshape(xb.size(0), model.emb_dim)
        z_eh    = model.embed_head(z_raw_rep)     # (batch, K*d_model)
        rec_raw = model.decode(z_raw_rep)         # (batch, B, d_in)
        rec_eh  = model.decode(z_eh)              # (batch, B, d_in)
        mean_embs.append(z_raw.cpu().numpy())     # save d_model-dim mean token
        recons_raw.append(rec_raw.cpu().numpy())
        recons_eh.append(rec_eh.cpu().numpy())

    mean_embs_np  = np.concatenate(mean_embs,   axis=0)
    recons_raw_np = np.concatenate(recons_raw,  axis=0)
    recons_eh_np  = np.concatenate(recons_eh,   axis=0)

    _lm = latent_mask[np.newaxis] if latent_mask is not None else None
    _ld = float(latent_mask.sum()) if latent_mask is not None else None

    def _mse(recons_arr, idx):
        if len(idx) == 0:
            return float("nan")
        r, t = recons_arr[idx], emb_block[idx]
        if _lm is not None:
            return float(np.sum((r - t) ** 2 * _lm) / (_ld * len(idx)))
        return float(np.mean((r - t) ** 2))

    raw_va = _mse(recons_raw_np, va_ix)
    raw_te = _mse(recons_raw_np, te_ix)
    eh_va  = _mse(recons_eh_np,  va_ix)
    eh_te  = _mse(recons_eh_np,  te_ix)

    np.save(out_dir / "mean_pool_subject_embeddings.npy", mean_embs_np.astype(np.float32))
    pd.DataFrame(
        mean_embs_np, columns=[f"d_{i}" for i in range(mean_embs_np.shape[1])]
    ).to_csv(out_dir / "mean_pool_subject_embeddings.csv", index=False)

    pd.DataFrame({
        "raw_mean_pool_val_recon_loss":       [round(raw_va, 6)],
        "raw_mean_pool_test_recon_loss":      [round(raw_te, 6) if np.isfinite(raw_te) else float("nan")],
        "embedhead_mean_pool_val_recon_loss":  [round(eh_va,  6)],
        "embedhead_mean_pool_test_recon_loss": [round(eh_te,  6) if np.isfinite(eh_te)  else float("nan")],
        "note": ["raw=mean_token+decoder; embedhead=mean_token+embed_head+decoder"],
    }).to_csv(out_dir / "mean_pool_baseline_summary.csv", index=False)

    return raw_va, raw_te, eh_va, eh_te


def _save_diagnostic_plots(block_diag_df, lt: str, out_dir: Path):
    """Scatter and bar plots for Phase 2 diagnostics.  Silent no-op if matplotlib absent."""
    if not HAS_PLT or block_diag_df is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    def _scatter(xcol, ycol, fname, xlabel, ylabel):
        if xcol not in block_diag_df.columns or ycol not in block_diag_df.columns:
            return
        mask = block_diag_df[xcol].notna() & block_diag_df[ycol].notna()
        x = block_diag_df.loc[mask, xcol].values.astype(float)
        y = block_diag_df.loc[mask, ycol].values.astype(float)
        if len(x) < 2:
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, s=30, alpha=0.7, color="steelblue")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{lt}")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    try:
        _scatter(
            "mean_pool_attn", "mean_context_delta_l2",
            "attention_vs_context_delta.png",
            "mean pool attention", "mean context delta L2",
        )
        _scatter(
            "n_active_latents", "mean_pool_attn",
            "attention_vs_phase1_active_latents.png",
            "n_active_latents (Phase 1)", "mean pool attention",
        )
        _scatter(
            "n_snps", "mean_pool_attn",
            "attention_vs_n_snps.png",
            "n_snps", "mean pool attention",
        )

        if "mean_context_delta_l2" in block_diag_df.columns:
            top25 = (
                block_diag_df[["block_id", "mean_context_delta_l2"]]
                .dropna()
                .nlargest(25, "mean_context_delta_l2")
            )
            if len(top25) > 0:
                fig, ax = plt.subplots(figsize=(10, max(4, len(top25) * 0.35)))
                y = np.arange(len(top25))
                ax.barh(y, top25["mean_context_delta_l2"].values, color="coral", alpha=0.85)
                ax.set_yticks(y)
                ax.set_yticklabels(top25["block_id"].values, fontsize=7)
                ax.set_xlabel("Mean context delta L2")
                ax.set_title(f"{lt} — top {len(top25)} blocks by contextualization change")
                ax.invert_yaxis()
                plt.tight_layout()
                plt.savefig(out_dir / "context_delta_by_block.png", dpi=150)
                plt.close()
    except Exception:
        pass


# ============================================================
# 10. MAIN PHASE-2 PIPELINE
# ============================================================
def run_phase2(cfg, *, config_path=None):
    t0_run = time.time()
    ac = cfg["attention"]
    cc = cfg["clustering"]

    set_seed(ac["seed"])
    dev = get_device(ac.get("device", "auto"))

    # Set deterministic behavior
    if dev.type == "cpu":
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    elif dev.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "config_phase2.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("\n══════ Step 8 · Loading Phase 1 frozen embeddings ══════")
    subjects, tr_ix, va_ix, te_ix, block_meta, embeddings, latent_dims_per_loss = load_phase1(
        cfg["phase1_dir"], cfg["loss_functions"]
    )

    N, B, d_in = list(embeddings.values())[0].shape
    block_names = block_meta["block_id"].values

    all_results = {}
    summary_rows = []

    for lt in cfg["loss_functions"]:
        print(f"\n{'═' * 55}")
        print(f"  Loss function: {lt}")
        print(f"{'═' * 55}")

        lt_dir = out / lt
        for sub in ("logs", "models", "embeddings", "clustering", "plots", "attention_maps",
                    "baselines", "diagnostics"):
            (lt_dir / sub).mkdir(parents=True, exist_ok=True)

        emb_block = embeddings[lt]  # (N, B, d_in)

        # Build per-loss block_dims and latent_mask.
        # Prefer all_blocks_latent_dims.npy (actual post-clamping dims) when available;
        # fall back to block_order.csv["latent_dim"] (requested dims) otherwise.
        _lt_dims = latent_dims_per_loss.get(lt)
        if _lt_dims is not None:
            block_dims = [int(d) for d in _lt_dims]
        elif "latent_dim" in block_meta.columns:
            block_dims = [int(d) for d in block_meta["latent_dim"]]
        else:
            block_dims = None

        if block_dims is not None:
            dim_counts = dict(pd.Series(block_dims).value_counts().sort_index())
            print(f"  N={N}  B={B}  max_d={d_in}  per-block dims={dim_counts}  [BlockProjector]")
            latent_mask = np.zeros((B, d_in), dtype=np.float32)
            for i, d_i in enumerate(block_dims):
                d_i = int(d_i)
                if d_i <= 0 or d_i > d_in:
                    raise ValueError(f"Invalid latent_dim for block {i}: {d_i}, max_d={d_in}")
                latent_mask[i, :d_i] = 1.0
            latent_mask_t = torch.tensor(latent_mask)
        else:
            latent_mask = None
            latent_mask_t = None
            print(f"  N={N}  B={B}  d_in={d_in}  [shared input_proj — legacy]")

        n_test_subj = len(te_ix)

        # ------------------ pre-training baselines ------------------
        # Zero baseline: predict zeros.  Train-mean baseline: predict training mean.
        # Both use the same latent mask as the model loss so units are comparable.
        # _masked_mse divides by mask.sum() (real dims) then by batch — same here.
        _tr_mean = emb_block[tr_ix].mean(axis=0, keepdims=True)  # (1, B, d_in)
        _va_zero_sq = emb_block[va_ix] ** 2
        _va_mean_sq = (emb_block[va_ix] - _tr_mean) ** 2

        if latent_mask is not None:
            _lm = latent_mask[np.newaxis]          # (1, B, d_in)
            _ld = float(latent_mask.sum())
            val_zero_baseline       = float(np.sum(_va_zero_sq * _lm) / (_ld * len(va_ix)))
            val_train_mean_baseline = float(np.sum(_va_mean_sq * _lm) / (_ld * len(va_ix)))
            if n_test_subj > 0:
                _te_zero_sq = emb_block[te_ix] ** 2
                _te_mean_sq = (emb_block[te_ix] - _tr_mean) ** 2
                test_zero_baseline       = float(np.sum(_te_zero_sq * _lm) / (_ld * n_test_subj))
                test_train_mean_baseline = float(np.sum(_te_mean_sq * _lm) / (_ld * n_test_subj))
            else:
                test_zero_baseline = test_train_mean_baseline = float("nan")
        else:
            val_zero_baseline       = float(np.mean(_va_zero_sq))
            val_train_mean_baseline = float(np.mean(_va_mean_sq))
            if n_test_subj > 0:
                test_zero_baseline       = float(np.mean(emb_block[te_ix] ** 2))
                test_train_mean_baseline = float(np.mean((emb_block[te_ix] - _tr_mean) ** 2))
            else:
                test_zero_baseline = test_train_mean_baseline = float("nan")

        print(f"  val  zero-baseline loss:       {val_zero_baseline:.6f}")
        print(f"  val  train-mean-baseline loss: {val_train_mean_baseline:.6f}")
        if n_test_subj > 0:
            print(f"  test zero-baseline loss:       {test_zero_baseline:.6f}")
            print(f"  test train-mean-baseline loss: {test_train_mean_baseline:.6f}")

        # ------------------ Step 9: train ------------------
        print("\n  Step 9 · Training attention aggregator ...")
        tr_t = torch.tensor(emb_block[tr_ix], dtype=torch.float32)
        va_t = torch.tensor(emb_block[va_ix], dtype=torch.float32)

        K = int(ac.get("n_pool_tokens", 1))
        if K < 1:
            raise ValueError("attention.n_pool_tokens must be >= 1")

        model = AttentionAggregator(
            n_blocks=B,
            d_in=d_in,
            d_model=ac["d_model"],
            n_heads=ac["n_heads"],
            n_layers=ac["n_layers"],
            d_ff=ac["d_ff"],
            dropout=ac["dropout"],
            block_dims=block_dims,
            n_pool_tokens=K,
        )

        npar = sum(p.numel() for p in model.parameters())
        print(
            f"    architecture: {B}x{d_in} -> d_model={ac['d_model']}  "
            f"heads={ac['n_heads']}  layers={ac['n_layers']}  "
            f"n_pool_tokens={K}  emb_dim={K * ac['d_model']}  "
            f"decoder_hidden={model.decoder_hidden}  params={npar:,}"
        )

        t0 = time.time()
        log, best_epoch, best_val_loss = train_attention_model(
            model, tr_t, va_t, cfg, dev,
            lt_dir / "logs" / "attention_training.csv",
            latent_mask_t=latent_mask_t,
        )
        dt = time.time() - t0
        print(f"    done in {dt:.1f}s  ({len(log)} epochs, best at {best_epoch})")

        torch.save(model.state_dict(), lt_dir / "models" / "attention_aggregator.pt")

        # ------------------ held-out test reconstruction loss ------------------
        model.eval()
        if n_test_subj > 0:
            te_t = torch.tensor(emb_block[te_ix], dtype=torch.float32)
            te_dl = DataLoader(TensorDataset(te_t), batch_size=256, shuffle=False)
            te_loss_acc = 0.0
            with torch.no_grad():
                for (xb,) in te_dl:
                    recon_te, _, _ = model(xb, return_self_attn=False)
                    loss_te = (
                        _masked_mse(recon_te, xb, latent_mask_t)
                        if latent_mask_t is not None
                        else F.mse_loss(recon_te, xb)
                    )
                    te_loss_acc += loss_te.item() * xb.size(0)
            te_recon_loss = te_loss_acc / n_test_subj
        else:
            te_recon_loss = float("nan")
        print(f"    test  reconstruction loss: {te_recon_loss:.6f}")

        # ------------------ Step 10: extract ------------------
        print("\n  Step 10 · Extracting embeddings, pooling attention, and self-attention ...")
        final_emb, pool_attn, recon, block_repr, self_attn_mean, self_attn_full = extract_all(
            model,
            emb_block,
            batch_size=256,
            return_self_attn=ac.get("extract_self_attn", True),
            save_full_self_attn=ac.get("save_full_self_attn", False),
        )

        if latent_mask is not None:
            sq = (recon - emb_block) ** 2  # (N, B, max_d)
            global_mse = float(np.sum(sq * latent_mask[np.newaxis]) / (latent_mask.sum() * N))
        else:
            global_mse = float(np.mean((recon - emb_block) ** 2))
        blk_mse = per_block_mse(recon, emb_block, block_dims=block_dims)

        print(f"    individual embedding : {final_emb.shape}")
        print(f"    pooling attention    : {pool_attn.shape}")
        print(f"    reconstruction MSE   : {global_mse:.6f}")
        print(f"    worst block MSE      : {block_names[blk_mse.argmax()]} ({blk_mse.max():.6f})")

        # ---- save arrays ----
        np.save(lt_dir / "embeddings" / "individual_embeddings.npy", final_emb)
        np.save(lt_dir / "embeddings" / "pooling_attention_weights.npy", pool_attn)
        np.save(lt_dir / "embeddings" / "reconstructions.npy", recon)
        np.save(lt_dir / "embeddings" / "block_contextual_repr.npy", block_repr)

        # For K>1 also save per-token pooling attention (N, K, B) alongside the mean (N, B)
        pool_attn_by_token = None
        if K > 1:
            pool_attn_by_token = _extract_pool_attn_by_token(model, emb_block)
            np.save(
                lt_dir / "embeddings" / "pooling_attention_weights_by_token.npy",
                pool_attn_by_token,
            )

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

        # ── Phase 2 diagnostics and baselines ────────────────────────────────
        dc = cfg.get("diagnostics", DEFAULT_CFG.get("diagnostics", {}))
        bc = cfg.get("baselines",   DEFAULT_CFG.get("baselines",   {}))

        # Attention entropy / top-k mass (always computed — cheap)
        _entropy   = _pool_attn_entropy(pool_attn)
        _top1_mass = _pool_attn_topk_mass(pool_attn, 1)
        _top5_mass = _pool_attn_topk_mass(pool_attn, min(5, B))
        _token_entropy = float("nan")
        _token_pairwise_corr = float("nan")
        _token_diversity = float("nan")

        if pool_attn_by_token is not None:
            _token_entropy = _pool_attn_token_entropy(pool_attn_by_token)
            _token_pairwise_corr = _pool_attn_token_pairwise_corr(pool_attn_by_token)
            _token_diversity = _pool_attn_token_diversity(pool_attn_by_token)
        # Self-attention mass statistics (layer 0, averaged over heads and subjects)
        _sa_diag_entry_l0 = _sa_offdiag_entry_l0 = float("nan")
        _sa_diag_mass_l0  = _sa_offdiag_mass_l0  = float("nan")
        if self_attn_mean is not None and len(self_attn_mean) > 0:
            _sa0     = self_attn_mean[0].mean(axis=0)   # (B, B), averaged over heads
            _eye     = np.eye(B, dtype=bool)
            _sa0_tot = float(_sa0.sum())
            _sa_diag_entry_l0    = float(_sa0[_eye].mean())
            _sa_offdiag_entry_l0 = float(_sa0[~_eye].mean()) if B > 1 else float("nan")
            if _sa0_tot > 0:
                _sa_diag_mass_l0    = float(_sa0[_eye].sum()  / _sa0_tot)
                _sa_offdiag_mass_l0 = float(_sa0[~_eye].sum() / _sa0_tot) if B > 1 else float("nan")

        # Part A — initial (pre-attention) block tokens
        initial_repr = None
        if dc.get("enabled", True) and dc.get("save_initial_block_repr", True):
            initial_repr = _extract_initial_tokens(model, emb_block)
            np.save(lt_dir / "embeddings" / "block_initial_repr.npy", initial_repr)

        # Part B — contextualization-change diagnostics
        ctx_per_block_df = ctx_per_subj_df = None
        if initial_repr is not None and dc.get("compute_contextualization_change", True):
            ctx_per_block_df, ctx_per_subj_df = compute_contextualization_change(
                initial_repr, block_repr, block_names, block_meta
            )
            ctx_per_block_df.to_csv(
                lt_dir / "embeddings" / "per_block_contextualization_change.csv",
                index=False,
            )
            ctx_per_subj_df.to_csv(
                lt_dir / "embeddings" / "per_subject_contextualization_change.csv",
                index=False,
            )

        # Part C — Phase 2 × Phase 1 block diagnostics join
        block_diag_df = None
        if dc.get("enabled", True) and dc.get("compute_phase1_phase2_join", True):
            block_diag_df = run_phase2_block_diagnostics(
                block_names, pool_attn, blk_mse, ctx_per_block_df,
                cfg["phase1_dir"], lt, lt_dir,
            )

        # Part D — PCA subject-embedding baseline
        # replace existing single-k PCA call with sweep
        _pca_sweep_list = bc.get("pca_sweep")
        if _pca_sweep_list:
            pca_sweep_df = run_pca_baseline_sweep(
                emb_block, tr_ix, va_ix, te_ix, block_dims, latent_mask,
                _pca_sweep_list, lt_dir / "baselines",
            )
            # for backward-compat reporting, pick k closest to d_model
            _target = ac["d_model"]
            closest = pca_sweep_df.iloc[(pca_sweep_df["n_components_req"] - _target).abs().argsort()[:1]]
            pca_n_components = int(closest["n_components_used"].iloc[0])
            pca_val_recon = float(closest["val_recon_mse"].iloc[0])
            pca_te_recon = float(closest["test_recon_mse"].iloc[0])
        # pca_n_components = pca_val_recon = pca_te_recon = float("nan")
        # if bc.get("enabled", True) and bc.get("run_pca", True):
        #     try:
        #         _pca_req = bc.get("pca_n_components")
        #         _pca_target = int(_pca_req) if _pca_req is not None else ac["d_model"]
        #         pca_n_components, pca_val_recon, pca_te_recon = run_pca_baseline(
        #             emb_block, tr_ix, va_ix, te_ix, block_dims, latent_mask,
        #             _pca_target, lt_dir / "baselines",
        #         )
        #         print(
        #             f"    PCA baseline      : n_comp={pca_n_components}"
        #             f"  val={pca_val_recon:.6f}  test={pca_te_recon:.6f}"
        #         )
        #     except Exception as _e:
        #         warnings.warn(f"PCA baseline failed: {_e}")

        # Part E — mean-pool baselines (raw and embed_head variants)
        mp_raw_va = mp_raw_te = mp_eh_va = mp_eh_te = float("nan")
        if bc.get("enabled", True) and bc.get("run_mean_pool", True):
            try:
                mp_raw_va, mp_raw_te, mp_eh_va, mp_eh_te = run_mean_pool_baseline(
                    model, emb_block, tr_ix, va_ix, te_ix,
                    latent_mask, latent_mask_t, lt_dir / "baselines",
                )
                print(
                    f"    mean-pool raw      : val={mp_raw_va:.6f}  test={mp_raw_te:.6f}"
                )
                print(
                    f"    mean-pool embedhead: val={mp_eh_va:.6f}  test={mp_eh_te:.6f}"
                )
            except Exception as _e:
                warnings.warn(f"Mean-pool baseline failed: {_e}")

        # Part F — diagnostic scatter / bar plots
        _save_diagnostic_plots(block_diag_df, lt, lt_dir / "plots")

        # Context-change aggregates for summary row
        _mean_ctx_l2 = _mean_ctx_cos = _max_ctx_l2 = float("nan")
        if ctx_per_block_df is not None:
            _mean_ctx_l2  = round(float(ctx_per_block_df["mean_context_delta_l2"].mean()),  6)
            _mean_ctx_cos = round(float(ctx_per_block_df["mean_context_delta_cosine"].mean()), 6)
            _max_ctx_l2   = round(float(ctx_per_block_df["mean_context_delta_l2"].max()),   6)

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
            "test_recon_loss": round(te_recon_loss, 6) if not np.isnan(te_recon_loss) else float("nan"),
            "val_zero_baseline_loss": round(val_zero_baseline, 6),
            "test_zero_baseline_loss": round(test_zero_baseline, 6) if not np.isnan(test_zero_baseline) else float("nan"),
            "val_train_mean_baseline_loss": round(val_train_mean_baseline, 6),
            "test_train_mean_baseline_loss": round(test_train_mean_baseline, 6) if not np.isnan(test_train_mean_baseline) else float("nan"),
            "model_vs_mean_baseline_ratio": round(best_val_loss / val_train_mean_baseline, 4) if val_train_mean_baseline > 0 else float("nan"),
            "recon_mse": round(global_mse, 6),
            "best_silhouette": round(best_sil, 4),
            # ── attention diagnostics ──────────────────────────────────────────
            "mean_pool_attn_entropy":        round(_entropy,   6),
            "mean_pool_attn_top1_mass":      round(_top1_mass, 6),
            "mean_pool_attn_top5_mass":      round(_top5_mass, 6),
            "pool_token_entropy":            round(_token_entropy, 6) if np.isfinite(_token_entropy) else float("nan"),
            "pool_token_pairwise_corr":      round(_token_pairwise_corr, 6) if np.isfinite(_token_pairwise_corr) else float("nan"),
            "pool_token_diversity":          round(_token_diversity, 6) if np.isfinite(_token_diversity) else float("nan"),
            "mean_context_delta_l2":         _mean_ctx_l2,
            "mean_context_delta_cosine":     _mean_ctx_cos,
            "max_context_delta_l2":          _max_ctx_l2,
            "self_attn_diag_entry_mean_layer0":    round(_sa_diag_entry_l0,    6) if np.isfinite(_sa_diag_entry_l0)    else float("nan"),
            "self_attn_offdiag_entry_mean_layer0": round(_sa_offdiag_entry_l0, 6) if np.isfinite(_sa_offdiag_entry_l0) else float("nan"),
            "self_attn_diag_total_mass_layer0":    round(_sa_diag_mass_l0,     6) if np.isfinite(_sa_diag_mass_l0)     else float("nan"),
            "self_attn_offdiag_total_mass_layer0": round(_sa_offdiag_mass_l0,  6) if np.isfinite(_sa_offdiag_mass_l0)  else float("nan"),
            # ── PCA baseline ───────────────────────────────────────────────────
            "pca_n_components":              int(pca_n_components) if np.isfinite(float(pca_n_components)) else float("nan"),
            "pca_val_recon_loss":            round(pca_val_recon, 6) if np.isfinite(pca_val_recon) else float("nan"),
            "pca_test_recon_loss":           round(pca_te_recon,  6) if np.isfinite(pca_te_recon)  else float("nan"),
            "transformer_vs_pca_val_ratio":  round(best_val_loss / pca_val_recon, 4) if (np.isfinite(pca_val_recon) and pca_val_recon > 0) else float("nan"),
            "transformer_vs_pca_test_ratio": round(te_recon_loss / pca_te_recon, 4)  if (np.isfinite(pca_te_recon)  and pca_te_recon  > 0 and np.isfinite(te_recon_loss)) else float("nan"),
            # ── mean-pool baselines (raw and embed_head variants) ──────────────
            "raw_mean_pool_val_recon_loss":            round(mp_raw_va, 6) if np.isfinite(mp_raw_va) else float("nan"),
            "raw_mean_pool_test_recon_loss":           round(mp_raw_te, 6) if np.isfinite(mp_raw_te) else float("nan"),
            "embedhead_mean_pool_val_recon_loss":      round(mp_eh_va,  6) if np.isfinite(mp_eh_va)  else float("nan"),
            "embedhead_mean_pool_test_recon_loss":     round(mp_eh_te,  6) if np.isfinite(mp_eh_te)  else float("nan"),
            "transformer_vs_raw_mean_pool_val_ratio":  round(best_val_loss / mp_raw_va, 4) if (np.isfinite(mp_raw_va) and mp_raw_va > 0) else float("nan"),
            "transformer_vs_raw_mean_pool_test_ratio": round(te_recon_loss / mp_raw_te, 4) if (np.isfinite(mp_raw_te) and mp_raw_te > 0 and np.isfinite(te_recon_loss)) else float("nan"),
            # ── run info ──────────────────────────────────────────────────────
            "seconds": round(dt, 1),
            "seed": ac["seed"],
            "device": str(dev),
            "n_subjects": N,
            "n_train": len(tr_ix),
            "n_val": len(va_ix),
            "n_test": n_test_subj,
            "n_blocks": B,
            "d_in": d_in,
            "n_pool_tokens": K,
            "embedding_dim": K * ac["d_model"],
            "decoder_hidden": model.decoder_hidden,
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

    _write_run_metadata(
        out,
        config_path=config_path or out / "config_phase2.yaml",
        cfg=cfg,
        t0=t0_run,
        t1=time.time(),
    )

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

    for f in ["subjects.csv", "train_idx.npy", "val_idx.npy", "test_idx.npy", "block_order.csv"]:
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
    run_phase2(cfg, config_path=config_path)

    # Post-run output validation
    expected = [out_dir / "phase2_summary.csv"]
    for lt in cfg.get("loss_functions", []):
        expected.append(out_dir / lt / "clustering" / "cluster_labels.csv")
    for p in expected:
        if not p.exists():
            raise FileNotFoundError(f"Expected output missing after Phase2: {p}")

    print(f"\n[phase2] complete (took {time.time() - t0:.1f}s)")