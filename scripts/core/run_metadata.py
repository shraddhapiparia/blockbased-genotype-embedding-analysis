#!/usr/bin/env python3
"""run_metadata.py — Write rich provenance JSON for each pipeline run.

Usage (from Phase 1 / Phase 2 entrypoints):

    from scripts.core.run_metadata import write_run_metadata
    out_path = write_run_metadata(
        out_dir,
        config_path=resolved_config,
        cfg=cfg,
        t0=t_start,
        t1=time.time(),
    )

Each run_metadata.json captures: git commit + dirty flag, config path and
SHA256, random seed, Python version, key library versions, full pip freeze
output, sys.argv command, UTC timestamps with Z suffix, elapsed seconds,
and SHA256 hashes of small config-defined input files.
"""

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def git_info() -> dict:
    """Return {commit, dirty}; graceful when git is absent or not a git repo.

    Uses `git status --porcelain` so that untracked files also set dirty=True,
    unlike `git diff --quiet` which only detects tracked modifications.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        return {"commit": sha, "dirty": bool(status.strip())}
    except Exception:
        return {"commit": "unavailable", "dirty": None}


def pip_freeze() -> list:
    """Return sorted pip freeze output as a list of requirement strings."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
        ).decode()
        return sorted(line for line in out.strip().splitlines() if line)
    except Exception:
        return ["<unavailable>"]


def file_sha256(path) -> str | None:
    """Return hex SHA256 of a file; None if the path is absent or is a directory."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(val: str) -> Path:
    """Resolve a path string against cwd when relative."""
    p = Path(val)
    return p if p.is_absolute() else Path.cwd() / p


def lib_versions() -> dict:
    """Return installed version strings for key scientific libraries."""
    import importlib
    result = {}
    for name in ("torch", "numpy", "pandas", "sklearn", "scipy", "statsmodels", "yaml"):
        try:
            mod = importlib.import_module(name)
            result[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            result[name] = "not installed"
    return result


def _utc_iso(epoch: float) -> str:
    """Return UTC ISO-8601 string with Z suffix for a Unix timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def write_run_metadata(
    out_dir,
    *,
    config_path,
    cfg: dict,
    command=None,
    t0: float = None,
    t1: float = None,
) -> Path:
    """Write run_metadata.json into out_dir and return the path.

    Parameters
    ----------
    out_dir     : output directory; created if absent
    config_path : path to the YAML config used for this run
    cfg         : parsed config dict (seed and input paths extracted from here)
    command     : recorded invocation; defaults to sys.argv
    t0, t1      : epoch timestamps (float) bracketing the run; both default to now
    """
    now = time.time()
    t0 = t0 if t0 is not None else now
    t1 = t1 if t1 is not None else now

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Seed lives under cfg["vae"]["seed"] (Phase 1) or cfg["attention"]["seed"] (Phase 2)
    seed = None
    for key in ("vae", "attention", "clustering"):
        sub = cfg.get(key)
        if isinstance(sub, dict) and "seed" in sub:
            seed = sub["seed"]
            break

    # Hash small config-defined input files; skip large directories.
    # Relative paths are resolved against cwd (the repo root when scripts are
    # invoked as: python scripts/core/VAE_phase1.py --config ...).
    input_hashes: dict = {}
    data_cfg = cfg.get("data") or {}
    for field in ("block_def", "eigenvec"):
        val = data_cfg.get(field)
        if not val:
            continue
        resolved = _resolve(val)
        if resolved.is_file():
            input_hashes[field] = file_sha256(resolved)
        elif resolved.exists():
            input_hashes[field] = f"<directory: {val}>"
        else:
            input_hashes[field] = f"<not found: {val}>"

    # config_path may also be relative; resolve before hashing
    config_resolved = _resolve(str(config_path))

    meta = {
        **git_info(),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_resolved),
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "libs": lib_versions(),
        "pip_freeze": pip_freeze(),
        "command": command if command is not None else sys.argv[:],
        "start_time": _utc_iso(t0),
        "end_time": _utc_iso(t1),
        "elapsed_seconds": round(t1 - t0, 2),
        "input_hashes": input_hashes,
    }

    out_path = out / "run_metadata.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[metadata]   written: {out_path}")
    return out_path
