"""
Tests that synthetic YAML configs parse without error and that the data paths
they reference resolve to existing locations under the repo root.
"""
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_P1_CFG = _ROOT / "configs" / "config_phase1_synthetic.yaml"
_P2_CFG = _ROOT / "configs" / "config_phase2_synthetic.yaml"


def _load(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def test_phase1_synthetic_config_parses():
    cfg = _load(_P1_CFG)
    assert "data" in cfg
    assert "vae" in cfg


def test_phase2_synthetic_config_parses():
    cfg = _load(_P2_CFG)
    assert "attention" in cfg
    assert "loss_functions" in cfg


def test_phase1_raw_dir_resolves():
    """raw_dir in the Phase 1 synthetic config must point to a real directory."""
    cfg = _load(_P1_CFG)
    raw_dir = _ROOT / cfg["data"]["raw_dir"]
    assert raw_dir.exists(), f"raw_dir from config does not exist: {raw_dir}"


def test_phase1_block_def_resolves():
    """block_def (manifest) in the Phase 1 synthetic config must exist."""
    cfg = _load(_P1_CFG)
    block_def = _ROOT / cfg["data"]["block_def"]
    assert block_def.exists(), f"block_def from config does not exist: {block_def}"


def test_phase2_loss_functions_nonempty():
    """Phase 2 synthetic config must specify at least one loss function."""
    cfg = _load(_P2_CFG)
    loss_functions = cfg.get("loss_functions", [])
    assert len(loss_functions) > 0, "loss_functions must not be empty"
