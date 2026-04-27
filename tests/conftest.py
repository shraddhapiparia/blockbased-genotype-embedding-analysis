import sys
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

_SYNTH_SENTINEL = _ROOT / "data" / "synthetic" / "region_blocks" / "synth_block_A.raw"


@pytest.fixture(scope="session", autouse=True)
def ensure_synthetic_data():
    """Generate synthetic .raw files if they are absent (gitignored, generated on demand)."""
    if not _SYNTH_SENTINEL.exists():
        script = _ROOT / "data" / "synthetic" / "create_synthetic_data.py"
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(_ROOT))


@pytest.fixture(scope="session")
def repo_root():
    return _ROOT
