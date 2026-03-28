from __future__ import annotations

import sys
from pathlib import Path

# Make tabflow_slurm importable as a top-level package by adding the repo root to sys.path.
_REPO_ROOT = str(Path(__file__).parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
