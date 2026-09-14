"""Repository-wide pytest configuration.

The warm-up helpers may start Ray and a worker pool in a benchmark job. No test may do that, so the
kill switch ``TABARENA_DISABLE_RAY_WARMUP`` is set for the whole session before any test module is
collected (see ``tabarena.utils.ray_utils``). Tests that exercise the Ray path inject a fake Ray and
clear the variable with ``monkeypatch.delenv``.
"""

from __future__ import annotations

import os

os.environ.setdefault("TABARENA_DISABLE_RAY_WARMUP", "1")
