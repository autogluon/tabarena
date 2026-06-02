from __future__ import annotations

import json
import os

AUX_METRIC_ENV_VAR = "TABARENA_AUX_METRIC_MAP"


def get_aux_metric_map() -> dict[str, str] | None:
    """Return a ``problem_type -> metric_name`` mapping from the ``TABARENA_AUX_METRIC_MAP`` env var.

    The env var must be a JSON object, e.g.
    ``{"binary": "balanced_accuracy", "multiclass": "balanced_accuracy", "regression": "r2"}``.
    Returns ``None`` when the env var is unset or empty, which disables aux-metric computation.
    """
    raw = os.environ.get(AUX_METRIC_ENV_VAR)
    if not raw:
        return None
    return json.loads(raw)
