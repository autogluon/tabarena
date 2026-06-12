"""TabArena's default evaluation metric per problem type — defined once.

Every consumer that needs the fallback metric for a task without an explicit
``eval_metric`` (the task wrapper, the experiment runner, metadata defaults)
resolves it here, so the mapping cannot drift between them.
"""

from __future__ import annotations

#: Default AutoGluon eval metric per problem type, used when none is specified.
DEFAULT_EVAL_METRIC_BY_PROBLEM_TYPE: dict[str, str] = {
    "binary": "roc_auc",
    "multiclass": "log_loss",
    "regression": "rmse",
}


def default_eval_metric(problem_type: str) -> str:
    """The TabArena default eval metric for ``problem_type`` (AutoGluon metric name)."""
    return DEFAULT_EVAL_METRIC_BY_PROBLEM_TYPE[problem_type]


#: Alias -> canonical TabArena metric name. AutoGluon registers several names per
#: metric (e.g. ``root_mean_squared_error`` == ``rmse``); results ingestion
#: canonicalizes so downstream comparisons join on one name.
EVAL_METRIC_ALIASES: dict[str, str] = {
    "root_mean_squared_error": "rmse",
}


def normalize_eval_metric(eval_metric: str) -> str:
    """Map a metric-name alias to its canonical TabArena name (identity otherwise)."""
    return EVAL_METRIC_ALIASES.get(eval_metric, eval_metric)
