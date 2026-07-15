from __future__ import annotations

from autogluon.core.metrics import make_scorer

from ._cpp_metrics import CppMetrics

# Fast RMSE for ensemble simulation: a fused single-pass C++ kernel, ~3-4x faster
# than the numpy expression in AutoGluon's rmse scorer (which makes three passes
# and two temporaries). No support for sample weights.
# Requires compiled C++ code, refer to `_cpp_metrics/README.md` for details.
fast_rmse = make_scorer("root_mean_squared_error", CppMetrics().rmse, optimum=0, greater_is_better=False)
fast_rmse.add_alias("rmse")
