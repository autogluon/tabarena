# bencheval

Benchmark evaluation for tabular machine learning: turns a table of per-task results into a
leaderboard with Elo, win-rates, average ranks, improvability, and bootstrap confidence intervals.
It is the leaderboard engine behind [TabArena](https://tabarena.ai/) and is developed in the
[TabArena repository](https://github.com/autogluon/tabarena), but it does not depend on the
`tabarena` package, so any benchmark that produces `(method, task, metric_error)` rows can use it.

> **Experimental.** The API is still moving. Pin the version you evaluate with; every `tabarena`
> release pins the matching `bencheval` release.

## Install

```bash
pip install --pre bencheval          # metrics + leaderboards
pip install --pre "bencheval[plot]"  # + matplotlib / seaborn / plotly for the plotting mixin
```

Releases on PyPI are pre-releases for now, so `--pre` (or `uv pip install --prerelease=allow`) is
needed. The git checkout is the recommended install for development; see the
[TabArena README](https://github.com/autogluon/tabarena#readme).

## Usage

```python
import pandas as pd
from bencheval.evaluator import BenchmarkEvaluator

# One row per (method, task[, seed]); lower `metric_error` is better.
data = pd.DataFrame(
    {
        "method": ["A", "B", "A", "B"],
        "task": ["t1", "t1", "t2", "t2"],
        "seed": [0, 0, 0, 0],
        "metric_error": [0.10, 0.12, 0.30, 0.25],
        "time_train_s": [1.0, 2.0, 1.5, 2.5],
        "time_infer_s": [0.1, 0.2, 0.1, 0.2],
    }
)

evaluator = BenchmarkEvaluator(seed_column="seed")
leaderboard = evaluator.leaderboard(data, include_error=True)
print(leaderboard)
```

`examples/plots/run_generate_custom_leaderboard.py` in the TabArena repository shows custom
leaderboard metrics, Elo calibration against a reference method, and seed averaging.

## License

Apache-2.0. See the `LICENSE` file shipped with the distribution.
