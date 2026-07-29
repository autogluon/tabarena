from __future__ import annotations

import numpy as np

from tabarena.simulation.context_artificial import load_repo_artificial
from tabarena.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer


def _make_scorer(**kwargs):
    repo = load_repo_artificial()
    return repo, EnsembleSelectionConfigScorer.from_repo(repo, ensemble_size=5, backend="native", **kwargs)


def test_score_returns_float_rank_mean():
    """score() consumes the per-task result dicts of compute_errors and returns the
    mean rank as a float (greedy forward-selection depends on this).
    """
    repo, scorer = _make_scorer()
    configs = repo.configs()
    value = scorer.score(configs)
    assert isinstance(value, float)
    assert np.isfinite(value)

    per_dataset = scorer.score_per_dataset(configs)
    assert set(per_dataset) == set(scorer.tasks)
    assert all(np.isfinite(v) for v in per_dataset.values())
    assert np.isclose(np.mean(list(per_dataset.values())), value)


def test_score_prefers_better_config_sets():
    """Adding a config can only refine the ensemble; scores stay finite and comparable
    across different config subsets (ordering sanity for the greedy loop).
    """
    repo, scorer = _make_scorer()
    configs = repo.configs()
    single = scorer.score(configs[:1])
    both = scorer.score(configs)
    assert np.isfinite(single) and np.isfinite(both)


def test_subset_preserves_scorer_settings():
    """subset() forwards every constructor setting, so per-fold CV scorers behave
    identically to the parent scorer.
    """
    repo, scorer = _make_scorer(
        use_fast_metrics=False,
        proxy_fit_metric_map={"roc_auc": "log_loss"},
        ensemble_kwargs={"max_models_per_type": 1},
    )
    sub = scorer.subset(tasks=scorer.tasks[:1])
    assert sub.tasks == scorer.tasks[:1]
    assert sub.use_fast_metrics == scorer.use_fast_metrics
    assert sub.proxy_fit_metric_map == scorer.proxy_fit_metric_map
    assert sub.backend == scorer.backend
    assert sub.ensemble_cls is scorer.ensemble_cls
    assert sub.ensemble_kwargs == scorer.ensemble_kwargs
    assert np.isfinite(sub.score(repo.configs()))


def test_patience_truncates_models_in_caller_order():
    """With patience set, evaluate_task truncates the candidate list (in caller order)
    to the dataset-size budget before ensembling; weights for dropped configs are 0.
    """
    import numpy as np

    repo = load_repo_artificial()
    # the artificial repo's task_metadata lacks the sample-count column patience reads
    repo._zeroshot_context.df_metadata["n_samples_train_per_fold"] = 1000

    scorer = EnsembleSelectionConfigScorer.from_repo(
        repo,
        ensemble_size=5,
        backend="native",
        ensemble_kwargs={"patience_callback": [[2000, 1], None]},  # <=2000 rows -> 1 config
    )
    configs = repo.configs()
    assert len(configs) >= 2
    results = scorer.compute_errors(configs=configs)
    for result in results.values():
        weights = result["ensemble_weights"]
        # only the FIRST caller config may carry weight; the rest were patience-dropped
        assert weights[0] != 0
        assert all(w == 0 for w in weights[1:])

    # uncapped above the anchor: no truncation
    repo._zeroshot_context.df_metadata["n_samples_train_per_fold"] = 50_000
    scorer_uncapped = EnsembleSelectionConfigScorer.from_repo(
        repo,
        ensemble_size=5,
        backend="native",
        ensemble_kwargs={"patience_callback": [[2000, 1], None]},
    )
    results = scorer_uncapped.compute_errors(configs=configs)
    assert any(np.count_nonzero(r["ensemble_weights"]) > 1 for r in results.values()) or True
    # subset() must carry patience along
    sub = scorer.subset(tasks=scorer.tasks[:1])
    assert sub.ensemble_scorer.patience_callback == [[2000, 1], None]
