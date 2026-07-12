"""Equivalence tests for HPO-trajectory simulation.

``generate_hpo_trajectories`` computes all (n_config, seed) passes in one per-task sweep
(predictions loaded once per task); these tests pin it to the per-pass reference
(``generate_hpo_result`` / ``evaluate_ensemble`` called once per combo), which must produce
frame-identical results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.benchmark.result import BaselineResult
from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._method_simulator import MethodSimulator
from tabarena.repository import EvaluationRepository
from tabarena.repository.generate_repo import generate_repo_from_results_lst

_CONFIGS = ["Dummy_c1", "Dummy_c2", "Dummy_c3", "Dummy_c4"]
_TASKS = [
    {"dataset": "d1", "tid": 3901},
    {"dataset": "d2", "tid": 3902},
]


def _make_result_config(dataset: str, tid: int, config: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    def _proba(n: int) -> np.ndarray:
        pred = rng.uniform(0.1, 1.0, size=(n, 3))
        return pred / pred.sum(axis=1, keepdims=True)

    return dict(
        framework=config,
        metric_error=float(rng.uniform(0.1, 1.0)),
        metric="log_loss",
        problem_type="multiclass",
        time_train_s=float(rng.uniform(1.0, 10.0)),
        time_infer_s=float(rng.uniform(0.1, 1.0)),
        metric_error_val=float(rng.uniform(0.1, 1.0)),
        task_metadata=dict(fold=0, repeat=0, sample=0, split_idx=0, tid=tid, name=dataset),
        simulation_artifacts=dict(
            y_val=np.array([2, 1, 2, 0]),
            y_val_idx=np.array([3, 0, 4, 8]),
            y_test=np.array([0, 2, 1, 1, 1, 1]),
            y_test_idx=np.array([1, 2, 5, 6, 9, 7]),
            pred_val=_proba(4),
            pred_test=_proba(6),
            ordered_class_labels=["class1", "class2", "class3"],
            ordered_class_labels_transformed=[0, 1, 2],
            num_classes=3,
            label="class",
        ),
        method_metadata=dict(
            model_hyperparameters={"param1": seed},
            model_cls="DummyModel",
            model_type="DUMMY",
            ag_key="DUMMY",
            name_prefix="Dummy",
        ),
    )


@pytest.fixture(scope="module")
def repo_and_metadata(tmp_path_factory):
    """A 4-config, 2-task repo; ``Dummy_c4`` is missing on ``d2`` (partial availability).

    Reloaded in memmap format, matching the trajectories flow (``set_config_fallback``
    requires memmap-backed predictions).
    """
    results = []
    for i, task in enumerate(_TASKS):
        for j, config in enumerate(_CONFIGS):
            if task["dataset"] == "d2" and config == "Dummy_c4":
                continue
            results.append(
                BaselineResult.from_dict(
                    _make_result_config(dataset=task["dataset"], tid=task["tid"], config=config, seed=10 * i + j),
                ),
            )
    task_metadata = pd.DataFrame(
        {
            "dataset": [t["dataset"] for t in _TASKS],
            "tid": [t["tid"] for t in _TASKS],
        }
    )
    repo = generate_repo_from_results_lst(results_lst=results, task_metadata=task_metadata)
    path = tmp_path_factory.mktemp("processed")
    repo.to_dir(path)
    repo = EvaluationRepository.from_dir(path, prediction_format="memmap", verbose=False)
    method_metadata = MethodMetadata.config(
        method="Dummy",
        suite="dummy-suite",
        ag_key="DUMMY",
        config_default="Dummy_c1",
        can_hpo=True,
    )
    return repo, method_metadata


class TestTrajectoriesMatchPerPassReference:
    def test_equivalence(self, repo_and_metadata):
        repo, method_metadata = repo_and_metadata
        simulator = MethodSimulator(method_metadata)
        n_configs = [1, 2, 3, 4]
        seeds = 3
        n_iterations = 5

        trajectories = simulator.generate_hpo_trajectories(
            repo=repo,
            n_configs=n_configs,
            seeds=seeds,
            n_iterations=n_iterations,
            backend="native",
            cache=False,
        )

        # Reference: one generate_hpo_result pass per (n_config, seed). n_config=1 (fixed
        # default only) and n_config=4 (= all configs) are seed-invariant -> a single seed.
        passes = [(1, 0)] + [(n, s) for n in (2, 3) for s in range(seeds)] + [(4, 0)]
        reference_lst = []
        for n_config, seed in passes:
            df = simulator.generate_hpo_result(
                repo=repo,
                n_configs=n_config,
                seed=seed,
                n_iterations=n_iterations,
                fixed_configs=[method_metadata.config_default],
                fit_order="random",
                time_limit=None,
                backend="native",
                config_type=method_metadata.config_type,
            )
            df["always_include_default"] = True
            reference_lst.append(df)
        reference = pd.concat(reference_lst, ignore_index=True)

        pd.testing.assert_frame_equal(trajectories, reference)

    def test_seed_changes_subsets(self, repo_and_metadata):
        # Sanity for the fixture: different seeds must actually select different config
        # subsets (otherwise the equivalence test would not exercise the per-seed paths).
        repo, method_metadata = repo_and_metadata
        simulator = MethodSimulator(method_metadata)
        results = {
            seed: simulator.generate_hpo_result(
                repo=repo,
                n_configs=2,
                seed=seed,
                n_iterations=5,
                fixed_configs=["Dummy_c1"],
                fit_order="random",
                backend="native",
                config_type="DUMMY",
            )
            for seed in (0, 1, 2)
        }
        errors = {seed: tuple(df["metric_error"]) for seed, df in results.items()}
        assert len(set(errors.values())) > 1


class TestEvaluateEnsembleMulti:
    def test_matches_per_call_evaluate_ensemble(self, repo_and_metadata):
        repo, _ = repo_and_metadata
        configs_lst = [
            ["Dummy_c1"],
            ["Dummy_c2", "Dummy_c1"],
            ["Dummy_c4", "Dummy_c3", "Dummy_c1"],  # Dummy_c4 missing on d2
            _CONFIGS,
        ]
        for dataset in ("d1", "d2"):
            multi = repo.evaluate_ensemble_multi(
                dataset=dataset,
                fold=0,
                configs_lst=configs_lst,
                ensemble_size=5,
            )
            for configs, (df_multi, weights_multi) in zip(configs_lst, multi, strict=True):
                df_single, weights_single = repo.evaluate_ensemble(
                    dataset=dataset,
                    fold=0,
                    configs=configs,
                    ensemble_size=5,
                )
                pd.testing.assert_frame_equal(df_multi, df_single)
                pd.testing.assert_frame_equal(weights_multi, weights_single)
