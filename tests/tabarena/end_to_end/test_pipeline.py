"""Integration tests for the end-to-end pipeline on synthetic raw results.

Covers the full raw -> (raw cache, processed repo, results) flow with ``backend="native"``
(no ray, no network: task metadata is passed explicitly), including the per-task processed
writes + merged context that back ``run_process_method.py --process``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.benchmark.result import BaselineResult
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.end_to_end import EndToEnd
from tabarena.models._method_metadata import MethodMetadata

_TASKS = [
    {"dataset": "d1", "tid": 3901},
    {"dataset": "d2", "tid": 3902},
]


def _task_metadata() -> TaskMetadataCollection:
    legacy = pd.DataFrame(
        {
            "dataset": [t["dataset"] for t in _TASKS],
            "name": [t["dataset"] for t in _TASKS],
            "tid": [t["tid"] for t in _TASKS],
            "problem_type": ["multiclass"] * len(_TASKS),
            "n_folds": [1] * len(_TASKS),
            "n_repeats": [1] * len(_TASKS),
            "n_features": [3] * len(_TASKS),
            "n_classes": [3] * len(_TASKS),
            "NumberOfInstances": [10] * len(_TASKS),
            "n_samples_train_per_fold": [6] * len(_TASKS),
            "n_samples_test_per_fold": [4] * len(_TASKS),
            "target_feature": ["class"] * len(_TASKS),
        }
    )
    return TaskMetadataCollection.from_legacy_df(legacy)


def _make_result_baseline(dataset: str, tid: int, framework: str = "B1") -> dict:
    return dict(
        framework=framework,
        metric_error=0.5,
        metric="log_loss",
        problem_type="multiclass",
        time_train_s=1.2,
        time_infer_s=1.6,
        task_metadata=dict(
            fold=0,
            repeat=0,
            sample=0,
            split_idx=0,
            tid=tid,
            name=dataset,
        ),
    )


def _make_result_config(dataset: str, tid: int, config: str, seed: int) -> dict:
    """A minimal ConfigResult dict (predictions + hyperparameters) for one (task, config)."""
    rng = np.random.default_rng(seed)

    def _proba(n: int) -> np.ndarray:
        pred = rng.uniform(0.1, 1.0, size=(n, 3))
        return pred / pred.sum(axis=1, keepdims=True)

    result = _make_result_baseline(dataset=dataset, tid=tid, framework=config)
    result["metric_error_val"] = 0.5
    result["simulation_artifacts"] = dict(
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
    )
    result["method_metadata"] = dict(
        model_hyperparameters={"param1": config},
        model_cls="DummyModel",
        model_type="DUMMY",
        ag_key="DUMMY",
        name_prefix="Dummy",
    )
    return result


def _config_results() -> list[dict]:
    return [
        _make_result_config(dataset=task["dataset"], tid=task["tid"], config=config, seed=i)
        for i, (task, config) in enumerate((task, config) for task in _TASKS for config in ("Dummy_c1", "Dummy_c2"))
    ]


def _write_raw(results: list[dict], path_raw) -> None:
    """Write results as ``<framework>/<tid>/<repeat>_<fold>/results.pkl`` (the run layout)."""
    for result in results:
        BaselineResult.from_dict(result).to_dir(path=path_raw)


class TestFromPathRawFullTiers:
    """`--process`-style run: explicit metadata, all artifact tiers, per-task native backend."""

    @pytest.fixture
    def artifacts(self, tmp_path):
        path_raw = tmp_path / "raw_in"
        artifact_dir = tmp_path / "artifacts"
        _write_raw(_config_results(), path_raw)
        method_metadata = MethodMetadata.config(
            method="Dummy",
            suite="dummy-suite",
            ag_key="DUMMY",
            config_default="Dummy_c1",
            can_hpo=True,
            artifact_dir=artifact_dir,
        )
        results = EndToEnd.from_path_raw(
            path_raw=path_raw,
            method_metadata=method_metadata,
            task_metadata=_task_metadata(),
            cache=True,
            cache_raw=True,
            cache_processed=True,
            cache_hpo_trajectories=True,
            backend="native",
            verbose=False,
        )
        return results, method_metadata, artifact_dir

    def test_results_frames(self, artifacts):
        results, _, _ = artifacts
        assert len(results.method_results_lst) == 1
        method_results = results.method_results_lst[0]
        assert set(method_results.model_results["dataset"]) == {"d1", "d2"}
        assert set(method_results.model_results["method"]) == {"Dummy_c1", "Dummy_c2"}
        # HPO simulation emits the default + tuned(+ensemble) methods per task.
        assert set(method_results.hpo_results["dataset"]) == {"d1", "d2"}
        assert (method_results.hpo_results["ta_name"] == "Dummy").all()
        assert (method_results.hpo_results["ta_suite"] == "dummy-suite").all()

    def test_cached_artifact_tiers(self, artifacts):
        _, method_metadata, artifact_dir = artifacts
        assert (artifact_dir / "metadata.yaml").is_file()
        assert (artifact_dir / "results" / "model_results.parquet").is_file()
        assert (artifact_dir / "results" / "hpo_results.parquet").is_file()
        assert (artifact_dir / "results" / "hpo_trajectories.parquet").is_file()
        # Raw copies: one results.pkl per (config, task).
        raw_files = sorted(p.relative_to(artifact_dir / "raw") for p in (artifact_dir / "raw").rglob("results.pkl"))
        assert len(raw_files) == 4

    def test_processed_roundtrip(self, artifacts):
        _, _, artifact_dir = artifacts
        # The per-task slices + merged context must load as a regular processed repo.
        reloaded = MethodMetadata.from_yaml(path=artifact_dir)
        repo = reloaded.load_processed()
        assert sorted(repo.datasets()) == ["d1", "d2"]
        assert sorted(repo.configs()) == ["Dummy_c1", "Dummy_c2"]
        assert repo.config_hyperparameters(config="Dummy_c1") == {"param1": "Dummy_c1"}
        # Predictions round-trip (float32 storage) for a known (task, config).
        expected = _config_results()[0]["simulation_artifacts"]["pred_test"]
        loaded = repo.predict_test(dataset="d1", fold=0, config="Dummy_c1")
        np.testing.assert_allclose(loaded, expected, rtol=1e-6)

    def test_results_match_cache(self, artifacts):
        results, method_metadata, _ = artifacts
        cached = EndToEnd.from_cache(methods=[method_metadata])
        in_memory = results.method_results_lst[0].hpo_results
        reloaded = cached.method_results_lst[0].hpo_results

        def _normalized(df: pd.DataFrame) -> pd.DataFrame:
            # Parquet round-trips None <-> nan in object columns; normalize before comparing.
            df = df.sort_values(["dataset", "method"]).reset_index(drop=True)
            return df.where(df.notna(), np.nan)

        pd.testing.assert_frame_equal(_normalized(in_memory), _normalized(reloaded))


class TestFromRawMatchesFromPathRaw:
    def test_equivalent_results(self, tmp_path):
        path_raw = tmp_path / "raw_in"
        _write_raw(_config_results(), path_raw)

        from_raw = EndToEnd.from_raw(
            results_lst=_config_results(),
            task_metadata=_task_metadata(),
            cache=False,
            backend="native",
            verbose=False,
        )
        from_path = EndToEnd.from_path_raw(
            path_raw=path_raw,
            task_metadata=_task_metadata(),
            cache=False,
            backend="native",
            verbose=False,
        )

        # Same inferred method identity (two configs -> tunable, no single default).
        for results in (from_raw, from_path):
            (method_metadata,) = results.method_metadata_lst
            assert method_metadata.method == "Dummy"
            assert method_metadata.can_hpo is True
            assert method_metadata.config_default is None

        # The per-task pipeline must produce the same frames as the single in-memory pass.
        def _sorted(df: pd.DataFrame) -> pd.DataFrame:
            return df.sort_values(["dataset", "method"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(_sorted(from_raw.get_results()), _sorted(from_path.get_results()))
        pd.testing.assert_frame_equal(
            _sorted(from_raw.method_results_lst[0].model_results),
            _sorted(from_path.method_results_lst[0].model_results),
        )

    def test_from_raw_keeps_repo_in_memory(self):
        results = EndToEnd.from_raw(
            results_lst=_config_results(),
            task_metadata=_task_metadata(),
            cache=False,
            backend="native",
            verbose=False,
        )
        repo = results.method_results_lst[0].repo
        assert repo is not None
        assert sorted(repo.configs()) == ["Dummy_c1", "Dummy_c2"]


class TestScanAndFilePaths:
    def test_scan_raw_info_rows(self, tmp_path):
        from tabarena.benchmark.result.raw_loading import scan_raw_info

        path_raw = tmp_path / "raw_in"
        _write_raw(_config_results(), path_raw)
        info_df = scan_raw_info(path_raw=path_raw, engine="sequential", progress_bar=False)
        assert len(info_df) == 4  # one row per results.pkl
        assert set(info_df["framework"]) == {"Dummy_c1", "Dummy_c2"}
        assert set(info_df["ag_key"]) == {"DUMMY"}
        assert set(info_df["method_type"]) == {"config"}

    def test_from_path_raw_with_prediscovered_file_paths(self, tmp_path):
        # Pre-discovered paths (e.g. from the inspect scan) must skip the walk, same results.
        from tabarena.benchmark.result.raw_loading import fetch_raw_result_paths

        path_raw = tmp_path / "raw_in"
        _write_raw(_config_results(), path_raw)
        file_paths = fetch_raw_result_paths(path_raw=path_raw)
        results = EndToEnd.from_path_raw(
            path_raw=path_raw,
            file_paths=file_paths,
            task_metadata=_task_metadata(),
            cache=False,
            backend="native",
            verbose=False,
        )
        (method_results,) = results.method_results_lst
        assert set(method_results.model_results["method"]) == {"Dummy_c1", "Dummy_c2"}
        assert set(method_results.model_results["dataset"]) == {"d1", "d2"}

    def test_file_paths_and_name_prefix_raw_are_exclusive(self, tmp_path):
        path_raw = tmp_path / "raw_in"
        _write_raw(_config_results(), path_raw)
        with pytest.raises(ValueError, match="not both"):
            EndToEnd.from_path_raw(
                path_raw=path_raw,
                file_paths=[path_raw / "x" / "0" / "0_0" / "results.pkl"],
                name_prefix_raw="Dummy",
                task_metadata=_task_metadata(),
                backend="native",
                verbose=False,
            )


class TestMultiMethod:
    def test_config_and_baseline_are_split(self, tmp_path):
        path_raw = tmp_path / "raw_in"
        baselines = [_make_result_baseline(dataset=t["dataset"], tid=t["tid"]) for t in _TASKS]
        _write_raw(_config_results() + baselines, path_raw)

        results = EndToEnd.from_path_raw(
            path_raw=path_raw,
            task_metadata=_task_metadata(),
            cache=False,
            backend="native",
            verbose=False,
        )
        by_method = {r.method_metadata.method: r for r in results.method_results_lst}
        assert set(by_method) == {"Dummy", "B1"}
        assert by_method["B1"].method_metadata.method_type == "baseline"
        assert by_method["B1"].hpo_results is None
        assert set(by_method["B1"].model_results["dataset"]) == {"d1", "d2"}
        assert set(by_method["Dummy"].model_results["method"]) == {"Dummy_c1", "Dummy_c2"}

    def test_single_method_args_rejected_for_multi_method(self, tmp_path):
        path_raw = tmp_path / "raw_in"
        baselines = [_make_result_baseline(dataset=t["dataset"], tid=t["tid"]) for t in _TASKS]
        _write_raw(_config_results() + baselines, path_raw)

        with pytest.raises(ValueError, match="apply to a single method"):
            EndToEnd.from_path_raw(
                path_raw=path_raw,
                task_metadata=_task_metadata(),
                method="my-method",
                cache=False,
                backend="native",
                verbose=False,
            )
