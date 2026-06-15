from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.models._in_memory_method_metadata import InMemoryMethodMetadata
from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._method_metadata_collection import MethodMetadataCollection
from tabarena.nips2025_utils.abstract_arena_context import AbstractArenaContext
from tabarena.nips2025_utils.end_to_end_single import EndToEndResultsSingle


def _results_frame(method: str, *, datasets=("d1",), folds=(0,), config_type="MM", ta_name="M", ta_suite="M"):
    rows = [
        {
            "method": method,
            "dataset": dataset,
            "fold": fold,
            "metric_error": 0.1,
            "metric": "roc_auc",
            "problem_type": "binary",
            "method_type": "config",
            "method_subtype": "default",
            "config_type": config_type,
            "ta_name": ta_name,
            "ta_suite": ta_suite,
        }
        for dataset in datasets
        for fold in folds
    ]
    return pd.DataFrame(rows)


def _task_metadata() -> TaskMetadataCollection:
    legacy = pd.DataFrame(
        {
            "tid": [0, 1],
            "dataset": ["d1", "d2"],
            "name": ["d1", "d2"],
            "n_folds": [1, 1],
            "n_repeats": [1, 1],
            "n_samples_train_per_fold": [100, 100],
            "n_samples_test_per_fold": [50, 50],
            "NumberOfInstances": [150, 150],
            "problem_type": ["binary", "binary"],
            "n_features": [10, 10],
            "n_classes": [2, 2],
            "target_feature": ["t", "t"],
        },
    )
    return TaskMetadataCollection.from_legacy_df(legacy)


class TestInMemoryArtifacts:
    def test_is_in_memory_marker(self):
        assert MethodMetadata.is_in_memory is False
        assert InMemoryMethodMetadata.is_in_memory is True

    def test_load_results_returns_a_copy(self):
        df = _results_frame("MM (default)")
        im = InMemoryMethodMetadata(results=df, method="M", method_type="config", model_key="MM")
        out = im.load_results()
        assert out.equals(df)
        out.loc[0, "metric_error"] = 999.0
        assert im.load_results().loc[0, "metric_error"] == 0.1  # original untouched

    def test_to_info_dict_excludes_in_memory_slots(self):
        im = InMemoryMethodMetadata(results=_results_frame("MM (default)"), method="M", method_type="baseline")
        info = im.to_info_dict()
        assert "_results" not in info and "_repo" not in info
        assert info["method"] == "M"

    def test_disabled_disk_ops_raise(self):
        im = InMemoryMethodMetadata(results=_results_frame("MM (default)"), method="M", method_type="baseline")
        for op in ("to_yaml", "to_yaml_fileobj", "load_raw", "generate_repo", "method_downloader", "method_uploader"):
            with pytest.raises(NotImplementedError):
                getattr(im, op)()

    def test_load_processed_requires_repo(self):
        im = InMemoryMethodMetadata(results=_results_frame("MM (default)"), method="M", method_type="config")
        with pytest.raises(NotImplementedError, match="no in-memory repo"):
            im.load_processed()
        sentinel = object()
        im_with_repo = InMemoryMethodMetadata(
            results=_results_frame("MM (default)"),
            repo=sentinel,
            method="M",
            method_type="config",
        )
        assert im_with_repo.load_processed() is sentinel


class TestCollectionInfo:
    def test_info_does_not_embed_dataframe(self):
        im = InMemoryMethodMetadata(results=_results_frame("MM (default)"), method="M", method_type="baseline")
        info = MethodMetadataCollection([im]).info()
        assert "_results" not in info.columns
        assert "_repo" not in info.columns
        assert info.loc[0, "method"] == "M"


class TestFromResultsSinglePrefixIdentity:
    """new_result_prefix must be baked into identity so the website merge key still matches."""

    def _results_single(self) -> EndToEndResultsSingle:
        base = MethodMetadata(
            method="MyModel",
            artifact_name="MyModel",
            method_type="config",
            ag_key="MM",
            model_key="MM",
            display_name="MyModel",
        )
        hpo = _results_frame("MM (default)", config_type="MM", ta_name="MyModel", ta_suite="MyModel")
        return EndToEndResultsSingle(method_metadata=base, model_results=hpo, hpo_results=hpo)

    def test_prefix_applied_to_identity_fields(self):
        im = self._results_single().to_method_metadata(new_result_prefix="[New] ")
        assert im.method == "[New] MyModel"
        assert im.artifact_name == "[New] MyModel"
        assert im.display_name == "[New] MyModel"
        assert im.config_type == "[New] MM"  # model_key was prefixed too

    def test_frame_and_identity_share_the_website_merge_key(self):
        # leaderboard_to_website_format merges leaderboard <-> info on (ta_name, ta_suite),
        # where info's ta_name/ta_suite == method/artifact_name. They must equal the frame's.
        im = self._results_single().to_method_metadata(new_result_prefix="[New] ")
        frame = im.load_results()
        assert list(frame["ta_name"].unique()) == [im.method]
        assert list(frame["ta_suite"].unique()) == [im.artifact_name]
        assert list(frame["method"].unique()) == ["[New] MM (default)"]

    def test_no_prefix_is_identity(self):
        im = self._results_single().to_method_metadata()
        assert im.method == "MyModel"
        assert im.config_type == "MM"


class TestContextRegistration:
    def _ctx(self, *extra) -> AbstractArenaContext:
        return AbstractArenaContext(methods=[], task_metadata=_task_metadata(), extra_methods=list(extra))

    def _in_memory(self, method: str, datasets) -> InMemoryMethodMetadata:
        return InMemoryMethodMetadata(
            results=_results_frame(f"{method} (default)", datasets=datasets, ta_name=method, ta_suite=method),
            method=method,
            artifact_name=method,
            method_type="config",
            model_key=method,
        )

    def test_registered_method_is_listed_and_loadable(self):
        im = self._in_memory("NewA", datasets=("d1",))
        ctx = self._ctx(im)
        assert "NewA" in ctx.methods
        loaded = ctx.load_results()
        assert set(loaded["method"]) == {"NewA (default)"}

    def test_registered_new_results_concats_only_in_memory(self):
        im_a = self._in_memory("NewA", datasets=("d1",))
        im_b = self._in_memory("NewB", datasets=("d1", "d2"))
        ctx = self._ctx(im_a, im_b)
        new = ctx._registered_new_results()
        assert set(new["method"]) == {"NewA (default)", "NewB (default)"}

    def test_registered_new_results_none_without_in_memory(self):
        assert self._ctx()._registered_new_results() is None


class TestResolveOnlyValidTasks:
    def _ctx_with(self, im) -> AbstractArenaContext:
        return AbstractArenaContext(methods=[], task_metadata=_task_metadata(), extra_methods=[im])

    def _im(self) -> InMemoryMethodMetadata:
        return InMemoryMethodMetadata(
            results=_results_frame("NewA (default)", datasets=("d1",), ta_name="NewA", ta_suite="NewA"),
            method="NewA",
            artifact_name="NewA",
            method_type="config",
            model_key="NewA",
        )

    def test_false_is_no_restriction(self):
        ctx = self._ctx_with(self._im())
        assert ctx._resolve_only_valid_tasks(False, None) == (None, None)

    def test_true_uses_registered_in_memory_methods(self):
        ctx = self._ctx_with(self._im())
        df_filter, names = ctx._resolve_only_valid_tasks(True, None)
        assert names is None
        assert set(df_filter["dataset"]) == {"d1"}  # only the task NewA ran

    def test_true_prefers_explicit_new_results(self):
        ctx = self._ctx_with(self._im())
        new_results = _results_frame("X (default)", datasets=("d2",))
        df_filter, _ = ctx._resolve_only_valid_tasks(True, new_results)
        assert set(df_filter["dataset"]) == {"d2"}

    def test_true_without_any_source_raises(self):
        ctx = AbstractArenaContext(methods=[], task_metadata=_task_metadata())
        with pytest.raises(ValueError, match="only_valid_tasks=True needs"):
            ctx._resolve_only_valid_tasks(True, None)

    def test_method_metadata_object_resolves_to_its_tasks(self):
        im = self._im()
        ctx = self._ctx_with(im)
        df_filter, names = ctx._resolve_only_valid_tasks(im, None)
        assert names is None
        assert set(df_filter["dataset"]) == {"d1"}

    def test_string_names_pass_through(self):
        ctx = self._ctx_with(self._im())
        df_filter, names = ctx._resolve_only_valid_tasks(["NewA (default)"], None)
        assert df_filter is None
        assert names == ["NewA (default)"]

    def test_numpy_array_of_names_passes_through_as_list(self):
        ctx = self._ctx_with(self._im())
        df_filter, names = ctx._resolve_only_valid_tasks(np.array(["NewA (default)"]), None)
        assert df_filter is None
        assert names == ["NewA (default)"]

    def test_mixed_list_raises(self):
        im = self._im()
        ctx = self._ctx_with(im)
        with pytest.raises(TypeError, match="not mixed"):
            ctx._resolve_only_valid_tasks([im, "NewA (default)"], None)
