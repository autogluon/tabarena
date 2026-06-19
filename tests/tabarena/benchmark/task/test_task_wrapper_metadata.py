"""TaskWrapper <-> TabArenaTaskMetadata unification.

Covers the three guarantees added by the metadata-first wrapper:

* a wrapper built with a ``metadata`` reads ``problem_type`` / ``label`` /
  ``eval_metric`` / ``has_text`` from it (single source of truth);
* ``TaskWrapper.compute_metadata`` reproduces, out of the box, exactly what the
  creation-time ``TabArenaTaskMetadataMixin.compute_metadata`` stored (both
  delegate to ``compute_task_metadata``);
* ``TaskWrapper.validate_metadata`` passes on the round-trip and reports the
  diverging fields on a stale/tampered stored metadata.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabarena.benchmark.task import UserTask, from_sklearn_splits_to_user_task_splits

if TYPE_CHECKING:
    from tabarena.benchmark.task.metadata import TabArenaTaskMetadata


def _toy_frame(*, classification: bool) -> pd.DataFrame:
    maker = make_classification if classification else make_regression
    kwargs = {"n_classes": 2} if classification else {}
    X, y = maker(n_samples=120, n_features=8, n_informative=5, random_state=0, **kwargs)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    df["cat"] = pd.Categorical(["a"] * 40 + ["b"] * 40 + ["c"] * 40)
    return df.assign(target=y)


def _make_user_task(tmp_path, *, classification: bool) -> tuple[UserTask, TabArenaTaskMetadata]:
    """A cached UserTask plus its creation-time metadata (the example-script flow)."""
    dataset = _toy_frame(classification=classification)
    if classification:
        splits = from_sklearn_splits_to_user_task_splits(
            StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(
                dataset.drop(columns="target"),
                dataset["target"],
            ),
            n_splits=3,
        )
    else:
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.33, random_state=0)
        splits = {0: {0: (train_idx, test_idx)}}

    name = "toy_classification" if classification else "toy_regression"
    task = UserTask(task_name=name, task_cache_path=tmp_path)
    oml_task = task.create_local_openml_task(
        dataset=dataset,
        target_feature="target",
        problem_type="classification" if classification else "regression",
        splits=splits,
    )
    metadata = oml_task.compute_metadata(
        tabarena_task_name=task.tabarena_task_name,
        task_id_str=task.task_id_str,
    )
    task.save_local_openml_task(oml_task)
    return task, metadata


@pytest.mark.parametrize("classification", [True, False], ids=["classification", "regression"])
def test_wrapper_metadata_round_trip(tmp_path, classification):
    """Stored metadata == wrapper-recomputed metadata, on the wrapper as actually loaded."""
    task, stored = _make_user_task(tmp_path, classification=classification)
    wrapper = task.with_task_metadata(stored).load()

    # Metadata-first problem attributes.
    assert wrapper.metadata is stored
    assert wrapper.problem_type == stored.problem_type
    assert wrapper.label == "target"
    assert wrapper.dataset_name == stored.dataset_name

    computed = wrapper.validate_metadata()  # raises on any non-identity field diff
    # Identity defaults come from the attached metadata.
    assert computed.tabarena_task_name == stored.tabarena_task_name
    assert computed.task_id_str == stored.task_id_str


def test_wrapper_compute_metadata_without_attached_metadata(tmp_path):
    """compute_metadata works out of the box on a wrapper loaded with no metadata."""
    task, stored = _make_user_task(tmp_path, classification=True)
    wrapper = task.load()  # no metadata attached
    assert wrapper.metadata is None

    computed = wrapper.compute_metadata()
    assert computed.tabarena_task_name is None  # no identity available
    expected = dataclasses.replace(stored, tabarena_task_name=None, task_id_str=None)
    assert computed.to_dict() == expected.to_dict()


def test_validate_metadata_reports_diverging_fields(tmp_path):
    task, stored = _make_user_task(tmp_path, classification=True)
    stale = dataclasses.replace(stored, num_instances=999, has_text=True)
    wrapper = task.load()

    with pytest.raises(AssertionError) as exc_info:
        wrapper.validate_metadata(stale)
    message = str(exc_info.value)
    assert "num_instances" in message
    assert "has_text" in message
    # Non-diverging fields are not reported.
    assert "num_features" not in message


def test_validate_metadata_requires_some_expectation(tmp_path):
    task, _ = _make_user_task(tmp_path, classification=True)
    with pytest.raises(ValueError, match="No metadata to validate against"):
        task.load().validate_metadata()


def test_metadata_first_eval_metric_and_has_text(tmp_path):
    """Problem metadata is read from the attached metadata, not re-derived."""
    task, stored = _make_user_task(tmp_path, classification=True)
    # The toy task has no eval metric (None -> per-problem-type default) and no text.
    assert task.with_task_metadata(stored).load().eval_metric == "roc_auc"

    tampered = dataclasses.replace(stored, eval_metric="log_loss", has_text=True)
    wrapper = task.with_task_metadata(tampered).load()
    assert wrapper.eval_metric == "log_loss"
    assert wrapper.has_text is True  # metadata definition wins over the dtype scan
    assert wrapper._has_text is False  # the scan itself is unchanged


def test_batch_runner_attaches_collection_metadata(tmp_path):
    """ExperimentBatchRunner hands each resolved spec its collection entry."""
    from tabarena.benchmark.experiment import ExperimentBatchRunner
    from tabarena.benchmark.task.metadata import TaskMetadataCollection

    task, stored = _make_user_task(tmp_path, classification=True)
    collection = TaskMetadataCollection.from_source([stored])
    runner = ExperimentBatchRunner(expname=str(tmp_path / "exp"), task_metadata=collection)

    spec = runner._resolve_task(task.tabarena_task_name)
    assert isinstance(spec, UserTask)
    # The collection may copy entries on ingest; the attached metadata must be equal.
    assert spec.task_metadata is not None
    assert spec.task_metadata.to_dict() == stored.to_dict()
    assert spec.load().problem_type == stored.problem_type
