"""End to end: the arena context asserts the validation protocol, the results record it.

Runs AutoGluon's ``DummyModel`` (constant / majority predictor, no GPU) on two toy ``UserTask``
datasets through the same context API the quick starts use, fully local and network-free, and pins
what a user of the official pipeline gets:

* TabArena's protocol stamps 8 bagging children per fit and records ``8x1`` on every result, the
  results-cache side record and the registered method;
* BeyondArena's protocol switches these tiny tasks to the 5x5 regime (25 children);
* a pre-built bagged experiment carrying another protocol is refused before any result is written;
* ``official_validation_protocol=False`` runs it and records ``3x1`` as a custom protocol;
* outer fits run without a flag and register as ``outer``;
* a rerun of the same ``expname`` under another protocol is refused by the cache guard;
* a bagged experiment without a protocol and without a context is refused at fit time.
"""

from __future__ import annotations

import pandas as pd
import pytest
from autogluon.core.models import DummyModel
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabarena.benchmark.experiment import (
    AGModelBagExperiment,
    BeyondArenaExperimentBundle,
    ExperimentBatchRunner,
    TabArenaV0pt1ExperimentBundle,
    ValidationProtocol,
    ValidationProtocolError,
    build_jobs,
)
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.metadata import TabArenaTaskMetadata, TaskMetadataCollection
from tabarena.benchmark.task.user_task import from_sklearn_splits_to_user_task_splits
from tabarena.benchmark.validation_protocol import (
    BEYONDARENA_VALIDATION_PROTOCOL,
    TABARENA_V0PT1_VALIDATION_PROTOCOL,
)
from tabarena.contexts import AbstractArenaContext
from tabarena.utils.config_utils import ConfigGenerator


def _toy_frame(*, classification: bool) -> pd.DataFrame:
    maker = make_classification if classification else make_regression
    kwargs = {"n_classes": 2} if classification else {}
    X, y = maker(n_samples=120, n_features=8, n_informative=5, random_state=0, **kwargs)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    df["cat"] = pd.Categorical(["a"] * 40 + ["b"] * 40 + ["c"] * 40)
    return df.assign(target=y)


def _make_classification_task(task_cache_dir) -> tuple[UserTask, TabArenaTaskMetadata]:
    dataset = _toy_frame(classification=True)
    n_splits = 3
    splits = from_sklearn_splits_to_user_task_splits(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(
            dataset.drop(columns="target"), dataset["target"]
        ),
        n_splits=n_splits,
    )
    task = UserTask(task_name="toy_classification", task_cache_path=task_cache_dir)
    task_wrapper = task.create_task(
        dataset=dataset, target_feature="target", problem_type="classification", splits=splits
    )
    task.save_task(task_wrapper)
    return task, task_wrapper.metadata


def _make_regression_task(task_cache_dir) -> tuple[UserTask, TabArenaTaskMetadata]:
    dataset = _toy_frame(classification=False)
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.33, random_state=0, shuffle=True)
    task = UserTask(task_name="toy_regression", task_cache_path=task_cache_dir)
    task_wrapper = task.create_task(
        dataset=dataset, target_feature="target", problem_type="regression", splits={0: {0: (train_idx, test_idx)}}
    )
    task.save_task(task_wrapper)
    return task, task_wrapper.metadata


@pytest.fixture(scope="module")
def toy(tmp_path_factory) -> dict:
    task_cache_dir = tmp_path_factory.mktemp("task_cache")
    clf_task, clf_meta = _make_classification_task(task_cache_dir)
    reg_task, reg_meta = _make_regression_task(task_cache_dir)
    return {
        "tasks": [clf_task, reg_task],
        "collection": TaskMetadataCollection.from_source([clf_meta, reg_meta]),
    }


def _generator() -> ConfigGenerator:
    return ConfigGenerator(search_space={}, model_cls=DummyModel, manual_configs=[{}])


def _context(toy: dict, **kwargs) -> AbstractArenaContext:
    return AbstractArenaContext(methods=[], task_metadata=toy["collection"], backend="native", **kwargs)


def _run(context: AbstractArenaContext, experiments: list, *, toy: dict, expname) -> list[dict]:
    return context.build_and_run_jobs(
        experiments,
        expname=str(expname),
        subset="lite",
        user_tasks=toy["tasks"],
        new_result_prefix="[New] ",
        debug_mode=True,
    )


def _registered(context: AbstractArenaContext) -> list:
    return [m for m in context.method_metadata_collection.method_metadata_lst if m.method in context._new_method_names]


def test_official_tabarena_pipeline_records_eight_by_one(toy, tmp_path):
    context = _context(toy, validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL)
    experiments = TabArenaV0pt1ExperimentBundle(models=[(_generator(), 0)]).build_experiments()
    expname = tmp_path / "tabarena"

    results = _run(context, experiments, toy=toy, expname=expname)

    assert len(results) == 2  # one lite split per toy dataset
    for result in results:
        record = result["validation_protocol"]
        assert record["flavour"] == "bagged"
        assert record["key"] == "8x1"
        assert record["protocol"]["enforced"] is True
        assert record["protocol"]["arena"] == "Arena"
        assert record["regime"] == "default"
        assert (record["num_bag_folds_resolved"], record["num_bag_sets_resolved"]) == (8, 1)
        assert (record["num_bag_folds_fitted"], record["num_bag_sets_fitted"]) == (8, 1)
        assert record["num_child_models"] == 8
        assert record["child_oof"] is False
    # The side record next to every cached result, and the registered method's protocol key.
    side_records = list(expname.rglob("validation_protocol.json"))
    assert len(side_records) == 2
    (method,) = _registered(context)
    assert method.validation_protocol == "8x1"
    assert context.validation_protocol_status(method) == "official"


def test_beyondarena_protocol_switches_tiny_tasks_to_five_by_five(toy, tmp_path):
    context = _context(toy, validation_protocol=BEYONDARENA_VALIDATION_PROTOCOL)
    experiments = BeyondArenaExperimentBundle(models=[(_generator(), 0)]).build_experiments()

    results = _run(context, experiments, toy=toy, expname=tmp_path / "beyondarena")

    for result in results:
        record = result["validation_protocol"]
        assert record["key"] == BEYONDARENA_VALIDATION_PROTOCOL.key()
        assert record["regime"] == "tiny"
        assert record["num_group_instances"] <= 500
        assert (record["num_bag_folds_resolved"], record["num_bag_sets_resolved"]) == (5, 5)
        assert record["num_child_models"] == 25
        assert record["custom_splits"] is False  # IID toy data: nothing group- or time-aware to honour
        assert record["clamps"] == []


def test_a_foreign_bagged_protocol_is_refused_before_any_result(toy, tmp_path):
    context = _context(toy, validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL)
    experiment = AGModelBagExperiment(
        name="Dummy_custom_BAG_L1",
        model_cls=DummyModel,
        model_hyperparameters={},
        validation_protocol=ValidationProtocol.custom(3),
    )
    expname = tmp_path / "refused"
    with pytest.raises(ValidationProtocolError, match="official_validation_protocol=False"):
        _run(context, [experiment], toy=toy, expname=expname)
    assert not list(expname.rglob("results.pkl"))


def test_opting_out_runs_the_custom_protocol_and_marks_it(toy, tmp_path):
    context = _context(toy, validation_protocol=ValidationProtocol.custom(3), official_validation_protocol=False)
    experiments = TabArenaV0pt1ExperimentBundle(models=[(_generator(), 0)]).build_experiments()

    results = _run(context, experiments, toy=toy, expname=tmp_path / "custom")

    for result in results:
        record = result["validation_protocol"]
        assert record["key"] == "3x1"
        assert record["protocol"]["enforced"] is False
        assert record["num_child_models"] == 3
    (method,) = _registered(context)
    assert method.validation_protocol == "3x1"
    # An opted-out bare context has no official protocol to judge against.
    assert context.validation_protocol_status(method) == "recorded"

    # Registered into a context that enforces TabArena's protocol, the same results are custom and warn.
    official = _context(toy, validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL)
    with pytest.warns(UserWarning, match=r"outside the official validation protocol \[8x1\]"):
        official.register(results, new_result_prefix="[Custom] ")
    (registered,) = _registered(official)
    assert registered.validation_protocol == "3x1"
    assert official.validation_protocol_status(registered) == "custom"


def test_outer_fits_run_without_a_flag_and_register_as_outer(toy, tmp_path):
    context = _context(toy, validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL)
    experiments = TabArenaV0pt1ExperimentBundle(models=[(_generator(), 0)], outer_experiments=True).build_experiments()

    with pytest.warns(UserWarning, match="outside the official validation protocol"):
        results = _run(context, experiments, toy=toy, expname=tmp_path / "outer")

    for result in results:
        assert result["validation_protocol"] == {"flavour": "outer", "protocol": None, "key": "outer"}
    (method,) = _registered(context)
    assert method.validation_protocol == "outer"


def test_a_rerun_under_another_protocol_is_refused_by_the_cache(toy, tmp_path):
    expname = tmp_path / "shared_expname"
    official = _context(toy, validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL)
    _run(
        official,
        TabArenaV0pt1ExperimentBundle(models=[(_generator(), 0)]).build_experiments(),
        toy=toy,
        expname=expname,
    )

    custom = _context(toy, validation_protocol=ValidationProtocol.custom(3), official_validation_protocol=False)
    with pytest.raises(ValidationProtocolError, match="was fit under validation protocol '8x1'"):
        _run(
            custom,
            TabArenaV0pt1ExperimentBundle(models=[(_generator(), 0)]).build_experiments(),
            toy=toy,
            expname=expname,
        )


def test_a_bagged_experiment_without_protocol_and_context_is_refused_at_fit(toy, tmp_path):
    experiments = TabArenaV0pt1ExperimentBundle(models=[(_generator(), 0)]).build_experiments()
    collection = toy["collection"].subset_tasks(split_indices="lite")
    runner = ExperimentBatchRunner(
        expname=str(tmp_path / "no_context"), task_metadata=collection, debug_mode=True, user_tasks=toy["tasks"]
    )
    with pytest.raises(ValidationProtocolError, match="arena context"):
        runner.run_jobs(build_jobs(experiments, collection))
