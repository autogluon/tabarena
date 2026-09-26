from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from autogluon.common.features.feature_metadata import FeatureMetadata

from tabarena.benchmark.preprocessing import TabArenaModelAgnosticPreprocessing
from tabarena.benchmark.preprocessing.group_feature_generators import S_GROUP_ID, GroupIdCodes
from tabarena.benchmark.preprocessing.pipeline import TABARENA_DEFAULT_GROUP_ID, resolve_preprocessing_pipeline
from tabarena.benchmark.task.metadata import GroupLabelTypes


def _grouped_frame(groups: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = len(groups)
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n), "patient": groups})
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def _pipeline(group_labels: GroupLabelTypes, *, expose_group_id: bool) -> TabArenaModelAgnosticPreprocessing:
    return TabArenaModelAgnosticPreprocessing(
        enable_sematic_text_features=False,
        group_cols="patient",
        group_labels=group_labels,
        expose_group_id=expose_group_id,
        verbosity=0,
    )


def test_group_id_codes_fit_block_levels_and_unseen_nan():
    codes = GroupIdCodes().fit(pd.Series(["p2", "p2", "p1", "p3"]))
    out = codes.transform(pd.Series(["p1", "p9", "p2"]))
    assert out.dtype == np.float64
    assert out.tolist()[0] == 1.0
    assert np.isnan(out.tolist()[1])
    assert out.tolist()[2] == 0.0


def test_per_sample_task_exposes_group_as_tagged_float_codes():
    X, y = _grouped_frame(["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"])
    gen = _pipeline(GroupLabelTypes.PER_SAMPLE, expose_group_id=True)
    out = gen.fit_transform(X, y=y)

    assert "patient" in out.columns
    assert out["patient"].dtype == np.float64
    assert out["patient"].tolist() == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert gen.feature_metadata.get_features(required_special_types=[S_GROUP_ID]) == ["patient"]
    assert gen.feature_metadata.get_feature_type_raw("patient") == "float"

    X_test, _ = _grouped_frame(["p2", "p9", "p9"])
    codes = gen.transform(X_test)["patient"]
    assert codes.tolist()[0] == 1.0
    assert codes.isna().tolist() == [False, True, True]


def test_singleton_groups_keep_their_code():
    # every group appears once; a category-count filter would erase them, float codes survive
    X, y = _grouped_frame([f"p{i}" for i in range(8)])
    out = _pipeline(GroupLabelTypes.PER_SAMPLE, expose_group_id=True).fit_transform(X, y=y)
    assert out["patient"].notna().all()
    assert out["patient"].nunique() == 8


def test_per_group_task_is_not_exposed():
    X, y = _grouped_frame(["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"])
    gen = _pipeline(GroupLabelTypes.PER_GROUP, expose_group_id=True)
    out = gen.fit_transform(X, y=y)
    assert "patient" not in out.columns
    assert gen.feature_metadata.get_features(required_special_types=[S_GROUP_ID]) == []


def test_default_pipeline_drops_the_group_column():
    X, y = _grouped_frame(["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"])
    out = _pipeline(GroupLabelTypes.PER_SAMPLE, expose_group_id=False).fit_transform(X, y=y)
    assert "patient" not in out.columns


def test_composite_group_key_is_refused():
    with pytest.raises(NotImplementedError):
        TabArenaModelAgnosticPreprocessing(
            enable_sematic_text_features=False,
            group_cols=["a", "b"],
            group_labels=GroupLabelTypes.PER_SAMPLE,
            expose_group_id=True,
        )


def test_named_pipeline_resolves_to_exposure_kwargs():
    default = resolve_preprocessing_pipeline("tabarena_default")
    exposed = resolve_preprocessing_pipeline(TABARENA_DEFAULT_GROUP_ID)
    assert exposed.feature_generator_cls is default.feature_generator_cls
    assert default.feature_generator_kwargs == {}
    assert exposed.feature_generator_kwargs == {"expose_group_id": True}


def test_tabpfn_wrapper_declares_tagged_group_column_categorical():
    from tabarena.models.tabpfn_3_5.model import TabPFN35Model

    model = TabPFN35Model(path="", name="m", problem_type="binary", eval_metric="log_loss", hyperparameters={})
    X = pd.DataFrame({"a": [0.1, 0.2, 0.3], "patient": [0.0, 1.0, np.nan]})
    model._feature_metadata = FeatureMetadata(
        type_map_raw={"a": "float", "patient": "float"}, type_group_map_special={S_GROUP_ID: ["patient"]}
    )
    out = model._preprocess(X, is_train=True)
    assert str(out["patient"].dtype) == "category"
    assert model._categorical_indices == [1]

    model._feature_metadata = FeatureMetadata(type_map_raw={"a": "float", "patient": "float"})
    out = model._preprocess(X, is_train=True)
    assert out["patient"].dtype == np.float64
    assert model._categorical_indices is None
