from __future__ import annotations

import pytest

pytest.importorskip("fintfm")

from tabarena.models.fintfm.checkpoints import (
    BINARY_CHECKPOINT,
    MULTICLASS_CHECKPOINT_UNPUBLISHED,
    REGRESSION_CHECKPOINT_UNPUBLISHED,
    CheckpointNotPublishedError,
    resolve_checkpoint,
)
from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel

ENV_VARS = (
    "FINTFM_PARTITION",
    "FINTFM_PROBE",
    "FINTFM_PARAMS",
    "FINTFM_REG_BINS",
    "FINTFM_REG_THRESHOLDS",
    "FINTFM_MAX_CONTEXT",
    "FINTFM_CHECKPOINT",
    "FINTFM_CHECKPOINT_MULTICLASS",
    "FINTFM_CHECKPOINT_REGRESSION",
    "FINTFM_PARTITION_SCALE_N",
    "FINTFM_PROBE_BUDGET",
    "FINTFM_PROB_FLOOR",
    "FINTFM_REG_READOUT",
    "FINTFM_REG_RFF_MAX_ROWS",
    "FINTFM_PROBE_ONEHOT",
)

#: The measured entry per problem type (fintfm FINDINGS §232, §235, §237).
ENTRY = {
    "binary": {
        "checkpoint": BINARY_CHECKPOINT,
        "partition": {},
        "probe": {"scale": True, "cv": True, "raw": True},
        "n_bins": 10,
    },
    "multiclass": {
        "checkpoint": MULTICLASS_CHECKPOINT_UNPUBLISHED,
        "partition": {"n_bags": 8},
        "probe": {"scale": True, "cv": True, "raw": True},
        "n_bins": 10,
    },
    "regression": {
        "checkpoint": REGRESSION_CHECKPOINT_UNPUBLISHED,
        "partition": {"n_bags": 4},
        "probe": {"raw": True},
        "n_bins": 10,
    },
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def make(problem_type: str, tmp_path, **hyperparameters) -> FinTFMLinearProbeModel:
    model = FinTFMLinearProbeModel(
        path=str(tmp_path),
        name="fintfm_test",
        problem_type=problem_type,
        eval_metric={"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}[problem_type],
        hyperparameters=hyperparameters,
    )
    model._init_params()  # AutoGluon calls this when a fit starts
    return model


def resolve(model: FinTFMLinearProbeModel) -> tuple[dict, dict]:
    params = model._get_model_params()
    entry = model._resolve_entry(params)
    return entry, params


@pytest.mark.parametrize("problem_type", sorted(ENTRY))
def test_default_is_the_entry_without_environment(problem_type, tmp_path):
    entry, rest = resolve(make(problem_type, tmp_path))
    assert entry == ENTRY[problem_type]
    # What is left is what FinancialTFMClassifier receives.
    assert rest["feature_transform"] == "squash"
    assert rest["max_features_strategy"] == "select"
    assert rest["max_context"] == 1000
    assert rest["n_ensemble"] == 8
    assert not any(k.startswith(("checkpoint_", "partition_", "probe_", "model_path")) for k in rest)
    assert "n_bins" not in rest


def test_supports_all_three_problem_types():
    assert FinTFMLinearProbeModel._supported_problem_types == ["binary", "multiclass", "regression"]


def test_default_hyperparameter_names_and_values(tmp_path):
    params = make("binary", tmp_path)._get_model_params()
    assert params["partition_binary"] == {}
    assert params["partition_multiclass"] == {"n_bags": 8}
    assert params["partition_regression"] == {"n_bags": 4}
    assert params["probe_regression"] == {"raw": True}
    assert params["checkpoint_binary"] == BINARY_CHECKPOINT
    assert params["n_bins"] == 10


def test_environment_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("FINTFM_PARTITION", '{"n_bags": 2}')
    monkeypatch.setenv("FINTFM_PROBE", '{"raw": true, "head": true}')
    monkeypatch.setenv("FINTFM_PARAMS", '{"feature_transform": "rank"}')
    monkeypatch.setenv("FINTFM_REG_BINS", "20")
    monkeypatch.setenv("FINTFM_MAX_CONTEXT", "500")
    monkeypatch.setenv("FINTFM_CHECKPOINT", "/some/generic.pt")
    monkeypatch.setenv("FINTFM_CHECKPOINT_REGRESSION", "/some/reg.pt")
    for problem_type in ENTRY:
        # head is refused beside cross-fitting, so this case turns cross-fitting off.
        entry, rest = resolve(make(problem_type, tmp_path, probe_crossfit_folds=0))
        assert entry["partition"] == {"n_bags": 2}
        assert entry["probe"] == {"raw": True, "head": True}
        assert entry["n_bins"] == 20
        assert entry["checkpoint"] == ("/some/reg.pt" if problem_type == "regression" else "/some/generic.pt")
        assert rest["feature_transform"] == "rank"
        assert rest["max_features_strategy"] == "select"
        assert rest["max_context"] == 500


def test_null_partition_and_empty_probe_switch_the_entry_off(monkeypatch, tmp_path):
    monkeypatch.setenv("FINTFM_PARTITION", "null")
    monkeypatch.setenv("FINTFM_PROBE", "{}")
    entry, _ = resolve(make("regression", tmp_path))
    assert entry["partition"] is None
    assert entry["probe"] == {}


def test_type_specific_checkpoint_env_beats_generic(monkeypatch, tmp_path):
    monkeypatch.setenv("FINTFM_CHECKPOINT", "/generic.pt")
    monkeypatch.setenv("FINTFM_CHECKPOINT_MULTICLASS", "/mc.pt")
    assert resolve(make("multiclass", tmp_path))[0]["checkpoint"] == "/mc.pt"
    assert resolve(make("binary", tmp_path))[0]["checkpoint"] == "/generic.pt"


def test_explicit_hyperparameter_beats_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("FINTFM_PARTITION", '{"n_bags": 2}')
    monkeypatch.setenv("FINTFM_PROBE", '{"raw": true}')
    monkeypatch.setenv("FINTFM_CHECKPOINT", "/env.pt")
    monkeypatch.setenv("FINTFM_REG_BINS", "20")
    model = make(
        "binary",
        tmp_path,
        partition_binary={"n_bags": 3},
        probe_binary={"scale": True},
        checkpoint_binary="/hp.pt",
        n_bins=5,
    )
    entry, _ = resolve(model)
    assert entry == {
        "checkpoint": "/hp.pt",
        "partition": {"n_bags": 3},
        "probe": {"scale": True},
        "n_bins": 5,
    }


def test_explicit_model_path_beats_checkpoint_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("FINTFM_CHECKPOINT", "/env.pt")
    assert resolve(make("binary", tmp_path, model_path="/hp.pt"))[0]["checkpoint"] == "/hp.pt"
    assert resolve(make("multiclass", tmp_path, model_path_multiclass="/mc.pt"))[0]["checkpoint"] == "/mc.pt"


def test_thresholds_env_drops_the_default_regression_partition(monkeypatch, tmp_path):
    monkeypatch.setenv("FINTFM_REG_THRESHOLDS", "15")
    assert resolve(make("regression", tmp_path))[0]["partition"] is None
    assert resolve(make("binary", tmp_path))[0]["partition"] == {}


@pytest.mark.parametrize(
    ("hyperparameters", "match"),
    [
        ({"partition_binary": {"n_bgas": 4}}, "n_bgas"),
        ({"partition_binary": {"estimator": None}}, "estimator"),
        ({"partition_binary": 4}, "JSON object or null"),
        ({"n_bins": 1}, "at least 2"),
        ({"checkpoint_binary": {"repo_id": "x"}}, "match neither"),
    ],
)
def test_unknown_or_malformed_hyperparameters_raise(hyperparameters, match, tmp_path):
    with pytest.raises(ValueError, match=match):
        resolve(make("binary", tmp_path, **hyperparameters))


def test_unknown_probe_key_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        FinTFMLinearProbeModel._validated_probe({"scal": True}, source="probe_binary")
    with pytest.raises(ValueError, match="no_rep needs raw"):
        FinTFMLinearProbeModel._validated_probe({"no_rep": True}, source="probe_binary")


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("FINTFM_PARTITION", '{"n_bgas": 4}', "n_bgas"),
        ("FINTFM_PARTITION", "{not json", "not valid JSON"),
        ("FINTFM_PROBE", '{"scal": true}', "unknown key"),
        ("FINTFM_PARAMS", '{"no_such_arg": 1}', "no_such_arg"),
        ("FINTFM_REG_BINS", "ten", "integer"),
    ],
)
def test_unknown_environment_values_raise(monkeypatch, tmp_path, name, value, match):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=match):
        resolve(make("binary", tmp_path))


@pytest.mark.parametrize("problem_type", ["multiclass", "regression"])
def test_unpublished_checkpoint_raises_a_precise_error(problem_type):
    ref = ENTRY[problem_type]["checkpoint"]
    with pytest.raises(CheckpointNotPublishedError, match=ref["unpublished"]) as err:
        resolve_checkpoint(ref, problem_type=problem_type)
    assert ref["sha256"] in str(err.value)
    assert f"FINTFM_CHECKPOINT_{problem_type.upper()}" in str(err.value)


def test_local_path_resolves_to_itself():
    assert resolve_checkpoint("/a/b.pt", problem_type="binary") == "/a/b.pt"


def test_published_checkpoint_checks_the_sha256(monkeypatch, tmp_path):
    fake = tmp_path / "weights.pt"
    fake.write_bytes(b"not the checkpoint")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(fake))
    with pytest.raises(ValueError, match="expected d231b062"):
        resolve_checkpoint(BINARY_CHECKPOINT, problem_type="binary")


def test_search_space_keys_are_read_and_its_first_values_are_the_entry(tmp_path):
    from tabarena.models.fintfm_linprobe.hpo import proposed_search_space_additions, search_space

    params = make("binary", tmp_path)._get_model_params()
    for key, space in search_space.items():
        assert key in params, f"{key} is not a hyperparameter the model reads"
        assert space.default == params[key], f"the first value of {key} is not the measured entry's"
        assert params[key] in list(space.data)
    # The proposal names a hyperparameter the model reads, and is not sampled.
    assert set(proposed_search_space_additions) == {"probe_crossfit_folds"}
    assert set(proposed_search_space_additions).isdisjoint(search_space)
    assert set(proposed_search_space_additions) <= set(params)


#: The measured entry's levers per problem type: binary ``fintfm_lp_bin_both_kaggle`` (§254, §257),
#: multiclass ``fintfm_lp_mc_floor_cpu`` and ``fintfm_lp_mc_floor_sdss17_mac`` (§255, §265, §269),
#: regression ``fintfm_lp_reg_rawpart4_pure_1114`` (§255). Each run's recorded hyperparameters give these.
MEASURED = {
    "binary": {"scale_with_n": True, "budget": 50_000, "crossfit": 5, "floor": None},
    "multiclass": {"scale_with_n": False, "budget": 20_000, "crossfit": 0, "floor": "cv"},
    "regression": {"scale_with_n": False, "budget": 20_000, "crossfit": 0, "floor": None},
}

#: The §283 levers of the entry per problem type: multiclass one-hot indicators, regression
#: random-Fourier-feature readout capped at 5,000 training rows; binary has neither.
ONEHOT_ENTRY = {"min_count": 3, "max_columns": 256, "max_levels": 64}
MEASURED_GATED = {
    "binary": {"onehot": None, "readout": "ridge", "rff_max_rows": None},
    "multiclass": {"onehot": ONEHOT_ENTRY, "readout": "ridge", "rff_max_rows": None},
    "regression": {"onehot": None, "readout": "rff", "rff_max_rows": 5_000},
}


def levers(model: FinTFMLinearProbeModel) -> dict:
    return {
        "scale_with_n": model._scale_with_n,
        "budget": model._probe_budget_setting,
        "crossfit": model._crossfit_folds,
        "floor": model._prob_floor_setting,
    }


def gated(model: FinTFMLinearProbeModel) -> dict:
    return {"onehot": model._onehot_cfg, "readout": model._reg_readout, "rff_max_rows": model._reg_rff_max_rows}


@pytest.mark.parametrize("problem_type", sorted(ENTRY))
def test_resolved_entry_equals_the_measured_entry(problem_type, tmp_path):
    model = make(problem_type, tmp_path)
    entry, _ = resolve(model)
    assert entry == ENTRY[problem_type]
    assert levers(model) == MEASURED[problem_type]


@pytest.mark.parametrize(
    ("problem_type", "hyperparameters", "expected"),
    [
        ("binary", {"partition_scale_with_n": False}, {"scale_with_n": False}),
        ("binary", {"probe_budget": 20_000}, {"budget": 20_000}),
        ("binary", {"probe_crossfit_folds": 0}, {"crossfit": 0}),
        ("binary", {"probe_prob_floor": "cv"}, {"floor": "cv"}),
        ("multiclass", {"probe_crossfit_folds": 5}, {"crossfit": 5}),
        ("multiclass", {"probe_prob_floor": None}, {"floor": None}),
        (
            "multiclass",
            {"partition_scale_with_n": True, "probe_budget": 50_000},
            {"scale_with_n": True, "budget": 50_000},
        ),
        ("regression", {"probe_crossfit_folds": 5}, {"crossfit": 5}),
        (
            "regression",
            {"partition_scale_with_n": True, "probe_budget": "all"},
            {"scale_with_n": True, "budget": "all"},
        ),
    ],
)
def test_explicit_hyperparameters_override_the_entry_levers(problem_type, hyperparameters, expected, tmp_path):
    model = make(problem_type, tmp_path, **hyperparameters)
    resolve(model)
    assert levers(model) == {**MEASURED[problem_type], **expected}


@pytest.mark.parametrize(
    ("variable", "value", "problem_type", "expected"),
    [
        ("FINTFM_PARTITION_SCALE_N", "off", "binary", {"scale_with_n": False}),
        ("FINTFM_PARTITION_SCALE_N", "on", "regression", {"scale_with_n": True}),
        ("FINTFM_PROBE_BUDGET", "30000", "binary", {"budget": 30_000}),
        ("FINTFM_PROBE_BUDGET", "30000", "multiclass", {"budget": 30_000}),
        ("FINTFM_PROB_FLOOR", "off", "multiclass", {"floor": None}),
    ],
)
def test_environment_overrides_the_entry_levers(monkeypatch, variable, value, problem_type, expected, tmp_path):
    monkeypatch.setenv(variable, value)
    model = make(problem_type, tmp_path)
    resolve(model)
    assert levers(model) == {**MEASURED[problem_type], **expected}


@pytest.mark.parametrize("problem_type", sorted(ENTRY))
def test_gated_levers_default_to_the_entry_per_problem_type(problem_type, tmp_path):
    model = make(problem_type, tmp_path)
    resolve(model)
    assert gated(model) == MEASURED_GATED[problem_type]


@pytest.mark.parametrize(
    ("problem_type", "hyperparameters", "expected"),
    [
        ("multiclass", {"probe_onehot": None}, {"onehot": None}),
        ("multiclass", {"probe_onehot": True}, {"onehot": {**ONEHOT_ENTRY, "max_levels": None}}),
        ("multiclass", {"probe_onehot": {"max_levels": 8}}, {"onehot": {**ONEHOT_ENTRY, "max_levels": 8}}),
        ("regression", {"reg_readout": "ridge"}, {"readout": "ridge"}),
        ("regression", {"reg_rff_max_rows": 20_000}, {"rff_max_rows": 20_000}),
        ("regression", {"reg_rff_max_rows": None}, {"rff_max_rows": None}),
    ],
)
def test_explicit_hyperparameters_override_the_gated_levers(problem_type, hyperparameters, expected, tmp_path):
    model = make(problem_type, tmp_path, **hyperparameters)
    resolve(model)
    assert gated(model) == {**MEASURED_GATED[problem_type], **expected}


@pytest.mark.parametrize(
    ("variable", "value", "problem_type", "expected"),
    [
        ("FINTFM_PROBE_ONEHOT", "off", "multiclass", {"onehot": None}),
        (
            "FINTFM_PROBE_ONEHOT",
            '{"min_count": 5, "max_columns": 32}',
            "multiclass",
            {"onehot": {"min_count": 5, "max_columns": 32, "max_levels": None}},
        ),
        ("FINTFM_REG_READOUT", "ridge", "regression", {"readout": "ridge"}),
        ("FINTFM_REG_RFF_MAX_ROWS", "20000", "regression", {"rff_max_rows": 20_000}),
        ("FINTFM_REG_RFF_MAX_ROWS", "none", "regression", {"rff_max_rows": None}),
    ],
)
def test_environment_overrides_the_gated_levers(monkeypatch, variable, value, problem_type, expected, tmp_path):
    monkeypatch.setenv(variable, value)
    model = make(problem_type, tmp_path)
    resolve(model)
    assert gated(model) == {**MEASURED_GATED[problem_type], **expected}


def test_explicit_hyperparameter_beats_the_gated_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("FINTFM_PROBE_ONEHOT", "off")
    monkeypatch.setenv("FINTFM_REG_READOUT", "ridge")
    mc = make("multiclass", tmp_path, probe_onehot=True)
    resolve(mc)
    assert gated(mc)["onehot"] is not None
    reg = make("regression", tmp_path, reg_readout="rff")
    resolve(reg)
    assert gated(reg)["readout"] == "rff"
