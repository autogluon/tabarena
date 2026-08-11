from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSmokeTest:
    """Per-model overrides for the generic model smoke test (``test_all_models.py``).

    The default reproduces the behaviour of the old per-model ``test_<model>.py``
    files: an empty hyperparameters dict (the model's own defaults) and all
    problem types. Only models that need *fast* toy hyperparameters or a
    restricted set of problem types appear in ``SMOKE_OVERRIDES`` below.

    Attributes:
        hyperparameters: passed to ``FitHelper.verify_model(model_hyperparameters=...)``.
        problem_types: restricts the tested problem types; ``None`` tests all
            of binary + multiclass + regression (AutoGluon's default).
        verify_single_prediction_equivalent_to_multi: whether predicting one row must match
            that row's value from a batched predict (AutoGluon's default is True, at
            ``atol=1e-5``). Set False only for a model shown to satisfy it on CPU and to
            miss it on GPU, and say so in a comment.
    """

    hyperparameters: dict = field(default_factory=dict)
    problem_types: tuple[str, ...] | None = None
    verify_single_prediction_equivalent_to_multi: bool = True


# Keyed by the registry method name (``MethodMetadata.method`` -- the same key
# used by ``get_model_registry()``). These exist only to keep the smoke test
# fast (tiny epochs / single estimator), not to exercise real configs. cpu/gpu
# variants of the same model share params; the gpu variant only runs when a
# CUDA device is available (see ``test_all_models.py``).
#
# Adding a model: only add an entry here if its smoke fit needs non-default
# (faster) hyperparameters or a restricted problem-type set. Otherwise the
# default (empty params, all problem types) is used automatically.
SMOKE_OVERRIDES: dict[str, ModelSmokeTest] = {
    "PerpetualBooster": ModelSmokeTest({"iteration_limit": 10, "budget": 0.1}),
    "ChimeraBoost": ModelSmokeTest({"n_estimators": 100}),
    "ModernNCA": ModelSmokeTest({"n_epochs": 10}),
    "ModernNCA_GPU": ModelSmokeTest({"n_epochs": 10}),
    "RealMLP": ModelSmokeTest({"n_epochs": 10}),
    "RealMLP_GPU": ModelSmokeTest({"n_epochs": 10}),
    "TabM": ModelSmokeTest({"n_epochs": 10, "tabm_k": 2, "n_bins": 8, "num_emb_type": "none"}),
    "TabPFN-v2.6": ModelSmokeTest({"n_estimators": 1}),
    "RealTabPFN-v2.5": ModelSmokeTest({"n_estimators": 1}),
    "TabPFN-3": ModelSmokeTest({"n_estimators": 1, "device": "cpu"}),
    "TabPFN-Wide": ModelSmokeTest({"device": "cpu"}),
    "TabICL_GPU": ModelSmokeTest({"n_estimators": 1}),
    "TabICLv2": ModelSmokeTest({"n_estimators": 1}),
    # Single-row vs batched predictions agree exactly on CPU but drift ~2.4e-4 on GPU (the
    # tolerance is 1e-5), so the mismatch is float non-determinism in the CUDA kernels rather
    # than batch-dependent preprocessing. Verified by running `FitHelper.verify_model` with
    # CUDA_VISIBLE_DEVICES="" — it passes with the check on.
    "TabSwift": ModelSmokeTest({"n_estimators": 1}, verify_single_prediction_equivalent_to_multi=False),
    "TabSTAR": ModelSmokeTest({"max_epochs": 1}),
    "TabFM": ModelSmokeTest({"n_estimators": 1}),
    "Nori": ModelSmokeTest(problem_types=("regression",)),
    "Nori-30M": ModelSmokeTest(problem_types=("regression",)),
    "iLTM": ModelSmokeTest({"finetuning_max_steps": 1, "n_ensemble": 1, "tree_n_estimators": 1}),
    "OrionMSP": ModelSmokeTest({"n_estimators": 1}),
    "EXAONE-Tabular": ModelSmokeTest({"ensemble_count": 1}),
}


def smoke_for(method: str) -> ModelSmokeTest:
    """Return the smoke-test config for a registry ``method``, or the default."""
    return SMOKE_OVERRIDES.get(method, ModelSmokeTest())
