from __future__ import annotations

import logging
import tempfile
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.tabular import __version__
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models._shared_weights_model import CheckpointSpec, SharedWeightsModelMixin, SharedWeightsSpec

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.models._weights import WeightsKey

logger = logging.getLogger(__name__)


class TabICLModelBase(SharedWeightsModelMixin, AbstractTorchModel):
    """TabICL is a foundation model for tabular data using in-context learning
    that is scalable to larger datasets than TabPFNv2. It is pretrained purely on synthetic data.
    TabICL currently only supports classification tasks.

    TabICL is one of the top performing methods overall on TabArena-v0.1: https://tabarena.ai

    The checkpoint's network is shared through the weights registry (see
    :mod:`tabarena.models._shared_weights_model`). Configurations with ``kv_cache`` (the module-level
    KV cache written by ``forward_with_cache`` is per estimator) or a user ``model_path`` keep the
    library's own per-estimator load. The module's inference managers are reconfigured from the
    calling estimator on every forward, so the children of a bag use it one after another.

    Paper: TabICL: A Tabular Foundation Model for In-Context Learning on Large Data
    Authors: Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan
    Codebase: https://github.com/soda-inria/tabicl
    License: BSD-3-Clause
    """

    ag_key = "NOTSET"
    #: ``import tabicl`` pulls in both estimators, torch, sklearn and the Hub client; the second entry
    #: is TabArena's registry-aware estimator module, imported lazily by the fit.
    warmup_modules: ClassVar[tuple[str, ...]] = ("tabicl", "tabarena.models.tabicl._estimators")
    #: Cheapness knob for the warm-up dummy fit; the number of ensemble views never touches the network.
    warmup_dummy_fit_hyperparameters: ClassVar[dict] = {"n_estimators": 1}
    ag_name = "NOTSET"
    ag_priority = 65
    seed_name = "random_state"

    default_classification_model: str | None = None
    default_regression_model: str | None = None
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1

    shared_weights_spec: ClassVar[SharedWeightsSpec] = SharedWeightsSpec(
        library="tabicl",
        checkpoint=CheckpointSpec(repo_id="jingang/TabICL", filename_param="checkpoint_version"),
        default_params=lambda cls: {
            "checkpoint_version": (cls.default_classification_model, cls.default_regression_model)
        },
        disable_when=("kv_cache", "model_path"),
        download_param="allow_auto_download",
        network_attr="model_",
        detach_attr="library",
        seam="load_model",
        post_attach_calls=("_build_inference_config",),
        device_attrs=(
            ("device_", "torch"),
            ("device", "str"),
            ("inference_config_.COL_CONFIG.device", "torch"),
            ("inference_config_.ROW_CONFIG.device", "torch"),
            ("inference_config_.ICL_CONFIG.device", "torch"),
        ),
    )
    # Better to refit the model for faster inference and similar quality as the bag.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"refit_folds": True}

    @classmethod
    def _build_shared_weights(cls, key: WeightsKey):
        from tabarena.models.tabicl._estimators import build_module

        return build_module(key.checkpoint, key.device)

    def get_model_cls(self):
        from tabarena.models.tabicl._estimators import estimator_cls

        return estimator_cls(self.shared_weights_spec.variant_for(self.problem_type))

    def get_checkpoint_version(self, hyperparameter: dict) -> str:
        clf_checkpoint = self.default_classification_model
        reg_checkpoint = self.default_regression_model

        # Resolve HPO
        if "checkpoint_version" in hyperparameter:
            if isinstance(hyperparameter["checkpoint_version"], str):
                clf_checkpoint = hyperparameter["checkpoint_version"]
                reg_checkpoint = hyperparameter["checkpoint_version"]
            elif isinstance(hyperparameter["checkpoint_version"], tuple):
                clf_checkpoint = hyperparameter["checkpoint_version"][0]
                reg_checkpoint = hyperparameter["checkpoint_version"][1]
            else:
                raise ValueError(
                    "checkpoint_version hyperparameter must be either a string or a tuple of two strings (clf, reg).",
                )

        if self.problem_type in ["binary", "multiclass"]:
            return clf_checkpoint

        return reg_checkpoint

    # TODO: is this still correct for TabICLv2?
    @staticmethod
    def _get_batch_size(n_cells: int):
        if n_cells <= 4_000_000:
            return 8
        if n_cells <= 6_000_000:
            return 4
        if n_cells <= 500_000_000:
            return 2
        return 1

    @staticmethod
    def _get_n_estimators_override(n_rows: int) -> int | None:
        # Avoid OOM and time issues
        if n_rows >= 700_000:
            return 1
        return None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        try:
            import tabicl  # noqa: F401
        except ImportError as err:
            logger.log(
                40,
                f"\tFailed to import tabicl! To use the TabICL model, "
                f"do: `pip install autogluon.tabular[tabicl]=={__version__}`.",
            )
            raise err

        device = self._resolve_fit_device(num_gpus)

        model_cls = self.get_model_cls()
        hyp = self._get_model_params()
        hyp["batch_size"] = hyp.get(
            "batch_size",
            self._get_batch_size(X.shape[0] * X.shape[1]),
        )
        hyp["checkpoint_version"] = self.get_checkpoint_version(hyperparameter=hyp)

        n_estimators_override = self._get_n_estimators_override(n_rows=X.shape[0])
        if n_estimators_override is not None:
            hyp["n_estimators"] = n_estimators_override

        # Unique per-instance offload dir inside the OS tempdir (node-local on
        # clusters). tabicl removes the offloaded files itself;
        # the empty dir is left to the OS tempdir reaper.
        disk_offload_dir = tempfile.mkdtemp(prefix="tabicl_")

        key, payload = self._acquire_shared_weights(device=device)
        self.model = model_cls(
            **hyp,
            device=device,
            n_jobs=num_cpus,
            disk_offload_dir=disk_offload_dir,
            verbose=X.shape[0] > 250_000,
            inference_config=dict(COL_CONFIG=dict(cpu_safety_factor=0.75)),
        )
        if key is not None:
            self.model.use_shared_weights(key, payload, type(self)._load_shared_weights)
        X = self.preprocess(X, y=y)
        self.model = self.model.fit(
            X=X,
            y=y,
        )

    # TODO: move memory estimate to specific models below.
    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Heuristic memory estimate that is very primitive.
        Can be vastly improved.
        """
        if hyperparameters is None:
            hyperparameters = {}

        dataset_size_mem_est = 3 * get_approximate_df_mem_usage(X).sum()  # roughly 3x DataFrame memory size
        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead

        n_rows = X.shape[0]
        n_features = X.shape[1]
        batch_size = hyperparameters.get(
            "batch_size",
            cls._get_batch_size(X.shape[0] * X.shape[1]),
        )
        embedding_dim = 128
        bytes_per_float = 4
        model_mem_estimate = 2 * batch_size * embedding_dim * bytes_per_float * (4 + n_rows) * n_features

        model_mem_estimate *= 1.3  # add 30% buffer

        # TODO: Observed memory spikes above expected values on large datasets, increasing mem estimate to compensate
        model_mem_estimate *= 2.0  # Note: 1.5 is not large enough, still gets OOM

        return model_mem_estimate + dataset_size_mem_est + baseline_overhead_mem_est

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def checkpoint_search_space(cls) -> list[str | tuple[str, str]]:
        """The checkpoint alternatives of this variant's search space, default first (from the spec)."""
        return list(cls.shared_weights_spec.checkpoint_choices["checkpoint_version"])


class TabICLModel(TabICLModelBase):
    """TabICLv1.1 model as used on TabArena."""

    ag_key = "TA-TABICL"
    ag_name = "TA-TabICL"

    default_classification_model: str | None = "tabicl-classifier-v1.1-20250506.ckpt"
    _supported_problem_types = ["binary", "multiclass"]

    shared_weights_spec: ClassVar[SharedWeightsSpec] = replace(
        TabICLModelBase.shared_weights_spec,
        checkpoint_choices={
            "checkpoint_version": (
                "tabicl-classifier-v1.1-20250506.ckpt",
                "tabicl-classifier-v1-20250208.ckpt",
            )
        },
    )

    def _set_default_params(self):
        default_params = {
            "n_estimators": 32,  # default of TabICLv1
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)


class TabICLv2Model(TabICLModelBase):
    """TabICLv2 model as used on TabArena."""

    ag_key = "TA-TABICLv2"
    ag_name = "TA-TabICLv2"

    default_classification_model: str | None = "tabicl-classifier-v2-20260212.ckpt"
    default_regression_model: str | None = "tabicl-regressor-v2-20260212.ckpt"
    _supported_problem_types = ["binary", "multiclass", "regression"]

    # TODO: search over v1 checkpoints too?
    shared_weights_spec: ClassVar[SharedWeightsSpec] = replace(
        TabICLModelBase.shared_weights_spec,
        checkpoint_choices={
            "checkpoint_version": (
                (
                    "tabicl-classifier-v2-20260212.ckpt",
                    "tabicl-regressor-v2-20260212.ckpt",
                ),
            )
        },
    )

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Memory estimate for v2 and large data is not supported yet.
        We ignore it for now, moreover as we refit_folds=True, there is no benefit yet.

        Problems are: GPU memory est, how to handle off-loading logic, ...
        """
        dataset_size_mem_est = 3 * get_approximate_df_mem_usage(X).sum()
        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead
        return dataset_size_mem_est + baseline_overhead_mem_est
