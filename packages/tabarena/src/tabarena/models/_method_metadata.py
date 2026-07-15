from __future__ import annotations

import io
import json
import os
import re
import warnings
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import pandas as pd
import yaml

from tabarena.benchmark.result.raw_loading import get_info_from_result, load_raw, results_to_holdout
from tabarena.loaders import get_tabarena_cache_root
from tabarena.repository.evaluation_repository import EvaluationRepository
from tabarena.repository.generate_repo import generate_repo_from_results_lst
from tabarena.utils.pickle_utils import fetch_all_pickles

if TYPE_CHECKING:
    from tabarena.benchmark.result import BaselineResult
    from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
    from tabarena.models._artifacts.downloader import MethodDownloader
    from tabarena.models._artifacts.uploader import MethodUploader


class MethodType(StrEnum):
    """Canonical set of valid ``MethodMetadata.method_type`` values.

    A ``str`` mixin so members compare equal to (and serialize as) their plain-string value
    — ``MethodType.CONFIG == "config"`` — which keeps YAML and round-trip behavior unchanged.
    This is the single source of truth for the ``method_type`` values the
    :class:`MethodMetadata` constructor accepts (see :meth:`values`).
    """

    CONFIG = "config"
    BASELINE = "baseline"
    PORTFOLIO = "portfolio"

    @classmethod
    def values(cls) -> list[str]:
        """The valid ``method_type`` strings, in declaration order."""
        return [m.value for m in cls]


#: String form of :class:`MethodType`, for annotating ``method_type`` fields and parameters.
#: Mirrors the enum members above (kept honest at runtime by the ``MethodType.values()`` check in
#: :meth:`MethodMetadata.__post_init__`).
MethodTypeLiteral = Literal["config", "baseline", "portfolio"]


# FIXME: Implement `best` and `best-N`
@dataclass(eq=False)
class MethodMetadata:
    """Identity, artifact layout, and storage/transport config for one benchmarked method.

    A ``@dataclass`` (``eq=False`` to preserve identity-based equality/hashing, matching the
    pre-dataclass class): the field declarations below are the canonical schema, and
    :meth:`to_info_dict` derives the serialized YAML / info-table surface from those fields (an
    allowlist) rather than from ``self.__dict__``. :meth:`__post_init__` fills the derived
    defaults (``suite`` <- ``method``, ``model_key`` <- ``ag_key``, ``can_hpo`` <-
    ``method_type``, ``display_name``) and validates the inputs.
    """

    #: Whether this method's artifacts live in memory (no on-disk/S3 backing). False for the
    #: disk-backed base class; True for :class:`InMemoryMethodMetadata`. Arena contexts treat
    #: in-memory methods as the locally-produced "new" results (e.g. ``compare`` resolves
    #: ``only_valid_tasks=True`` against them). A ``ClassVar`` (not a dataclass field), so it
    #: stays out of :meth:`to_info_dict` / the metadata info table.
    is_in_memory: ClassVar[bool] = False

    # Fields are grouped by how a method author supplies them: (1) identity & on-disk location,
    # (2) inferable from raw results, (3) manual, (4) purely informative. Order is otherwise free —
    # every caller constructs MethodMetadata by keyword, and to_info_dict is field-driven.

    # -- (1) Identity & on-disk location ------------------------------------------------------
    #: ``method``, ``suite``, ``artifact_dir``, and ``cache_root`` determine where this method's
    #: artifacts live on disk (see :attr:`path`):
    #:
    #:     <tabarena_cache>/artifacts/<suite>/methods/<method>/   (default cache layout)
    #:     <cache_root>/artifacts/<suite>/methods/<method>/       (explicit ``cache_root`` override)
    #:     <artifact_dir>/                                        (explicit ``artifact_dir`` override)
    #:
    #: ``method`` (the only required field) is the unique method identifier, e.g. ``"TabPFN-3"``.
    #: ``suite`` is the artifact set the method belongs to — normally the dated benchmark run,
    #: e.g. ``"tabarena-2026-05-13"`` — surfaced downstream as the ``ta_suite`` column;
    #: ``(method, suite)`` is the unique key within a :class:`MethodMetadataCollection`. ``suite``
    #: defaults to ``method`` (in :meth:`__post_init__`) when omitted, so a one-off method needs
    #: only ``method``.
    #:
    #: ``cache_root`` and ``artifact_dir`` are mutually-exclusive, runtime-only path overrides
    #: (kept out of :meth:`to_info_dict`, so neither serializes into the committed ``metadata.yaml``
    #: or the info table). ``cache_root`` overrides the root of the TabArena cache this method
    #: resolves against — the dir under which ``artifacts/<suite>/methods/<method>/`` lives —
    #: defaulting to the global :func:`get_tabarena_cache_root` when unset. Set it to read a method
    #: from a *specific* cache (e.g. a teammate's cache on a shared drive) without mutating global
    #: state; the loaded instance carries it, so deferred ``load_results`` / :attr:`path` keep
    #: resolving into that cache. ``artifact_dir`` instead points *directly* at this method's
    #: artifact directory (the dir holding ``metadata.yaml`` + ``results/``), bypassing the
    #: ``artifacts/<suite>/methods/`` layout entirely — e.g. a committed copy in a repo's ``data/``
    #: folder; when set, :attr:`path` is ``artifact_dir`` itself.
    method: str
    suite: str | None = None
    artifact_dir: str | Path | None = None
    cache_root: str | Path | None = None

    # -- (2) Inferable from raw results -------------------------------------------------------
    #: Populated automatically by :meth:`from_raw` (and the ``run_process_local_raw_data.py``
    #: inspector) from the raw result frame, so they can be left to inference when authoring from
    #: raw artifacts. ``model_key`` has no independent raw signal and is derived in
    #: :meth:`__post_init__` from ``ag_key``; ``can_hpo`` is inferred from the raw results by
    #: :meth:`from_raw` and only falls back to a ``method_type``-based default in
    #: :meth:`__post_init__` when left ``None`` (see its own note below).
    method_type: MethodTypeLiteral = "config"
    ag_key: str | None = None
    model_key: str | None = None
    config_default: str | None = None
    #: Whether this method has more than one config result — i.e. whether it can be HPO-tuned in
    #: simulation, and so supports the derived tuned / tuned+ensemble methods. This is NOT an
    #: intrinsic model capability: it literally means ">1 config was run". :meth:`from_raw` sets it
    #: from the count of distinct configs in the raw results (1 config -> ``False``, >1 -> ``True``),
    #: so the *same* model can have ``can_hpo=True`` in one suite and ``False`` in another purely
    #: from how many configs that suite ran (e.g. TabDPT: 200 configs in TabArena -> ``True``, 1
    #: config in BeyondArena -> ``False`` — same model, not a capability pin). When left ``None`` it
    #: defaults in :meth:`__post_init__` to ``method_type == "config"``.
    can_hpo: bool | None = None
    compute: Literal["cpu", "gpu"] = "cpu"
    #: Whether the model was trained with bagging (cross-validation across folds). When ``True``,
    #: the raw data contains per-fold test *and* validation predictions, and the reported test
    #: predictions are the average of the per-fold test predictions. Config-only by convention
    #: (enforced in :meth:`__post_init__`) — baselines/portfolios are recorded as ``False``.
    is_bag: bool = False
    #: Whether an artifact of each tier exists for this method. Default ``True``: configs and
    #: baselines have raw + processed + results. Set ``False`` only when the tier never exists for
    #: the method at all — e.g. a portfolio, which only ever has results (no raw/processed). It is
    #: about whether the artifact exists, not whether it is currently uploaded (so an as-yet-unhosted
    #: config is still all ``True``). Descriptive metadata only — serialized into the info table,
    #: not used to gate loading.
    has_raw: bool = True
    has_processed: bool = True
    has_results: bool = True

    # -- (3) Manual (not inferable) -----------------------------------------------------------
    #: Specified by hand when relevant: display-name overrides (``name`` / ``name_suffix``) and
    #: the storage/transport config below (where and how artifacts are cached and uploaded).
    name: str | None = None
    name_suffix: str | None = None
    #: Storage backend for this method's artifacts. ``None`` (the default) infers it in
    #: :meth:`__post_init__`: ``"r2"`` when ``cache_kwargs`` carries a remote location
    #: (``bucket`` + ``prefix``), else ``"local"``. ``"s3"``/``"r2"`` require that location;
    #: ``"local"`` forbids it.
    cache_type: Literal["local", "r2", "s3"] | None = None
    #: All ``cache_type``-specific configuration, kept out of the core schema so casual users (who
    #: only ever use ``"local"``) aren't shown remote-storage knobs and so future backends can add
    #: their own keys without new fields. For ``"s3"``/``"r2"`` it carries the remote location
    #: ``{"bucket": ..., "prefix": ...}`` (the s3-compatible coordinates used by both, formerly the
    #: top-level ``s3_bucket`` / ``s3_prefix`` fields); ``"s3"`` may additionally set
    #: ``{"upload_as_public": True}`` for a public-read ACL, and ``"r2"`` may set
    #: ``{"base_url": ...}`` to override the public download domain (default
    #: ``"https://data.tabarena.ai/"``). The uploader/downloader factories read what they need from
    #: here. Empty for ``"local"``.
    cache_kwargs: dict = field(default_factory=dict)

    # -- (4) Purely informative ---------------------------------------------------------------
    # Descriptive only — no effect on behavior, paths, or loading.
    #: The date the method was run (the benchmark run that produced its results), as a
    #: ``"YYYY-MM-DD"`` string. Validated in :meth:`__post_init__` when set.
    date: str | None = None
    #: The date the method/algorithm was first introduced (paper or library release), e.g.
    #: ``"2017-06"`` or ``"2001"`` — distinct from :attr:`date` (when it was run on TabArena).
    #: Precision varies: year for classical methods, month for arXiv-anchored ones. Usually
    #: anchored to :attr:`reference_url`. Validated in :meth:`__post_init__` when set.
    date_introduced: str | None = None
    reference_url: str | None = None
    display_name: str | None = None
    #: Whether this method's results are verified / signed-off. Default ``True``; set ``False`` for
    #: methods that are not yet verified (e.g. newly-added or not-yet-released models). A manual
    #: trust flag, not (yet) read anywhere in code.
    verified: bool = True

    def __post_init__(self):
        if self.suite is None:
            self.suite = self.method
        if self.model_key is None:
            self.model_key = self.ag_key
        if self.can_hpo is None:
            self.can_hpo = self.method_type == "config"
        self.artifact_dir = Path(self.artifact_dir) if self.artifact_dir is not None else None
        self.cache_root = Path(self.cache_root) if self.cache_root is not None else None
        if self.artifact_dir is not None and self.cache_root is not None:
            raise AssertionError(
                "Specify at most one of `artifact_dir` (a method's exact artifact directory) and "
                "`cache_root` (a cache root to resolve <artifacts/suite/methods/method> under); "
                f"got both (method={self.method!r})."
            )

        assert isinstance(self.method, str)
        assert len(self.method) > 0
        assert isinstance(self.suite, str)
        assert len(self.suite) > 0
        # Accept a MethodType member or its plain string, normalize to the string for storage,
        # then validate against the canonical set (the constructor's previous hardcoded list).
        if isinstance(self.method_type, MethodType):
            self.method_type = self.method_type.value
        assert self.method_type in MethodType.values(), (
            f"Unknown method_type: {self.method_type!r}. Valid values: {MethodType.values()}"
        )
        assert self.compute in ["cpu", "gpu"]
        # When set, `date` must be a real calendar date in YYYY-MM-DD format.
        if self.date is not None:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.date):
                raise AssertionError(
                    f"date must be in 'YYYY-MM-DD' format, got {self.date!r} (method={self.method!r})."
                )
            try:
                datetime.strptime(self.date, "%Y-%m-%d")
            except ValueError as e:
                raise AssertionError(
                    f"date {self.date!r} is not a valid calendar date (method={self.method!r})."
                ) from e
        # When set, `date_introduced` allows variable precision: 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD'.
        if self.date_introduced is not None:
            if not re.fullmatch(r"\d{4}(-\d{2}){0,2}", self.date_introduced):
                raise AssertionError(
                    f"date_introduced must be 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD', "
                    f"got {self.date_introduced!r} (method={self.method!r})."
                )
        # Guard against arguments that belong to a different method_type, so a mismatched field is
        # surfaced at construction rather than silently ignored. `name` is the baseline/portfolio
        # display-name override; `ag_key` / `model_key` / `config_default` / `name_suffix` are
        # config-only, as are `can_hpo=True` and `is_bag=True` (only configs are bagged by
        # convention). (Checks run after the derived defaults above.)
        if self.name is not None and self.method_type == "config":
            raise AssertionError("Cannot specify `name` for method_type: 'config'.")
        if self.name is not None and self.name_suffix is not None:
            raise AssertionError("Must only specify one of `name` and `name_suffix`.")
        if self.method_type != "config":
            config_only_set = [
                field_name
                for field_name in ("ag_key", "model_key", "config_default", "name_suffix")
                if getattr(self, field_name) is not None
            ]
            if config_only_set:
                raise AssertionError(
                    f"Fields {config_only_set} are only valid for method_type='config', but "
                    f"method_type={self.method_type!r} (method={self.method!r})."
                )
            for flag in ("can_hpo", "is_bag"):
                if getattr(self, flag):
                    raise AssertionError(
                        f"{flag}=True is only valid for method_type='config', but "
                        f"method_type={self.method_type!r} (method={self.method!r})."
                    )

        if self.display_name is None:
            self.display_name = self._compute_display_name()

        assert isinstance(self.display_name, str)
        assert len(self.display_name) > 0

        # Resolve the storage backend: None infers "r2" when cache_kwargs carries a remote location
        # (bucket + prefix), else "local". (Reads cache_kwargs directly, not has_remote_cache,
        # which keys off cache_type.)
        ck_bucket = self.cache_kwargs.get("bucket")
        ck_prefix = self.cache_kwargs.get("prefix")
        has_remote_location = ck_bucket is not None and ck_prefix is not None
        if self.cache_type is None:
            self.cache_type = "r2" if has_remote_location else "local"
        assert self.cache_type in ("local", "r2", "s3"), f"Unknown `cache_type`: {self.cache_type!r}"
        # s3/r2 require a remote location in cache_kwargs (bucket + prefix); local must not have one.
        if self.cache_type in ("r2", "s3"):
            if not has_remote_location:
                raise AssertionError(
                    f"cache_type={self.cache_type!r} requires cache_kwargs to contain 'bucket' and "
                    f"'prefix' (method={self.method!r}, cache_kwargs={self.cache_kwargs!r})."
                )
        elif ck_bucket is not None or ck_prefix is not None:
            raise AssertionError(
                f"cache_type='local' must not set 'bucket'/'prefix' in cache_kwargs "
                f"(method={self.method!r}, cache_kwargs={self.cache_kwargs!r})."
            )

    def _compute_display_name(self) -> str:
        if self.name is not None:
            display_name = self.name
        elif self.config_type is not None:
            display_name = self.config_type
        else:
            display_name = self.method
        return display_name

    @property
    def config_type(self) -> str | None:
        if self.method_type != "config":
            return None
        if self.name_suffix is not None:
            return f"{self.model_key}{self.name_suffix}"
        return self.model_key

    # -- type-specific constructors -----------------------------------------------------------
    # Thin wrappers over the dataclass constructor that expose only the arguments relevant to each
    # method_type, so a caller is never presented with fields that belong to a different type.
    # Each sets ``method_type`` and forwards the shared fields (identity, ``compute``, ``has_*``,
    # storage/transport, informative) via ``**kwargs``; ``__post_init__`` still validates the
    # result. Orthogonal to :meth:`tabarena_legacy_s3`, the legacy public-s3 *storage* preset.

    @classmethod
    def config(
        cls,
        *,
        method: str,
        ag_key: str | None = None,
        model_key: str | None = None,
        config_default: str | None = None,
        name_suffix: str | None = None,
        can_hpo: bool | None = None,
        is_bag: bool = False,
        **kwargs,
    ) -> Self:
        """A ``config`` method (a tunable model with one or more configs; the default type).

        Exposes the config-only fields (``ag_key`` / ``model_key`` / ``config_default`` /
        ``name_suffix`` / ``can_hpo`` / ``is_bag``); ``name`` is rejected (it is
        baseline/portfolio-only).
        """
        return cls(
            method=method,
            method_type="config",
            ag_key=ag_key,
            model_key=model_key,
            config_default=config_default,
            name_suffix=name_suffix,
            can_hpo=can_hpo,
            is_bag=is_bag,
            **kwargs,
        )

    @classmethod
    def baseline(cls, *, method: str, name: str | None = None, **kwargs) -> Self:
        """A ``baseline`` method (e.g. an AutoGluon preset).

        Exposes ``name`` (the display-name override); the config-only fields are rejected.
        """
        return cls(method=method, method_type="baseline", name=name, **kwargs)

    @classmethod
    def portfolio(cls, *, method: str, name: str | None = None, has_raw: bool = False, **kwargs) -> Self:
        """A ``portfolio`` method (a fixed selection/ensemble over configs).

        Exposes ``name`` and defaults ``has_raw=False`` (portfolios usually have no raw
        artifacts); the config-only fields are rejected.
        """
        return cls(method=method, method_type="portfolio", name=name, has_raw=has_raw, **kwargs)

    # TODO: Also support baseline methods
    @classmethod
    def from_raw(
        cls,
        results_lst: list[BaselineResult],
        method: str | None = None,
        suite: str | None = None,
        config_default: str | None = None,
        compute: Literal["cpu", "gpu"] | None = None,
        artifact_dir: str | Path | None = None,
    ) -> Self:
        result_lst_dict = []

        for r in results_lst:
            cur_result = get_info_from_result(result=r)
            result_lst_dict.append(cur_result)
        result_df = pd.DataFrame(result_lst_dict)

        unique_method_types = result_df["method_type"].unique()
        assert len(unique_method_types) == 1
        method_type = unique_method_types[0]

        if method_type == "config":
            method_metadata = cls._from_raw_config(
                result_df=result_df,
                method=method,
                suite=suite,
                config_default=config_default,
                compute=compute,
                artifact_dir=artifact_dir,
            )
        elif method_type == "baseline":
            method_metadata = cls._from_raw_baseline(
                result_df=result_df,
                method=method,
                suite=suite,
                compute=compute,
                artifact_dir=artifact_dir,
            )
        else:
            raise ValueError(f"Unknown method_type: {method_type}")

        return method_metadata

    @classmethod
    def _from_raw_baseline(
        cls,
        result_df: pd.DataFrame,
        method: str | None = None,
        suite: str | None = None,
        compute: Literal["cpu", "gpu"] | None = None,
        artifact_dir: str | Path | None = None,
    ) -> Self:
        unique_method_types = result_df["method_type"].unique()
        assert len(unique_method_types) == 1
        method_type = unique_method_types[0]

        assert method_type == "baseline"

        unique_methods = result_df["framework"].unique()
        assert len(unique_methods) == 1
        if method is None:
            method = unique_methods[0]

        unique_num_gpus = result_df["num_gpus"].unique()
        assert len(unique_num_gpus) == 1
        num_gpus = unique_num_gpus[0]

        if compute is None:
            compute: Literal["cpu", "gpu"] = "cpu" if num_gpus == 0 else "gpu"

        if suite is None:
            suite = method

        return cls.baseline(
            method=method,
            suite=suite,
            compute=compute,
            artifact_dir=artifact_dir,
        )

    @classmethod
    def compute_method_name(
        cls,
        method: str,
        method_type: str,
        method_subtype: str | None,
        config_type: str | None,
        display_name: str | None,
    ) -> str:
        subtype_to_suffix_map = {
            "default": " (default)",
            "tuned": " (tuned)",
            "tuned_ensemble": " (tuned + ensemble)",
        }
        valid_method_types = ["config", "baseline", "hpo", "portfolio"]
        if method_type not in valid_method_types:
            raise ValueError(f"Unknown {method_type=}. Valid values: {valid_method_types}")
        if pd.isna(display_name):
            if method_type in ["baseline", "portfolio"]:
                display_name = method
            else:
                assert isinstance(config_type, str)
                if method_subtype is None:
                    display_name = method
                else:
                    display_name = config_type

        name_suffix = None
        if method_type in ["config", "hpo"]:
            if method_subtype is None:
                name_suffix = None
            else:
                assert method_subtype in subtype_to_suffix_map, (
                    f"Unknown {method_subtype=}. Valid values: {list(subtype_to_suffix_map.keys())}"
                )
                name_suffix = subtype_to_suffix_map[method_subtype]
        if name_suffix:
            display_name = display_name + name_suffix
        return display_name

    @classmethod
    def _from_raw_config(
        cls,
        result_df: pd.DataFrame,
        method: str | None = None,
        suite: str | None = None,
        config_default: str | None = None,
        compute: Literal["cpu", "gpu"] | None = None,
        artifact_dir: str | Path | None = None,
    ) -> Self:
        unique_method_types = result_df["method_type"].unique()
        assert len(unique_method_types) == 1
        method_type = unique_method_types[0]

        assert method_type == "config"

        unique_model_types = result_df["model_type"].unique()
        assert len(unique_model_types) == 1, (
            f"MethodMetadata requires exactly 1 model type, found: {unique_model_types}"
        )
        # `model_key` is the simulation/comparison family key (what the repo groups configs by via
        # `model_type`); `ag_key` is the AutoGluon model-class key (maps back to the model impl) and
        # may differ — e.g. after a re-key (name_prefix/model_key) that keeps the same backbone but
        # a distinct family. Source `model_key` from `model_type`, NOT from `ag_key` (the latter
        # would conflate the two and mis-key re-keyed families during simulate/compare).
        model_key = unique_model_types[0]

        unique_num_gpus = result_df["num_gpus"].unique()
        if len(unique_num_gpus) != 1:
            warnings.warn(
                f"MethodMetadata found more than one unique num_gpus value, found: {unique_num_gpus}. "
                "Using max number of groups as official compute value.",
                stacklevel=2,
            )
        num_gpus = unique_num_gpus.max()

        if compute is None:
            compute: Literal["cpu", "gpu"] = "cpu" if num_gpus == 0 else "gpu"

        unique_ag_key = result_df["ag_key"].unique()
        assert len(unique_ag_key) == 1
        ag_key = unique_ag_key[0]

        is_bag = bool(result_df["is_bag"].any())

        unique_name_prefix = result_df["name_prefix"].unique()
        assert len(unique_name_prefix) == 1
        name_prefix = unique_name_prefix[0]

        unique_methods = result_df["framework"].unique()
        if len(unique_methods) == 1:
            _config_default = unique_methods[0]
            can_hpo = False
        else:
            _config_default = None
            can_hpo = True
        if config_default is None:
            config_default = _config_default

        if method is None:
            method = name_prefix

        if suite is None:
            suite = method

        return cls.config(
            method=method,
            suite=suite,
            compute=compute,
            config_default=config_default,
            ag_key=ag_key,
            model_key=model_key,
            can_hpo=can_hpo,
            is_bag=is_bag,
            artifact_dir=artifact_dir,
        )

    @classmethod
    def tabarena_legacy_s3(
        cls,
        *,
        method: str,
        suite: str,
        has_raw: bool = True,
        has_processed: bool = True,
        has_results: bool = True,
        bucket: str = "tabarena",
        prefix: str = "cache",
        **kwargs,
    ) -> Self:
        """Build a :class:`MethodMetadata` for a method whose artifacts live in the legacy public
        TabArena **S3** store.

        Fills the boilerplate shared by every such artifact — the ``tabarena`` / ``cache`` S3
        location, ``cache_type="s3"``, the public-read ACL (``cache_kwargs={"upload_as_public":
        True}``), and a fully-cached ``has_raw``/``has_processed``/``has_results`` — so callers
        specify only what is method-specific. ``cache_type`` and the ACL are fixed (this preset is
        s3-and-public by definition); ``has_*`` and ``bucket``/``prefix`` can be overridden
        via keyword. The single source of truth for the public-artifact preset that the
        per-artifact ``_common_*_kwargs`` dicts used to re-declare.
        """
        return cls(
            method=method,
            suite=suite,
            has_raw=has_raw,
            has_processed=has_processed,
            has_results=has_results,
            cache_type="s3",
            cache_kwargs={"bucket": bucket, "prefix": prefix, "upload_as_public": True},
            **kwargs,
        )

    @property
    def has_remote_cache(self) -> bool:
        """Whether this method's artifacts live in a remote store (an ``"s3"``/``"r2"`` backend)
        rather than ``cache_type="local"`` (no remote backing). ``__post_init__`` guarantees a
        remote backend always has a remote location (``bucket`` + ``prefix``).
        """
        return self.cache_type in ("r2", "s3")

    @property
    def has_configs_hyperparameters(self) -> bool:
        return self.method_type == "config"

    @property
    def _path_root(self) -> Path:
        return self.path_cache_root / "artifacts"

    @property
    def path_cache_root(self) -> Path:
        """The TabArena cache root this method resolves against: its ``cache_root`` override if
        set, else the global :func:`get_tabarena_cache_root`.
        """
        return self.cache_root if self.cache_root is not None else get_tabarena_cache_root()

    @property
    def path(self) -> Path:
        """Root directory of this method's artifacts on disk.

        Default layout: ``<path_cache_root>/artifacts/<suite>/methods/<method>/``, derived from
        ``(suite, method)`` under :attr:`path_cache_root` (the ``cache_root`` override, else the
        global cache). When ``artifact_dir`` is set it *is* the artifact directory
        (``metadata.yaml`` + ``results/`` live directly under it) and is returned as-is —
        ``suite``/``method``/``cache_root`` do not contribute — so committed copies can live at an
        arbitrary location.
        """
        if self.artifact_dir is not None:
            return self.artifact_dir
        return self._path_root / self.suite / "methods" / self.method

    @property
    def path_raw(self) -> Path:
        return self.path / "raw"

    @property
    def path_processed(self) -> Path:
        return self.path / "processed"

    @property
    def path_results(self) -> Path:
        return self.path / "results"

    @property
    def path_raw_exists(self) -> bool:
        return self.path_raw.is_dir()

    @property
    def path_processed_exists(self) -> bool:
        return self.path_processed.is_dir()

    @property
    def path_results_exists(self) -> bool:
        return self.path_results.is_dir()

    def path_results_hpo(self) -> Path:
        return self.path_results / "hpo_results.parquet"

    def path_results_model(self) -> Path:
        return self.path_results / "model_results.parquet"

    def path_results_portfolio(self) -> Path:
        return self.path_results / "portfolio_results.parquet"

    def path_results_hpo_trajectories(self) -> Path:
        return self.path_results / "hpo_trajectories.parquet"

    def relative_to_cache_root(self, path: Path) -> Path:
        return path.relative_to(self.path_cache_root)

    def relative_to_root(self, path: Path) -> Path:
        return path.relative_to(self._path_root)

    def relative_to_method(self, path: Path) -> Path:
        return path.relative_to(self.path)

    def method_downloader(
        self,
        cache_type: str = "auto",
        verbose: bool = False,
    ) -> MethodDownloader:
        if cache_type == "auto":
            cache_type = self.cache_type
        if not self.has_remote_cache:
            raise AssertionError(
                f"Tried to get MethodDownloader from MethodMetadata, but cache_type={self.cache_type!r} "
                f"has no remote store to download from."
                f"\n\t(method={self.method}, suite={self.suite}, cache_type={self.cache_type})"
                f"\nUse a remote backend ('s3'/'r2') with bucket + prefix to enable artifact download.",
            )

        bucket = self.cache_kwargs["bucket"]
        prefix = self.cache_kwargs["prefix"]
        if cache_type == "r2":
            from tabarena.models._artifacts.downloader_public_r2 import MethodDownloaderPublicR2

            return MethodDownloaderPublicR2(
                method_metadata=self,
                base_url=self.cache_kwargs.get("base_url", "https://data.tabarena.ai/"),
                r2_prefix=prefix,
                verbose=verbose,
                clear_dirs=False,
            )
        if cache_type == "s3":
            from tabarena.models._artifacts.downloader_s3 import MethodDownloaderS3

            return MethodDownloaderS3(
                method_metadata=self,
                s3_bucket=bucket,
                s3_prefix=prefix,
                verbose=verbose,
                clear_dirs=False,
            )
        raise ValueError(f"Invalid cache_type for downloads: {cache_type}")

    @staticmethod
    def r2_credentials_help() -> str:
        """How to obtain and set the R2 upload credentials (``R2_ACCOUNT_ID`` / ``R2_ACCESS_KEY_ID``
        / ``R2_SECRET_ACCESS_KEY``).

        Shared by :meth:`method_uploader`'s missing-credentials error and the upload script's
        dry-run output, so the instructions live in one place.
        """
        return (
            "Where to find these values in the Cloudflare R2 dashboard "
            "(https://dash.cloudflare.com/ -> R2 Object Storage):\n"
            "  - R2_ACCOUNT_ID: shown as 'Account ID' on the R2 overview page, "
            "and embedded in the dashboard URL (dash.cloudflare.com/<account_id>/r2).\n"
            "  - R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY: open 'Manage R2 API Tokens' "
            "-> 'Create API Token', choose the required permission (e.g. Object Read & "
            "Write) and bucket scope, then submit. Cloudflare displays both the Access "
            "Key ID and the Secret Access Key on the success page; the secret is shown "
            "only once, so copy it immediately. If you've lost an existing secret, "
            "create a new token (or roll the existing one) from the same page.\n"
            "\n"
            "For the official TabArena R2 account, go to "
            "https://dash.cloudflare.com/<account_id>/r2/api-tokens and create a new user "
            "API token with Object Read & Write permissions scoped to only the 'tabarena' "
            "bucket. If such a token already exists, use it instead and select 'Roll' to "
            "obtain new credentials for the API token. Once viewing the credentials, use "
            "the 'Access Key ID' and 'Secret Access Key' values for R2_ACCESS_KEY_ID and "
            "R2_SECRET_ACCESS_KEY respectively."
        )

    def method_uploader(self, cache_type: str = "auto") -> MethodUploader:
        if cache_type == "auto":
            cache_type = self.cache_type
        if not self.has_remote_cache:
            raise AssertionError(
                f"Tried to get MethodUploader from MethodMetadata, but cache_type={self.cache_type!r} "
                f"has no remote store to upload to."
                f"\n\t(method={self.method}, suite={self.suite}, cache_type={self.cache_type})"
                f"\nUse a remote backend ('s3'/'r2') with bucket + prefix to enable artifact upload.",
            )

        bucket = self.cache_kwargs["bucket"]
        prefix = self.cache_kwargs["prefix"]
        if cache_type == "r2":
            from tabarena.models._artifacts.uploader_r2 import MethodUploaderR2

            r2_env_vars = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
            missing = [v for v in r2_env_vars if not os.environ.get(v)]
            if missing:
                raise OSError(
                    f"Missing required environment variable(s) for R2 uploads: {missing}.\n"
                    f"Set {', '.join(r2_env_vars)} in your shell (or a .env file) before "
                    f"calling method_uploader() with cache_type='r2'.\n\n" + self.r2_credentials_help()
                )

            return MethodUploaderR2(
                method_metadata=self,
                r2_account_id=os.environ["R2_ACCOUNT_ID"],
                r2_bucket=bucket,
                r2_prefix=prefix,
                r2_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                r2_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            )
        if cache_type == "s3":
            from tabarena.models._artifacts.uploader_s3 import MethodUploaderS3

            return MethodUploaderS3(
                method_metadata=self,
                s3_bucket=bucket,
                s3_prefix=prefix,
                upload_as_public=self.cache_kwargs.get("upload_as_public", False),
            )
        raise ValueError(f"Invalid cache_type for uploads: {cache_type}")

    def load_model_results(self) -> pd.DataFrame:
        return pd.read_parquet(path=self.path_results_model())

    def load_hpo_results(self) -> pd.DataFrame:
        return pd.read_parquet(path=self.path_results_hpo())

    def load_portfolio_results(self) -> pd.DataFrame:
        return pd.read_parquet(path=self.path_results_portfolio())

    def load_results(self) -> pd.DataFrame:
        if self.method_type == "config":
            df_results = self.load_hpo_results()
        elif self.method_type == "baseline":
            df_results = self.load_model_results()
        elif self.method_type == "portfolio":
            df_results = self.load_portfolio_results()
        else:
            raise ValueError(f"Unknown method_type: {self.method_type} for method {self.method}")
        return df_results

    def path_configs_hyperparameters(self) -> Path:
        return self.path_processed / "configs_hyperparameters.json"

    def load_configs_hyperparameters(self, download: str | bool = "auto") -> dict[str, dict]:
        if download == "auto":
            try:
                return self.load_configs_hyperparameters(download=False)
            except FileNotFoundError:
                print(
                    f"Cache miss detected for configs_hyperparameters.json "
                    f"(method={self.method}), attempting download...",
                )
                out = self.load_configs_hyperparameters(download=True)
                print("\tDownload successful")
                return out
        elif isinstance(download, bool) and download:
            self.download_configs_hyperparameters()
        with open(self.path_configs_hyperparameters()) as f:
            return json.load(f)

    def download_configs_hyperparameters(self):
        method_downloader = self.method_downloader()
        method_downloader.download_configs_hyperparameters()

    def load_raw_file_paths(self, max_files: int | None = None) -> list[Path]:
        return fetch_all_pickles(dir_path=self.path_raw, suffix="results.pkl", max_files=max_files)

    def load_raw(
        self,
        path_raw: str | Path | None = None,
        engine: str = "ray",
        as_holdout: bool = False,
    ) -> list[BaselineResult]:
        """Loads the raw results artifacts from all `results.pkl` files in the `path_raw` directory.

        Parameters
        ----------
        path_raw
        engine
        as_holdout

        Returns:
        -------

        """
        if path_raw is None:
            path_raw = self.path_raw
        return load_raw(path_raw=path_raw, engine=engine, as_holdout=as_holdout)

    def load_processed(
        self,
        path_processed: str | Path | None = None,
        prediction_format: Literal["memmap", "memopt", "mem"] = "memmap",
        verbose: bool = False,
    ) -> EvaluationRepository:
        if path_processed is None:
            path_processed = self.path_processed
        return EvaluationRepository.from_dir(
            path=path_processed,
            prediction_format=prediction_format,
            verbose=verbose,
        )

    def generate_repo(
        self,
        results_lst: list[BaselineResult] | None = None,
        task_metadata: pd.DataFrame | TaskMetadataCollection = None,
        cache: bool = False,
        engine: str = "ray",
    ) -> EvaluationRepository:
        if results_lst is None:
            results_lst = self.load_raw(engine=engine)

        if self.name is not None:
            # note: this is an in-place operation, but results_lst can be very large, so duplicating has drawbacks.
            for r in results_lst:
                r.update_name(name=self.name)

        repo: EvaluationRepository = generate_repo_from_results_lst(
            results_lst=results_lst,
            task_metadata=task_metadata,
            name_suffix=self.name_suffix,
        )

        if cache:
            repo.to_dir(self.path_processed)
        return repo

    def generate_repo_holdout(
        self,
        results_lst: list[BaselineResult] | None = None,
        task_metadata: pd.DataFrame | TaskMetadataCollection = None,
        engine: str = "ray",
    ) -> EvaluationRepository:
        if results_lst is None:
            results_lst = self.load_raw(engine=engine)
        results_holdout_lst = results_to_holdout(result_lst=results_lst)
        return generate_repo_from_results_lst(
            results_lst=results_holdout_lst,
            task_metadata=task_metadata,
            name_suffix=self.name_suffix,
        )

    @property
    def has_hpo_trajectories(self) -> bool:
        """Whether an HPO-trajectories artifact is locally available (no download attempted)."""
        return self.path_results_hpo_trajectories().exists()

    def load_hpo_trajectories(self, download: bool | str = "auto") -> pd.DataFrame:
        path_local = self.path_results_hpo_trajectories()
        if download == "auto":
            download = not path_local.exists()
            if download:
                print(f"Downloading hpo trajectories for {self.method}...")
        if download:
            self.method_downloader().download_results_hpo_trajectories()
        return pd.read_parquet(path=path_local)

    def to_info_dict(self) -> dict:
        """The flat field dict used to build the metadata info table / YAML.

        Derived from the declared dataclass fields (an allowlist), in declaration order, minus
        the runtime-only ``artifact_dir`` / ``cache_root`` overrides — local paths that must not
        leak into committed YAML or the info table. Field-driven rather than
        ``self.__dict__``-driven so that non-field instance state never lands in the info table
        (e.g. :class:`InMemoryMethodMetadata`'s in-memory results frame, set as a plain attribute,
        not a field). Remains an overridable hook for subclasses.
        """
        info = {f.name: getattr(self, f.name) for f in fields(self)}
        info.pop("artifact_dir", None)
        info.pop("cache_root", None)
        return info

    @property
    def path_metadata(self) -> Path:
        return self.path / "metadata.yaml"

    def to_yaml(self, path: Path | str | None = None):
        if path is None:
            path = self.path_metadata
        assert str(path).endswith(".yaml")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as outfile:
            yaml.dump(self.to_info_dict(), outfile, default_flow_style=False)

    def to_yaml_fileobj(self) -> io.BytesIO:
        """Serialize this object to YAML and return a BytesIO buffer suitable for
        s3_client.upload_fileobj, without writing to local disk.

        Returns:
        -------
        io.BytesIO
            Buffer positioned at start containing UTF-8 encoded YAML.
        """
        yaml_str = yaml.safe_dump(
            self.to_info_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        buf = io.BytesIO(yaml_str.encode("utf-8"))
        buf.seek(0)
        return buf

    @staticmethod
    def _migrate_legacy_kwargs(kwargs: dict) -> dict:
        """Fix up a kwargs dict loaded from a serialized ``metadata.yaml`` so yaml written under
        an older schema still constructs. Mutates and returns ``kwargs``. All backwards-compat
        handling for the on-disk format lives here:

        - ``artifact_name``: renamed to ``suite``. Fold a legacy ``artifact_name`` key onto
          ``suite`` so metadata.yaml written before the rename still loads. (We only ever wrote
          one of the two, so this never clobbers an explicit ``suite``.)
        - ``use_artifact_name_in_prefix``: field removed from the schema; drop it if present.
        - ``upload_as_public``: moved into ``cache_kwargs`` (an s3-specific public-read ACL knob).
          Fold a ``True`` value into ``cache_kwargs`` only when ``cache_type == "s3"``; otherwise
          (e.g. ``"r2"``) drop it regardless of value, since it is not a valid key there.
        - top-level ``s3_bucket`` / ``s3_prefix`` (oldest) and ``bucket`` / ``prefix`` (interim,
          before they moved into ``cache_kwargs``): fold onto ``cache_kwargs["bucket"]`` /
          ``cache_kwargs["prefix"]`` so previously-written metadata.yaml still loads.
        """
        if "artifact_name" in kwargs:
            kwargs.setdefault("suite", kwargs.pop("artifact_name"))
        kwargs.pop("use_artifact_name_in_prefix", None)
        if kwargs.pop("upload_as_public", False) and kwargs.get("cache_type") == "s3":
            kwargs.setdefault("cache_kwargs", {}).setdefault("upload_as_public", True)
        for legacy_key, ck_key in (
            ("s3_bucket", "bucket"),
            ("bucket", "bucket"),
            ("s3_prefix", "prefix"),
            ("prefix", "prefix"),
        ):
            if legacy_key in kwargs:
                kwargs.setdefault("cache_kwargs", {}).setdefault(ck_key, kwargs.pop(legacy_key))
        return kwargs

    @classmethod
    def from_yaml(
        cls,
        path: Path | str | None = None,
        method: str | None = None,
        suite: str | None = None,
        cache_root: str | Path | None = None,
    ) -> Self:
        """Load a method's metadata from YAML, in one of two forms:

        * ``from_yaml(path=...)`` loads committed artifacts from a self-contained directory.
          ``path`` may be that directory **or** the ``metadata.yaml`` inside it (anything not
          ending in ``.yaml`` is treated as the directory); either way the method's
          :attr:`artifact_dir` is set to the directory, so ``results/`` resolve next to the
          metadata and no TabArena cache is consulted. This is how committed copies in a repo's
          ``data/`` folder are loaded.
        * ``from_yaml(method=..., suite=...)`` looks the method up in a TabArena cache at
          ``<cache>/artifacts/<suite>/methods/<method>/metadata.yaml``. ``cache_root`` selects
          *which* cache (e.g. a teammate's cache on a shared drive); it is stamped onto the loaded
          instance so its deferred ``load_results`` / :attr:`path` keep resolving into that cache.
          Omit ``cache_root`` to use the global :func:`get_tabarena_cache_root`.

        ``path`` and ``cache_root`` are mutually exclusive: ``path`` already pins an exact location.
        """
        if path is not None:
            if cache_root is not None:
                raise ValueError(
                    "Pass either `path` (a committed artifact dir/metadata.yaml) or `cache_root` "
                    "(a cache root to look up `method`/`suite` under), not both."
                )
            path = Path(path)
            # `path` is either the artifact directory or the metadata.yaml inside it. Discriminate
            # on the suffix (not ``is_dir()``) so it works before the dir exists and so method dirs
            # whose name contains a dot (e.g. ``TA-TabPFN-2.6``) are still read as directories.
            is_yaml_file = path.suffix == ".yaml"
            artifact_dir = path.parent if is_yaml_file else path
            yaml_path = path if is_yaml_file else path / "metadata.yaml"
            override = {"artifact_dir": artifact_dir}
        else:
            assert method is not None, "method must be specified if path is not specified"
            assert suite is not None, "suite must be specified if path is not specified"
            root = Path(cache_root) if cache_root is not None else get_tabarena_cache_root()
            yaml_path = root / "artifacts" / suite / "methods" / method / "metadata.yaml"
            override = {"cache_root": cache_root} if cache_root is not None else {}

        with open(yaml_path) as file:
            kwargs = yaml.safe_load(file)
        kwargs.update(override)
        cls._migrate_legacy_kwargs(kwargs)
        return cls(**kwargs)

    def cache_raw(
        self,
        results_lst: list[BaselineResult],
    ):
        path = self.path_raw
        for result in results_lst:
            result.to_dir(path=path)

    def cache_processed(self, repo: EvaluationRepository):
        repo.to_dir(self.path_processed)

    def path_results_files(self) -> list[Path]:
        if self.method_type == "portfolio":
            file_names = [
                self.path_results_portfolio(),
            ]
        else:
            file_names = [
                self.path_results_model(),
            ]

        if self.method_type == "config":
            file_names.append(self.path_results_hpo())
        return file_names


@dataclass(frozen=True)
class ModelDescriptor:
    """Run-independent, *intrinsic* facts about a model: how it is displayed and cited, its
    compute class, and whether it bags.

    Declared once per model (in its ``models/<key>/info.py``) and reused by every
    :class:`MethodMetadata` that benchmarks the model — the TabArena run in that ``info.py``
    *and* the Beyond-IID re-run in
    :mod:`tabarena.contexts.beyondarena.methods` — instead of being
    re-typed per artifact. It carries only the fields that are identical across those runs;
    everything run-specific (``method`` name, ``ag_key``, ``suite``, ``config_default``,
    ``can_hpo``, ``date``, storage) is supplied per call to :meth:`method_metadata`.
    """

    display_name: str
    compute: Literal["cpu", "gpu"] = "cpu"
    is_bag: bool = False
    reference_url: str | None = None
    date_introduced: str | None = None

    def method_metadata(self, *, method: str, **kwargs) -> MethodMetadata:
        """Build a :class:`MethodMetadata` for one benchmark run of this model.

        The descriptor's intrinsic fields (``display_name``, ``compute``, ``is_bag``,
        ``reference_url``, ``date_introduced``) are supplied as defaults; pass any of them in ``**kwargs`` to
        override for a variant (e.g. a CPU build of a GPU model, which keeps the same paper
        but a different ``display_name`` and ``compute``). All other (run-specific)
        :class:`MethodMetadata` fields come from ``**kwargs``.

        A descriptor always describes a ``config`` method, so this routes through
        :meth:`MethodMetadata.config`; an explicit ``method_type="config"`` is accepted (and
        dropped) for back-compat, but any other ``method_type`` is rejected.
        """
        method_type = kwargs.pop("method_type", "config")
        if method_type != "config":
            raise ValueError(
                f"ModelDescriptor describes config methods; got method_type={method_type!r} (method={method!r})."
            )
        fields_from_descriptor = dict(
            display_name=self.display_name,
            compute=self.compute,
            is_bag=self.is_bag,
            reference_url=self.reference_url,
            date_introduced=self.date_introduced,
        )
        # Caller-provided values win, so a variant can override an intrinsic default.
        return MethodMetadata.config(method=method, **{**fields_from_descriptor, **kwargs})
