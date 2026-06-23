from __future__ import annotations

import io
import json
import os
import warnings
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import pandas as pd
import yaml

from tabarena.loaders import get_tabarena_cache_root
from tabarena.nips2025_utils.generate_repo import generate_repo_from_results_lst
from tabarena.nips2025_utils.load_artifacts import results_to_holdout
from tabarena.nips2025_utils.method_processor import get_info_from_result, load_raw
from tabarena.repository.evaluation_repository import EvaluationRepository
from tabarena.utils.pickle_utils import fetch_all_pickles
from tabarena.utils.s3_utils import s3_get_object

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


# FIXME: Implement `best` and `best-N`
@dataclass(eq=False)
class MethodMetadata:
    """Identity, artifact layout, and storage/transport config for one benchmarked method.

    A ``@dataclass`` (``eq=False`` to preserve identity-based equality/hashing, matching the
    pre-dataclass class): the field declarations below are the canonical schema, and
    :meth:`to_info_dict` derives the serialized YAML / info-table surface from those fields (an
    allowlist) rather than from ``self.__dict__``. :meth:`__post_init__` fills the derived
    defaults (``artifact_name`` <- ``method``, ``model_key`` <- ``ag_key``, ``can_hpo`` <-
    ``method_type``, ``display_name``) and validates the inputs.
    """

    #: Whether this method's artifacts live in memory (no on-disk/S3 backing). False for the
    #: disk-backed base class; True for :class:`InMemoryMethodMetadata`. Arena contexts treat
    #: in-memory methods as the locally-produced "new" results (e.g. ``compare`` resolves
    #: ``only_valid_tasks=True`` against them). A ``ClassVar`` (not a dataclass field), so it
    #: stays out of :meth:`to_info_dict` / the metadata info table.
    is_in_memory: ClassVar[bool] = False

    method: str
    artifact_name: str | None = None
    date: str | None = None
    method_type: str = "config"
    ag_key: str | None = None
    model_key: str | None = None
    name: str | None = None
    name_suffix: str | None = None
    config_default: str | None = None
    compute: Literal["cpu", "gpu"] = "cpu"
    is_bag: bool = False
    has_raw: bool = False
    has_processed: bool = False
    has_results: bool = False
    verified: bool = False
    use_artifact_name_in_prefix: bool = False
    can_hpo: bool | None = None
    s3_bucket: str | None = None
    s3_prefix: str | None = None
    upload_as_public: bool = False
    reference_url: str | None = None
    #: Storage backend for this method's artifacts. Defaults to ``"r2"`` (the public TabArena
    #: data bucket); methods whose artifacts predate this default explicitly pass ``"s3"``.
    cache_type: Literal["s3", "r2", "local"] = "r2"
    #: Optional override pointing at a self-contained, *flat* method directory
    #: (``<cache_root>/<method>/`` holding ``metadata.yaml`` + ``results/``), used to load a
    #: method's committed artifacts from an arbitrary location (e.g. a repo's ``data/`` folder)
    #: without touching the global TabArena cache root. ``None`` => derive paths from the cache
    #: root as usual. Kept out of ``to_info_dict`` so this local path is never serialized into
    #: the committed ``metadata.yaml`` or the metadata info table.
    cache_root: str | Path | None = None
    display_name: str | None = None

    def __post_init__(self):
        if self.artifact_name is None:
            self.artifact_name = self.method
        if self.model_key is None:
            self.model_key = self.ag_key
        if self.can_hpo is None:
            self.can_hpo = self.method_type == "config"
        self.cache_root = Path(self.cache_root) if self.cache_root is not None else None

        assert isinstance(self.method, str)
        assert len(self.method) > 0
        assert isinstance(self.artifact_name, str)
        assert len(self.artifact_name) > 0
        # Accept a MethodType member or its plain string, normalize to the string for storage,
        # then validate against the canonical set (the constructor's previous hardcoded list).
        if isinstance(self.method_type, MethodType):
            self.method_type = self.method_type.value
        assert self.method_type in MethodType.values(), (
            f"Unknown method_type: {self.method_type!r}. Valid values: {MethodType.values()}"
        )
        assert self.compute in ["cpu", "gpu"]
        if self.name is not None and self.method_type == "config":
            raise AssertionError("Cannot specify `name` for method_type: 'config'.")
        if self.name is not None and self.name_suffix is not None:
            raise AssertionError("Must only specify one of `name` and `name_suffix`.")

        if self.display_name is None:
            self.display_name = self._compute_display_name()

        assert isinstance(self.display_name, str)
        assert len(self.display_name) > 0
        assert self.cache_type in ["s3", "r2", "local"], f"Unknown `cache_type`: {self.cache_type}"

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

    # TODO: Also support baseline methods
    @classmethod
    def from_raw(
        cls,
        results_lst: list[BaselineResult],
        method: str | None = None,
        artifact_name: str | None = None,
        config_default: str | None = None,
        compute: Literal["cpu", "gpu"] | None = None,
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
                artifact_name=artifact_name,
                config_default=config_default,
                compute=compute,
            )
        elif method_type == "baseline":
            method_metadata = cls._from_raw_baseline(
                result_df=result_df,
                method=method,
                artifact_name=artifact_name,
                compute=compute,
            )
        else:
            raise ValueError(f"Unknown method_type: {method_type}")

        return method_metadata

    @classmethod
    def _from_raw_baseline(
        cls,
        result_df: pd.DataFrame,
        method: str | None = None,
        artifact_name: str | None = None,
        compute: Literal["cpu", "gpu"] | None = None,
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

        if artifact_name is None:
            artifact_name = method

        return cls(
            method=method,
            artifact_name=artifact_name,
            method_type=method_type,
            compute=compute,
            config_default=None,
            can_hpo=False,
            is_bag=False,
            has_raw=True,
            has_processed=True,
            has_results=True,
            # Preserve the historical default now that the field default is "r2".
            cache_type="s3",
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
        artifact_name: str | None = None,
        config_default: str | None = None,
        compute: Literal["cpu", "gpu"] | None = None,
    ) -> Self:
        unique_method_types = result_df["method_type"].unique()
        assert len(unique_method_types) == 1
        method_type = unique_method_types[0]

        assert method_type == "config"

        unique_model_types = result_df["model_type"].unique()
        assert len(unique_model_types) == 1, (
            f"MethodMetadata requires exactly 1 model type, found: {unique_model_types}"
        )

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

        if artifact_name is None:
            artifact_name = method

        return cls(
            method=method,
            artifact_name=artifact_name,
            method_type=method_type,
            compute=compute,
            config_default=config_default,
            ag_key=ag_key,
            can_hpo=can_hpo,
            is_bag=is_bag,
            has_raw=True,
            has_processed=True,
            has_results=True,
        )

    @classmethod
    def tabarena_public(
        cls,
        *,
        method: str,
        artifact_name: str,
        has_raw: bool = True,
        has_processed: bool = True,
        has_results: bool = True,
        upload_as_public: bool = True,
        s3_bucket: str = "tabarena",
        s3_prefix: str = "cache",
        **kwargs,
    ) -> Self:
        """Build a :class:`MethodMetadata` for a method whose artifacts live in the public
        TabArena store.

        Fills the boilerplate shared by every public TabArena artifact — the ``tabarena`` /
        ``cache`` S3 location, ``upload_as_public=True``, and a fully-cached
        ``has_raw``/``has_processed``/``has_results`` — so callers specify only what is
        method-specific. Any of these defaults can be overridden via keyword (e.g.
        ``has_raw=False`` for a portfolio with no raw artifacts, or ``s3_prefix=...`` for a
        non-standard prefix). The single source of truth for the public-artifact preset that the
        per-artifact ``_common_*_kwargs`` dicts used to re-declare.
        """
        return cls(
            method=method,
            artifact_name=artifact_name,
            has_raw=has_raw,
            has_processed=has_processed,
            has_results=has_results,
            upload_as_public=upload_as_public,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            **kwargs,
        )

    @property
    def has_s3_cache(self) -> bool:
        return self.s3_bucket is not None and self.s3_prefix is not None

    @property
    def has_configs_hyperparameters(self) -> bool:
        return self.method_type == "config"

    @property
    def _path_root(self) -> Path:
        return get_tabarena_cache_root() / "artifacts"

    @property
    def path_cache_root(self) -> Path:
        return get_tabarena_cache_root()

    @property
    def path(self) -> Path:
        # When ``cache_root`` is set, artifacts live in a flat ``<cache_root>/<method>/`` dir
        # (no ``artifacts/<artifact_name>/methods/`` infix) so committed copies stay shallow.
        if self.cache_root is not None:
            return self.cache_root / self.method
        return self._path_root / self.artifact_name / "methods" / self.method

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

    def to_s3_cache_loc(self, path: Path, s3_cache_root: str) -> str:
        path_suffix: str = self.relative_to_cache_root(path=path).as_posix()
        return f"{s3_cache_root}/{path_suffix}"

    def method_downloader(
        self,
        cache_type: str = "auto",
        verbose: bool = False,
    ) -> MethodDownloader:
        if cache_type == "auto":
            cache_type = self.cache_type
        if not self.has_s3_cache:
            raise AssertionError(
                f"Tried to get MethodDownloader from MethodMetadata, "
                f"but s3_bucket and/or s3_prefix were not specified!"
                f"\n\t(method={self.method}, artifact_name={self.artifact_name}, "
                f"s3_bucket={self.s3_bucket}, s3_prefix={self.s3_prefix})"
                f"\nEnsure you initialize MethodMetadata with s3_bucket and s3_prefix to enable s3 artifact download.",
            )

        if cache_type == "r2":
            from tabarena.models._artifacts.downloader_public_r2 import MethodDownloaderPublicR2

            return MethodDownloaderPublicR2(
                method_metadata=self,
                base_url="https://data.tabarena.ai/",
                r2_prefix=self.s3_prefix,
                verbose=verbose,
                clear_dirs=False,
            )
        if cache_type == "s3":
            from tabarena.models._artifacts.downloader_s3 import MethodDownloaderS3

            return MethodDownloaderS3(
                method_metadata=self,
                s3_bucket=self.s3_bucket,
                s3_prefix=self.s3_prefix,
                verbose=verbose,
                clear_dirs=False,
            )
        raise ValueError(f"Invalid cache_type for downloads: {cache_type}")

    def method_uploader(self, cache_type: str = "auto") -> MethodUploader:
        if cache_type == "auto":
            cache_type = self.cache_type
        if not self.has_s3_cache:
            raise AssertionError(
                f"Tried to get MethodUploader from MethodMetadata, "
                f"but s3_bucket and/or s3_prefix were not specified!"
                f"\n\t(method={self.method}, artifact_name={self.artifact_name}, "
                f"s3_bucket={self.s3_bucket}, s3_prefix={self.s3_prefix})"
                f"\nEnsure you initialize MethodMetadata with s3_bucket and s3_prefix to enable s3 artifact upload.",
            )

        if cache_type == "r2":
            from tabarena.models._artifacts.uploader_r2 import MethodUploaderR2

            r2_env_vars = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
            missing = [v for v in r2_env_vars if not os.environ.get(v)]
            if missing:
                raise OSError(
                    f"Missing required environment variable(s) for R2 uploads: {missing}.\n"
                    f"Set {', '.join(r2_env_vars)} in your shell (or a .env file) before "
                    f"calling method_uploader() with cache_type='r2'.\n"
                    f"\n"
                    f"Where to find these values in the Cloudflare R2 dashboard "
                    f"(https://dash.cloudflare.com/ -> R2 Object Storage):\n"
                    f"  - R2_ACCOUNT_ID: shown as 'Account ID' on the R2 overview page, "
                    f"and embedded in the dashboard URL (dash.cloudflare.com/<account_id>/r2).\n"
                    f"  - R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY: open 'Manage R2 API Tokens' "
                    f"-> 'Create API Token', choose the required permission (e.g. Object Read & "
                    f"Write) and bucket scope, then submit. Cloudflare displays both the Access "
                    f"Key ID and the Secret Access Key on the success page; the secret is shown "
                    f"only once, so copy it immediately. If you've lost an existing secret, "
                    f"create a new token (or roll the existing one) from the same page.",
                )

            return MethodUploaderR2(
                method_metadata=self,
                r2_account_id=os.environ["R2_ACCOUNT_ID"],
                r2_bucket=self.s3_bucket,
                r2_prefix=self.s3_prefix,
                r2_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                r2_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            )
        if cache_type == "s3":
            from tabarena.models._artifacts.uploader_s3 import MethodUploaderS3

            return MethodUploaderS3(
                method_metadata=self,
                s3_bucket=self.s3_bucket,
                s3_prefix=self.s3_prefix,
                upload_as_public=self.upload_as_public,
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
        the runtime-only ``cache_root`` override — a local path that must not leak into committed
        YAML or the info table. Field-driven rather than ``self.__dict__``-driven so that
        non-field instance state never lands in the info table (e.g.
        :class:`InMemoryMethodMetadata`'s in-memory results frame, set as a plain attribute,
        not a field). Remains an overridable hook for subclasses.
        """
        info = {f.name: getattr(self, f.name) for f in fields(self)}
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

    @classmethod
    def from_yaml(
        cls,
        path: Path | str | None = None,
        method: str | None = None,
        artifact_name: str | None = None,
        *,
        cache_root: str | Path | None = None,
        relative_cache: bool | Literal["auto"] = "auto",
    ) -> Self:
        """Load a method's metadata from YAML.

        ``cache_root`` points the loaded method at a self-contained, flat artifact directory
        (``<cache_root>/<method>/`` holding ``metadata.yaml`` + ``results/``) instead of the
        global TabArena cache — e.g. to load committed results from a repo's ``data/`` folder.
        When given alongside an explicit ``path``, the two must describe the same location
        (``<cache_root>/<method>/metadata.yaml``); a mismatch raises so layout typos surface early.

        ``relative_cache=True`` infers ``cache_root`` from ``path`` itself (as ``path.parent.parent``,
        i.e. the directory containing the ``<method>/`` folder), so callers loading a method from an
        explicit ``path`` need only pass ``path`` — no separately-computed ``cache_root``. Requires
        an explicit ``path`` and is mutually exclusive with ``cache_root``.

        The default ``"auto"`` applies that inference exactly when it is safe and unambiguous: an
        explicit ``path`` is given and no ``cache_root`` was passed. It is a no-op for the
        ``method``/``artifact_name`` lookup forms (so those callers are unaffected), and the
        inference is correct for both the flat committed layout and the global cache layout, since
        in both the yaml's parent directory is named ``<method>``.
        """
        if relative_cache == "auto":
            relative_cache = path is not None and cache_root is None

        if relative_cache:
            if path is None:
                raise ValueError("relative_cache=True requires an explicit `path`.")
            if cache_root is not None:
                raise ValueError("Pass either `cache_root` or `relative_cache=True`, not both.")
            cache_root = Path(path).parent.parent

        if path is None:
            assert method is not None, "method must be specified if path is not specified"
            assert artifact_name is not None, "artifact_name must be specified if path is not specified"
            if cache_root is not None:
                path = Path(cache_root) / method / "metadata.yaml"
            else:
                path = get_tabarena_cache_root() / "artifacts" / artifact_name / "methods" / method / "metadata.yaml"

        assert str(path).endswith(".yaml")
        with open(path) as file:
            kwargs = yaml.safe_load(file)
        if cache_root is not None:
            kwargs["cache_root"] = cache_root
        if "cache_type" not in kwargs:
            # yaml created before the cache_type default flipped to "r2"; preserve old "s3" default
            kwargs["cache_type"] = "s3"
        method_metadata = cls(**kwargs)
        if cache_root is not None and Path(method_metadata.path_metadata).resolve() != Path(path).resolve():
            raise ValueError(
                f"cache_root/method layout mismatch: with cache_root={cache_root!r} the metadata for "
                f"method={method_metadata.method!r} is expected at {method_metadata.path_metadata}, "
                f"but it was loaded from {path}. Lay artifacts out as <cache_root>/<method>/metadata.yaml.",
            )
        return method_metadata

    @classmethod
    def from_s3_cache(
        cls,
        method: str,
        s3_bucket: str,
        s3_prefix: str = "cache",
        artifact_name: str | None = None,
    ) -> Self:
        metadata = MethodMetadata(
            method=method,
            artifact_name=artifact_name,
        )
        path_local = Path(metadata.path_metadata)
        # The remote key is just the prefix joined with the path's location relative to the cache
        # root — computed directly rather than by building and re-parsing an "s3://..." URI.
        s3_key = (Path(s3_prefix) / metadata.relative_to_cache_root(path_local)).as_posix()
        # Stream into memory
        try:
            obj = s3_get_object(Bucket=s3_bucket, Key=s3_key)
        except Exception as e:
            print(
                f"Failed to fetch MethodMetadata yaml file from s3! Maybe it doesn't exist or is not public?"
                f'\n\t(method="{method}", artifact_name="{artifact_name}", '
                f's3_bucket="{s3_bucket}", s3_prefix="{s3_prefix}")',
            )
            raise e

        body = obj["Body"]  # file-like object (StreamingBody, BytesIO, etc.)
        kwargs = yaml.safe_load(body)

        if "s3_bucket" not in kwargs:
            # yaml created before s3_bucket existed
            kwargs["s3_bucket"] = s3_bucket
        if "s3_prefix" not in kwargs:
            # yaml created before s3_prefix existed
            kwargs["s3_prefix"] = s3_prefix
        if "cache_type" not in kwargs:
            # yaml created before the cache_type default flipped to "r2"; preserve old "s3" default
            kwargs["cache_type"] = "s3"

        return cls(**kwargs)

    def cache_raw(
        self,
        results_lst: list[BaselineResult],
    ):
        path = self.path_raw
        for result in results_lst:
            result.to_dir(path=path)

    def load_end_to_end_results(self):
        model_results = self.load_model_results()
        hpo_results = self.load_hpo_results()
        from tabarena.nips2025_utils.end_to_end_single import EndToEndResultsSingle

        return EndToEndResultsSingle(
            method_metadata=self,
            model_results=model_results,
            hpo_results=hpo_results,
        )

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
    :mod:`tabarena.nips2025_utils.artifacts._beyond_method_metadata` — instead of being
    re-typed per artifact. It carries only the fields that are identical across those runs;
    everything run-specific (``method`` name, ``ag_key``, ``artifact_name``, ``config_default``,
    ``can_hpo``, ``date``, storage) is supplied per call to :meth:`method_metadata`.
    """

    display_name: str
    compute: Literal["cpu", "gpu"] = "cpu"
    is_bag: bool = False
    reference_url: str | None = None

    def method_metadata(self, *, method: str, **kwargs) -> MethodMetadata:
        """Build a :class:`MethodMetadata` for one benchmark run of this model.

        The descriptor's intrinsic fields (``display_name``, ``compute``, ``is_bag``,
        ``reference_url``) are supplied as defaults; pass any of them in ``**kwargs`` to
        override for a variant (e.g. a CPU build of a GPU model, which keeps the same paper
        but a different ``display_name`` and ``compute``). All other (run-specific)
        :class:`MethodMetadata` fields come from ``**kwargs``.
        """
        fields_from_descriptor = dict(
            display_name=self.display_name,
            compute=self.compute,
            is_bag=self.is_bag,
            reference_url=self.reference_url,
        )
        # Caller-provided values win, so a variant can override an intrinsic default.
        return MethodMetadata(method=method, **{**fields_from_descriptor, **kwargs})
