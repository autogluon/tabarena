from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from tabarena.models._method_metadata import MethodMetadata
from tabarena.nips2025_utils.abstract_arena_context import AbstractArenaContext
from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection
from tabarena.nips2025_utils.eval_all import evaluate_all
from tabarena.nips2025_utils.per_dataset_tables import get_per_dataset_tables
from tabarena.nips2025_utils.subset_predicate import SubsetPredicate
from tabarena.paper.paper_runner_tabarena import PaperRunTabArena
from tabarena.paper.tabarena_evaluator import TabArenaEvaluator
from tabarena.repository import EvaluationRepository, EvaluationRepositoryCollection

if TYPE_CHECKING:
    from tabarena.benchmark.result import BaselineResult
    from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
    from tabarena.repository.abstract_repository import AbstractRepository

_methods_paper = [
    "AutoGluon_v140_bq_4h8c",
    # "AutoGluon_v140_eq_4h8c",
    "AutoGluon_v150_eq_4h8c",
    # "Portfolio-N200-4h",
    "CatBoost",
    # "Dummy",
    "ExplainableBM",
    "ExtraTrees",
    "KNeighbors",
    "LightGBM",
    "LinearModel",
    # "ModernNCA",
    "NeuralNetFastAI",
    "NeuralNetTorch",
    "RandomForest",
    # "RealMLP",
    # "TabM",
    "XGBoost",
    "Mitra_GPU",
    "ModernNCA_GPU",
    "RealMLP_GPU",
    "TabDPT_GPU",
    "TabICL_GPU",
    "TabM_GPU",
    "TabPFNv2_GPU",
    "xRFM_GPU",
    "BetaTabPFN_GPU",
    "TabFlex_GPU",
    "RealTabPFN-v2.5",
    "SAP-RPT-OSS",
    "TabICLv2",
    "TabSTAR",
    "PerpetualBooster",
    "TabPFN-v2.6",
    "LimiX",
    "OrionMSP",
    "TabPFN-3",
    "iLTM",
]


class TabArenaContext(AbstractArenaContext):
    """Reference arena context: TabArena v0.1 task/method presets + the paper workflow.

    Implements the :class:`AbstractArenaContext` hooks against the committed TabArena v0.1
    suite and the paper's method metadata, overrides :meth:`load_baseline_results` to load the
    paper baseline results, and adds the full reproduction workflow (HPO / portfolio
    simulation, plotting, repo generation). ``BeyondArenaContext`` subclasses this.
    """

    SUBSET_PREDICATES: dict[str, SubsetPredicate] = {
        "all": SubsetPredicate(lambda df: pd.Series(True, index=df.index)),
        # problem_type
        "binary": SubsetPredicate(lambda df: df["problem_type"] == "binary", ("problem_type",)),
        "multiclass": SubsetPredicate(lambda df: df["problem_type"] == "multiclass", ("problem_type",)),
        "classification": SubsetPredicate(
            lambda df: df["problem_type"].isin(["binary", "multiclass"]), ("problem_type",)
        ),
        "regression": SubsetPredicate(lambda df: df["problem_type"] == "regression", ("problem_type",)),
        # size buckets keyed on training rows
        "medium": SubsetPredicate(lambda df: df["max_train_rows"].between(10_001, 100_000), ("max_train_rows",)),
        "small": SubsetPredicate(lambda df: df["max_train_rows"] <= 10_000, ("max_train_rows",)),
        "tiny": SubsetPredicate(lambda df: df["max_train_rows"] <= 2_000, ("max_train_rows",)),
        # foundation-model compatibility (operates on tabarena task_metadata columns)
        "tabpfn": SubsetPredicate(
            lambda df: (df["max_train_rows"] <= 10_000) & (df["n_features"] <= 500) & (df["n_classes"] <= 10),
            ("max_train_rows", "n_features", "n_classes"),
        ),
        "tabicl": SubsetPredicate(
            lambda df: (df["max_train_rows"] <= 100_000) & (df["n_features"] <= 500) & (df["n_classes"] > 0),
            ("max_train_rows", "n_features", "n_classes"),
        ),
        # split-level filter: keeps split 0 == (fold 0, repeat 0). Evaluated on the task grid's
        # "split" column (see TaskMetadataCollection.task_grid); a results frame's "fold" is the
        # split, so this maps to fold == 0 there.
        "lite": SubsetPredicate(lambda df: df["split"] == 0, ("split",)),
    }

    def __init__(
        self,
        methods: list[MethodMetadata] | str = "tabarena",
        task_metadata: str | TaskMetadataCollection = "tabarena",
        *,
        extra_methods: list[MethodMetadata] | None = None,
        include_unverified: bool = False,
        backend: Literal["ray", "native"] = "ray",
        fillna_method: str | None = "RF (default)",
        calibration_method: str | None = "RF (default)",
    ):
        super().__init__(
            methods=methods,
            task_metadata=task_metadata,
            extra_methods=extra_methods,
            include_unverified=include_unverified,
            backend=backend,
            fillna_method=fillna_method,
            calibration_method=calibration_method,
        )

    def _resolve_task_metadata_preset(self, name: str) -> TaskMetadataCollection:
        if name != "tabarena":
            raise ValueError(f"Unknown task_metadata preset {name!r}; expected 'tabarena'.")
        # Native default: the committed TabArena v0.1 suite (metadata only, no downloads).
        from tabarena.benchmark.task.metadata import TaskMetadataCollection

        return TaskMetadataCollection.from_preset("TabArena-v0.1")

    def _resolve_methods_preset(self, name: str, *, include_unverified: bool) -> list[MethodMetadata]:
        if name != "tabarena":
            raise ValueError(f"Unknown methods preset '{name}'.")
        methods = copy.deepcopy(_methods_paper)
        method_metadata_lst: list[MethodMetadata] = copy.deepcopy(
            tabarena_method_metadata_collection.method_metadata_lst,
        )
        method_metadata_lst = [m for m in method_metadata_lst if m.method in methods]
        if not include_unverified:
            method_metadata_lst = [m for m in method_metadata_lst if m.verified]
        method_metadata_name_set = {m.method for m in method_metadata_lst}
        methods = [m for m in methods if m in method_metadata_name_set]
        return [m for m in method_metadata_lst if m.method in set(methods)]

    @property
    def _default_subsets(self):
        return [
            [],
            ["tiny"],
            ["small"],
            ["medium"],
            ["binary"],
            ["multiclass"],
            ["classification"],
            ["regression"],
        ]

    # FIXME: Finish this, it is WIP
    def generate_all_figs(
        self,
        output_dir,
        subsets: list[list[str] | tuple[str, list[str]]] | str = "auto",
        new_results=None,
        compare_kwargs=None,
        tuning_trajectory_kwargs=None,
        plot_compare: bool = True,
        plot_runtime_per_method: bool = False,
        plot_tuning_trajectories: bool = False,
        save_website_leaderboard: bool = False,
        website_leaderboard_kwargs: dict | None = None,
        website_leaderboard_filename: str = "leaderboard_website.csv",
    ) -> None:
        if compare_kwargs is None:
            compare_kwargs = {}
        if tuning_trajectory_kwargs is None:
            tuning_trajectory_kwargs = {}
        if website_leaderboard_kwargs is None:
            website_leaderboard_kwargs = {}
        if subsets == "auto":
            subsets = self._default_subsets
        for subset in subsets:
            output_suffix = None
            if subset is None:
                subset = []
            if isinstance(subset, tuple):
                assert len(subset) == 2
                output_suffix, subset = subset
            if isinstance(subset, str):
                subset = [subset]
            if isinstance(subset, list):
                if output_suffix is None:
                    output_suffix = "all" if not subset else "&".join(subset)
            else:
                raise ValueError(f"Unknown subset: {subset!r}")
            output_dir_subset = output_dir / output_suffix

            # FIXME: new_results
            if plot_compare:
                lb_df = self.compare(
                    output_dir=output_dir_subset,
                    subset=subset,
                    new_results=new_results,
                    subset_label=output_suffix,
                    **compare_kwargs,
                )
                if save_website_leaderboard and lb_df is not None:
                    lb_website = self.leaderboard_to_website_format(
                        leaderboard=lb_df,
                        **website_leaderboard_kwargs,
                    )
                    output_dir_subset.mkdir(parents=True, exist_ok=True)
                    lb_website.to_csv(
                        output_dir_subset / website_leaderboard_filename,
                        index=False,
                    )
            if plot_tuning_trajectories:
                self.plot_tuning_trajectories(
                    save_path=output_dir_subset / "tuning_trajectories",
                    subset=subset,
                    extra_results=new_results,
                    **tuning_trajectory_kwargs,
                )
            if plot_runtime_per_method:
                self.plot_runtime_per_method(
                    save_path=output_dir_subset / "ablation" / "all-runtimes",
                    # new_results=new_results,
                    subset=subset,
                )

    def load_raw(self, method: str, as_holdout: bool = False) -> list[BaselineResult]:
        metadata: MethodMetadata = self.method_metadata(method=method)
        return metadata.load_raw(engine=self.engine, as_holdout=as_holdout)

    def load_repo(
        self, methods: list[str | MethodMetadata] | None = None, config_fallback: str | None = None
    ) -> EvaluationRepositoryCollection:
        if methods is None:
            methods = self.methods
        repos = []
        for method in methods:
            metadata = method if isinstance(method, MethodMetadata) else self.method_metadata(method=method)
            cur_repo = metadata.load_processed()
            repos.append(cur_repo)
        return EvaluationRepositoryCollection(repos=repos, config_fallback=config_fallback)

    def generate_repo(self, method: str) -> Path:
        metadata = self.method_metadata(method=method)
        metadata.generate_repo(
            results_lst=None,
            task_metadata=self.task_metadata_collection,
            cache=True,
            engine=self.engine,
        )
        return metadata.path_processed

    # FIXME: This is a hacky approach, refactor
    def generate_hpo_trajectories(
        self,
        methods: list[str | MethodMetadata],
        n_configs: list[int | None] | str = "auto",
        seeds: int | list[int] = 20,
        n_iterations: int = 40,
        default_method: str | None = None,
        always_include_default: bool = True,
        fixed_configs: list[str] | None = None,
        fit_order: Literal["original", "random"] = "random",
        time_limit: float | None = None,
        backend: Literal["ray", "native"] = "ray",
        repo: EvaluationRepository | None = None,
        folds: list[int] | None = None,
        ta_name: str | None = None,
        ta_suite: str | None = None,
        display_name: str | None = None,
    ) -> pd.DataFrame:
        methods: list[MethodMetadata] = [
            m if isinstance(m, MethodMetadata) else self.method_metadata(m) for m in methods
        ]
        if repo is None:
            repo = self.load_repo(methods=methods)
            if folds is not None:
                repo = repo.subset(folds=folds)
        if not default_method:
            default_method = methods[0]
        else:
            for method in methods:
                if method.method == default_method:
                    default_method = method
                    break
        hpo_trajectory = default_method.generate_hpo_trajectories(
            n_configs=n_configs,
            repo=repo,
            seeds=seeds,
            n_iterations=n_iterations,
            always_include_default=always_include_default,
            fixed_configs=fixed_configs,
            fit_order=fit_order,
            time_limit=time_limit,
            backend=backend,
            config_type=repo.config_types(),
            cache=False,
        )

        hpo_trajectory["ta_name"] = ta_name
        hpo_trajectory["ta_suite"] = ta_suite
        hpo_trajectory["display_name"] = display_name
        return hpo_trajectory

    # TODO: Refine this
    def generate_portfolio_trajectories(
        self,
        configs: list[str],
        config_fallback: str | None = None,
        n_configs: list[int | None] | str = "auto",
        seeds: int | list[int] = 1,
        n_iterations: int = 40,
        fit_order: Literal["original", "random"] = "original",
        time_limit: float | None = None,
        methods: str | None = None,
        repo: EvaluationRepository | None = None,
        folds: list[int] | None = None,
        name: str | None = None,
        ta_name: str | None = None,
        ta_suite: str | None = None,
        display_name: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Given a list of configs, compute the tuning trajectories
        for the first N configs for each N in n_configs.
        """
        if n_configs == "auto":
            n_configs = [
                1,
                2,
                5,
                10,
                25,
                50,
                100,
                150,
                None,  # all configs
            ]
        if isinstance(seeds, int):
            seeds = list(range(seeds))

        if repo is None:
            if methods is not None:
                methods: list[MethodMetadata] = [
                    m if isinstance(m, MethodMetadata) else self.method_metadata(m) for m in methods
                ]
            repo = self.load_repo(methods=methods, config_fallback=config_fallback)

        # TODO: also include config_fallback
        configs_w_fallback = copy.deepcopy(configs)
        if config_fallback is not None:
            if config_fallback not in configs_w_fallback:
                configs_w_fallback.append(config_fallback)
            repo.set_config_fallback(config_fallback=config_fallback)
        repo = repo.subset(configs=configs_w_fallback, folds=folds)

        df_results_hpo_lst = []

        n_config_total = len(configs)

        n_configs = [n_config if n_config is not None else n_config_total for n_config in n_configs]
        n_configs = [n_config for n_config in n_configs if n_config <= n_config_total]
        n_configs = sorted(set(n_configs))

        for n_config in n_configs:
            print(f"Running n_config={n_config}")
            for seed in seeds:
                df_results_hpo = self.simulate_portfolio_from_configs(
                    n_iterations=n_iterations,
                    configs=configs[:n_config],
                    repo=repo,
                    folds=folds,
                    seed=seed,
                    fit_order=fit_order,
                    time_limit=time_limit,
                    **kwargs,
                )

                if name is not None:
                    df_results_hpo["method"] = f"HPO-N{n_config}-{name}"
                df_results_hpo["n_configs"] = n_config
                df_results_hpo["n_iterations"] = n_iterations
                df_results_hpo_lst.append(df_results_hpo)
        hpo_trajectory = pd.concat(df_results_hpo_lst, ignore_index=True)

        hpo_trajectory["ta_name"] = ta_name
        hpo_trajectory["ta_suite"] = ta_suite
        hpo_trajectory["display_name"] = display_name

        return hpo_trajectory

    def combine_hpo(
        self,
        methods: list[str],
        new_config_type: str,
        ta_name: str,
        ta_suite: str,
        method_default: str | None = None,
        repo: EvaluationRepository | None = None,
        n_configs: int | None = None,
        time_limit: float | None = None,
        fit_order: Literal["original", "random"] = "original",
        default_always_first: bool = True,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Perform HPO across multiple methods.

        Returns default, tuned, and tuned + ensembled results.
        """
        if method_default is None:
            method_default = methods[0]
        if repo is None:
            repo = self.load_repo(methods=methods)

        config_type_default = self.method_metadata(method_default).config_type
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)
        config_default = simulator._config_default(config_type=config_type_default, use_first_if_missing=True)
        if config_default is not None:
            default = simulator.run_config_default(model_type=config_type_default)
            default = default.rename(columns={"framework": "method"})
            default["ta_name"] = ta_name
            default["ta_suite"] = ta_suite
            default["config_type"] = new_config_type
            default["method"] = f"{new_config_type} (default)"
        else:
            default = None

        fixed_configs = [config_default] if default_always_first and config_default else None

        tuned = self.run_hpo(
            method=methods,
            repo=repo,
            n_iterations=1,
            n_configs=n_configs,
            time_limit=time_limit,
            fit_order=fit_order,
            seed=seed,
            fixed_configs=fixed_configs,
        )

        tuned_ens = self.run_hpo(
            method=methods,
            repo=repo,
            n_iterations=40,
            n_configs=n_configs,
            time_limit=time_limit,
            fit_order=fit_order,
            seed=seed,
            fixed_configs=fixed_configs,
        )

        tuned["ta_name"] = ta_name
        tuned["ta_suite"] = ta_suite
        tuned["config_type"] = new_config_type
        tuned["method"] = f"{new_config_type} (tuned)"
        tuned_ens["ta_name"] = ta_name
        tuned_ens["ta_suite"] = ta_suite
        tuned_ens["config_type"] = new_config_type
        tuned_ens["method"] = f"{new_config_type} (tuned + ensemble)"

        return pd.concat(
            [
                default,
                tuned,
                tuned_ens,
            ],
            ignore_index=True,
        )

    def run_hpo(
        self,
        method: str | list[str],
        repo: EvaluationRepository = None,
        n_iterations: int = 40,
        n_configs: int | None = None,
        time_limit: float | None = None,
        fit_order: Literal["original", "random"] = "original",
        seed: int = 0,
        **kwargs,
    ) -> pd.DataFrame:
        if not isinstance(method, list):
            method = [method]
        valid_methods = self.methods
        if repo is None:
            repo = self.load_repo(methods=method)
        method_new = []
        for m in method:
            if m in valid_methods:
                method_metadata = self.method_metadata(method=m)
                config_type = method_metadata.config_type
            else:
                config_type = m
            method_new.append(config_type)
        method = method_new
        if len(method) == 1:
            method = method[0]
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)
        df_results_family_hpo = simulator.run_ensemble_config_type(
            config_type=method,
            n_iterations=n_iterations,
            n_configs=n_configs,
            time_limit=time_limit,
            fit_order=fit_order,
            seed=seed,
            **kwargs,
        )
        df_results_family_hpo = df_results_family_hpo.rename(
            columns={
                "framework": "method",
            }
        )
        name = "HPO"
        if n_configs is not None:
            name += f"-N{n_configs}"
        name += f"-{method}"
        df_results_family_hpo["method"] = name
        return df_results_family_hpo

    # FIXME: WIP
    def _run_compare_pca(
        self,
        configs: list[str] | None = None,
        repo: EvaluationRepository | None = None,
        config_fallback: str | None = None,
    ):
        if repo is None:
            repo = self.load_repo(config_fallback=config_fallback)
        if configs is None:
            configs = self._get_config_defaults()

        simulator = PaperRunTabArena(repo=repo, backend=self.backend)
        simulator.evaluator.compute_avg_config_prediction_delta(configs=configs)

    # FIXME: WIP
    def _get_config_defaults(self):
        config_defaults = []
        for m in self.method_metadata_collection.method_metadata_lst:
            if m.method_type != "config":
                continue
            config_defaults.append(m.get_config_default())
        return config_defaults

    def simulate_portfolio_from_configs(
        self,
        configs: list[str],
        config_fallback: str | None = None,
        repo: EvaluationRepositoryCollection = None,
        **kwargs,
    ):
        if repo is None:
            repo = self.load_repo(config_fallback=config_fallback)
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)

        results = simulator.evaluate_ensembles(
            configs=configs,
            **kwargs,
        )

        return results.rename(columns={"framework": "method"})

    def simulate_portfolio_from_configs_per(
        self,
        df_info: pd.DataFrame,
        config_fallback: str | None = None,
        repo: EvaluationRepositoryCollection = None,
        **kwargs,
    ):
        if repo is None:
            repo = self.load_repo(config_fallback=config_fallback)
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)

        results = simulator.evaluate_ensembles_per(
            df_info=df_info,
            **kwargs,
        )

        return results.rename(columns={"framework": "method"})

    def simulate_portfolio_search(
        self,
        methods: list[str],
        config_fallback: str,
        result_baselines: pd.DataFrame,
        repo: EvaluationRepositoryCollection = None,
        config_types: list[str] | None = None,
        selected_types: list[str] | None = None,
        n_portfolio: int = 25,
        n_ensemble: int = 40,
        time_limit: float | None = 14400,
        average_seeds: bool = False,
    ):
        if repo is None:
            repo = self.load_repo(methods=methods, config_fallback=config_fallback)
        if config_types is None:
            config_types = repo.config_types()
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)

        simulator.run_portfolio_search(
            model_types=config_types,
            selected_types=selected_types,
            result_baselines=result_baselines,
            n_portfolio=n_portfolio,
            n_ensemble=n_ensemble,
            time_limit=time_limit,
            average_seeds=average_seeds,
        )

    def run_portfolio(
        self,
        repo: AbstractRepository,
        configs: list[str],
        n_portfolio: int,
        n_ensemble: int | None = None,
        time_limit: int | None = 14400,
    ) -> pd.DataFrame:
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)
        return simulator.run_zs(
            configs=configs,
            n_portfolios=n_portfolio,
            n_ensemble=n_ensemble,
            n_ensemble_in_name=True,
            time_limit=time_limit,
        )

    def simulate_portfolio(
        self, methods: list[str], config_fallback: str, repo: EvaluationRepositoryCollection = None, **kwargs
    ):
        if repo is None:
            repo = self.load_repo(methods=methods, config_fallback=config_fallback)
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)

        df_results_n_portfolio = []
        n_portfolios = [200] if "n_portfolios" not in kwargs else kwargs.pop("n_portfolios")
        for n_portfolio in n_portfolios:
            df_results_n_portfolio.append(
                simulator.run_zs(n_portfolios=n_portfolio, n_ensemble=None, n_ensemble_in_name=False, **kwargs)
            )
        return pd.concat(df_results_n_portfolio, ignore_index=True)

    def run_portfolio_from_config_types(
        self,
        repo: AbstractRepository,
        config_types: list[str],
        n_portfolio: int,
        n_ensemble: int | None = None,
        time_limit: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        simulator = PaperRunTabArena(repo=repo, backend=self.backend)
        return simulator.run_zs_from_types(
            config_types=config_types,
            n_portfolios=n_portfolio,
            n_ensemble=n_ensemble,
            n_ensemble_in_name=True,
            time_limit=time_limit,
            **kwargs,
        )

    def load_hpo_results(self, method: str) -> pd.DataFrame:
        metadata = self.method_metadata(method=method)
        return metadata.load_hpo_results()

    def load_config_results(self, method: str) -> pd.DataFrame:
        metadata = self.method_metadata(method=method)
        return metadata.load_model_results()

    def load_portfolio_results(self, method: str) -> pd.DataFrame:
        metadata = self.method_metadata(method=method)
        return metadata.load_portfolio_results()

    def load_baseline_results(
        self,
        methods: list[str] | None = None,
        download_results: str | bool = "auto",
        methods_drop: list[str] | None = None,
    ) -> pd.DataFrame:
        if methods is None:
            methods = self.methods
        if methods_drop is not None:
            for method in methods_drop:
                if method not in methods:
                    raise AssertionError(
                        f"Specified '{method}' in `methods_drop`, but '{method}' is not present in methods: {methods}",
                    )
            methods = [method for method in methods if method not in methods_drop]

        df_results_lst = []
        for method in methods:
            method_metadata = self.method_metadata(method=method)
            if isinstance(download_results, bool) and download_results:
                method_downloader = method_metadata.method_downloader()
                method_downloader.download_results()

            try:
                df_results = method_metadata.load_results()
            except FileNotFoundError as err:
                if isinstance(download_results, str) and download_results == "auto":
                    print(
                        f"Missing local results files for method! "
                        f"Attempting to download from s3 and retry... "
                        f'(download_results={download_results}, method="{method_metadata.method}")',
                    )
                    method_downloader = method_metadata.method_downloader()
                    method_downloader.download_results()
                    df_results = method_metadata.load_results()
                else:
                    print(
                        f"Missing local results files for method {method_metadata.method}! "
                        f"Try setting `download_results=True` to get the required files.",
                    )
                    raise err
            df_results_lst.append(df_results)

        return pd.concat(df_results_lst, ignore_index=True)

    def load_configs_hyperparameters(
        self,
        methods: list[str] | None = None,
        download: bool | str = False,
    ) -> dict[str, dict]:
        if methods is None:
            methods = self.methods
            methods = [m for m in methods if self.method_metadata(m).method_type == "config"]
        configs_hyperparameters_lst = []
        for method in methods:
            metadata = self.method_metadata(method=method)
            configs_hyperparameters = metadata.load_configs_hyperparameters(download=download)
            configs_hyperparameters_lst.append(configs_hyperparameters)

        def merge_dicts_no_duplicates(dicts: list[dict]) -> dict:
            merged = {}
            for d in dicts:
                for key in d:
                    if key in merged:
                        raise KeyError(
                            f"Duplicate key found in configs_hyperparameters: {key}\n"
                            f"This should never happen and may mean that a given config name "
                            f"belongs to multiple different hyperparameters!",
                        )
                merged.update(d)
            return merged

        return merge_dicts_no_duplicates(configs_hyperparameters_lst)

    def evaluate_all(
        self,
        save_path: str | Path,
        df_results: pd.DataFrame = None,
        df_results_cpu: pd.DataFrame = None,
        configs_hyperparameters: dict[str, dict] | None = None,
        include_portfolio: bool = False,
        elo_bootstrap_rounds: int = 200,
        use_latex: bool = False,
        fillna_method: str | None = "auto",
        use_website_folder_names: bool = False,
        evaluator_kwargs: dict | None = None,
        engine: str = "auto",
        progress_bar: bool = True,
    ):
        if df_results is None:
            df_results = self.load_baseline_results(download_results="auto")

        if fillna_method == "auto":
            fillna_method = self.fillna_method
        if fillna_method is not None:
            df_results = TabArenaContext.fillna_metrics(
                df_to_fill=df_results,
                df_fillna=df_results[df_results["method"] == fillna_method],
            )

        evaluate_all(
            tabarena_context=self,  # FIXME: Don't do this in future, clean up
            df_results=df_results,
            # configs_hyperparameters=configs_hyperparameters,  # TODO: Add back later
            eval_save_path=save_path,
            elo_bootstrap_rounds=elo_bootstrap_rounds,
            use_latex=use_latex,
            use_website_folder_names=use_website_folder_names,
            evaluator_kwargs=evaluator_kwargs,
            engine=engine,
            progress_bar=progress_bar,
        )

    def plot_tuning_trajectories(
        self,
        save_path: str | Path,
        subset: list[str] | None = None,
        **kwargs,
    ):
        from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import plot_tuning_trajectories

        plot_tuning_trajectories(
            tabarena_context=self,
            fig_save_dir=save_path,
            subset_map=subset,
            **kwargs,
        )

    def plot_tuning_trajectories_per_dataset(
        self,
        save_path: str | Path,
        file_ext: str = ".pdf",
        to_grid: bool = False,
        **kwargs,
    ):
        if to_grid:
            assert file_ext == ".png", f"to_grid=True only works with file_ext={'.png'!r}"
        from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import plot_tuning_trajectories_per_dataset

        plot_tuning_trajectories_per_dataset(
            tabarena_context=self,
            fig_save_dir=save_path,
            file_ext=file_ext,
            **kwargs,
        )

        if to_grid:
            self._make_png_grid(
                save_path=save_path,
            )

    def _make_png_grid(
        self,
        save_path: str | Path,
        suffix: str | Path = "tuning_trajectories/pareto_n_configs_err_tot_train.png",
        output_suffix: str | Path = "per_dataset_train_vs_error.png",
        n_cols: int = 5,
        datasets: list[str] | None = None,
    ):
        from tabarena.plot.png_to_grid import make_png_grid

        if not datasets:
            datasets = sorted(self.task_metadata_collection.dataset_names())

        n_datasets = len(datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols

        prefix = save_path
        output_path = save_path.parent / output_suffix

        png_files = [prefix / dataset / suffix for dataset in datasets]
        make_png_grid(
            image_paths=png_files,
            output_path=output_path,
            n_rows=n_rows,
            n_cols=n_cols,
            padding=12,
            bg_color=(255, 255, 255, 255),
            resize_mode="fit",
            scale=0.33,
        )

    def plot_runtime_per_method(
        self,
        save_path: str | Path,
        df_results_configs: pd.DataFrame = None,
        subset: list[str] | None = None,
        **kwargs,
    ):
        if df_results_configs is None:
            df_results_configs = self.load_config_results_multi()
        else:
            df_results_configs = df_results_configs.copy()
        if "imputed" in df_results_configs.columns:
            # Remove imputed results
            df_results_configs["imputed"] = df_results_configs["imputed"].fillna(0)
            df_results_configs = df_results_configs[df_results_configs["imputed"] == 0]

        if subset:
            df_results_configs = self.subset_results(df_results=df_results_configs, subset=subset)

        # Group/legend by per-method display_name (falling back to the method name)
        # rather than config_type, so methods sharing a config_type (e.g. CPU/GPU
        # variants like RealMLP and RealMLP_GPU) get distinct lines.
        method_to_display_name = {
            m.method: m.display_name
            for m in self.method_metadata_collection.method_metadata_lst
            if m.method_type == "config"
        }
        df_results_configs["config_type"] = (
            df_results_configs["ta_name"].map(method_to_display_name).fillna(df_results_configs["ta_name"])
        )

        deep_dive_kwargs = dict(kwargs.pop("deep_dive_kwargs", None) or {})

        evaluator = TabArenaEvaluator(output_dir=save_path, task_metadata=self.task_metadata_collection)
        evaluator.generate_runtime_plot(
            df_results=df_results_configs,
            deep_dive_kwargs=deep_dive_kwargs,
            **kwargs,
        )

    def generate_per_dataset_tables(
        self,
        save_path: str | Path,
        df_results: pd.DataFrame = None,
        fillna_method: str | None = "auto",  # FIXME: Don't hardcode
        per_dataset_dir: str | Path | None = None,
        method_order: list[str] | None = None,
        use_display_names: bool = False,
    ):
        if fillna_method == "auto":
            fillna_method = self.fillna_method
        if df_results is None:
            df_results = self.load_baseline_results(download_results="auto")

        if use_display_names:
            rename_map = self._method_rename_map_to_display_names()
            if rename_map:
                df_results = df_results.copy()
                df_results["method"] = df_results["method"].replace(rename_map)
                if fillna_method in rename_map:
                    fillna_method = rename_map[fillna_method]

        if fillna_method is not None:
            df_results = TabArenaContext.fillna_metrics(
                df_to_fill=df_results,
                df_fillna=df_results[df_results["method"] == fillna_method],
            )

        get_per_dataset_tables(
            df_results=df_results,
            save_path=Path(save_path),
            task_metadata=self.task_metadata_collection,
            per_dataset_dir=Path(per_dataset_dir) if per_dataset_dir is not None else None,
            method_order=method_order,
        )

    def load_config_results_multi(
        self,
        method_metadata_lst: list[MethodMetadata] | None = None,
    ) -> pd.DataFrame:
        if method_metadata_lst is None:
            method_metadata_lst = self.method_metadata_collection.method_metadata_lst
        df_results_configs_lst = []
        for method_metadata in method_metadata_lst:
            if method_metadata.method_type == "config":
                df_results_configs_lst.append(method_metadata.load_model_results())
        return pd.concat(df_results_configs_lst, ignore_index=True)

    def find_missing(self, method: str):
        metadata = self.method_metadata(method=method)
        repo = EvaluationRepository.from_dir(path=metadata.path_processed)

        tasks = repo.tasks()
        n_tasks = len(tasks)
        print(f"Method: {method} | n_tasks={n_tasks}")

        metrics = repo.metrics()
        metrics = metrics.reset_index(drop=False)

        configs = repo.configs()

        n_configs = len(configs)

        runs_missing_lst = []

        fail_dict = {}
        for i, config in enumerate(configs):
            metrics_config = metrics[metrics["framework"] == config]
            n_tasks_config = len(metrics_config)

            tasks_config = list(metrics_config[["dataset", "fold"]].values)
            tasks_config = {tuple(t) for t in tasks_config}

            n_tasks_missing = n_tasks - n_tasks_config
            tasks_missing = [t for t in tasks if t not in tasks_config] if n_tasks_missing != 0 else []

            for dataset, fold in tasks_missing:
                runs_missing_lst.append(
                    (dataset, fold, config),
                )

            print(f"{n_tasks_missing}\t{config}\t{i + 1}/{n_configs}")
            fail_dict[config] = n_tasks_missing

        import pandas as pd
        # fail_series = pd.Series(fail_dict).sort_values()

        df_missing = pd.DataFrame(data=runs_missing_lst, columns=["dataset", "fold", "framework"])
        df_missing = df_missing.rename(columns={"framework": "method"})
        print(df_missing)

        # save_pd.save(path="missing_runs.csv", df=df_missing)

        return df_missing

    @classmethod
    def fillna_metrics(cls, df_to_fill: pd.DataFrame, df_fillna: pd.DataFrame) -> pd.DataFrame:
        """Fills missing (dataset, fold, framework) rows in df_to_fill with the (dataset, fold) row in df_fillna.

        Parameters
        ----------
        df_to_fill
        df_fillna

        Returns:
        -------

        """
        method_col = "method"
        split_col = "fold"
        dataset_col = "dataset"

        columns_to_keep = [
            "method_type",
            "method_subtype",
            "config_type",
            "ta_name",
            "ta_suite",
        ]
        columns_to_keep = [c for c in columns_to_keep if c in df_to_fill]
        per_column: dict[str, dict] = {}
        for c in columns_to_keep:
            groupby_method = df_to_fill.groupby(method_col)[c]
            nunique = groupby_method.nunique(dropna=False)
            invalid = nunique[nunique != 1]
            df_to_fill_invalid = df_to_fill[df_to_fill[method_col].isin(invalid.index)]
            groupby_method_invalid = df_to_fill_invalid.groupby(method_col)[c]
            if not invalid.empty:
                raise AssertionError(
                    f"Found a method with multiple values for column {c} (must be unique):\n"
                    f"{groupby_method_invalid.value_counts(dropna=False)}",
                )

            # Using .first() is safe because nunique == 1 for every method
            per_column[c] = groupby_method.first().to_dict()

        df_to_fill = df_to_fill.set_index([dataset_col, split_col, method_col], drop=True)
        df_fillna = df_fillna.set_index([dataset_col, split_col], drop=True).drop(columns=[method_col])

        unique_frameworks = list(df_to_fill.index.unique(level=method_col))

        df_filled = df_fillna.index.to_frame().merge(
            pd.Series(data=unique_frameworks, name=method_col),
            how="cross",
        )
        df_filled = df_filled.set_index(keys=list(df_filled.columns))

        # missing results
        nan_vals = df_filled.index.difference(df_to_fill.index)

        # fill valid values
        fill_cols = list(df_to_fill.columns)
        df_filled[fill_cols] = np.nan
        df_filled[fill_cols] = df_filled[fill_cols].astype(df_to_fill.dtypes)
        df_filled.loc[df_to_fill.index] = df_to_fill

        df_fillna_to_use = df_fillna.loc[nan_vals.droplevel(level=method_col)].copy()
        df_fillna_to_use.index = nan_vals
        df_filled.loc[nan_vals] = df_fillna_to_use

        if "imputed" not in df_filled.columns:
            df_filled["imputed"] = False
        df_filled.loc[nan_vals, "imputed"] = True
        df_filled["imputed"] = df_filled["imputed"].fillna(0).astype(bool)

        df_filled = df_filled.reset_index(drop=False)

        # Overwrite values column-by-column while preserving order
        for c in columns_to_keep:
            mapping = per_column[c]
            df_filled[c] = df_filled[method_col].map(mapping)

        return df_filled
