"""``MethodSimulator`` — TabArena HPO / model-selection simulation for one method.

This behavior used to live on :class:`~tabarena.models._method_metadata.MethodMetadata`; it is
split out here so the metadata stays a serializable *record* (fields + paths + IO) and the
simulation logic (which drives :class:`~tabarena.paper.paper_runner_tabarena.PaperRunTabArena`
over the method's processed results) is a separate concern.

A ``MethodSimulator`` wraps a ``MethodMetadata`` and reads its identity (``method``,
``artifact_name``, ``config_type``, ``method_type``, ``can_hpo``, ``config_default``), its
artifact-path helpers, and its ``load_processed`` repo loader. Construct one as
``MethodSimulator(method_metadata)`` and call the ``generate_*`` / ``get_config_default``
methods that previously hung off the metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
from autogluon.common.savers import save_pd

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata
    from tabarena.repository.evaluation_repository import EvaluationRepository


class MethodSimulator:
    """Runs TabArena HPO/model-selection simulations for a single method's processed results."""

    def __init__(self, method_metadata: MethodMetadata):
        self.method_metadata = method_metadata

    # FIXME: TMP, pre-calculate and cache this in MethodMetadata!
    def get_config_default(self, repo: EvaluationRepository | None = None):
        mm = self.method_metadata
        if repo is None:
            repo = mm.load_processed()
        from tabarena.paper.paper_runner_tabarena import PaperRunTabArena

        if mm.config_type is None:
            config_types = repo.config_types()
            assert len(config_types) == 1
            config_type = repo.config_types()[0]
        else:
            config_type = mm.config_type
        return PaperRunTabArena(repo=repo)._config_default(config_type=config_type, use_first_if_missing=True)

    def generate_results(
        self,
        repo: EvaluationRepository | None = None,
        backend: Literal["ray", "native"] = "ray",
        cache: bool = False,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame]:
        mm = self.method_metadata
        save_file = str(mm.path_results_hpo())
        save_file_model = str(mm.path_results_model())
        if repo is None:
            repo = mm.load_processed()

        if mm.method_type == "config":
            model_types = repo.config_types()
            assert len(model_types) == 1
            model_type = model_types[0]
        else:
            model_type = None

        from tabarena.paper.paper_runner_tabarena import PaperRunTabArena

        simulator = PaperRunTabArena(repo=repo, backend=backend)

        if mm.method_type == "config":
            hpo_results = simulator.run_minimal_single(model_type=model_type, tune=mm.can_hpo)
            hpo_results["ta_name"] = mm.method
            hpo_results["ta_suite"] = mm.artifact_name
            hpo_results = hpo_results.rename(
                columns={"framework": "method"}
            )  # FIXME: Don't do this, make it method by default
            if cache:
                save_pd.save(path=save_file, df=hpo_results)
            config_results = simulator.run_config_family(config_type=model_type)
            baseline_results = None
        else:
            hpo_results = None
            config_results = None
            baseline_results = simulator.run_baselines()

        results_lst = [config_results, baseline_results]
        results_lst = [r for r in results_lst if r is not None]
        model_results = pd.concat(results_lst, ignore_index=True)

        model_results["ta_name"] = mm.method
        model_results["ta_suite"] = mm.artifact_name
        model_results = model_results.rename(
            columns={"framework": "method"}
        )  # FIXME: Don't do this, make it method by default
        if cache:
            save_pd.save(path=save_file_model, df=model_results)

        return hpo_results, model_results

    def generate_hpo_result(
        self,
        repo: EvaluationRepository = None,
        n_iterations: int = 40,
        n_configs: int | None = None,
        time_limit: float | None = None,
        fixed_configs: list[str] | None = None,
        fit_order: Literal["original", "random"] = "random",
        config_type: str | list[str] | None = None,
        backend: Literal["ray", "native"] = "ray",
        seed: int = 0,
        **kwargs,
    ) -> pd.DataFrame:
        mm = self.method_metadata
        if repo is None:
            repo = mm.load_processed()
        if config_type is None:
            assert mm.config_type is not None
            config_type = mm.config_type
        from tabarena.paper.paper_runner_tabarena import PaperRunTabArena

        simulator = PaperRunTabArena(repo=repo, backend=backend)
        df_results_hpo = simulator.run_ensemble_config_type(
            config_type=config_type,
            n_iterations=n_iterations,
            n_configs=n_configs,
            time_limit=time_limit,
            fixed_configs=fixed_configs,
            fit_order=fit_order,
            seed=seed,
            **kwargs,
        )
        df_results_hpo = df_results_hpo.rename(
            columns={
                "framework": "method",
            }
        )
        df_results_hpo["method"] = f"HPO-N{n_configs}-{config_type}"
        df_results_hpo["n_configs"] = n_configs
        df_results_hpo["n_iterations"] = n_iterations
        df_results_hpo["seed"] = seed
        df_results_hpo["ta_name"] = mm.method
        df_results_hpo["ta_suite"] = mm.artifact_name
        return df_results_hpo

    def generate_hpo_trajectories(
        self,
        n_configs: list[int | None] | str = "auto",
        seeds: int | list[int] = 20,
        n_iterations: int = 40,
        fixed_configs: list[str] | None = None,
        always_include_default: bool = True,
        fit_order: Literal["original", "random"] = "random",
        time_limit: float | None = None,
        backend: Literal["ray", "native"] = "ray",
        config_type: str | list[str] | None = None,
        repo: EvaluationRepository | None = None,
        cache: bool = False,
    ) -> pd.DataFrame:
        mm = self.method_metadata
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

        df_results_hpo_lst = []
        if repo is None:
            repo = mm.load_processed()

        # FIXME: Needed for TabPFN-2.5
        repo.set_config_fallback(config_fallback=mm.config_default)

        n_config_total = repo.n_configs()

        n_configs = [n_config if n_config is not None else n_config_total for n_config in n_configs]
        n_configs = [n_config for n_config in n_configs if n_config <= n_config_total]
        n_configs = sorted(set(n_configs))

        if always_include_default and fixed_configs is None:
            config_default = mm.config_default
            assert config_default is not None
            fixed_configs = [config_default]

        for n_config in n_configs:
            print(f"Running n_config={n_config} ({mm.method})")
            assert n_config <= n_config_total
            # The trajectory is independent of `seed` whenever no random config
            # selection occurs: either the single config is a fixed one (e.g. the
            # always-included default) or every config is included. In those cases
            # `seed` only shuffles configs that are then either discarded (n_config
            # == 1) or fully retained (n_config == n_config_total), so every seed
            # yields an identical result -- run a single seed to avoid redundant work.
            seed_invariant = (n_config == 1 and fixed_configs) or n_config == n_config_total
            seeds_for_n_config = seeds[:1] if seed_invariant else seeds
            for seed in seeds_for_n_config:
                df_results_hpo = self.generate_hpo_result(
                    repo=repo,
                    n_configs=n_config,
                    seed=seed,
                    n_iterations=n_iterations,
                    fixed_configs=fixed_configs,
                    fit_order=fit_order,
                    time_limit=time_limit,
                    backend=backend,
                    config_type=config_type,
                )
                df_results_hpo["always_include_default"] = always_include_default
                df_results_hpo_lst.append(df_results_hpo)
        df_results_hpo_combined = pd.concat(df_results_hpo_lst, ignore_index=True)

        if cache:
            save_pd.save(path=mm.path_results_hpo_trajectories(), df=df_results_hpo_combined)

        return df_results_hpo_combined

    def generate_best(
        self,
        repo: EvaluationRepository = None,
        n_configs: int = 1,
        n_iterations: int = 40,
        backend: Literal["ray", "native"] = "ray",
        time_limit: float | None = None,
        **kwargs,
    ):
        mm = self.method_metadata
        if repo is None:
            repo = mm.load_processed()
        from tabarena.paper.paper_runner_tabarena import PaperRunTabArena

        simulator = PaperRunTabArena(repo=repo, backend=backend)
        df_results_best = simulator.run_zs(
            n_portfolios=n_configs,
            n_ensemble=n_iterations,
            n_ensemble_in_name=True,
            time_limit=time_limit,
            **kwargs,
        )
        df_results_best["method"] = f"{mm.config_type} (best)"
        df_results_best["method_subtype"] = "best"
        df_results_best["n_configs"] = n_configs
        df_results_best["n_iterations"] = n_iterations
        df_results_best["ta_name"] = mm.method
        df_results_best["ta_suite"] = mm.artifact_name
        return df_results_best
