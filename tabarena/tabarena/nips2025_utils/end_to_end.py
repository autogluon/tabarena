from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from typing_extensions import Self

import pandas as pd

from tabarena.benchmark.result import BaselineResult
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabarena.nips2025_utils.compare import compare, compare_on_tabarena
from tabarena.nips2025_utils.end_to_end_single import (
    EndToEndResultsSingle,
    EndToEndSingle,
)
from tabarena.nips2025_utils.fetch_metadata import load_task_metadata
from tabarena.nips2025_utils.method_processor import (
    generate_task_metadata,
    load_all_artifacts,
)
from tabarena.nips2025_utils.load_metadata_from_raw import load_from_raw_all_metadata
from bencheval.tabarena import TabArena
from tabarena.utils.pickle_utils import fetch_all_pickles
from tabarena.utils.ray_utils import ray_map_list


class EndToEnd:
    def __init__(
        self,
        end_to_end_lst: list[EndToEndSingle],
    ):
        self.end_to_end_lst = end_to_end_lst

    def configs_hyperparameters(self) -> dict[str, dict | None]:
        configs_hyperparameters_per_method = [
            e2e.configs_hyperparameters() for e2e in self.end_to_end_lst
        ]
        configs_hyperparameters = {}
        for d in configs_hyperparameters_per_method:
            for k, v in d.items():
                if k in configs_hyperparameters:
                    raise ValueError(f"Duplicate key detected: {k!r}")
                configs_hyperparameters[k] = v
        return configs_hyperparameters

    @classmethod
    def from_raw(
        cls,
        results_lst: list[BaselineResult | dict],
        task_metadata: pd.DataFrame | None = None,
        cache: bool = True,
        cache_raw: bool = True,
        name: str | None = None,
        name_prefix: str | None = None,
        name_suffix: str | None = None,
        model_key: str | None = None,
        artifact_name: str | None = None,
        backend: Literal["ray", "native"] = "ray",
        verbose: bool = True,
    ) -> Self:
        log = print if verbose else (lambda *a, **k: None)

        # raw
        results_lst: list[BaselineResult] = EndToEndSingle.clean_raw(
            results_lst=results_lst
        )

        if task_metadata is None:
            tids = list({r.task_metadata["tid"] for r in results_lst})
            task_metadata = generate_task_metadata(tids=tids)

        result_types_dict = {}
        for r in results_lst:
            cur_method_metadata = MethodMetadata.from_raw(results_lst=[r])
            cur_tuple = (cur_method_metadata.method, cur_method_metadata.artifact_name, cur_method_metadata.method_type)
            if cur_tuple not in result_types_dict:
                result_types_dict[cur_tuple] = []
            result_types_dict[cur_tuple].append(r)

        unique_types = list(result_types_dict.keys())

        log(
            f"Constructing EndToEnd from raw results... Found {len(unique_types)} unique methods: {unique_types}"
        )
        end_to_end_lst = []
        for cur_type in unique_types:
            cur_results_lst = result_types_dict[cur_type]
            cur_end_to_end = EndToEndSingle.from_raw(
                results_lst=cur_results_lst,
                task_metadata=task_metadata,
                cache=cache,
                cache_raw=cache_raw,
                name=name,
                name_prefix=name_prefix,
                name_suffix=name_suffix,
                model_key=model_key,
                artifact_name=artifact_name,
                backend=backend,
                verbose=verbose,
            )
            end_to_end_lst.append(cur_end_to_end)
        return cls(end_to_end_lst=end_to_end_lst)

    @classmethod
    def from_path_raw(
        cls,
        path_raw: str | Path | list[str | Path],
        task_metadata: pd.DataFrame | None = None,
        cache: bool = True,
        cache_raw: bool = True,
        name: str | None = None,
        name_prefix: str | None = None,
        name_suffix: str | None = None,
        model_key: str | None = None,
        artifact_name: str | None = None,
        backend: Literal["ray", "native"] = "ray",
        verbose: bool = True,
    ) -> Self:
        log = print if verbose else (lambda *a, **k: None)

        engine = "ray" if backend == "ray" else "sequential"
        file_paths = fetch_all_pickles(
            dir_path=path_raw, suffix="results.pkl"
        )
        results_metadata = load_from_raw_all_metadata(file_paths=file_paths, engine=engine)
        unique_types_dict = dict()
        unique_tids = set()
        for r, file_path in zip(results_metadata, file_paths):
            r_type = r[0].method
            r_tid = r[1]["tid"]
            if r_type not in unique_types_dict:
                unique_types_dict[r_type] = []
            unique_tids.add(r_tid)
            unique_types_dict[r_type].append(file_path)
        unique_types = list(unique_types_dict.keys())

        if task_metadata is None:
            task_metadata = generate_task_metadata(tids=list(unique_tids))

        log(
            f"Constructing EndToEnd from raw results... Found {len(unique_types)} unique methods: {unique_types}"
        )
        end_to_end_lst = []
        for cur_type in unique_types:
            cur_path_raw_lst = unique_types_dict[cur_type]
            cur_end_to_end = EndToEndSingle.from_path_raw(
                path_raw=cur_path_raw_lst,
                task_metadata=task_metadata,
                cache=cache,
                cache_raw=cache_raw,
                name=name,
                name_prefix=name_prefix,
                name_suffix=name_suffix,
                model_key=model_key,
                artifact_name=artifact_name,
                backend=backend,
                verbose=verbose,
            )
            end_to_end_lst.append(cur_end_to_end)
        return cls(end_to_end_lst=end_to_end_lst)

    @classmethod
    def from_cache(cls, methods: list[str | MethodMetadata | tuple[str, str]]) -> Self:
        end_to_end_lst = []
        for method in methods:
            if isinstance(method, tuple):
                method, artifact_name = method
            else:
                artifact_name = None
            end_to_end_single = EndToEndSingle.from_cache(
                method=method, artifact_name=artifact_name
            )
            end_to_end_lst.append(end_to_end_single)
        return cls(end_to_end_lst=end_to_end_lst)

    @staticmethod
    def from_path_raw_to_results(
        path_raw: str | Path | list[str | Path],
        task_metadata: pd.DataFrame | None = None,
        cache: bool = True,
        name: str | None = None,
        name_prefix: str | None = None,
        name_suffix: str | None = None,
        model_key: str | None = None,
        artifact_name: str | None = None,
        num_cpus: int | None = None,
    ) -> EndToEndResults:
        """
        Create and cache end-to-end results for all methods in the given directory.
        Will not cache raw or processed data. To cache all artifacts, call `from_path_raw` instead.
        This is ~10x+ faster than calling `from_path_raw` for large artifacts, but will not cache raw or processed data.

        Will process each task separately in parallel using ray, minimizing disk operations and memory burden.

        Parameters
        ----------
        path_raw : str | Path | list[str | Path]
            Path to the directory containing raw results.
        task_metadata : pd.DataFrame or None = None
            The task_metadata containing information for each task,
            such as the target evaluation metric and problem_type.
            If unspecified, will be inferred from ``results_lst``.
        cache : bool = True
            If True, will cache method metadata and results to disk.
            This function will never cache raw and processed data.
        name : str or None = None
            If specified, will overwrite the name of the method.
            Will raise an exception if more than one config is present.
        name_prefix : str or None = None
            If specified, will be prepended to the name of the method (including all configs of the method).
            Useful for ensuring a unique name compared to prior results for a given model type,
            such as when re-running LightGBM.
            Will also update the model_key.
        name_suffix : str or None = None
            If specified, will be appended to the name of the method (including all configs of the method).
            Useful for ensuring a unique name compared to prior results for a given model type,
            such as when re-running LightGBM.
            Will also update the model_key.
        model_key : str or None = None
            If specified, will overwrite the model_key of the method (result.model_type).
            This is the `ag_key` value, used to distinguish between different config families
            during portfolio simulation.
        artifact_name : str or None = None
            The name of the upper directory in the cache:
                ~/.cache/tabarena/artifacts/{artifact_name}/methods/{method}/
            If unspecified, will default to ``{method}``
        num_cpus : int or None = None
            Number of CPUs to use for parallel processing.
            If None, it will use all available CPUs.
        """
        if num_cpus is None:
            num_cpus = len(os.sched_getaffinity(0))

        print("Get results paths...")
        file_paths = fetch_all_pickles(
            dir_path=path_raw, suffix="results.pkl"
        )

        all_file_paths_method = {}
        for file_path in file_paths:
            did_sid = f"{file_path.parts[-3]}/{file_path.parts[-2]}"
            if did_sid not in all_file_paths_method:
                all_file_paths_method[did_sid] = []
            all_file_paths_method[did_sid].append(file_path)

        if task_metadata is None:
            print("Get task metadata...")
            task_metadata = load_task_metadata()
            # Below is too slow to use by default, TODO: get logic for any task that is fast
            # task_metadata = generate_task_metadata(tids=list({r.split("/")[0] for r in all_file_paths_method}))

        results: list[EndToEndResults] = ray_map_list(
            list_to_map=list(all_file_paths_method.values()),
            func=_process_result_list,
            func_element_key_string="file_paths_method",
            num_workers=num_cpus,
            num_cpus_per_worker=1,
            func_put_kwargs=dict(
                task_metadata=task_metadata,
                name=name,
                name_prefix=name_prefix,
                name_suffix=name_suffix,
                model_key=model_key,
                artifact_name=artifact_name,
            ),
            track_progress=True,
            tqdm_kwargs={"desc": "Processing Results"},
            ray_remote_kwargs={"max_calls": 0},
        )
        results: list[EndToEndResultsSingle] = [
            e2e_single for e2e in results for e2e_single in e2e.end_to_end_results_lst
        ]

        print("Merging results...")
        results_per_method = {}
        for e2e_single in results:
            method = e2e_single.method_metadata.method
            if method not in results_per_method:
                results_per_method[method] = []
            results_per_method[method].append(e2e_single)
        e2e_single_lst = []
        for method, e2e_lst in results_per_method.items():
            cur_e2e = EndToEndResultsSingle.concat(e2e_lst=e2e_lst)
            e2e_single_lst.append(cur_e2e)
        e2e_results = EndToEndResults(end_to_end_results_lst=e2e_single_lst)

        if cache:
            print(f"Caching metadata and results...")
            e2e_results.cache()
        return e2e_results

    def to_results(self) -> EndToEndResults:
        return EndToEndResults(
            end_to_end_results_lst=[
                end_to_end.to_results() for end_to_end in self.end_to_end_lst
            ],
        )


class EndToEndResults:
    def __init__(
        self,
        end_to_end_results_lst: list[EndToEndResultsSingle],
    ):
        self.end_to_end_results_lst = end_to_end_results_lst

    @property
    def model_results(self) -> pd.DataFrame | None:
        model_results_lst = [e2e.model_results for e2e in self.end_to_end_results_lst]
        model_results_lst = [
            model_results
            for model_results in model_results_lst
            if model_results is not None
        ]
        if not model_results_lst:
            return None

        return pd.concat(model_results_lst, ignore_index=True)

    @property
    def hpo_results(self) -> pd.DataFrame | None:
        hpo_results_lst = [e2e.hpo_results for e2e in self.end_to_end_results_lst]
        hpo_results_lst = [
            hpo_results for hpo_results in hpo_results_lst if hpo_results is not None
        ]
        if not hpo_results_lst:
            return None

        return pd.concat(hpo_results_lst, ignore_index=True)

    # FIXME: WIP (add more)
    def leaderboard(
        self,
    ) -> pd.DataFrame:
        from bencheval.tabarena import TabArena
        results = self.get_results(
            # new_result_prefix=new_result_prefix,
            # use_artifact_name_in_prefix=use_artifact_name_in_prefix,
            # use_model_results=use_model_results,
            # fillna=False,
        )

        tabarena = TabArena(
            method_col="method",
            task_col="dataset",
            seed_column="fold",
            error_col="metric_error",
            columns_to_agg_extra=[
                "time_train_s",
                "time_infer_s",
                # "imputed",
            ],
            groupby_columns=[
                "metric",
                "problem_type",
            ],
        )

        leaderboard_kwargs = {}
        leaderboard_kwargs.setdefault("include_elo", True)
        leaderboard_kwargs.setdefault("include_winrate", True)
        leaderboard_kwargs.setdefault("include_mrr", True)
        leaderboard_kwargs.setdefault("include_rank_counts", True)
        # leaderboard_kwargs.setdefault("baseline_method", calibration_framework)
        # leaderboard_kwargs.setdefault("elo_kwargs", elo_kwargs)
        # leaderboard_kwargs.setdefault("average_seeds", average_seeds)

        leaderboard = tabarena.leaderboard(
            data=results,
            **leaderboard_kwargs,
        )
        return leaderboard

    def compare(
        self,
        output_dir: str | Path,
        task_metadata: pd.DataFrame,
        *,
        fillna: str | None = None,
        only_valid_tasks: str | list[str] | None = None,
        new_result_prefix: str | None = None,
        use_artifact_name_in_prefix: bool | None = None,
        use_model_results: bool = False,
        score_on_val: bool = False,
        average_seeds: bool = True,
        leaderboard_kwargs: dict | None = None,
    ):
        results = self.get_results(
            new_result_prefix=new_result_prefix,
            use_artifact_name_in_prefix=use_artifact_name_in_prefix,
            use_model_results=use_model_results,
            fillna=False,
        )

        return compare(
            df_results=results,
            output_dir=output_dir,
            task_metadata=task_metadata,
            fillna=fillna,
            only_valid_tasks=only_valid_tasks,
            score_on_val=score_on_val,
            average_seeds=average_seeds,
            leaderboard_kwargs=leaderboard_kwargs,
        )

    def compare_on_tabarena(
        self,
        output_dir: str | Path,
        *,
        only_valid_tasks: bool = False,
        subset: str | None | list = None,
        new_result_prefix: str | None = None,
        use_artifact_name_in_prefix: bool | None = None,
        use_model_results: bool = False,
        score_on_val: bool = False,
        average_seeds: bool = True,
        leaderboard_kwargs: dict | None = None,
        tabarena_context_kwargs: dict | None = None,
    ) -> pd.DataFrame:
        """Compare results on TabArena leaderboard.

        Args:
            output_dir (str | Path): Directory to save the results.
            subset (str | None | list): Subset of tasks to evaluate on.
                Options are "classification", "regression", "lite"  for TabArena-Lite,
                "tabicl", "tabpfn", "tabpfn/tabicl", or None for all tasks.
                Or a list of subset names to filter for.
            new_result_prefix (str | None): If not None, add a prefix to the new
                results to distinguish new results from the original TabArena results.
                Use this, for example, if you re-run a model from TabArena.
        """
        results = self.get_results(
            new_result_prefix=new_result_prefix,
            use_artifact_name_in_prefix=use_artifact_name_in_prefix,
            use_model_results=use_model_results,
            fillna=not only_valid_tasks,
        )

        return compare_on_tabarena(
            new_results=results,
            output_dir=output_dir,
            only_valid_tasks=only_valid_tasks,
            subset=subset,
            score_on_val=score_on_val,
            average_seeds=average_seeds,
            leaderboard_kwargs=leaderboard_kwargs,
            tabarena_context_kwargs=tabarena_context_kwargs,
        )

    def get_results(
        self,
        new_result_prefix: str | None = None,
        use_artifact_name_in_prefix: bool | None = None,
        use_model_results: bool = False,
        fillna: bool = False,
    ) -> pd.DataFrame:
        df_results_lst = []
        for result in self.end_to_end_results_lst:
            df_results_lst.append(
                result.get_results(
                    new_result_prefix=new_result_prefix,
                    use_artifact_name_in_prefix=use_artifact_name_in_prefix,
                    use_model_results=use_model_results,
                    fillna=fillna,
                )
            )
        df_results = pd.concat(df_results_lst, ignore_index=True)
        return df_results

    @classmethod
    def from_cache(cls, methods: list[str | MethodMetadata | tuple[str, str]]) -> Self:
        end_to_end_results_lst = []
        for method in methods:
            if isinstance(method, tuple):
                method, artifact_name = method
            else:
                artifact_name = None
            end_to_end_results = EndToEndResultsSingle.from_cache(
                method=method, artifact_name=artifact_name
            )
            end_to_end_results_lst.append(end_to_end_results)
        return cls(end_to_end_results_lst=end_to_end_results_lst)

    def cache(self):
        for e2e_results_single in self.end_to_end_results_lst:
            e2e_results_single.cache()


def _process_result_list(
    *,
    file_paths_method: list[Path],
    task_metadata: pd.DataFrame,
    name: str = None,
    name_prefix: str = None,
    name_suffix: str = None,
    model_key: str = None,
    artifact_name: str | None = None,
) -> EndToEndResults:
    results_lst = load_all_artifacts(
        file_paths=file_paths_method, engine="sequential", progress_bar=False
    )

    e2e = EndToEnd.from_raw(
        results_lst=results_lst,
        task_metadata=task_metadata,
        name=name,
        name_prefix=name_prefix,
        name_suffix=name_suffix,
        model_key=model_key,
        artifact_name=artifact_name,
        cache=False,
        cache_raw=False,
        backend="native",
        verbose=False,
    )
    e2e_results = e2e.to_results()

    return e2e_results
