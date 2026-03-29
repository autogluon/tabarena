"""Minimal script to read all results.pkl produced by run_experiments_new,
parse basic information from each, and save a summary DataFrame to disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).parent


def _parse_pickle_batch(file_paths: list[str]) -> list[dict]:
    """Load a batch of results.pkl files and return one record dict per file."""
    from tabarena.benchmark.result.baseline_result import BaselineResult

    records = []
    for path in file_paths:
        result = BaselineResult.from_pickle(path)
        task_meta = result.task_metadata
        preprocessing_results = result.result["preprocessing_artifacts"].get("feature_selection", None)
        if preprocessing_results is None:
            assert "default" in result.framework
            fe_total_budget, fe_budget_index = None, None
            feature_selection_method, feature_selection_is_scoring_method = None, None
            original_feature_names, selected_feature_names, feature_scores = None, None, None
            max_features = None
        else:
            fe_total_budget = preprocessing_results["max_features_input"].keywords["b"]
            fe_budget_index = preprocessing_results["max_features_input"].keywords["idx"]
            feature_selection_method = preprocessing_results["method_type"]
            feature_selection_is_scoring_method = preprocessing_results["feature_scoring_method"]
            original_feature_names = preprocessing_results["original_feature_names"]
            selected_feature_names = preprocessing_results["selected_feature_names"]
            feature_scores = preprocessing_results["feature_scores"]
            max_features = preprocessing_results["max_features"]

        record = {
            "experiment_method_name_string": result.framework,
            # Model details
            "model_details": result.hyperparameters,
            # Task details
            **task_meta,
            # Run Results
            "metric_error": result.result["metric_error"],
            "metric_error_val": result.result.get("metric_error_val"),
            "metric": result.result["metric"],
            "problem_type": result.problem_type,
            "time_train_s": result.result["time_train_s"],
            "time_infer_s": result.result["time_infer_s"],
            # Preprocessing things
            "feature_selection_method": feature_selection_method,
            "feature_selection_is_scoring_method": feature_selection_is_scoring_method,
            "original_feature_names": original_feature_names,
            "selected_feature_names": selected_feature_names,
            "feature_scores": feature_scores,
            "max_features": max_features,
            "feature_selection_budget_total": fe_total_budget,
            "feature_selection_budget_index": fe_budget_index,
        }
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def read_benchmark_results(
    data_path: str | Path,
    *,
    benchmark_name: str | None = None,
    output_dir: str | Path = _SCRIPT_DIR,
    num_cpus: int = 8,
    batch_size: int = 1000,
    no_ray: bool = False,
) -> pd.DataFrame:
    """Find all results.pkl under *data_path*, parse them in parallel, return a DataFrame.

    Parameters
    ----------
    data_path : str | Path
        Root directory containing the benchmark output (typically the ``data/`` subdirectory
        produced by ``run_experiments_new``).
    benchmark_name : str | None
        Name used for the output CSV file: ``<output_dir>/<benchmark_name>.csv``.
        Defaults to the name of ``data_path``'s parent directory.
    output_dir : str | Path
        Directory where the CSV is saved. Defaults to the directory of this script.
    num_cpus : int
        Number of Ray workers. Each worker processes one batch of pickle files.
    batch_size : int
        Number of pickle files processed by a single Ray worker call.
    no_ray : bool
        If True, run sequentially in the current process without Ray (useful for debugging).

    Returns:
    -------
    pd.DataFrame
        One row per results.pkl, with columns: framework, dataset, tid, fold, repeat,
        split_idx, metric_error, metric_error_val, metric, problem_type,
        time_train_s, time_infer_s, path.
    """
    from tabarena.utils.pickle_utils import fetch_all_pickles

    if benchmark_name is None:
        raise ValueError("benchmark_name must be provided to determine the output file name.")
    output_path = Path(output_dir) / benchmark_name / "results_per_split.csv"
    data_path = Path(data_path)
    data_path = data_path / benchmark_name / "data"

    print(f"Searching for results.pkl under: {data_path}")
    file_paths = fetch_all_pickles(dir_path=data_path, suffix="results.pkl")
    print(f"Found {len(file_paths)} results.pkl files.")

    if not file_paths:
        return pd.DataFrame()

    result_paths = [str(p) for p in file_paths]
    batches = [result_paths[i : i + batch_size] for i in range(0, len(result_paths), batch_size)]

    if no_ray:
        print(f"Processing {len(batches)} batches sequentially (no_ray=True).")
        from tqdm import tqdm

        nested_records: list[list[dict]] = [
            _parse_pickle_batch(batch) for batch in tqdm(batches, desc="Parsing results")
        ]
    else:
        import ray
        from tabarena.utils.ray_utils import ray_map_list

        print(f"Processing {len(batches)} batches across up to {num_cpus} workers.")
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus, include_dashboard=False)

        nested_records = ray_map_list(
            batches,
            func=_parse_pickle_batch,
            func_element_key_string="file_paths",
            num_workers=num_cpus,
            num_cpus_per_worker=1,
            track_progress=True,
            tqdm_kwargs={"desc": "Parsing results"},
            ray_remote_kwargs={"max_calls": 0},
        )

    records = [record for batch in nested_records for record in batch]
    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} results.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return df


def main(
    data_path: str | Path | None = None,
    *,
    benchmark_name: str | None = None,
    output_dir: str | Path | None = None,
    num_cpus: int | None = None,
    batch_size: int | None = None,
    no_ray: bool = False,
) -> pd.DataFrame:
    """Entry point that works both as a direct call and as a CLI script.

    Any argument left as ``None`` is filled in from command-line arguments via argparse.
    When called programmatically with all arguments provided, argparse is never invoked.

    Parameters
    ----------
    data_path : str | Path | None
        Root directory containing results.pkl files (e.g. ``.../output/<run>/data``).
    benchmark_name : str | None
        Name for the output file: ``<output_dir>/<benchmark_name>.csv``.
        Defaults to the parent directory name of ``data_path``.
    output_dir : str | Path | None
        Directory to save the CSV. Defaults to the directory of this script.
    num_cpus : int | None
        Number of Ray workers. Defaults to 8.
    batch_size : int | None
        Pickle files per worker call. Defaults to 1000.
    no_ray : bool
        If True, run sequentially without Ray (useful for debugging).
    """
    if data_path is None or num_cpus is None or batch_size is None or output_dir is None:
        parser = argparse.ArgumentParser(description="Parse benchmark results.pkl files into a CSV.")
        parser.add_argument(
            "--data_path",
            type=str,
            default=None,
            help="Root directory containing results.pkl files (e.g. .../output/<run>/data).",
        )
        parser.add_argument(
            "--benchmark_name",
            type=str,
            default=None,
            help="Name for the output file: <output_dir>/<benchmark_name>.csv. "
            "Defaults to the parent directory name of data_path.",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help=f"Directory to save the CSV. Defaults to the script directory ({_SCRIPT_DIR}).",
        )
        parser.add_argument("--num_cpus", type=int, default=None, help="Number of Ray workers. Defaults to 8.")
        parser.add_argument(
            "--batch_size", type=int, default=None, help="Pickle files per worker call. Defaults to 1000."
        )
        parser.add_argument(
            "--no_ray", action="store_true", help="Run sequentially without Ray (useful for debugging)."
        )
        args = parser.parse_args()

        if data_path is None:
            data_path = args.data_path
        if benchmark_name is None:
            benchmark_name = args.benchmark_name
        if output_dir is None:
            output_dir = args.output_dir
        if num_cpus is None:
            num_cpus = args.num_cpus
        if batch_size is None:
            batch_size = args.batch_size
        if not no_ray:
            no_ray = args.no_ray

    if data_path is None:
        raise ValueError("data_path must be provided either as an argument or via --data_path.")

    return read_benchmark_results(
        data_path=data_path,
        benchmark_name=benchmark_name,
        output_dir=output_dir if output_dir is not None else _SCRIPT_DIR,
        num_cpus=num_cpus if num_cpus is not None else 8,
        batch_size=batch_size if batch_size is not None else 1000,
        no_ray=no_ray,
    )


if __name__ == "__main__":
    main(
        data_path="/work/dlclarge2/purucker-tabarena/output",
        benchmark_name="feature_selection_benchmark_example_2903",
        output_dir="./evals",
        # no_ray=True,
    )
