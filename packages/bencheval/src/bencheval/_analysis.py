from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import RANK


class DatasetAnalysisMixin:
    """Dataset / fold similarity, representativeness and jitter analytics (mixed into BenchmarkEvaluator)."""

    def dataset_representativeness(
        self,
        results_per_task: pd.DataFrame,
        value_col: str = RANK,
        *,
        similarity: str = "spearman",  # {"spearman", "pearson"}
        population_mode: str = "loo_mean",  # {"loo_mean", "global_mean"}
        score_mode: str = "to_population",  # {"to_population", "mean_pairwise"}
        min_methods: int | None = None,
    ) -> dict:
        """Quantify how similar each dataset is to the others based on model performance,
        then identify the most- and least-representative datasets.

        Intuition
        ---------
        Each dataset defines a vector over methods (e.g., ranks or errors).
        Two datasets are "similar" if these vectors are correlated across methods.
        Representativeness is measured by how well a dataset aligns with the population.

        Parameters
        ----------
        results_per_task : pd.DataFrame
            Output of `self.compute_results_per_task(data=data)` (may include seed col).
        value_col : str, default "rank"
            Column used to compare method performance. Common:
            - RANK (lower is better)
            - self.error_col (lower is better)
            - IMPROVABILITY (higher is better) [still fine; correlation handles direction]
            - LOSS_RESCALED (lower is better)
        similarity : {"spearman","pearson"}, default "spearman"
            Similarity metric for comparing datasets via correlation across methods.
            Spearman is recommended (rank-based; robust to scale).
        population_mode : {"loo_mean","global_mean"}, default "loo_mean"
            How to define the population reference vector:
            - "loo_mean": reference for dataset d is mean vector over all datasets except d
            - "global_mean": reference is mean vector over all datasets (includes d)
        score_mode : {"to_population","mean_pairwise"}, default "to_population"
            How to assign a single representativeness score per dataset:
            - "to_population": correlation(dataset_vector, population_reference_vector)
            - "mean_pairwise": average correlation to all other datasets
        min_methods : int | None
            If set, require at least this many methods to compute correlations.

        Returns:
        -------
        out : dict with keys
            - "representativeness": pd.DataFrame indexed by dataset with columns:
                ["score", "num_methods", "num_obs"]
            - "most_representative": str
            - "least_representative": str
            - "dataset_similarity_matrix": pd.DataFrame (dataset x dataset)
            - "method_matrix": pd.DataFrame (dataset x method), values = value_col (after seed-avg)
        """
        if similarity not in {"spearman", "pearson"}:
            raise ValueError(f"similarity must be 'spearman' or 'pearson', got {similarity!r}")
        if population_mode not in {"loo_mean", "global_mean"}:
            raise ValueError(f"population_mode must be 'loo_mean' or 'global_mean', got {population_mode!r}")
        if score_mode not in {"to_population", "mean_pairwise"}:
            raise ValueError(f"score_mode must be 'to_population' or 'mean_pairwise', got {score_mode!r}")

        required = {self.task_col, self.method_col, value_col}
        missing = [c for c in required if c not in results_per_task.columns]
        if missing:
            raise ValueError(
                f"results_per_task is missing required columns: {missing}\n"
                f"Available columns: {list(results_per_task.columns)}",
            )

        df = results_per_task.copy()

        # If seed column exists, average within (task, method) so each dataset has one value per method.
        group_cols = [self.task_col, self.method_col]
        if self.seed_column is not None and self.seed_column in df.columns:
            df = df.groupby(group_cols, as_index=False)[value_col].mean()

        # Pivot to dataset x method matrix
        M = df.pivot(index=self.task_col, columns=self.method_col, values=value_col)

        # Optional sanity: require sufficient overlap
        num_methods_per_dataset = M.notna().sum(axis=1)
        if min_methods is not None:
            keep = num_methods_per_dataset >= min_methods
            M = M.loc[keep]
            num_methods_per_dataset = num_methods_per_dataset.loc[keep]
            if M.shape[0] == 0:
                raise ValueError(f"No datasets have >= {min_methods} methods after filtering.")

        # If you're using BenchmarkEvaluator's verify_data_is_dense, M should be fully dense.
        # But we handle missingness by using pairwise corr + aligning vectors where needed.

        # 1) Dataset↔dataset similarity matrix (correlation across methods)
        # corr over rows => easiest via transpose (methods as rows, datasets as cols)
        dataset_sim = M.T.corr(method=similarity)

        # 2) Representativeness score per dataset
        if score_mode == "mean_pairwise":
            # Mean similarity to all other datasets (exclude self)
            score = (
                (dataset_sim.sum(axis=1) - 1.0) / (dataset_sim.shape[0] - 1)
                if dataset_sim.shape[0] > 1
                else dataset_sim.iloc[:, 0]
            )
        # Similarity to a "population reference vector"
        # Reference vector is mean performance across datasets (optionally leave-one-out).
        elif population_mode == "global_mean":
            ref = M.mean(axis=0)
            score = M.apply(lambda row: row.corr(ref, method=similarity), axis=1)
        else:
            # Leave-one-out mean reference: ref_d = mean of all datasets except d
            # Compute efficiently: total_sum - row, then divide by (n-1)
            n = M.shape[0]
            if n <= 1:
                score = pd.Series(index=M.index, data=np.nan, dtype=float)
            else:
                total = M.sum(axis=0)

                def loo_corr(row: pd.Series) -> float:
                    ref_d = (total - row) / (n - 1)
                    return row.corr(ref_d, method=similarity)

                score = M.apply(loo_corr, axis=1)

        rep = pd.DataFrame(
            {
                "score": score,
                "num_methods": num_methods_per_dataset,
                "num_obs": df.groupby(self.task_col).size().reindex(M.index).astype(int),
            }
        ).sort_values("score", ascending=False)

        most_rep = rep.index[0]
        least_rep = rep.index[-1]

        return {
            "representativeness": rep,
            "most_representative": most_rep,
            "least_representative": least_rep,
            "dataset_similarity_matrix": dataset_sim.loc[M.index, M.index],
            "method_matrix": M,
        }

    def dataset_fold_similarity(
        self,
        results_per_task: pd.DataFrame,
        dataset: str,
        value_col: str = RANK,
        *,
        similarity: str = "spearman",  # {"spearman", "pearson"}
        agg_across_methods: str = "mean",  # {"mean", "median"}
        min_methods: int | None = None,
        return_pairwise: bool = True,
    ) -> dict:
        """Quantify how similar a dataset's folds/seeds are, based on how methods perform.

        Assumes `results_per_task` was produced with `include_seed_col=True`, so that
        each row corresponds to (task, seed, method) with a metric like rank/error.

        Approach
        --------
        For the chosen dataset, build a matrix:
            rows = fold/seed
            cols = method
            values = value_col (e.g., rank or metric_error)

        Then compute fold↔fold similarity as correlation across methods.

        Parameters
        ----------
        results_per_task : pd.DataFrame
            Output of `self.compute_results_per_task(..., include_seed_col=True)`.
            Must include `self.seed_column` and `value_col`.
        dataset : str
            Task/dataset name (value from `self.task_col`) to analyze.
        value_col : str, default RANK
            Performance column to compare across folds. Good defaults:
            - RANK (ranking alignment)
            - self.error_col (raw metric error)
            - IMPROVABILITY, LOSS_RESCALED, etc.
        similarity : {"spearman","pearson"}, default "spearman"
            Correlation type for fold similarity.
        agg_across_methods : {"mean","median"}, default "mean"
            How to aggregate a per-fold similarity score from its similarities to other folds.
        min_methods : int | None
            If set, require each fold have at least this many methods present
            (after pivot) to be included.
        return_pairwise : bool, default True
            If True, also return a tidy dataframe of fold-pair similarities.

        Returns:
        -------
        out : dict
            - "fold_similarity_matrix": pd.DataFrame (fold x fold)
            - "fold_scores": pd.DataFrame indexed by fold with columns:
                ["score", "num_methods"]
            - "most_similar_pair": tuple | None  (fold_i, fold_j, sim)
            - "least_similar_pair": tuple | None (fold_i, fold_j, sim)
            - "fold_method_matrix": pd.DataFrame (fold x method)
            - "pairwise": pd.DataFrame (optional) columns [seed_i, seed_j, similarity]
        """
        if self.seed_column is None:
            raise ValueError("BenchmarkEvaluator.seed_column is None, but fold similarity requires a seed/fold column.")
        if self.seed_column not in results_per_task.columns:
            raise ValueError(
                f"results_per_task must include seed column {self.seed_column!r}. "
                "Ensure you called compute_results_per_task(..., include_seed_col=True).",
            )
        if similarity not in {"spearman", "pearson"}:
            raise ValueError(f"similarity must be 'spearman' or 'pearson', got {similarity!r}")
        if agg_across_methods not in {"mean", "median"}:
            raise ValueError(f"agg_across_methods must be 'mean' or 'median', got {agg_across_methods!r}")

        required = {self.task_col, self.method_col, self.seed_column, value_col}
        missing = [c for c in required if c not in results_per_task.columns]
        if missing:
            raise ValueError(
                f"results_per_task is missing required columns: {missing}\n"
                f"Available columns: {list(results_per_task.columns)}",
            )

        df = results_per_task.loc[results_per_task[self.task_col] == dataset].copy()
        if df.empty:
            raise ValueError(
                f"No rows found for {self.task_col} == {dataset!r}.\n"
                f"Available datasets: {sorted(results_per_task[self.task_col].unique())[:50]}",
            )

        # If duplicates exist for (task, seed, method), average them.
        df = df.groupby([self.task_col, self.seed_column, self.method_col], as_index=False)[value_col].mean()

        # Pivot -> folds x methods
        M = df.pivot(index=self.seed_column, columns=self.method_col, values=value_col)

        # Optionally filter folds with insufficient method coverage
        num_methods_per_fold = M.notna().sum(axis=1)
        if min_methods is not None:
            keep = num_methods_per_fold >= min_methods
            M = M.loc[keep]
            num_methods_per_fold = num_methods_per_fold.loc[keep]
            if M.shape[0] == 0:
                raise ValueError(f"No folds have >= {min_methods} methods after filtering.")

        # Correlation across methods between folds
        # corr over rows => transpose to make folds columns
        fold_sim = M.T.corr(method=similarity)

        # Per-fold "representativeness among folds": average similarity to other folds
        if fold_sim.shape[0] <= 1:
            scores = pd.Series(index=fold_sim.index, data=np.nan, dtype=float)
        else:
            off_diag = fold_sim.copy()
            np.fill_diagonal(off_diag.values, np.nan)
            if agg_across_methods == "mean":
                scores = off_diag.mean(axis=1, skipna=True)
            else:
                scores = off_diag.median(axis=1, skipna=True)

        fold_scores = pd.DataFrame(
            {
                "score": scores,
                "num_methods": num_methods_per_fold.reindex(fold_sim.index).astype(int),
            }
        ).sort_values("score", ascending=False)

        # Identify most/least similar fold pairs
        most_pair = None
        least_pair = None
        if fold_sim.shape[0] >= 2:
            # consider only upper triangle (excluding diagonal)
            sim_vals = fold_sim.where(np.triu(np.ones(fold_sim.shape), k=1).astype(bool))
            stacked = sim_vals.stack(future_stack=True)  # MultiIndex (seed_i, seed_j) -> similarity
            if len(stacked) > 0:
                (i_max, j_max), v_max = stacked.idxmax(), float(stacked.max())
                (i_min, j_min), v_min = stacked.idxmin(), float(stacked.min())
                most_pair = (i_max, j_max, v_max)
                least_pair = (i_min, j_min, v_min)

        out = {
            "fold_similarity_matrix": fold_sim,
            "fold_scores": fold_scores,
            "most_similar_pair": most_pair,
            "least_similar_pair": least_pair,
            "fold_method_matrix": M,
        }

        if return_pairwise and fold_sim.shape[0] >= 2:
            sim_vals = fold_sim.where(np.triu(np.ones(fold_sim.shape), k=1).astype(bool))

            # Ensure the row/col index level names are unique before stacking,
            # otherwise reset_index can try to insert a column that already exists.
            idx_name = fold_sim.index.name or self.seed_column or "fold"
            col_name = fold_sim.columns.name or self.seed_column or "fold"

            # Use internal unique names for stack/reset_index
            sim_vals_safe = sim_vals.copy()
            sim_vals_safe.index = sim_vals_safe.index.rename(f"__{idx_name}_i")
            sim_vals_safe.columns = sim_vals_safe.columns.rename(f"__{col_name}_j")

            # pandas >= 2.1: support future_stack=True (silences FutureWarning)
            try:
                s = sim_vals_safe.stack(future_stack=True)
            except TypeError:
                s = sim_vals_safe.stack(dropna=True)
            s = s.dropna()

            pairwise = s.rename("similarity").reset_index()

            # Rename back to desired output names (avoid collisions on rename too)
            col_i = f"{self.seed_column}_i"
            col_j = f"{self.seed_column}_j"
            pairwise = pairwise.rename(
                columns={
                    f"__{idx_name}_i": col_i,
                    f"__{col_name}_j": col_j,
                }
            )

            pairwise = pairwise.sort_values("similarity", ascending=False).reset_index(drop=True)
            out["pairwise"] = pairwise
        return out

    def rank_datasets_by_fold_similarity(
        self,
        results_per_task: pd.DataFrame,
        value_col: str = RANK,
        *,
        similarity: str = "spearman",  # {"spearman", "pearson"}
        agg_fold_score: str = "mean_pairwise",  # {"mean_pairwise", "median_pairwise"}
        min_folds: int = 2,
        min_methods: int | None = None,
        include_pairwise_extremes: bool = True,
        # --- new: stability estimation controls ---
        target_reliability: float = 0.90,
        stability_cap_folds: int = 100,
        stability_conservative: bool = True,
        stability_rho_floor: float = 0.01,
    ) -> dict:
        """Rank datasets by how consistently their folds/seeds agree with each other,
        and estimate how many folds are needed to get a stable global ordering.

        Assumes `results_per_task` was computed with `include_seed_col=True` so that
        each row corresponds to (task, seed, method) with a metric like rank/error.

        For each dataset:
          1) build fold x method matrix
          2) compute fold-fold similarity matrix (correlation across methods)
          3) summarize it into a single "fold agreement" score
          4) estimate stability/reliability of the k-fold aggregate and folds needed
             to hit `target_reliability`, using Spearman–Brown style extrapolation:
                Rel(k) = (k*rho) / (1 + (k-1)*rho)
             where rho ~= fold_agreement.

        Parameters
        ----------
        results_per_task : pd.DataFrame
            Output of `self.compute_results_per_task(..., include_seed_col=True)`.
            Must include `self.seed_column`.
        value_col : str, default RANK
            Column used to compare method performance across folds.
        similarity : {"spearman","pearson"}, default "spearman"
            Similarity metric for fold-fold correlation across methods.
        agg_fold_score : {"mean_pairwise","median_pairwise"}, default "mean_pairwise"
            How to aggregate fold-fold similarities into a dataset score.
        min_folds : int, default 2
            Require at least this many folds/seeds for a dataset to be scored.
        min_methods : int | None
            If set, require each fold have at least this many methods present.
            Passed through to `dataset_fold_similarity`.
        include_pairwise_extremes : bool, default True
            If True, include the most/least similar fold pair per dataset.
        target_reliability : float, default 0.90
            Stability threshold τ in (0,1). Higher = stricter stability requirement.
        stability_cap_folds : int, default 100
            Maximum folds to return for `folds_needed_for_stability@τ`.
        stability_conservative : bool, default True
            If True, treat rho<=0 as "not stably orderable" and return cap (with rho floored).
        stability_rho_floor : float, default 0.01
            Minimum rho used when conservative and rho is extremely small/negative.

        Returns:
        -------
        out : dict
            - "dataset_ranking": pd.DataFrame indexed by dataset with columns:
                ["fold_agreement", "num_folds", "num_methods_min", "num_methods_mean",
                 "rho_used", "stability_at_num_folds@{τ}", "folds_needed_for_stability@{τ}",
                 "most_similar_pair", "least_similar_pair" (optional), "error" (optional)]
              sorted descending by fold_agreement.
            - "per_dataset": dict[str, dict]
              Lightweight per-dataset summary.
            - "stability_params": dict
              Echo of stability configuration.
        """
        if self.seed_column is None:
            raise ValueError(
                "BenchmarkEvaluator.seed_column is None, but fold similarity ranking requires a seed/fold column."
            )
        if self.seed_column not in results_per_task.columns:
            raise ValueError(
                f"results_per_task must include seed column {self.seed_column!r}. "
                "Ensure you called compute_results_per_task(..., include_seed_col=True).",
            )
        if agg_fold_score not in {"mean_pairwise", "median_pairwise"}:
            raise ValueError(
                f"agg_fold_score must be 'mean_pairwise' or 'median_pairwise', got {agg_fold_score!r}",
            )
        if similarity not in {"spearman", "pearson"}:
            raise ValueError(f"similarity must be 'spearman' or 'pearson', got {similarity!r}")
        if not (0 < float(target_reliability) < 1):
            raise ValueError(f"target_reliability must be in (0,1), got {target_reliability!r}")

        tau = float(target_reliability)
        tau_tag = f"{tau:.2f}".rstrip("0").rstrip(".")  # e.g. "0.9" or "0.95"
        col_stability_at_k = "stability_at_num_folds"
        col_folds_needed = f"folds_needed_for_stability@{tau_tag}"

        def _rel(k: int, rho: float) -> float:
            if k <= 0 or not np.isfinite(rho) or rho <= 0:
                return 0.0
            return (k * rho) / (1.0 + (k - 1) * rho)

        def _folds_needed(rho: float) -> int:
            # k >= tau(1-rho)/(rho(1-tau))
            if not np.isfinite(rho) or rho <= 0:
                return int(stability_cap_folds)
            k_float = (tau * (1.0 - rho)) / (rho * (1.0 - tau))
            k_req = int(np.ceil(k_float))
            return max(1, min(int(stability_cap_folds), k_req))

        datasets = list(pd.Index(results_per_task[self.task_col].unique()).sort_values())

        rows: list[dict] = []
        per_dataset: dict[str, dict] = {}

        for ds in datasets:
            try:
                out_ds = self.dataset_fold_similarity(
                    results_per_task=results_per_task,
                    dataset=ds,
                    value_col=value_col,
                    similarity=similarity,
                    agg_across_methods="mean",  # not used for dataset score
                    min_methods=min_methods,
                    return_pairwise=False,
                )
            except Exception as e:
                rows.append(
                    {
                        self.task_col: ds,
                        "fold_agreement": np.nan,
                        "num_folds": 0,
                        "num_methods_min": np.nan,
                        "num_methods_mean": np.nan,
                        "rho_used": np.nan,
                        col_stability_at_k: np.nan,
                        col_folds_needed: np.nan,
                        "error": repr(e),
                    }
                )
                continue

            fold_sim = out_ds["fold_similarity_matrix"]
            fold_method = out_ds["fold_method_matrix"]

            num_folds = int(fold_sim.shape[0])
            methods_per_fold = fold_method.notna().sum(axis=1)
            num_methods_min = int(methods_per_fold.min()) if len(methods_per_fold) else np.nan
            num_methods_mean = float(methods_per_fold.mean()) if len(methods_per_fold) else np.nan

            if num_folds < min_folds:
                rows.append(
                    {
                        self.task_col: ds,
                        "fold_agreement": np.nan,
                        "num_folds": num_folds,
                        "num_methods_min": num_methods_min,
                        "num_methods_mean": num_methods_mean,
                        "rho_used": np.nan,
                        col_stability_at_k: np.nan,
                        col_folds_needed: np.nan,
                        "error": f"Too few folds (min_folds={min_folds})",
                    }
                )
                continue

            # Aggregate off-diagonal similarities into dataset agreement score (rho estimate)
            sim_vals = fold_sim.to_numpy(copy=True)
            np.fill_diagonal(sim_vals, np.nan)

            rho = float(np.nanmean(sim_vals)) if agg_fold_score == "mean_pairwise" else float(np.nanmedian(sim_vals))

            rho_used = rho
            notes = None
            if stability_conservative:
                # clamp to [-1,1]
                if rho_used > 1:
                    rho_used = 1.0
                if rho_used < -1:
                    rho_used = -1.0
                if rho_used <= 0:
                    # treat as not reliably aggregatable; use floor for rel(k) curve but folds_needed -> cap
                    notes = "rho<=0; stability may not be achievable by averaging folds alone"
                    rho_used = max(float(stability_rho_floor), 0.0)

            stability_at_k = float(_rel(num_folds, rho_used))
            folds_needed = int(_folds_needed(rho_used))

            row = {
                self.task_col: ds,
                "fold_agreement": rho,  # raw estimate from observed fold similarities
                "num_folds": num_folds,
                "num_methods_min": num_methods_min,
                "num_methods_mean": num_methods_mean,
                "rho_used": rho_used,  # what we used for stability extrapolation
                col_stability_at_k: stability_at_k,
                col_folds_needed: folds_needed,
            }
            if notes is not None:
                row["stability_note"] = notes

            if include_pairwise_extremes and num_folds >= 2:
                tri = np.triu(np.ones_like(sim_vals, dtype=bool), k=1)
                vals = sim_vals[tri]
                if np.isfinite(vals).any():
                    fold_labels = list(fold_sim.index)
                    ij = np.argwhere(tri)
                    k_max = int(np.nanargmax(vals))
                    i_max, j_max = map(int, ij[k_max])
                    row["most_similar_pair"] = (fold_labels[i_max], fold_labels[j_max], float(vals[k_max]))

                    k_min = int(np.nanargmin(vals))
                    i_min, j_min = map(int, ij[k_min])
                    row["least_similar_pair"] = (fold_labels[i_min], fold_labels[j_min], float(vals[k_min]))
                else:
                    row["most_similar_pair"] = None
                    row["least_similar_pair"] = None

            rows.append(row)

            per_dataset[ds] = {
                "fold_agreement": rho,
                "num_folds": num_folds,
                "num_methods_min": num_methods_min,
                "num_methods_mean": num_methods_mean,
                "rho_used": rho_used,
                col_stability_at_k: stability_at_k,
                col_folds_needed: folds_needed,
            }
            if notes is not None:
                per_dataset[ds]["stability_note"] = notes
            if include_pairwise_extremes:
                per_dataset[ds]["most_similar_pair"] = row.get("most_similar_pair")
                per_dataset[ds]["least_similar_pair"] = row.get("least_similar_pair")

        ranking = pd.DataFrame(rows).set_index(self.task_col)
        ranking = ranking.sort_values(by="fold_agreement", ascending=False)

        return {
            "dataset_ranking": ranking,
            "per_dataset": per_dataset,
            "stability_params": {
                "target_reliability": tau,
                "cap_folds": stability_cap_folds,
                "conservative": stability_conservative,
                "rho_floor": stability_rho_floor,
                "similarity": similarity,
                "agg_fold_score": agg_fold_score,
            },
        }

    def jitter_all_datasets(
        self,
        results_per_task: pd.DataFrame,
        *,
        rank_col: str = RANK,
        metric: str = "winrate",
        datasets: list[str] | None = None,
        return_per_method: bool = False,
        sort_by: str = "jitter_mean",
        ascending: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, dict] | None]:
        """Compute fold jitter metrics for each dataset.

        This is a wrapper around `self.dataset_jitter(...)`.

        Parameters
        ----------
        results_per_task : pd.DataFrame
            Output of `self.compute_results_per_task(..., include_seed_col=True)`.
        rank_col : str, default RANK
            Column read for per-fold ranks (rescaled to winrate when metric="winrate").
        metric : {"winrate", "rank"}, default "winrate"
            Per-method per-fold score used to compute jitter. ``"winrate"`` rescales the
            rank to ``1 - (rank - 1)/(M - 1)`` so jitter values are in [0, 1] and
            comparable across benchmarks with different method counts.
        datasets : list[str] | None
            If specified, only compute for these datasets. Otherwise uses all unique datasets.
        return_per_method : bool, default False
            If True, also return a dict keyed by dataset with per-method jitter outputs.
        sort_by : str, default "jitter_mean"
            Column name to sort the returned DataFrame by.
        ascending : bool, default True
            Sort direction.

        Returns:
        -------
        df : pd.DataFrame
            One row per dataset with:
            ["dataset", "metric", "num_folds", "num_methods", "jitter_mean", "pairwise_jitter_mean"]
            plus "error" if a dataset fails.
        per_method : dict[str, dict] | None
            If return_per_method=True, a dict of per-dataset detailed outputs; else None.
        """
        if self.seed_column is None:
            raise ValueError("seed_column must be set to compute fold jitter.")
        if self.seed_column not in results_per_task.columns:
            raise ValueError(
                f"results_per_task must include seed column {self.seed_column!r}. "
                "Ensure you called compute_results_per_task(..., include_seed_col=True).",
            )

        if datasets is None:
            datasets = list(pd.Index(results_per_task[self.task_col].unique()).sort_values())

        rows: list[dict] = []
        per_method: dict[str, dict] | None = {} if return_per_method else None

        for ds in datasets:
            try:
                out = self.dataset_jitter(
                    results_per_task=results_per_task,
                    dataset=ds,
                    rank_col=rank_col,
                    metric=metric,
                    return_per_method=return_per_method,
                )
                # ensure consistent row schema
                rows.append(
                    {
                        "dataset": out["dataset"],
                        "metric": out["metric"],
                        "num_folds": out["num_folds"],
                        "num_methods": out["num_methods"],
                        "jitter_mean": out["jitter_mean"],
                        "pairwise_jitter_mean": out["pairwise_jitter_mean"],
                    }
                )

                if return_per_method:
                    assert per_method is not None
                    # Keep only the per-method parts to avoid duplication
                    per_method[ds] = {
                        "per_method_jitter_mean": out.get("per_method_jitter_mean", None),
                        "per_method_pairwise_jitter_mean": out.get("per_method_pairwise_jitter_mean", None),
                    }

            except Exception as e:
                rows.append(
                    {
                        "dataset": ds,
                        "metric": metric,
                        "num_folds": np.nan,
                        "num_methods": np.nan,
                        "jitter_mean": np.nan,
                        "pairwise_jitter_mean": np.nan,
                        "error": repr(e),
                    }
                )
                if return_per_method:
                    assert per_method is not None
                    per_method[ds] = {"error": repr(e)}

        df = pd.DataFrame(rows)

        if sort_by is not None and sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending, na_position="last").reset_index(drop=True)

        return df, per_method

    def dataset_jitter(
        self,
        results_per_task: pd.DataFrame,
        dataset: str,
        *,
        rank_col: str = RANK,
        metric: str = "winrate",
        return_per_method: bool = False,
    ) -> dict:
        """Compute fold-instability metrics for a dataset.

        ``metric`` selects the per-fold per-method score:
          - ``"winrate"`` (default): per-fold winrate ``1 - (rank - 1) / (M - 1)``,
            in [0, 1] regardless of M. Comparable across benchmarks with different
            method counts.
          - ``"rank"``: raw per-fold rank from ``rank_col``, scale depends on M.

        Metrics returned:
          - jitter_mean:
                E_{fold, method}[ |score_fold - score_global| ]
          - pairwise_jitter_mean:
                E_{fold1<fold2, method}[ |score_f1 - score_f2| ]

        Assumes results_per_task was computed with include_seed_col=True.

        Parameters
        ----------
        results_per_task : pd.DataFrame
        dataset : str
        rank_col : str
            Source rank column (default: RANK).
        metric : {"winrate", "rank"}, default "winrate"
        return_per_method : bool
            If True, also return per-method jitter statistics.

        Returns:
        -------
        dict
        """
        if metric not in ("winrate", "rank"):
            raise ValueError(f"metric must be 'winrate' or 'rank', got {metric!r}")
        if self.seed_column is None:
            raise ValueError("seed_column must be set to compute fold jitter.")
        if self.seed_column not in results_per_task.columns:
            raise ValueError(
                "results_per_task must include seed column. Call compute_results_per_task(..., include_seed_col=True)",
            )

        df = results_per_task.loc[results_per_task[self.task_col] == dataset].copy()

        if df.empty:
            raise ValueError(f"No rows found for dataset {dataset!r}")

        # Pivot: folds x methods (values are ranks; rescaled to winrate below if requested)
        M = df.pivot(
            index=self.seed_column,
            columns=self.method_col,
            values=rank_col,
        )

        # Drop methods with any missing folds
        M = M.dropna(axis=1)

        folds = M.index.tolist()
        methods = M.columns.tolist()

        num_folds = len(folds)
        num_methods = len(methods)

        if num_folds < 2:
            raise ValueError("Need at least 2 folds to compute jitter.")
        if num_methods < 2:
            raise ValueError("Need at least 2 methods to compute jitter.")

        if metric == "winrate":
            # winrate = 1 - (rank - 1) / (M - 1); in [0, 1] regardless of M.
            M = 1.0 - (M - 1.0) / (num_methods - 1)

        # ---------------------------------------------------------
        # 1) Expected jitter: deviation of per-fold score from the global mean
        # ---------------------------------------------------------

        global_score = M.mean(axis=0)

        abs_dev = (M - global_score).abs()

        jitter_mean = abs_dev.values.mean()

        # ---------------------------------------------------------
        # 2) Pairwise jitter: average absolute change between folds
        # ---------------------------------------------------------

        pairwise_changes = []

        for i in range(num_folds):
            for j in range(i + 1, num_folds):
                diff = (M.iloc[i] - M.iloc[j]).abs()
                pairwise_changes.append(diff.values.mean())

        pairwise_jitter_mean = float(np.mean(pairwise_changes))

        result = {
            "dataset": dataset,
            "metric": metric,
            "num_folds": num_folds,
            "num_methods": num_methods,
            "jitter_mean": float(jitter_mean),
            "pairwise_jitter_mean": pairwise_jitter_mean,
        }

        if return_per_method:
            result["per_method_jitter_mean"] = abs_dev.mean(axis=0)
            result["per_method_pairwise_jitter_mean"] = pd.concat(
                [(M.iloc[i] - M.iloc[j]).abs() for i in range(num_folds) for j in range(i + 1, num_folds)],
                axis=1,
            ).mean(axis=1)

        return result

    def dataset_jitter_bootstrap_curve(
        self,
        results_per_task: pd.DataFrame,
        dataset: str,
        *,
        rank_col: str = RANK,
        metric: str = "winrate",
        k_values: list[int] | None = None,
        n_bootstrap: int = 500,
        seed: int = 0,
        drop_methods_with_any_missing: bool = True,
    ) -> pd.DataFrame:
        """Bootstrap how the k-fold *mean per-method score* converges to the global mean.

        For each bootstrap replicate:
          - sample k folds WITH replacement
          - compute per-method mean score across those k folds  (mu_hat_k)
          - compare to per-method global mean score across ALL folds (mu_global)
          - jitter_k = mean_m |mu_hat_k[m] - mu_global[m]|

        ``metric`` selects the per-fold per-method score:
          - ``"winrate"`` (default): per-fold winrate ``1 - (rank - 1) / (M - 1)``,
            in [0, 1] regardless of M. Comparable across datasets/benchmarks with
            different method counts.
          - ``"rank"``: raw per-fold rank from ``rank_col``, scale depends on M.

        Returns a DataFrame with mean/median/95% CI of jitter_k vs k, plus a
        ``metric`` column identifying which score was used.
        """
        if metric not in ("winrate", "rank"):
            raise ValueError(f"metric must be 'winrate' or 'rank', got {metric!r}")
        if self.seed_column is None:
            raise ValueError("seed_column must be set to compute fold jitter.")
        if self.seed_column not in results_per_task.columns:
            raise ValueError(
                f"results_per_task must include seed column {self.seed_column!r}. "
                "Ensure you called compute_results_per_task(..., include_seed_col=True).",
            )

        df = results_per_task.loc[results_per_task[self.task_col] == dataset].copy()
        if df.empty:
            raise ValueError(f"No rows found for dataset {dataset!r}")

        # folds x methods matrix of ranks (the canonical per-fold score; winrate is a
        # per-fold linear rescale of rank, see below)
        M_df = df.pivot(index=self.seed_column, columns=self.method_col, values=rank_col)

        if drop_methods_with_any_missing:
            M_df = M_df.dropna(axis=1)
        elif M_df.isna().any().any():
            raise ValueError(
                "Found NaNs in fold×method matrix. Set drop_methods_with_any_missing=True "
                "or impute missing values before bootstrapping.",
            )

        M_full = M_df.to_numpy(dtype=float)  # shape (F, M), values are ranks
        F, M = M_full.shape
        if F < 1:
            raise ValueError("No folds available after filtering.")
        if M < 2:
            raise ValueError("Need at least 2 methods after filtering.")

        if metric == "winrate":
            # winrate = 1 - (rank - 1) / (M - 1); in [0, 1] regardless of M.
            # Equivalent to the pairwise winrate convention with average-rank tie handling.
            M_full = 1.0 - (M_full - 1.0) / (M - 1.0)

        # Global per-method mean score across ALL folds (fixed reference)
        mu_global = M_full.mean(axis=0)  # shape (M,)

        if k_values is None:
            k_values = list(range(1, F + 1))
        else:
            k_values = [int(k) for k in k_values]
            if any(k < 1 for k in k_values):
                raise ValueError("All k_values must be >= 1.")
            if any(k > F for k in k_values):
                raise ValueError(f"All k_values must be <= number of available folds ({F}).")

        rng = np.random.default_rng(seed)

        def _summ(x: np.ndarray) -> dict:
            return dict(
                mean=float(np.mean(x)),
                p50=float(np.quantile(x, 0.50)),
                p025=float(np.quantile(x, 0.025)),
                p975=float(np.quantile(x, 0.975)),
            )

        rows: list[dict] = []

        for k in k_values:
            jit = np.empty(n_bootstrap, dtype=float)

            for b in range(n_bootstrap):
                # Sample folds WITH replacement: classical bootstrap of the k-fold
                # mean against the all-folds reference. At k=F this still produces
                # non-zero jitter (Monte-Carlo resampling noise of the all-folds
                # mean estimate) — without that, Φ=1 collapses to 0 in the strip.
                idx = rng.choice(F, size=k, replace=True)
                M_sel = M_full[idx, :]  # (k, M)

                mu_hat = M_sel.mean(axis=0)  # per-method mean score across sampled folds
                jit[b] = float(np.abs(mu_hat - mu_global).mean())

            s = _summ(jit)
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "k": k,
                    "n_bootstrap": n_bootstrap,
                    "n_folds_available": F,
                    "n_methods": M,
                    "jitter_mean": s["mean"],
                    "jitter_p50": s["p50"],
                    "jitter_p025": s["p025"],
                    "jitter_p975": s["p975"],
                }
            )

        return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    def jitter_bootstrap_curve_all_datasets(
        self,
        results_per_task: pd.DataFrame,
        *,
        rank_col: str = RANK,
        metric: str = "winrate",
        k_values: list[int] | None = None,
        n_bootstrap: int = 300,
        seed: int = 0,
        drop_methods_with_any_missing: bool = True,
    ) -> pd.DataFrame:
        datasets = list(pd.Index(results_per_task[self.task_col].unique()).sort_values())
        dfs: list[pd.DataFrame] = []

        for ds in datasets:
            try:
                d = self.dataset_jitter_bootstrap_curve(
                    results_per_task=results_per_task,
                    dataset=ds,
                    rank_col=rank_col,
                    metric=metric,
                    k_values=k_values,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    drop_methods_with_any_missing=drop_methods_with_any_missing,
                )
                dfs.append(d)
            except Exception as e:
                dfs.append(
                    pd.DataFrame(
                        [
                            {
                                "dataset": ds,
                                "metric": metric,
                                "k": np.nan,
                                "n_bootstrap": n_bootstrap,
                                "error": repr(e),
                            }
                        ]
                    )
                )

        return pd.concat(dfs, axis=0, ignore_index=True)

    @staticmethod
    def estimate_folds_for_stable_ordering(
        fold_agreement: float,
        num_folds: int,
        *,
        target_reliability: float = 0.90,  # "stable ordering" threshold
        cap: int = 100,  # safety cap
        conservative: bool = True,
        rho_floor: float = 0.01,
    ) -> dict:
        """Estimate how many folds are needed to get a stable global ordering on a dataset,
        based on observed inter-fold agreement.

        Uses Spearman–Brown style extrapolation:
            Rel(k) = (k*rho) / (1 + (k-1)*rho)
        where rho ~ mean pairwise fold correlation ("fold_agreement").

        Parameters
        ----------
        fold_agreement : float
            Mean pairwise fold similarity (e.g., Spearman correlation across methods).
        num_folds : int
            Number of folds used to estimate fold_agreement.
            Used mainly for reporting; the extrapolation itself uses fold_agreement.
        target_reliability : float, default 0.90
            Target reliability for considering the ordering "stable".
        cap : int, default 100
            Maximum folds to return (avoid absurd numbers).
        conservative : bool, default True
            If True, clamp rho into a plausible range and warn on edge cases.
        rho_floor : float, default 0.01
            Minimum rho to use when conservative and rho is extremely small/negative.

        Returns:
        -------
        dict with:
            - "rho": used rho
            - "target_reliability": tau
            - "k_required": estimated folds needed (int, capped)
            - "rel_at_num_folds": estimated reliability at the observed num_folds
            - "rel_curve": optional small table (k, Rel(k)) for k in [1..min(cap, 20)]
            - "notes": list[str]
        """
        notes = []
        rho = float(fold_agreement)

        if not np.isfinite(rho):
            raise ValueError("fold_agreement must be finite.")

        # Correlation should be in [-1, 1]; we only have meaningful fold-averaging reliability for rho > 0
        if conservative:
            if rho > 1:
                notes.append("rho > 1 encountered; clamping to 1.")
                rho = 1.0
            if rho < -1:
                notes.append("rho < -1 encountered; clamping to -1.")
                rho = -1.0

            if rho <= 0:
                notes.append(
                    "fold_agreement <= 0 suggests folds disagree or are unrelated; "
                    "a stable global ordering may not be achievable by averaging folds alone. "
                    "Returning cap.",
                )
                rho = max(rho_floor, 0.0)

        tau = float(target_reliability)
        if not (0 < tau < 1):
            raise ValueError("target_reliability must be in (0, 1).")

        def rel(k: int) -> float:
            if k <= 0:
                return np.nan
            if rho <= 0:
                return 0.0
            return (k * rho) / (1.0 + (k - 1) * rho)

        rel_at_num = rel(int(num_folds))

        # Solve k >= tau(1-rho)/(rho(1-tau))
        if rho <= 0:
            k_req = cap
        else:
            k_float = (tau * (1.0 - rho)) / (rho * (1.0 - tau))
            k_req = int(np.ceil(k_float))
            k_req = max(1, k_req)

        if k_req > cap:
            notes.append(f"Estimated folds ({k_req}) exceed cap ({cap}); returning cap.")
            k_req = cap

        rel_curve = [(k, rel(k)) for k in range(1, min(cap, 20) + 1)]
        rel_curve_df = pd.DataFrame(rel_curve, columns=["k", "reliability"])

        return {
            "rho": rho,
            "target_reliability": tau,
            "k_required": k_req,
            "rel_at_num_folds": rel_at_num,
            "rel_curve": rel_curve_df,
            "notes": notes,
        }
