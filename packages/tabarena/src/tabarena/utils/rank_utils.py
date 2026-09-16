from __future__ import annotations

import numpy as np
import pandas as pd


def get_rank(
    error: float, error_lst: list[float], ties_win: bool = False, pct: bool = False, include_partial: bool = True
) -> float:
    """If ties_win=True, rank will equal a win if tied with an error in error_lst. (ex: rank 1.0)
    If ties_win=False, rank will equal a tie if tied with an error in error_lst. (ex: rank 1.5).

    If pct=True, rescales output to be between 0 and 1, with 0 = best, 1 = worst.

    If include_partial=True,
        a fractional rank between 0 and 0.5 will be added
        based on the linear distance between the two nearest results in error_lst
        If error is better than any result, it compares against an error of 0.
        If error is worse than any result, it compares against an error twice as much as the worst error in error_lst.
        When True, this increases the worst possible rank by `0.5`.
        Cannot be True when ties_win=True.
    """
    if ties_win and include_partial:
        raise AssertionError("ties_win and include_partial cannot both be True.")
    rank = 0
    prior_err = 0
    win = False
    for e in error_lst:
        if error == e:
            # tie
            if ties_win:
                pass  # count as a win
            else:
                rank += 0.5  # count as a tie
        elif error > e:
            rank += 1
        else:
            win = True
        if win:
            if include_partial and error > 0:
                # Add up to 0.5 rank based on distance between closest loss and closest win.
                divisor = e - prior_err
                partial_rank = 0.5 if divisor == 0 else (error - prior_err) / divisor / 2
                partial_rank = min(partial_rank, 0.5)
                rank += partial_rank
            # error_lst is assumed to be sorted, so we know that all future elements will be wins
            # once we find our first win, allowing us to break early
            break
        prior_err = e
    if not win and include_partial and prior_err != 0:
        # Error is worse than all results,
        #  double the error of the worst result in error_lst as a new rank to penalize up to 0.5 rank
        partial_rank = min((error - prior_err) / prior_err, 1) / 2
        rank += partial_rank

    if pct:
        max_rank = len(error_lst)
        if include_partial:
            max_rank += 0.5
        rank /= max_rank
    return rank


class RankScorer:
    def __init__(
        self,
        df_results: pd.DataFrame,
        tasks: list[str],
        metric_error_col: str = "metric_error",
        task_col: str = "task",
        framework_col: str = "framework",
        ties_win: bool = False,
        pct: bool = False,
        include_partial: bool = True,
    ):
        """:param df_results: Dataframe of method performance containing columns `metric_error_col`,
        `task_col` and `framework_col`.
        :param tasks: datasets to consider
        :param ties_win: whether ties count as a win (True) or a tie (False). Set False to ensure symmetric equivalence.
        :param pct: whether to display the returned rankings in percentile form.
        """
        assert all(col in df_results for col in [metric_error_col, task_col, framework_col])
        all_datasets = set(df_results[task_col].unique())
        for task in tasks:
            assert task in all_datasets, f"{task_col} {task} not present in passed evaluations"
        self.ties_win = ties_win
        self.pct = pct
        self.include_partial = include_partial
        df_pivot = df_results.pivot_table(values=metric_error_col, index=task_col, columns=framework_col)
        # Sort a materialized copy: under pandas copy-on-write `.values` is a read-only
        # view (in-place sort raises), and on multi-block frames it is a throwaway copy
        # (in-place sort silently no-ops, leaving the rows unsorted for `get_rank`).
        sorted_errors = df_pivot.to_numpy(dtype=np.float64, copy=True)
        sorted_errors.sort(axis=1)
        # NOTE: Framework columns are no longer meaningful after row-wise sort.
        row_by_task = {task: i for i, task in enumerate(df_pivot.index)}
        self.error_dict = {}
        for task in tasks:
            row = sorted_errors[row_by_task[task]]
            self.error_dict[task] = row[~np.isnan(row)].tolist()

    def rank_many(self, tasks, errors) -> np.ndarray:
        """Vectorized :meth:`rank` over aligned ``tasks`` and ``errors`` arrays.

        Returns the same values as calling :meth:`rank` per row (including the NaN handling of
        each branch), computed per task with ``searchsorted`` on the sorted error list.
        """
        errors = np.asarray(errors, dtype=np.float64)
        tasks = np.asarray(tasks)
        out = np.empty(len(errors), dtype=np.float64)
        codes, uniques = pd.factorize(tasks)
        for code, task in enumerate(uniques):
            idx = np.flatnonzero(codes == code)
            out[idx] = self._rank_array(task=task, errors=errors[idx])
        return out

    def _rank_array(self, task: str, errors: np.ndarray) -> np.ndarray:
        a = np.asarray(self.error_dict[task], dtype=np.float64)
        n = len(a)
        left = np.searchsorted(a, errors, side="left")
        if self.ties_win and not self.include_partial:
            # mirrors `rank`: a bare searchsorted, so NaN errors sort last (rank n)
            rank = left.astype(np.float64)
            return rank / n if self.pct else rank
        right = np.searchsorted(a, errors, side="right")
        rank = left.astype(np.float64) if self.ties_win else left + 0.5 * (right - left)
        if self.include_partial:
            if n == 0:
                pass  # nothing to compare against: no partial rank either way
            else:
                win = right < n
                # first win and the element processed just before it (0 when nothing precedes)
                first_win = a[np.minimum(right, n - 1)]
                prior = np.where(right > 0, a[np.maximum(right - 1, 0)], 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    divisor = first_win - prior
                    partial_win = np.where(divisor == 0, 0.5, (errors - prior) / divisor / 2)
                    partial_win = np.minimum(partial_win, 0.5)
                    partial_loss = np.minimum((errors - prior) / prior, 1) / 2
                rank = rank + np.where(win & (errors > 0), partial_win, 0.0)
                rank = rank + np.where(~win & (prior != 0), partial_loss, 0.0)
        # `get_rank`: a NaN error compares False against everything, so the first element counts
        # as a win with no partial rank; the result is 0.
        rank = np.where(np.isnan(errors), 0.0, rank)
        if self.pct:
            rank = rank / (n + 0.5 if self.include_partial else n)
        return rank

    def rank(self, task: str, error: float) -> float:
        """Get the rank of a result on a dataset given an error."""
        if self.ties_win and not self.include_partial:
            rank = np.searchsorted(self.error_dict[task], error)
            if self.pct:
                return rank / len(self.error_dict[task])
            return rank
        return get_rank(
            error=error,
            error_lst=self.error_dict[task],
            ties_win=self.ties_win,
            pct=self.pct,
            include_partial=self.include_partial,
        )
