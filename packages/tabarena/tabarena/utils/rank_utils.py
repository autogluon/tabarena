from typing import List

import numpy as np
import pandas as pd


def get_rank(error: float,
             error_lst: List[float],
             ties_win: bool = False,
             pct: bool = False,
             include_partial: bool = True) -> float:
    """
    If ties_win=True, rank will equal a win if tied with an error in error_lst. (ex: rank 1.0)
    If ties_win=False, rank will equal a tie if tied with an error in error_lst. (ex: rank 1.5)

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
        raise AssertionError('ties_win and include_partial cannot both be True.')
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
                if divisor == 0:
                    partial_rank = 0.5
                else:
                    partial_rank = ((error - prior_err) / divisor) / 2
                if partial_rank > 0.5:
                    # Safeguard against divide by 0 edge-cases
                    partial_rank = 0.5
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
        tasks: List[str],
        metric_error_col: str = 'metric_error',
        task_col: str = 'task',
        framework_col: str = 'framework',
        ties_win: bool = False,
        pct: bool = False,
        include_partial: bool = True,
    ):
        """
        :param df_results: Dataframe of method performance containing columns `metric_error_col`,
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
        df_pivot.values.sort(axis=1)  # NOTE: The framework columns are now no longer correct. Do not use them.

        # tolist to drop the framework col name, since it is no longer ordered.
        self.error_dict = {dataset: df_pivot.loc[dataset].dropna().tolist() for dataset in tasks}

    def rank(self, task: str, error: float) -> float:
        """
        Get the rank of a result on a dataset given an error.
        """
        if self.ties_win and not self.include_partial:
            rank = np.searchsorted(self.error_dict[task], error)
            if self.pct:
                return rank / len(self.error_dict[task])
            else:
                return rank
        else:
            return get_rank(error=error,
                            error_lst=self.error_dict[task],
                            ties_win=self.ties_win,
                            pct=self.pct,
                            include_partial=self.include_partial)
