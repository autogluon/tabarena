from __future__ import annotations

import pandas as pd

from tabarena.repository import EvaluationRepository
from tabarena.utils.normalized_scorer import NormalizedScorer
from tabarena.utils.rank_utils import RankScorer


def make_scorers(repo: EvaluationRepository, only_baselines=False):
    if only_baselines:
        df_results_baselines = repo._zeroshot_context.df_baselines
    else:
        dfs_to_concat = []
        if len(repo._zeroshot_context.df_configs_ranked) != 0:
            dfs_to_concat.append(repo._zeroshot_context.df_configs_ranked)
        if len(repo._zeroshot_context.df_baselines) != 0:
            dfs_to_concat.append(repo._zeroshot_context.df_baselines)
        if len(dfs_to_concat) > 1:
            df_results_baselines = pd.concat(dfs_to_concat, ignore_index=True)
        else:
            df_results_baselines = dfs_to_concat[0]

    unique_dataset_folds = [
        f"{repo.dataset_to_tid(dataset)}_{fold}"
        for dataset in repo.datasets()
        for fold in repo.dataset_to_folds(dataset=dataset)
    ]
    rank_scorer = RankScorer(df_results_baselines, tasks=unique_dataset_folds, pct=False)
    normalized_scorer = NormalizedScorer(df_results_baselines, tasks=unique_dataset_folds, baseline=None)
    return rank_scorer, normalized_scorer
