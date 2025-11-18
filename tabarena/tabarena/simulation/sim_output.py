import copy
from typing import List, Union

import pandas as pd

from tabarena.portfolio import Portfolio, PortfolioCV


# FIXME: Ideally this should be easier, but not really possible with current logic.
def metric_error_to_score(metric_error: float, metric: str):
    if metric.startswith('neg_'):
        metric_score = -metric_error
    elif metric == 'auc':
        metric_score = 1 - metric_error
    else:
        from autogluon.core.metrics import get_metric
        metric_func = get_metric(metric=metric)
        metric_score = metric_func.convert_error_to_score(metric_error)
    return metric_score


# FIXME: Make portfolio_cv its own class!
# FIXME: accurate time_infer_s
# FIXME: accurate score_val
class SimulationOutputGenerator:
    """
    Generate an output pandas DataFrame that is compatible with the AutoMLBenchmark results format.
    Useful to compare directly with AutoML frameworks and baselines.

    Given a portfolio, create an output DataFrame with the following columns:

    dataset:
        The original dataset name (as a string)
    fold:
        The fold (as an int)
    framework:
        Set to the user provided name for all rows.
        This is used to differentiate the portfolio from other portfolios.
    metric:
        Example: 'neg_rmse', 'auc', etc.
    metric_error:
        The metric error of the portfolio. Lower is better, 0 is perfect.
    metric_score:
        The metric score of the portfolio, defined identical to AutoMLBenchmark.
        Higher is better.
    problem_type
    score_val:
        # FIXME: Not correct currently, but not necessary for analysis purposes.
    tid
    time_infer_s:
        # FIXME: Not correct currently, instead figure out which configs were used in the ensemble.
    time_train_s:
        The sum of the training times of the individual models in the portfolio.

    """
    def __init__(self,
                 repo,
                 config_scorer_type='ensemble',
                 config_scorer_kwargs: dict = None):
        if config_scorer_kwargs is None:
            config_scorer_kwargs = {}
        else:
            config_scorer_kwargs = copy.deepcopy(config_scorer_kwargs)

        from tabarena.repository.evaluation_repository_zeroshot import EvaluationRepositoryZeroshot
        self.repo: EvaluationRepositoryZeroshot = repo

        self.config_scorer_type = config_scorer_type
        self.config_scorer_kwargs = config_scorer_kwargs

        assert self.config_scorer_type in ['ensemble', 'single']

    def from_portfolio(self,
                       portfolio: Union[List[str], Portfolio],
                       *,
                       name: str,
                       tasks: List[str] = None,
                       minimal_columns=True,) -> pd.DataFrame:
        """
        Create from a single portfolio (Not cross-validated)
        """
        if isinstance(portfolio, Portfolio):
            if tasks is None and portfolio.test_datasets_fold is not None:
                tasks = portfolio.test_datasets_fold
            portfolio = portfolio.configs
        if tasks is None:
            tasks = self.repo._zeroshot_context.get_tasks()

        # TODO: subset datasets
        repo = self.repo.subset(configs=portfolio, verbose=False)

        config_scorer_test = repo._construct_config_scorer(
            tasks=tasks,
            config_scorer_type=self.config_scorer_type,
            **self.config_scorer_kwargs
        )

        zsc = repo._zeroshot_context

        df_raw_subset = zsc.df_configs[zsc.df_configs['task'].isin(tasks)]
        df_raw_subset = df_raw_subset[df_raw_subset['framework'].isin(portfolio)]
        df_total_train_and_infer_times = df_raw_subset[['task', 'time_train_s', 'time_infer_s']].groupby('task').sum()
        df_raw_subset = df_raw_subset.drop_duplicates(subset=['task'])

        score_per_dataset, metadata = config_scorer_test.compute_errors(portfolio)

        df_raw_subset['metric_error'] = [score_per_dataset[row[0]] for row in zip(df_raw_subset['task'])]
        # TODO: Add back?
        # df_raw_subset['metric_score'] = [metric_error_to_score(row[0], row[1]) for row in
        #                                  zip(df_raw_subset['metric_error'], df_raw_subset['metric'])]
        df_raw_subset['framework'] = name
        df_raw_subset = df_raw_subset.set_index('task', drop=True)
        df_raw_subset['time_train_s'] = df_total_train_and_infer_times['time_train_s']

        # FIXME: time_infer_s is not correct since it assumes all models are used in final ensemble
        #  In reality the infer_speed is faster.
        df_raw_subset['time_infer_s'] = df_total_train_and_infer_times['time_infer_s']
        df_raw_subset["portfolio"] = [portfolio] * len(df_raw_subset)
        df_raw_subset = df_raw_subset.reset_index(drop=True)
        if minimal_columns:
            # TODO: Add val_error
            min_cols = [
                'dataset',
                'fold',
                'framework',
                'metric_error',
                'time_train_s',
                'time_infer_s',
                'metric',
                'problem_type',
                'tid',
                'portfolio',
            ]
            df_raw_subset = df_raw_subset[min_cols]
        return df_raw_subset

    def from_portfolio_cv(self, portfolio_cv: PortfolioCV, name: str, minimal_columns=True) -> pd.DataFrame:
        """
        Create from a cross-validated portfolio, using the results only from
        the holdout to construct an output that is not overfit.
        """
        assert portfolio_cv.are_test_folds_unique()

        portfolios = portfolio_cv.portfolios
        num_folds = len(portfolios)
        for i in range(len(portfolios)):
            print(f'Fold {portfolios[i].fold} Selected Configs: {portfolios[i].configs}')

        for i in range(len(portfolios)):
            print(f'Fold {portfolios[i].fold} Test Score: {portfolios[i].test_score}')

        print(f'Final Test Score: {portfolio_cv.get_test_score_overall()}')

        df_raw_all = []
        for f in range(num_folds):
            print(f'Computing scores for each dataset... (fold={f})')
            portfolio = portfolios[f]
            df_raw_subset = self.from_portfolio(portfolio=portfolio.configs,
                                                tasks=portfolio.test_datasets_fold,
                                                name=name,
                                                minimal_columns=minimal_columns)
            df_raw_all.append(df_raw_subset)
        df_raw_all = pd.concat(df_raw_all, ignore_index=True)
        return df_raw_all
