from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .evaluation_repository import EvaluationRepository
from ..portfolio import Portfolio, PortfolioCV
from ..simulation.sim_output import SimulationOutputGenerator
from ..simulation.sim_runner import run_zs_simulation


class EvaluationRepositoryZeroshot(EvaluationRepository):
    """An extension of EvaluationRepository that includes zeroshot simulation methods."""
    def generate_output_from_portfolio_cv(self,
                                          portfolio_cv: PortfolioCV,
                                          name: str,
                                          config_scorer_type: str = 'ensemble',
                                          config_scorer_kwargs: dict = None) -> pd.DataFrame:
        """
        Generates an AutoGluon-Benchmark compatible pandas DataFrame output that can be used to compare
        with existing baselines and other simulation results.

        The input portfolios for this method are cross-validated simulation portfolios, where each task may have
        a different portfolio used, which is defined in `portfolio_cv`.

        :param portfolio_cv: A PortfolioCV object that contains the selected portfolio for each task.
            This can be obtained via first calling `self.simulate_zeroshot(...)`.
        :param name: The name associated with this result. This name should be a unique identifier to this
            particular simulation, and will be the name used in downstream evaluation and comparison logic.
        :param config_scorer_type: One of ['ensemble', 'single']
            If 'ensemble', will compute results using the score achieved by val EnsembleSelection on the portfolio.
            If 'single', will compute results using the score achieved by the val single best model in the portfolio.
        :param config_scorer_kwargs: Optional kwargs used to initialize the config scorer.
            If config_scorer_type='ensemble', one example kwarg is 'ensemble_size',
            which dictates the number of greedy EnsembleSelection steps.
            To perfectly replicate AutoGluon's ensemble selection, set 'ensemble_size' to 100.
        :return: A pandas DataFrame of N rows, where N is the number of tasks in `portfolio_cv`.
            This output is compatible with evaluation and comparison logic implemented in the AutoGluon-Benchmark repo.
            Each row contains the following columns:
                ['dataset', 'fold', 'framework', 'metric_error', 'time_train_s', 'time_infer_s', 'metric', 'problem_type', 'tid']
            Below is an example row (name='dummy'):
                dataset           2dplanes
                fold                     0
                framework            dummy
                metric_error       0.01468
                time_train_s    112.329859
                time_infer_s      4.299721
                metric                 auc
                problem_type        binary
                tid                   3593
        """
        sog = SimulationOutputGenerator(repo=self,
                                        config_scorer_type=config_scorer_type,
                                        config_scorer_kwargs=config_scorer_kwargs)
        df_result = sog.from_portfolio_cv(portfolio_cv=portfolio_cv, name=name)
        return df_result

    def generate_output_from_portfolio(self,
                                       portfolio: Union[Portfolio, List[str]],
                                       name: str,
                                       config_scorer_type: str = 'ensemble',
                                       config_scorer_kwargs: dict = None) -> pd.DataFrame:
        """
        Generates an AutoGluon-Benchmark compatible pandas DataFrame output that can be used to compare
        with existing baselines and other simulation results.

        The input for this method is a single portfolio used for all valid tasks.
        Be careful of generating an output on tasks used to obtain the provided portfolio to avoid overfit results.

        This method is best used when using a train/test split and only generating the output on the holdout test tasks.
        If you need to generate output for all tasks, use `self.generate_output_from_portfolio_cv` instead.

        :param portfolio: A Portfolio object that contains the selected portfolio for each task.
            Can also be a list of model names. In this case, all tasks in the EvaluationRepository will have results computed.
        :param name: The name associated with this result. This name should be a unique identifier to this
            particular simulation, and will be the name used in downstream evaluation and comparison logic.
        :param config_scorer_type: One of ['ensemble', 'single']
            If 'ensemble', will compute results using the score achieved by val EnsembleSelection on the portfolio.
            If 'single', will compute results using the score achieved by the val single best model in the portfolio.
        :param config_scorer_kwargs: Optional kwargs used to initialize the config scorer.
            If config_scorer_type='ensemble', one example kwarg is 'ensemble_size',
            which dictates the number of greedy EnsembleSelection steps.
            To perfectly replicate AutoGluon's ensemble selection, set 'ensemble_size' to 100.
        :return: A pandas DataFrame with identical format to the output of `self.generate_output_from_portfolio_cv`.
            Refer to `self.generate_output_from_portfolio_cv` for details.
        """
        sog = SimulationOutputGenerator(repo=self,
                                        config_scorer_type=config_scorer_type,
                                        config_scorer_kwargs=config_scorer_kwargs)
        df_result = sog.from_portfolio(portfolio=portfolio, name=name)
        return df_result

    # TODO: add simulate_zeroshot_debug
    def simulate_zeroshot(self,
                          num_zeroshot: int = 10,
                          n_splits: int = 2,
                          backend: str = 'ray',
                          config_scorer_type: str = 'ensemble',
                          config_scorer_kwargs: dict = None) -> PortfolioCV:
        """
        Perform greedy-forward selection zeroshot simulation.

        :param num_zeroshot: The number of models in the portfolio to select.
        :param n_splits: The number of splits to perform in cross-validation.
            Larger values will take longer but produce better results.
        :param backend: One of ['ray', 'seq'].
            If 'ray', will parallelize across all cores to speed up the simulation.
            If 'seq', will only use a single process. Not recommended except for debugging.
        :param config_scorer_type: One of ['ensemble', 'single']
            If 'ensemble', will forward-select using the score achieved by val EnsembleSelection on the portfolio.
            If 'single', forward-select using the score achieved by the val single best model in the portfolio.
        :param config_scorer_kwargs: Optional kwargs used to initialize the config scorer.
            If config_scorer_type='ensemble', one example kwarg is 'ensemble_size',
            which dictates the number of greedy EnsembleSelection steps.
        :return: A PortfolioCV object that contains the final simulated portfolio for every task.
        """
        if config_scorer_kwargs is None:
            config_scorer_kwargs = {}
        if config_scorer_type == 'ensemble':
            config_scorer = self._construct_ensemble_selection_config_scorer(**config_scorer_kwargs)
        elif config_scorer_type == 'single':
            config_scorer = self._construct_single_best_config_scorer(**config_scorer_kwargs)
        else:
            raise ValueError(f'Unknown config_scorer_type: {config_scorer_type}')
        results_cv: PortfolioCV = run_zs_simulation(
            zsc=self._zeroshot_context,
            config_scorer=config_scorer,
            n_splits=n_splits,
            config_generator_kwargs={'num_zeroshot': num_zeroshot},
            backend=backend,
        )
        return results_cv
