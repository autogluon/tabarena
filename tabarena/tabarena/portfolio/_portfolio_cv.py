from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as st


# TODO: Consider including more, such as metric, metric_error, so that we don't need to calculate that later
class Portfolio:
    def __init__(self,
                 configs: List[str],
                 train_score: float = None,
                 test_score: float = None,
                 train_datasets: List[str] = None,
                 test_datasets: List[str] = None,
                 train_datasets_fold: List[str] = None,
                 test_datasets_fold: List[str] = None,
                 fold: int = None,
                 split: int = None,
                 repeat: int = None,
                 step: int = None,
                 n_configs_avail: int = None):
        self.configs = configs
        self.train_score = train_score
        self.test_score = test_score
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.train_datasets_fold = train_datasets_fold
        self.test_datasets_fold = test_datasets_fold
        self.fold = fold
        self.split = split
        self.repeat = repeat

        # Optional, the step in which this portfolio was generated
        # (relevant when portfolio is generated through iterative steps)
        self.step = step

        self.n_configs_avail = n_configs_avail  # Number of configs to choose from at fit time


class PortfolioCV:
    """
    Contains a list of Portfolios generated during cross-validation.
    """
    def __init__(self, portfolios: List[Portfolio]):
        self.portfolios = portfolios

    def is_dense(self) -> bool:
        """
        Return True if for each repeat all test datasets appear in exactly one split
        """
        portfolios_by_repeat = self.get_portfolios_by_repeat()
        for r in portfolios_by_repeat:
            is_dense_repeat = self.are_test_folds_unique(portfolios=portfolios_by_repeat[r])
            if not is_dense_repeat:
                return False
        return True

    def num_configs_max(self) -> int:
        return np.max([len(p.configs) for p in self.portfolios])

    def num_configs_avail_max(self) -> int:
        return np.max([p.n_configs_avail for p in self.portfolios])

    def split_max(self) -> int:
        return np.max([p.split for p in self.portfolios])

    def step_max(self) -> int:
        return np.max([p.step for p in self.portfolios])

    def print_summary(self):
        is_dense = self.is_dense()
        num_repeats = self.num_repeats()
        num_portfolios = len(self.portfolios)
        print(f'Summarizing PortfolioCV:\n'
              f'\tis_dense={is_dense}, num_portfolios={num_portfolios}, num_repeats={num_repeats}, '
              f'split_max={self.split_max()}, num_configs_max={self.num_configs_max()}, '
              f'num_configs_avail_max={self.num_configs_avail_max()}, step_max={self.step_max()}'
              f'\n'
              f'\ttrain_error   = {self.get_train_score_overall():.5f}'
              f'\ttrain_stddev  = {self.get_train_score_stddev():.5f}'
              f'\ttrain_95_conf = {self.get_train_score_conf_from_repeats():.5f}\n'
              f'\t test_error   = {self.get_test_score_overall():.5f}'
              f'\t test_stddev  = {self.get_test_score_stddev():.5f}'
              f'\t test_95_conf = {self.get_test_score_conf_from_repeats():.5f}'
              f'')

    def get_portfolios_by_repeat(self) -> Dict[int, List[Portfolio]]:
        portfolio_by_repeat_dict = defaultdict(list)
        for portfolio in self.portfolios:
            portfolio_by_repeat_dict[portfolio.repeat].append(portfolio)
        return dict(portfolio_by_repeat_dict)

    def repeats(self) -> List[int]:
        repeats = set()
        for p in self.portfolios:
            repeats.add(p.repeat)
        repeats = sorted(list(repeats))
        return repeats

    def num_repeats(self):
        return len(self.repeats())

    def has_test_score(self) -> bool:
        for portfolio in self.portfolios:
            if portfolio.test_score is None:
                return False
        return True

    def get_test_scores(self) -> List[float]:
        return [portfolio.test_score for portfolio in self.portfolios]

    def get_train_scores(self) -> List[float]:
        return [portfolio.train_score for portfolio in self.portfolios]

    def get_test_scores_per_repeat(self) -> List[float]:
        repeats = self.repeats()
        repeat_error_dict = defaultdict(list)
        for portfolio in self.portfolios:
            repeat_error_dict[portfolio.repeat].append(portfolio.test_score)
        return [np.mean(repeat_error_dict[repeat]) for repeat in repeats]

    def get_train_scores_per_repeat(self) -> List[float]:
        repeats = self.repeats()
        repeat_error_dict = defaultdict(list)
        for portfolio in self.portfolios:
            repeat_error_dict[portfolio.repeat].append(portfolio.train_score)
        return [np.mean(repeat_error_dict[repeat]) for repeat in repeats]

    def get_test_score_stddev(self) -> float:
        return np.std(self.get_test_scores())

    def get_train_score_stddev(self) -> float:
        return np.std(self.get_train_scores())

    def get_test_score_conf_from_folds(self, confidence=0.95):
        test_errors = self.get_test_scores()
        return self._error_to_t_interval(errors=test_errors, confidence=confidence)

    def get_train_score_conf_from_folds(self, confidence=0.95):
        train_errors = self.get_train_scores()
        return self._error_to_t_interval(errors=train_errors, confidence=confidence)

    def get_test_score_conf_from_repeats(self, confidence=0.95):
        """Most accurate way to compute conf bound, but requires multiple repeats"""
        test_errors = self.get_test_scores_per_repeat()
        return self._error_to_t_interval(errors=test_errors, confidence=confidence)

    def get_train_score_conf_from_repeats(self, confidence=0.95):
        """Most accurate way to compute conf bound, but requires multiple repeats"""
        train_errors = self.get_train_scores_per_repeat()
        return self._error_to_t_interval(errors=train_errors, confidence=confidence)

    def _error_to_t_interval(self, errors: List[float], confidence: float = 0.95) -> float:
        t_interval = st.t.interval(confidence=confidence,
                                   df=len(errors)-1,
                                   loc=np.mean(errors),
                                   scale=st.sem(errors))
        t_interval_mean = np.mean(t_interval)
        t_interval_deviation = t_interval_mean - t_interval[0]
        return t_interval_deviation

    def get_test_score_overall(self) -> float:
        total_num_datasets = 0
        total_test_score = 0
        for portfolio in self.portfolios:
            test_score = portfolio.test_score
            num_datasets = len(portfolio.test_datasets_fold)
            total_num_datasets += num_datasets
            total_test_score += test_score*num_datasets
        test_score = total_test_score / total_num_datasets
        return test_score

    def get_train_score_overall(self):
        total_num_datasets = 0
        total_train_score = 0
        for portfolio in self.portfolios:
            train_score = portfolio.train_score
            num_datasets = len(portfolio.train_datasets_fold)
            total_num_datasets += num_datasets
            total_train_score += train_score*num_datasets
        train_score = total_train_score / total_num_datasets
        return train_score

    def are_test_folds_unique(self, portfolios: Optional[List[Portfolio]] = None) -> bool:
        """
        Return True if each test dataset is only present in one fold.
        An example of when this would return False is repeated k-fold.
        """
        if portfolios is None:
            portfolios = self.portfolios
        seen_datasets_folds = set()
        for p in portfolios:
            for d in p.test_datasets_fold:
                if d not in seen_datasets_folds:
                    seen_datasets_folds.add(d)
                else:
                    return False
        return True

    @classmethod
    def combine(cls, portfolio_cv_list: list):
        """
        Combine a list of PortfolioCV objects into one
        """
        portfolios = []
        for p in portfolio_cv_list:
            portfolios += p.portfolios
        return PortfolioCV(portfolios=portfolios)

    def get_test_train_rank_diff(self) -> float:
        """
        Returns the amount of overfitting that has occurred.
        AKA: How overoptimistic the portfolio is on training compared to test
        Lower = Better
        """
        return self.get_test_score_overall() - self.get_train_score_overall()
