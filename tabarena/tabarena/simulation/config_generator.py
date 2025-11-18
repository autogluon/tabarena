import copy
from collections import defaultdict
import math
import random
import time
from typing import Any, Dict, List

import numpy as np
import ray
from sklearn.model_selection import RepeatedKFold

from .configuration_list_scorer import ConfigurationListScorer
from .simulation_context import ZeroshotSimulatorContext
from ..portfolio import Portfolio, PortfolioCV


@ray.remote
def score_config_ray(config_scorer, existing_configs, new_config) -> float:
    configs = existing_configs + [new_config]
    score = config_scorer.score(configs)
    return score


class ZeroshotConfigGenerator:
    def __init__(self, config_scorer, configs: List[str], backend='ray'):
        self.config_scorer = config_scorer
        self.all_configs = configs
        self.backend = backend

    def select_zeroshot_configs(self,
                                num_zeroshot: int,
                                zeroshot_configs: List[str] = None,
                                removal_stage=False,
                                removal_threshold=0,
                                config_scorer_test=None,
                                return_all_metadata: bool = False,
                                ) -> (List[Dict[str, Any]]):
        zeroshot_configs = [] if zeroshot_configs is None else copy.deepcopy(zeroshot_configs)
        metadata_list = []

        iteration = 0
        if self.backend == 'ray':
            if not ray.is_initialized():
                ray.init()
            config_scorer = ray.put(self.config_scorer)
            selector = self._select_ray
        else:
            config_scorer = self.config_scorer
            selector = self._select_sequential

        metadata_out = None
        while len(zeroshot_configs) < num_zeroshot:
            # greedily search the config that would yield the lowest average rank if we were to evaluate it in combination
            # with previously chosen configs.

            valid_configs = [c for c in self.all_configs if c not in zeroshot_configs]
            if not valid_configs:
                if not return_all_metadata:
                    metadata_list = [metadata_out]
                break
            iteration += 1

            time_start = time.time()
            best_next_config, best_train_score = selector(valid_configs, zeroshot_configs, config_scorer)
            time_end = time.time()

            zeroshot_configs.append(best_next_config)
            fit_time = time_end - time_start
            msg = f'{iteration}\t: Train: {round(best_train_score, 2)}'
            if config_scorer_test:
                test_score = config_scorer_test.score(zeroshot_configs)
                msg += f'\t| Test: {round(test_score, 2)} \t| Overfit: {round(test_score-best_train_score, 2)}'
            else:
                test_score = None
            msg += f' | {round(fit_time, 2)}s | {self.backend} | {best_next_config}'
            # print('here, make metadata')
            metadata_out = dict(
                configs=copy.deepcopy(zeroshot_configs),
                new_config=best_next_config,
                step=iteration,
                train_score=best_train_score,
                test_score=test_score,
                num_configs=len(zeroshot_configs),
                fit_time=fit_time,
                backend=self.backend,
            )
            is_last = len(zeroshot_configs) >= num_zeroshot
            if return_all_metadata or is_last:
                metadata_list.append(metadata_out)

            print(msg)
        if removal_stage:
            zeroshot_configs = self.prune_zeroshot_configs(zeroshot_configs, removal_threshold=removal_threshold)

            # FIXME: Get this from prune instead
            metadata_out = self._get_metadata_from_configs(
                configs=zeroshot_configs,
                step=iteration+1,
                train_score=None,
                test_score=None,
                config_scorer=self.config_scorer,
                config_scorer_test=config_scorer_test,
            )

            if return_all_metadata:
                metadata_list.append(metadata_out)
            else:
                metadata_list = [metadata_out]

        print(f"selected {zeroshot_configs}")
        # TODO: metadata_list not updated by prune_zeroshot_configs
        return metadata_list

    def _get_metadata_from_configs(self,
                                   configs,
                                   new_config=None,
                                   step=None,
                                   train_score=None,
                                   test_score=None,
                                   fit_time=None,
                                   config_scorer=None,
                                   config_scorer_test=None) -> Dict[str, Any]:
        if train_score is None and config_scorer is not None:
            train_score = config_scorer.score(configs)
        if test_score is None and config_scorer_test is not None:
            test_score = config_scorer_test.score(configs)
        return dict(
            configs=copy.deepcopy(configs),
            new_config=new_config,
            step=step,
            train_score=train_score,
            test_score=test_score,
            num_configs=len(configs),
            fit_time=fit_time,
            backend=self.backend,
        )

    @staticmethod
    def _select_sequential(configs: list, prior_configs: list, config_scorer):
        best_next_config = None
        # todo could use np.inf but would need unit-test (also to check that ray/sequential returns the same selection)
        best_score = 999999999
        for config in configs:
            config_selected = prior_configs + [config]
            config_score = config_scorer.score(config_selected)
            if config_score < best_score:
                best_score = config_score
                best_next_config = config
        return best_next_config, best_score

    @staticmethod
    def _select_ray(configs: list, prior_configs: list, config_scorer):
        # Create and execute all tasks in parallel
        results = []
        for i in range(len(configs)):
            results.append(score_config_ray.remote(
                config_scorer,
                prior_configs,
                configs[i],
            ))
        result = ray.get(results)
        result_idx_min = result.index(min(result))
        best_next_config = configs[result_idx_min]
        best_score = result[result_idx_min]
        return best_next_config, best_score

    def prune_zeroshot_configs(self, zeroshot_configs: List[str], removal_threshold=0) -> List[str]:
        zeroshot_configs = copy.deepcopy(zeroshot_configs)
        best_score = self.config_scorer.score(zeroshot_configs)
        finished_removal = False
        while not finished_removal:
            best_remove_config = None
            for config in zeroshot_configs:
                config_selected = [c for c in zeroshot_configs if c != config]
                config_score = self.config_scorer.score(config_selected)

                if best_remove_config is None:
                    if config_score <= (best_score + removal_threshold):
                        best_score = config_score
                        best_remove_config = config
                else:
                    if config_score <= best_score:
                        best_score = config_score
                        best_remove_config = config
            if best_remove_config is not None:
                print(f'REMOVING: {best_score} | {best_remove_config}')
                zeroshot_configs.remove(best_remove_config)
            else:
                finished_removal = True
        return zeroshot_configs


class ZeroshotConfigGeneratorCV:
    def __init__(self,
                 n_splits: int,
                 zeroshot_simulator_context: ZeroshotSimulatorContext,
                 config_scorer: ConfigurationListScorer,
                 config_generator_kwargs=None,
                 configs: List[str] = None,
                 n_repeats: int = 1,
                 backend='ray'):
        """
        Runs zero-shot selection on `n_splits` ("train", "test") folds of datasets.
        For each split, zero-shot configurations are selected using the datasets belonging on the "train" split and the
        performance of the zero-shot configuration is evaluated using the datasets in the "test" split.
        :param n_splits: number of splits for RepeatedKFold
        :param zeroshot_simulator_context:
        :param config_scorer:
        :param configs:
        :param n_repeats: number of repeats for RepeatedKFold
        :param backend:
        """
        assert n_splits >= 2
        assert n_repeats >= 1
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        if config_generator_kwargs is None:
            config_generator_kwargs = {}
        self.config_generator_kwargs = config_generator_kwargs
        self.backend = backend
        self.config_scorer = config_scorer
        self.unique_tasks = np.array(config_scorer.tasks)
        self.task_to_dataset_dict = zeroshot_simulator_context.task_to_dataset_dict
        self.unique_datasets = set()
        self.dataset_to_task_map = dict()
        for task in self.unique_tasks:
            dataset = self.task_to_dataset_dict[task]
            self.unique_datasets.add(dataset)
            if dataset in self.dataset_to_task_map:
                self.dataset_to_task_map[dataset].append(task)
            else:
                self.dataset_to_task_map[dataset] = [task]
        for dataset in self.dataset_to_task_map:
            self.dataset_to_task_map[dataset] = sorted(self.dataset_to_task_map[dataset])
        self.unique_datasets = np.array((sorted(list(self.unique_datasets))))

        if configs is None:
            configs = zeroshot_simulator_context.get_configs()
        self.configs = configs

        self.kf = RepeatedKFold(n_splits=self.n_splits, random_state=0, n_repeats=self.n_repeats)

    def get_n_tasks(self) -> int:
        return len(self.unique_tasks)

    def get_n_datasets(self) -> int:
        return len(self.unique_datasets)

    def get_n_configs(self) -> int:
        return len(self.configs)

    def _get_tasks_from_datasets(self, datasets: List[str], num_folds=None) -> List[str]:
        tasks = []
        for d in datasets:
            tasks += self._get_tasks_from_dataset(dataset=d, num_folds=num_folds)
        return tasks

    def _get_tasks_from_dataset(self, dataset: str, num_folds=None) -> List[str]:
        if num_folds is None:
            return self.dataset_to_task_map[dataset]
        else:
            return self.dataset_to_task_map[dataset][:num_folds]

    def _get_split(self, fold: int) -> int:
        """
        Note: fold 0 is the first fold, not fold 1.
        split 0 is the first split.
        """
        return fold % self.n_splits

    def _get_repeat(self, fold: int) -> int:
        """
        repeat 0 is the first repeat
        """
        return int(fold / self.n_splits)

    def run_and_return_all_steps(self,
                                 sample_train_folds: int = None,
                                 sample_train_ratio: float = None,
                                 sample_configs_ratio: float = None,
                                 # TODO: Sample configs
                                 score_all: bool = True,
                                 score_final: bool = True,
                                 return_all_metadata: bool = True) -> List[PortfolioCV]:
        """
        Run cross-validated zeroshot simulation.

        :param sample_train_folds:
            Number of folds to filter training data to for each fold. Used for debugging.
            Lower values should result in worse test scores and higher overfit scores
            If set to a value larger than the folds available in the datasets, it will have no effect.
        :param sample_train_ratio:
            Ratio of datasets to filter training data to for each fold. Used for debugging.
            Lower values should result in worse test scores and higher overfit scores
        :param sample_configs_ratio:
            Ratio of configs to filter to for each fold. Used for debugging.
            Lower values should result in worse test scores and lower overfit scores
        :param score_all: If True, calculates test score at each step of the zeroshot simulation process.
        :param score_final: If True, calculates test score for the final step of the zeroshot simulation process.
        :param return_all_metadata:
            If True, returns N elements in a list, with the index referring to the order of selection.
                Note: If folds differ in number of simulation steps, this will raise an exception.
                For example, config pruning as a post-processing step to greedy selection
                of N elements could have differing step counts.
            If False, returns a list with only 1 element corresponding to the final zeroshot config.
        """
        results_dict_by_len = defaultdict(list)
        n_configs_total = self.get_n_configs()
        is_first_fold = True
        configs = self.configs
        print(f'Fitting CV\n'
              f'\tscorer={self.config_scorer.__class__.__name__}\n'
              f'\tn_splits={self.n_splits}, n_repeats={self.n_repeats}, n_configs={n_configs_total}\n'
              f'\tn_datasets={self.get_n_datasets()}, n_tasks={self.get_n_tasks()}\n'
              f'\tsample_train_folds={sample_train_folds} '
              f'| sample_train_ratio={sample_train_ratio} '
              f'| sample_configs_ratio={sample_configs_ratio}'
              )
        for i, (train_index, test_index) in enumerate(self.kf.split(self.unique_datasets)):
            split = self._get_split(i)
            repeat = self._get_repeat(i)
            X_train, X_test = list(self.unique_datasets[train_index]), list(self.unique_datasets[test_index])
            len_X_train_og = len(X_train)
            len_X_test_og = len(X_test)
            if split == 0 or is_first_fold:
                if sample_configs_ratio is not None and sample_configs_ratio < 1:
                    random.seed(repeat)
                    num_samples = math.ceil(n_configs_total * sample_configs_ratio)
                    configs = random.sample(self.configs, num_samples)
            if sample_train_ratio is not None and sample_train_ratio < 1:
                random.seed(0)
                num_samples = math.ceil(len(X_train) * sample_train_ratio)
                X_train = random.sample(X_train, num_samples)
            random.seed(0)
            n_configs_avail = len(configs)
            train_tasks = self._get_tasks_from_datasets(datasets=X_train, num_folds=sample_train_folds)
            test_tasks = self._get_tasks_from_datasets(datasets=X_test)
            len_train_tasks = len(train_tasks)
            len_train_datasets = len(X_train)
            print(f'Fitting Fold {i + 1}/{self.n_splits*self.n_repeats} (R{repeat+1}S{split+1})...\n'
                  f'\ttrain_datasets: {len_train_datasets}/{len_X_train_og} | train_tasks: {len_train_tasks}\n'
                  f'\ttest_datasets : {len(X_test)}/{len_X_test_og} | test_tasks : {len(test_tasks)}\n'
                  f'\tn_configs={n_configs_avail}/{n_configs_total}'
                  )
            metadata_fold = self.run_fold(train_tasks,
                                          test_tasks,
                                          configs=configs,
                                          score_all=score_all,
                                          score_final=score_final,
                                          return_all_metadata=return_all_metadata)
            for j, m in enumerate(metadata_fold):
                # FIXME: It is possible not all folds will have results match up correctly
                #  if we introduce config pruning.
                #  This logic should probably only be present in scenarios where we are debugging
                #  Otherwise we should only take the final result of each fold.
                results_fold_i = Portfolio(
                    configs=m['configs'],
                    train_score=m['train_score'],
                    test_score=m['test_score'],
                    train_datasets=X_train,
                    test_datasets=X_test,
                    train_datasets_fold=train_tasks,
                    test_datasets_fold=test_tasks,
                    fold=i + 1,
                    split=split+1,
                    repeat=repeat+1,
                    step=m['step'],
                    n_configs_avail=n_configs_avail,
                )
                results_dict_by_len[j].append(results_fold_i)
            is_first_fold = False
        for val in results_dict_by_len.values():
            assert (self.n_splits * self.n_repeats) == len(val)  # Ensure no bugs such as only getting a subset of fold results
        portfolio_cv_list = [PortfolioCV(portfolios=v) for k, v in results_dict_by_len.items()]
        return portfolio_cv_list

    def run(self,
            sample_train_folds=None,
            sample_train_ratio=None,
            score_all=False,
            score_final=True) -> PortfolioCV:
        """
        Identical to `run_and_return_all_steps`, but the output is simply the final PortfolioCV.

        score_all is also set to False by default to speed up the simulation.
        """
        results_cv_list = self.run_and_return_all_steps(sample_train_folds=sample_train_folds,
                                                        sample_train_ratio=sample_train_ratio,
                                                        score_all=score_all,
                                                        score_final=score_final,
                                                        return_all_metadata=False)

        assert len(results_cv_list) == 1
        results_cv = results_cv_list[0]

        return results_cv

    def run_fold(self,
                 train_tasks: List[str],
                 test_tasks: List[str],
                 configs: List[str] = None,
                 score_all=False,
                 score_final=True,
                 return_all_metadata=False) -> List[Dict[str, Any]]:
        if configs is None:
            configs = self.configs
        config_scorer_train = self.config_scorer.subset(tasks=train_tasks)
        config_scorer_test = self.config_scorer.subset(tasks=test_tasks)

        zs_config_generator = ZeroshotConfigGenerator(config_scorer=config_scorer_train,
                                                      configs=configs,
                                                      backend=self.backend)

        num_zeroshot = self.config_generator_kwargs.get('num_zeroshot', 10)
        removal_stage = self.config_generator_kwargs.get('removal_stage', False)

        metadata_list = zs_config_generator.select_zeroshot_configs(
            num_zeroshot=num_zeroshot,
            removal_stage=removal_stage,
            config_scorer_test=config_scorer_test if score_all else None,
            return_all_metadata=return_all_metadata,
        )
        # deleting
        # FIXME: SPEEDUP WITH RAY
        # zeroshot_configs = zs_config_generator.prune_zeroshot_configs(zeroshot_configs, removal_threshold=0)

        if score_final and metadata_list[-1]['test_score'] is None:
            score = config_scorer_test.score(metadata_list[-1]['configs'])
            print(f'test_score: {score}')
            metadata_list[-1]['test_score'] = score

        return metadata_list
