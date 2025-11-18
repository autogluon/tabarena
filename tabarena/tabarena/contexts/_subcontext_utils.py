import random
from typing import Callable

from tabarena.repository import EvaluationRepository
from tabarena.utils.cache import cache_function


def gen_sample_repo(
        fun: Callable[[], EvaluationRepository],
        n_models: int = None,
        n_folds: int = None,
        n_datasets: int = None,
        random_seed: int = 0,
) -> EvaluationRepository:
    repo = fun()
    models = repo.configs()
    datasets = [repo.tid_to_dataset(taskid) for taskid in repo.tids()]
    folds_sample = None
    if n_folds is not None:
        folds_sample = list(range(n_folds))

    random.seed(random_seed)
    datasets_sample = None
    if n_datasets is not None:
        datasets_sample = random.sample(datasets, n_datasets)

    random.seed(random_seed + 1)
    models_sample = None
    if n_models is not None:
        models_sample = random.sample(models, n_models)

    return repo.subset(
        folds=folds_sample,
        datasets=datasets_sample,
        configs=models_sample,
    )


def gen_sample_repo_with_cache(
        fun: Callable[[], EvaluationRepository],
        cache_name_prefix: str,
        *,
        n_folds: int = None,
        n_datasets: int = None,
        n_models: int = None,
        random_seed: int = 0,
        ignore_cache: bool = False,
) -> EvaluationRepository:
    f"""
    Generate and cache a subset of a EvaluationRepository.
    Future calls will reload the cache, which is uniquely identified by the automatically generated cache file name.

    :param fun: The function call that will load the original repository prio to calling subset.
    :param cache_name_prefix: The name to use as the prefix of the cache name.
    :param n_folds: The number of folds to subset to. If None, will use all folds.
    :param n_datasets: The number of datasets to subset to. If None, will use all datasets.
    :param n_models: The number of models to subset to. If None, will use all models.
    :param random_seed: The random seed used when randomly selecting {n_datasets} and {n_models}.
    :param ignore_cache: If True, will compute the repo subset even if the cache already existed, and will overwrite it.
    :return: An EvaluationRepository object that is subset according to the parameters, and cached to disk.
    """
    cache_name = cache_name_prefix
    if n_datasets:
        cache_name += f'_D{n_datasets}'
    if n_folds:
        cache_name += f'_F{n_folds}'
    if n_models:
        cache_name += f'_C{n_models}'
    cache_name += f'_S{random_seed}'
    repo: EvaluationRepository = cache_function(lambda: gen_sample_repo(
        fun=fun,
        n_folds=n_folds,
        n_models=n_models,
        n_datasets=n_datasets,
        random_seed=random_seed,
    ), cache_name=cache_name, ignore_cache=ignore_cache)
    return repo
