from datetime import datetime
import math
import os
from typing import Dict, List

import boto3
import matplotlib.pyplot as plt

from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl

from .config_generator import ZeroshotConfigGeneratorCV
from .simulation_context import ZeroshotSimulatorContext
from ..portfolio import PortfolioCV
from ..utils import catchtime


# TODO: Return type hints + docstring
def run_zs_sim_end_to_end(subcontext_name: str,
                          num_zeroshot: int = 10,
                          n_splits: int = 2,
                          config_scorer_type: str = 'ensemble',
                          config_scorer_kwargs: dict = None,
                          subcontext_load_kwargs: dict = None,
                          backend='ray'):
    from ..contexts import get_subcontext
    from ..repository import EvaluationRepository
    if isinstance(subcontext_name, str):
        benchmark_subcontext = get_subcontext(subcontext_name)
        if subcontext_load_kwargs is None:
            subcontext_load_kwargs = dict()
        with catchtime(f"load {benchmark_subcontext.name}"):
            repo = benchmark_subcontext.load_from_parent(**subcontext_load_kwargs).to_zeroshot()
    elif isinstance(subcontext_name, EvaluationRepository):
        repo = subcontext_name.to_zeroshot()
    else:
        raise ValueError(f"Unsupported type for subcontext_name: {type(subcontext_name)}")

    repo.print_info()

    results_cv = repo.simulate_zeroshot(
        config_scorer_type=config_scorer_type,
        config_scorer_kwargs=config_scorer_kwargs,
        num_zeroshot=num_zeroshot,
        n_splits=n_splits,
        backend=backend,
    )
    print(f'Final Score: {results_cv.get_test_score_overall()}')
    return results_cv, repo


def run_zs_simulation(zsc: ZeroshotSimulatorContext,
                      config_scorer,
                      n_splits=10,
                      config_generator_kwargs=None,
                      configs=None,
                      backend='ray') -> PortfolioCV:
    zs_config_generator_cv = ZeroshotConfigGeneratorCV(
        n_splits=n_splits,
        zeroshot_simulator_context=zsc,
        config_scorer=config_scorer,
        config_generator_kwargs=config_generator_kwargs,
        configs=configs,
        backend=backend,
    )
    portfolio_cv = zs_config_generator_cv.run()

    portfolios = portfolio_cv.portfolios
    for i in range(len(portfolios)):
        print(f'Fold {portfolios[i].fold} Selected Configs: {portfolios[i].configs}')

    for i in range(len(portfolios)):
        print(f'Fold {portfolios[i].fold} | '
              f'Test Score: {portfolios[i].test_score} | '
              f'Train Score: {portfolios[i].train_score}')

    print(f'Overall (CV) | '
          f'Test Score: {portfolio_cv.get_test_score_overall()} | '
          f'Train Score: {portfolio_cv.get_train_score_overall()} | '
          f'Overfit Score: {portfolio_cv.get_test_train_rank_diff()}')

    return portfolio_cv


def plot_results_multi(portfolio_cv_lists: List[List[PortfolioCV]],
                       title: str = None,
                       footnote: str = None,
                       save_prefix: str = None,
                       save_to_s3: bool = True,
                       x_axis_col: str = 'step'):
    if title is None:
        title = f"Overfitting Delta"
    fig = plt.figure(dpi=300)
    ax = fig.subplots()
    for portfolio_cv_list in portfolio_cv_lists:
        df, num_train_tasks, num_test_tasks, n_configs_avail = get_test_train_rank_diff_df(portfolio_cv_list=portfolio_cv_list)
        ax.scatter(df[x_axis_col], df['overfit_delta'], alpha=0.5, label=f'tr_tasks={num_train_tasks}, n_conf={n_configs_avail}')
    ax.set_xlabel(x_axis_col)
    ax.set_ylabel('Overfit delta rank (lower is better)')
    ax.set_title(title)
    if footnote is not None:
        plt.figtext(0.99, 0.01, footnote, horizontalalignment='right')
    ax.grid()
    ax.legend()
    if save_prefix is not None:
        save_path = f'{save_prefix}overfit_delta_comparison.png'
        mkdir_path(save_path)
        plt.savefig(save_path)
    plt.show()

    # TODO: Split into separate function, avoid 2 plots in 1 function
    num_lists = len(portfolio_cv_lists)
    num_lists_sqrt = math.ceil(math.sqrt(num_lists))
    num_x = num_lists_sqrt
    num_y = num_lists_sqrt
    while num_x * num_y >= num_lists:
        num_y -= 1
    num_y += 1
    fig, axes = plt.subplots(num_y, num_x, sharex=True, sharey=True)

    try:
        axes_flat = axes.flat
    except AttributeError:
        # When only 1 subplot, returns an axes
        axes_flat = [axes]

    fig.set_size_inches(8, 8)
    fig.set_dpi(300)
    # ax.scatter(df['num_configs'], df['test_score'], alpha=0.5, label='test')
    # ax.scatter(df['num_configs'], df['train_score'], alpha=0.5, label='train')
    for ax, portfolio_cv_list in zip(axes_flat[:num_lists], portfolio_cv_lists):
        df, num_train_tasks, num_test_tasks, n_configs_avail = get_test_train_rank_diff_df(portfolio_cv_list=portfolio_cv_list)
        ax.errorbar(df[x_axis_col], df['test_error'], alpha=0.5, label=f'test', yerr=df['test_error_std'], fmt='o')
        ax.errorbar(df[x_axis_col], df['train_error'], alpha=0.5, label=f'train', yerr=df['train_error_std'], fmt='o')

        # ax.set_xlabel('num_configs')  # Add an x-label to the axes.
        # ax.set_ylabel('rank (lower is better)')  # Add a y-label to the axes.
        ax.set_title(f'tr_tasks={num_train_tasks}, n_conf={n_configs_avail}')  # Add a title to the axes.
        ax.grid()
    axes_flat[0].legend()
    # axes[0].set_xlabel('num_configs')  # Add an x-label to the axes.
    # axes.flat[0].set_ylabel('rank (lower is better)')  # Add a y-label to the axes.
    if footnote is not None:
        plt.figtext(0.99, 0.01, footnote, horizontalalignment='right')
    fig.suptitle(title)
    fig.supxlabel(x_axis_col)
    fig.supylabel('rank (lower is better)')
    if save_prefix is not None:
        save_path = f'{save_prefix}train_test_comparison.png'
        mkdir_path(save_path)
        plt.savefig(save_path)
    plt.show()

    if save_prefix is not None and save_to_s3:
        s3 = boto3.resource('s3')

        # FIXME: Won't work nicely if save_prefix is absolute path
        s3_bucket = 'autogluon-zeroshot'
        s3_prefix = f'{save_prefix}'

        for f in ['overfit_delta_comparison.png', 'train_test_comparison.png']:
            s3.Bucket(s3_bucket).upload_file(f"{save_prefix}{f}", f"{s3_prefix}{f}")


def get_test_train_rank_diff_df(portfolio_cv_list: List[PortfolioCV]):
    from collections import defaultdict
    df_dict = defaultdict(list)

    num_train_tasks = None
    num_test_tasks = None
    n_configs_avail = None

    for portfolio_cv in portfolio_cv_list:
        portfolio_cv.print_summary()
        assert portfolio_cv.is_dense(), f'PortfolioCV is not dense!'
        portfolios = portfolio_cv.portfolios
        if num_train_tasks is None:
            num_train_tasks = len(portfolios[0].train_datasets_fold)
        if num_test_tasks is None:
            num_test_tasks = len(portfolios[0].test_datasets_fold)
        if n_configs_avail is None:
            n_configs_avail = portfolios[0].n_configs_avail
        n_configs = len(portfolios[0].configs)
        step = portfolios[0].step
        # print(k)
        # for i in range(len(portfolios)):
        #     print(f'Fold {portfolios[i].fold} Selected Configs: {portfolios[i].configs}')
        #
        # for i in range(len(portfolios)):
        #     print(f'Fold {portfolios[i].fold} Test Score: {portfolios[i].test_score}')

        df_dict['n_configs'].append(n_configs)
        df_dict['step'].append(step)
        if portfolio_cv.has_test_score():
            df_dict['test_error'].append(portfolio_cv.get_test_score_overall())
            df_dict['test_error_std'].append(portfolio_cv.get_test_score_stddev())
        else:
            df_dict['test_error'].append(None)
            df_dict['test_error_std'].append(None)
        df_dict['train_error'].append(portfolio_cv.get_train_score_overall())
        df_dict['train_error_std'].append(portfolio_cv.get_train_score_stddev())


    import pandas as pd
    df = pd.DataFrame(df_dict)
    df['overfit_delta'] = df['test_error'] - df['train_error']
    return df, num_train_tasks, num_test_tasks, n_configs_avail


# TODO: Save list of list of porfolioCV, can then plot graphs separately from computing
def run_zs_simulation_debug(zsc: ZeroshotSimulatorContext,
                            config_scorer,
                            n_splits=10,
                            n_repeats=1,
                            config_generator_kwargs=None,
                            configs=None,
                            score_all=True,
                            backend='ray',
                            save_prefix=None,
                            num_halving=5) -> List[List[PortfolioCV]]:
    """
    num_halving:
        The number of times the training data is halved to measure increase in overfitting / decrease in test score

    """
    utcnow = datetime.utcnow()
    timestamp = utcnow.strftime("%Y%m%d_%H%M%S")
    zs_config_generator_cv = ZeroshotConfigGeneratorCV(
        n_splits=n_splits,
        n_repeats=n_repeats,
        zeroshot_simulator_context=zsc,
        config_scorer=config_scorer,
        config_generator_kwargs=config_generator_kwargs,
        configs=configs,
        backend=backend,
    )
    num_folds = len(zsc.folds)

    settings = [
        dict(
            sample_train_folds=None,
            sample_train_ratio=None,
        )
    ]

    if num_halving is not None and num_halving >= 1:
        cur_train_folds = num_folds
        cur_train_ratio = 1
        for i in range(num_halving):
            if cur_train_folds == 1:
                cur_train_ratio /= 2
            else:
                cur_train_folds = max(int(cur_train_folds / 2), 1)
            settings.append(
                dict(
                    sample_train_folds=cur_train_folds,
                    sample_train_ratio=cur_train_ratio,
                    # sample_configs_ratio=cur_train_ratio,
                )
            )

    portfolio_cv_lists = []
    for setting in settings:
        portfolio_cv_list = zs_config_generator_cv.run_and_return_all_steps(score_all=score_all, **setting)
        portfolio_cv_lists.append(portfolio_cv_list)

    # from autogluon.common.loaders import load_pkl
    # portfolio_cv_lists = load_pkl.load('s3://autogluon-zeroshot/output/unnamed/20230324_224848/pf_cv_lists.pkl')

    if save_prefix is None:
        save_prefix = 'unnamed'
    save_prefix = f'output/{save_prefix}/{timestamp}/'

    # TODO: Avoid hardcoding s3 bucket
    # TODO: Avoid forcing s3 save
    s3_bucket = 'autogluon-zeroshot'
    s3_save_path = f's3://{s3_bucket}/{save_prefix}'

    print(f'Saving output artifacts to local: {save_prefix}')
    print(f'Saving output artifacts to s3:    {s3_save_path}')

    # TODO: Instead of saving each file individually to s3, just copy the entire output dir to s3

    save_pkl.save(path=f'{save_prefix}pf_cv_lists.pkl', object=portfolio_cv_lists)
    save_pkl.save(path=f'{s3_save_path}pf_cv_lists.pkl', object=portfolio_cv_lists)

    plot_results_multi(portfolio_cv_lists,
                       title=f'Overfit Delta: n_configs={zs_config_generator_cv.get_n_configs()}, '
                             f'n_tasks={zs_config_generator_cv.get_n_tasks()}, '
                             f'n_splits={zs_config_generator_cv.n_splits}, '
                             f'n_repeats={zs_config_generator_cv.n_repeats}',
                       footnote=f'scorer={zs_config_generator_cv.config_scorer.__class__.__name__}',
                       save_prefix=f'{save_prefix}plots/')

    # plot_results(portfolio_cv_dict)

    return portfolio_cv_lists


def mkdir_path(path):
    path_parent = os.path.dirname(path)
    if path_parent == '':
        path_parent = '.'
    os.makedirs(path_parent, exist_ok=True)
