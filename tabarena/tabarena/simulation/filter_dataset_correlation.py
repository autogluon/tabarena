from typing import List
import seaborn as sns

from tabarena.contexts import get_context
from tabarena.simulation.simulation_context import ZeroshotSimulatorContext


def sort_datasets_linkage(zsc: ZeroshotSimulatorContext, datasets: List[float] = None) -> List[float]:
    """
    :param zsc:
    :param datasets: if passed only consider those datasets
    :return: datasets sorted by appearance order in hierarchical clustering linkage.
    Essentially, most typical datasets appear first and most outlier ones appear last.
    """
    df_pivot = zsc.df_configs_ranked.pivot_table(
        index="framework", columns="tid", values="metric_error"
    )
    df_rank = df_pivot.rank() / len(df_pivot)  # dataframe of ranks where columns are datasets and rows are models
    if datasets is not None:
        assert all(x in df_rank.columns for x in datasets)
        subcols = [x for x in df_rank.columns if x in datasets]
        df_rank = df_rank.loc[:, subcols]
    # TODO we could just call scipy
    cg = sns.clustermap(df_rank.corr())
    ordered_datasets_indices = cg.dendrogram_row.reordered_ind
    ordered_datasets = [df_rank.columns[i] for i in ordered_datasets_indices]
    return ordered_datasets


if __name__ == '__main__':
    bag = False
    if bag:
        context_name = 'BAG_D104_F10_C608_FULL'
    else:
        context_name = 'D104_F10_C608_FULL'
    benchmark_context = get_context(context_name)

    zsc, zeroshot_pred_proba, zeroshot_gt = benchmark_context.load(load_predictions=True, lazy_format=True)
    print(sort_datasets_linkage(zsc))

    sub_datasets = [359949.0, 359950.0, 359951.0, 359954.0, 359955.0, 359959.0,]
    res = sort_datasets_linkage(zsc, sub_datasets)
    print(res)
    for x in sub_datasets:
        assert x in res, x