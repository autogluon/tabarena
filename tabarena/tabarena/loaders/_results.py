import pandas as pd

from autogluon.common.loaders import load_pd


def load_results(
    path_configs: str,
    path_metadata: str = None,
    metadata_join_column: str = "dataset",
    require_tid_in_metadata: bool = False,
    fix_tabrepo_dataset_names: bool = True,
) -> (pd.DataFrame, pd.DataFrame):
    print(f'Loading input files...\n'
          f'\tconfigs :           {path_configs}\n'
          f'\tmetadata:           {path_metadata}')
    if path_metadata is not None:
        df_metadata = load_pd.load(path_metadata)
    else:
        df_metadata = None
    df_configs = load_pd.load(path_configs)
    df_configs = preprocess_configs(df_configs=df_configs, inplace=True)

    if require_tid_in_metadata:
        if df_metadata is None:
            raise AssertionError('`metadata` must not be None if `require_tid_in_metadata=True`')
        if metadata_join_column not in df_configs:
            raise AssertionError(f"`metadata` file was specified but `configs` is missing the required `{metadata_join_column}` "
                                 f"column to join with `metadata`.\n"
                                 f"\tmetadata_join_column: {metadata_join_column}\n"
                                 f"\tpath         configs: {path_configs}\n"
                                 f"\tpath        metadata: {path_metadata}\n"
                                 f"\tcolumns      configs: {list(df_configs.columns)}\n"
                                 f"\tcolumns     metadata: {list(df_metadata.columns)}")
        if metadata_join_column not in df_metadata:
            raise AssertionError(f"`metadata` file was specified but `metadata` is missing the required `{metadata_join_column}` "
                                 f"column to join with `configs`.\n"
                                 f"\tmetadata_join_column: {metadata_join_column}\n"
                                 f"\tpath         configs: {path_configs}\n"
                                 f"\tpath        metadata: {path_metadata}\n"
                                 f"\tcolumns      configs: {list(df_configs.columns)}\n"
                                 f"\tcolumns     metadata: {list(df_metadata.columns)}")
        valid_values = set(list(df_metadata[metadata_join_column].unique()))
        init_dataset_count = len(df_configs["dataset"].unique())
        df_configs = df_configs[df_configs[metadata_join_column].isin(valid_values)]
        post_dataset_count = len(df_configs["dataset"].unique())
        if init_dataset_count != post_dataset_count:
            print(f'Filtered datasets with tids that are missing from metadata. '
                  f'Filtered from {init_dataset_count} datasets -> {post_dataset_count} datasets.')

    if df_metadata is not None:
        if "dataset" != metadata_join_column:
            unique_metadata_join_column_vals = df_configs[[metadata_join_column, "dataset"]].drop_duplicates()
            assert unique_metadata_join_column_vals["dataset"].value_counts().values.max() == 1, (f"Metadata join key `{metadata_join_column}` does not produce a 1:1 mapping with `dataset`.\n"
                                                                                                  f"Counts post-join (should always be 1):\n"
                                                                                                  f"{unique_metadata_join_column_vals['dataset'].value_counts()}")
            df_metadata = df_metadata.drop(columns=["dataset"], errors="ignore")
        else:
            metadata_column_order = ["dataset"] + [c for c in df_metadata.columns if c != "dataset"]
            df_metadata = df_metadata[metadata_column_order]  # make dataset first

    if fix_tabrepo_dataset_names and "name" in df_metadata:
        # Fix for TabRepo v1 datasets
        rename_dict = {
           'GAMETES_Epistasis_2-Way_1000atts_0.4H_EDM-1_EDM-1_1': 'GAMETES_Epistasis_2-Way_1000atts_0_4H_EDM-1_EDM-1_1',
           'GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1': 'GAMETES_Epistasis_2-Way_20atts_0_1H_EDM-1_1',
           'GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1': 'GAMETES_Epistasis_2-Way_20atts_0_4H_EDM-1_1',
           'GAMETES_Epistasis_3-Way_20atts_0.2H_EDM-1_1': 'GAMETES_Epistasis_3-Way_20atts_0_2H_EDM-1_1',
           'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001': 'GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM-2_001',
           'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_75_EDM-2_001': 'GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM-2_001',
           'numerai28.6': 'numerai28_6',
        }
        df_metadata["name"] = df_metadata["name"].map(rename_dict).fillna(df_metadata["name"])

    return df_configs, df_metadata


def get_metric_name(metric: str) -> str:
    metric_map = {
        "auc": "roc_auc",
        "neg_rmse": "rmse",
        "neg_logloss": "log_loss",
    }
    return metric_map.get(metric, metric)


def get_metric_error_from_score(score: float, metric: str) -> float:
    from autogluon.core.metrics import get_metric
    metric = get_metric_name(metric=metric)
    return get_metric(metric=metric).convert_score_to_error(score=score)


def preprocess_configs(df_configs: pd.DataFrame, inplace=True) -> pd.DataFrame:
    if not inplace:
        df_configs = df_configs.copy(deep=True)
    if "tid" in df_configs:
        df_configs['tid'] = df_configs['tid'].astype(int)
    df_configs["metric"] = df_configs["metric"].apply(lambda m: get_metric_name(metric=m))
    if "metric_error_val" not in df_configs:
        df_configs["metric_error_val"] = df_configs[["score_val", "metric"]].apply(
            lambda row: get_metric_error_from_score(score=row["score_val"], metric=row["metric"]), axis=1,
        )
    if "model" in df_configs:
        df_configs['framework'] = df_configs['model']
    return df_configs


def preprocess_baselines(df_baselines: pd.DataFrame, inplace=True) -> pd.DataFrame:
    if not inplace:
        df_baselines = df_baselines.copy(deep=True)
    if "tid" in df_baselines:
        df_baselines['tid'] = df_baselines['tid'].astype(int)
    df_baselines["metric"] = df_baselines["metric"].apply(lambda m: get_metric_name(metric=m))
    return df_baselines
