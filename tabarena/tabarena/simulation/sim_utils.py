import pandas as pd


# FIXME: Doesn't work for multi-fold
def get_dataset_to_tid_dict(df: pd.DataFrame) -> dict:
    df_tid_to_dataset_map = df[['tid', 'dataset']].drop_duplicates(['tid', 'dataset'])
    dataset_to_tid_dict = df_tid_to_dataset_map.set_index('dataset')
    dataset_to_tid_dict = dataset_to_tid_dict['tid'].to_dict()
    return dataset_to_tid_dict


def get_task_to_dataset_dict(df: pd.DataFrame) -> dict:
    df_tid_to_dataset_map = df[['task', 'dataset']].drop_duplicates(['task', 'dataset'])
    dataset_to_tid_dict = df_tid_to_dataset_map.set_index('task')
    dataset_to_tid_dict = dataset_to_tid_dict['dataset'].to_dict()
    return dataset_to_tid_dict


def filter_datasets(df: pd.DataFrame, datasets: pd.DataFrame) -> pd.DataFrame:
    return df.merge(datasets, on=["dataset", "fold"])


def get_dataset_to_metric_problem_type(df_configs: pd.DataFrame, df_baselines: pd.DataFrame) -> pd.DataFrame:
    df_min = df_configs[["dataset", "metric", "problem_type"]].drop_duplicates()
    df_min_baselines = df_baselines[["dataset", "metric", "problem_type"]].drop_duplicates()
    df_min = pd.concat([df_min, df_min_baselines], ignore_index=True).drop_duplicates()
    counts = df_min["dataset"].value_counts().to_dict()
    for dataset in counts:
        if counts[dataset] != 1:
            df_dataset = df_min[df_min["dataset"] == dataset]
            raise AssertionError(f"Error: Multiple `problem_type` or `metric` values defined in the data for dataset {dataset}\n:"
                                 f"{df_dataset}")
    df_min = df_min.set_index("dataset")
    return df_min
