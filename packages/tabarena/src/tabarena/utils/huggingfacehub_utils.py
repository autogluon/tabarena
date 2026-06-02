import os
from pathlib import Path
from huggingface_hub import HfApi
from tqdm import tqdm

from tabarena.utils.result_utils import results_path

def upload_hugging_face(
        version: str,
        repo_id: str,
        local_dir: Path | None = None,
        override_existing_files: bool = True,
        continue_in_case_of_error: bool = True
):
    """
    Uploads tabrepo data to Hugging Face repository.
    You should set your env variable HF_TOKEN and ask write access to tabrepo before using the script.

    Args:
        version (str): The version of the data to be uploaded, the folder data/results/{version}/ should
        be present and should contain baselines.parquet, configs.parquet and model_predictions/ folder
        repo_id (str): The ID of the Hugging Face repository.
        local_dir (Path): path to load datasets, use tabrepo default if not specified
        override_existing_files (bool): Whether to re-upload files if they are already found in HuggingFace.
    Returns:
        None
    """
    commit_message = f"Upload tabrepo new version"
    if local_dir is None:
        local_dir = str(results_path())
    else:
        local_dir = str(local_dir)

    for filename in ["baselines.parquet", "configs.parquet", "model_predictions"]:
        assert (root / filename).exists(), f"Expected to found {filename} but could not be found in {root / filename}."
    api = HfApi()
    for filename in ["baselines.parquet", "configs.parquet"]:
        path_in_repo = str(Path(version) / filename)
        if api.file_exists(repo_id=repo_id, filename=path_in_repo, token=os.getenv("HF_TOKEN"), repo_type="dataset") and not override_existing_files:
            print(f"Skipping {path_in_repo} which already exists in the repo.")
            continue

        api.upload_file(
            path_or_fileobj=root / filename,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
            token=os.getenv("HF_TOKEN"),
        )
    files = list(sorted(Path(root / "model_predictions").glob("*")))
    for dataset_path in tqdm(files):
        print(dataset_path)
        try:
            path_in_repo = str(Path(version) / "model_predictions" / dataset_path.name)
            # ideally, we would just check if the folder exists but it is not possible AFAIK, we could alternatively
            # upload per file but it would create a lot of different commits.
            if api.file_exists(repo_id=repo_id, filename=str(Path(path_in_repo) / "0" / "metadata.json"), token=os.getenv("HF_TOKEN"),
                               repo_type="dataset") and not override_existing_files:
                print(f"Skipping {path_in_repo} which already exists in the repo.")
                continue
            api.upload_folder(
                folder_path=dataset_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                ignore_patterns="*DS_Store",
                commit_message=f"Upload tabrepo new version {dataset_path.name}",
                token=os.getenv("HF_TOKEN"),
            )
        except Exception as e:
            if continue_in_case_of_error:
                print(str(e))
            else:
                raise e


def download_from_huggingface(
        version: str = "2023_11_14",
        force_download: bool = False,
        local_files_only: bool = False,
        datasets: list[str] | None = None,
        local_dir: str | Path = None,
):
    """
    :param version: name of a tabrepo version such as `2023_11_14`
    :param local_files_only: whether to use local files with no internet check on the Hub
    :param force_download: forces files to be downloaded
    :param datasets: list of datasets to download, if not specified all datasets will be downloaded
    :param local_dir: where to download local files, if not specified all files will be downloaded under
     {tabrepo_root}/data/results
    :return:
    """
    # https://huggingface.co/datasets/Tabrepo/tabrepo/tree/main/2023_11_14/model_predictions
    api = HfApi()
    if local_dir is None:
        local_dir = str(results_path())
    else:
        local_dir = str(local_dir)
    print(f"Going to download tabrepo files to {local_dir}.")
    if datasets is None:
        allow_patterns = None
    else:
        allow_patterns = [f"*{version}*{d}*" for d in datasets]

    allow_patterns += [
        "*baselines.parquet",
        "*configs.parquet",
        "*task_metadata.csv",
    ]

    print(f"Allowed patterns: {allow_patterns}")
    api.snapshot_download(
        repo_id="Tabrepo/tabrepo",
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=local_dir,
        force_download=force_download,
        local_files_only=local_files_only,
    )
if __name__ == '__main__':
    # upload_hugging_face(
    #     version="2023_11_14",
    #     repo_id="tabrepo/tabrepo",
    #     override_existing_files=False,
    # )
    datasets = [
        'Australian',
    ]
    download_from_huggingface(
        datasets=datasets,
        version="2023_11_14",
    )