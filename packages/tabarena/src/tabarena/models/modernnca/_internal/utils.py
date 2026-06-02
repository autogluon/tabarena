# from https://github.com/LAMDA-Tabular/TALENT/blob/cb6cb0cc9d69ac75c467e8dae8ca5ac3d3beb2f2/TALENT/model/utils.py#L1
from __future__ import annotations

import errno
import json
import os
import random
import shutil
import time

import numpy as np
import sklearn.metrics as skm
import torch

THIS_PATH = os.path.dirname(__file__)


def mkdir(path):
    """Create a directory if it does not exist.

    :path: str, path to the directory
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def set_gpu(x):
    """Set environment variable CUDA_VISIBLE_DEVICES.

    :x: str, GPU id
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = x
    print("using gpu:", x)


def ensure_path(path, remove=True):
    """Ensure a path exists.

    path: str, path to the directory
    remove: bool, whether to remove the directory if it exists
    """
    if os.path.exists(path):
        if remove and input(f"{path} exists, remove? ([y]/n)") != "n":
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)


#  --- criteria helper ---
class Averager:
    """A simple averager."""

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        """:x: float, value to be added."""
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer:
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        """Measure the time since the last call to measure.

        :p: int, period of printing the time
        """
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return f"{x / 3600:.1f}h"
        if x >= 60:
            return f"{round(x / 60)}m"
        return f"{x}s"


#  ---- import from lib.util -----------
def set_seeds(base_seed: int, one_cuda_seed: bool = False) -> None:
    """Set random seeds for reproducibility.

    :base_seed: int, base seed
    :one_cuda_seed: bool, whether to set one seed for all GPUs
    """
    assert 0 <= base_seed < 2**32 - 10000
    random.seed(base_seed)
    np.random.seed(base_seed + 1)
    torch.manual_seed(base_seed + 2)
    cuda_seed = base_seed + 3
    if one_cuda_seed:
        torch.cuda.manual_seed_all(cuda_seed)
    elif torch.cuda.is_available():
        # the following check should never succeed since torch.manual_seed also calls
        # torch.cuda.manual_seed_all() inside; but let's keep it just in case
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        # Source: https://github.com/pytorch/pytorch/blob/2f68878a055d7f1064dded1afac05bb2cb11548f/torch/cuda/random.py#L109
        for i in range(torch.cuda.device_count()):
            default_generator = torch.cuda.default_generators[i]
            default_generator.manual_seed(cuda_seed + i)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rmse(y, prediction, y_info):
    """:y: np.ndarray, ground truth
    :prediction: np.ndarray, prediction
    :y_info: dict, information about the target variable
    :return: float, root mean squared error.
    """
    rmse = skm.mean_squared_error(y, prediction) ** 0.5  # type: ignore[code]
    if y_info["policy"] == "mean_std":
        rmse *= y_info["std"]
    return rmse


def load_config(args, config=None, config_name=None):
    """Load the config file.

    :args: argparse.Namespace, arguments
    :config: dict, config file
    :config_name: str, name of the config file
    :return: argparse.Namespace, arguments
    """
    if config is None:
        config_path = os.path.join(
            os.path.abspath(os.path.join(THIS_PATH, "..")),
            "configs",
            args.dataset,
            f"{args.model_type if args.config_name is None else args.config_name}.json",
        )
        with open(config_path) as fp:
            config = json.load(fp)

    # set additional parameters
    args.config = config

    # save the config files
    with open(
        os.path.join(args.save_path, "{}.json".format("config" if config_name is None else config_name)),
        "w",
    ) as fp:
        args_dict = vars(args)
        if "device" in args_dict:
            del args_dict["device"]
        json.dump(args_dict, fp, sort_keys=True, indent=4)

    return args


# parameter search
def sample_parameters(trial, space, base_config):
    """Sample hyper-parameters.

    :trial: optuna.trial.Trial, trial
    :space: dict, search space
    :base_config: dict, base configuration
    :return: dict, sampled hyper-parameters
    """

    def get_distribution(distribution_name):
        return getattr(trial, f"suggest_{distribution_name}")

    result = {}
    for label, subspace in space.items():
        if isinstance(subspace, dict):
            result[label] = sample_parameters(trial, subspace, base_config)
        else:
            assert isinstance(subspace, list)
            distribution, *args = subspace

            if distribution.startswith("?"):
                default_value = args[0]
                result[label] = (
                    get_distribution(distribution.lstrip("?"))(label, *args[1:])
                    if trial.suggest_categorical(f"optional_{label}", [False, True])
                    else default_value
                )

            elif distribution == "$mlp_d_layers":
                min_n_layers, max_n_layers, d_min, d_max = args
                n_layers = trial.suggest_int("n_layers", min_n_layers, max_n_layers)
                suggest_dim = lambda name: trial.suggest_int(name, d_min, d_max)  # noqa
                d_first = [suggest_dim("d_first")] if n_layers else []
                d_middle = [suggest_dim("d_middle")] * (n_layers - 2) if n_layers > 2 else []
                d_last = [suggest_dim("d_last")] if n_layers > 1 else []
                result[label] = d_first + d_middle + d_last

            elif distribution == "$d_token":
                assert len(args) == 2
                try:
                    n_heads = base_config["model"]["n_heads"]
                except KeyError:
                    n_heads = base_config["model"]["n_latent_heads"]

                for x in args:
                    assert x % n_heads == 0
                result[label] = trial.suggest_int("d_token", *args, n_heads)  # type: ignore[code]

            elif distribution in ["$d_ffn_factor", "$d_hidden_factor"]:
                if base_config["model"]["activation"].endswith("glu"):
                    args = (args[0] * 2 / 3, args[1] * 2 / 3)
                result[label] = trial.suggest_uniform("d_ffn_factor", *args)

            else:
                result[label] = get_distribution(distribution)(label, *args)
    return result


def merge_sampled_parameters(config, sampled_parameters):
    """Merge the sampled hyper-parameters.

    :config: dict, configuration
    :sampled_parameters: dict, sampled hyper-parameters
    """
    for k, v in sampled_parameters.items():
        if isinstance(v, dict):
            merge_sampled_parameters(config.setdefault(k, {}), v)
        else:
            # If there are parameters in the default config, the value of the parameter will be overwritten.
            config[k] = v


def show_results(args, info, metric_name, loss_list, results_list, time_list):
    """Show the results for deep learning models.

    :args: argparse.Namespace, arguments
    :info: dict, information about the dataset
    :metric_name: list, names of the metrics
    :loss_list: list, list of loss
    :results_list: list, list of results
    :time_list: list, list of time
    """
    metric_arrays = {name: [] for name in metric_name}

    for result in results_list:
        for idx, name in enumerate(metric_name):
            metric_arrays[name].append(result[idx])

    metric_arrays["Time"] = time_list
    metric_name = (*metric_name, "Time")

    mean_metrics = {name: np.mean(metric_arrays[name]) for name in metric_name}
    std_metrics = {name: np.std(metric_arrays[name]) for name in metric_name}
    mean_loss = np.mean(np.array(loss_list))

    # Printing results
    print(f"{args.model_type}: {args.seed_num} Trials")
    for name in metric_name:
        if info["task_type"] == "regression" and name != "Time":
            formatted_results = ", ".join([f"{e:.8e}" for e in metric_arrays[name]])
            print(f"{name} Results: {formatted_results}")
            print(f"{name} MEAN = {mean_metrics[name]:.8e} ± {std_metrics[name]:.8e}")
        else:
            formatted_results = ", ".join([f"{e:.8f}" for e in metric_arrays[name]])
            print(f"{name} Results: {formatted_results}")
            print(f"{name} MEAN = {mean_metrics[name]:.8f} ± {std_metrics[name]:.8f}")

    print(f"Mean Loss: {mean_loss:.8e}")

    print("-" * 20, "GPU info", "-" * 20)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus} GPU Available.")
        for i in range(num_gpus):
            gpu_info = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_info.name}")
            print(f"  Total Memory:          {gpu_info.total_memory / 1024**2} MB")
            print(f"  Multi Processor Count: {gpu_info.multi_processor_count}")
            print(f"  Compute Capability:    {gpu_info.major}.{gpu_info.minor}")
    else:
        print("CUDA is unavailable.")
    print("-" * 50)
