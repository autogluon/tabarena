from __future__ import annotations

import errno
import json
import os
import os.path as osp
import pprint
import random
import shutil
import time

import numpy as np
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


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


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


import sklearn.metrics as skm


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
        os.path.join(
            args.save_path,
            "{}.json".format("config" if config_name is None else config_name),
        ),
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
                d_middle = (
                    [suggest_dim("d_middle")] * (n_layers - 2) if n_layers > 2 else []
                )
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


def get_classical_args():
    """Get the arguments for classical models.

    :return: argparse.Namespace, arguments
    """
    import argparse
    import warnings

    warnings.filterwarnings("ignore")
    with open("configs/classical_configs.json") as file:
        default_args = json.load(file)
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument("--dataset", type=str, default=default_args["dataset"])
    parser.add_argument(
        "--model_type",
        type=str,
        default=default_args["model_type"],
        choices=[
            "LogReg",
            "NCM",
            "RandomForest",
            "xgboost",
            "catboost",
            "lightgbm",
            "svm",
            "knn",
            "NaiveBayes",
            "dummy",
            "LinearRegression",
        ],
    )

    # optimization parameters
    parser.add_argument(
        "--normalization",
        type=str,
        default=default_args["normalization"],
        choices=["none", "standard", "minmax", "quantile", "maxabs", "power", "robust"],
    )
    parser.add_argument(
        "--num_nan_policy",
        type=str,
        default=default_args["num_nan_policy"],
        choices=["mean", "median"],
    )
    parser.add_argument(
        "--cat_nan_policy",
        type=str,
        default=default_args["cat_nan_policy"],
        choices=["new", "most_frequent"],
    )
    parser.add_argument(
        "--cat_policy",
        type=str,
        default=default_args["cat_policy"],
        choices=[
            "indices",
            "ordinal",
            "ohe",
            "binary",
            "hash",
            "loo",
            "target",
            "catboost",
        ],
    )
    parser.add_argument(
        "--num_policy",
        type=str,
        default=default_args["num_policy"],
        choices=[
            "none",
            "Q_PLE",
            "T_PLE",
            "Q_Unary",
            "T_Unary",
            "Q_bins",
            "T_bins",
            "Q_Johnson",
            "T_Johnson",
        ],
    )
    parser.add_argument("--n_bins", type=int, default=default_args["n_bins"])
    parser.add_argument(
        "--cat_min_frequency", type=float, default=default_args["cat_min_frequency"]
    )

    # other choices
    parser.add_argument("--n_trials", type=int, default=default_args["n_trials"])
    parser.add_argument("--seed_num", type=int, default=default_args["seed_num"])
    parser.add_argument("--gpu", default=default_args["gpu"])
    parser.add_argument("--tune", action="store_true", default=default_args["tune"])
    parser.add_argument("--retune", action="store_true", default=default_args["retune"])
    parser.add_argument(
        "--dataset_path", type=str, default=default_args["dataset_path"]
    )
    parser.add_argument("--model_path", type=str, default=default_args["model_path"])
    parser.add_argument(
        "--evaluate_option", type=str, default=default_args["evaluate_option"]
    )
    args = parser.parse_args()

    set_gpu(args.gpu)
    save_path1 = "-".join([args.dataset, args.model_type])

    save_path2 = f"Norm-{args.normalization}"
    save_path2 += f"-Nan-{args.num_nan_policy}-{args.cat_nan_policy}"
    save_path2 += f"-Cat-{args.cat_policy}"

    if args.cat_min_frequency > 0.0:
        save_path2 += f"-CatFreq-{args.cat_min_frequency}"
    if args.tune:
        save_path1 += "-Tune"

    save_path = osp.join(save_path1, save_path2)
    args.save_path = osp.join(args.model_path, save_path)
    mkdir(args.save_path)

    # load config parameters
    args.seed = 0

    config_default_path = os.path.join("configs", "default", args.model_type + ".json")
    config_opt_path = os.path.join("configs", "opt_space", args.model_type + ".json")
    with open(config_default_path) as file:
        default_para = json.load(file)

    with open(config_opt_path) as file:
        opt_space = json.load(file)

    args.config = default_para[args.model_type]
    set_seeds(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    pprint(vars(args))

    args.config["fit"]["n_bins"] = args.n_bins
    return args, default_para, opt_space


def get_deep_args():
    """Get the arguments for deep learning models.

    :return: argparse.Namespace, arguments
    """
    import argparse
    import warnings

    warnings.filterwarnings("ignore")

    argparse.ArgumentParser()
    # basic parameters
    default_args = {
        "dataset": "yeast",
        "model_type": "Beta",
        "max_epoch": 200,
        "batch_size": 1024,
        "normalization": "standard",
        "num_nan_policy": "mean",
        "cat_nan_policy": "new",
        "cat_policy": "indices",
        "num_policy": "none",
        "n_bins": 2,
        "cat_min_frequency": 0.0,
        "n_trials": 100,
        "seed_num": 5,
        "workers": 0,
        "gpu": "0",
        "tune": False,
        "retune": False,
        "evaluate_option": "best-val",
        "dataset_path": "data",
        "model_path": "results_model",
    }
    from types import SimpleNamespace

    args = SimpleNamespace(**default_args)

    set_gpu(args.gpu)
    save_path1 = "-".join([args.dataset, args.model_type])
    save_path2 = f"Epoch{args.max_epoch}BZ{args.batch_size}"
    save_path2 += f"-Norm-{args.normalization}"
    save_path2 += f"-Nan-{args.num_nan_policy}-{args.cat_nan_policy}"
    save_path2 += f"-Cat-{args.cat_policy}"

    if args.cat_min_frequency > 0.0:
        save_path2 += f"-CatFreq-{args.cat_min_frequency}"
    if args.tune:
        save_path1 += "-Tune"

    save_path = osp.join(save_path1, save_path2)
    args.save_path = osp.join(args.model_path, save_path)
    # mkdir(args.save_path)

    # load config parameters
    default_para = {
        "Beta": {
            "model": {
                "arch_type": "tabm-mini",
                "k": 16,
                "num_embeddings": {
                    "type": "PLREmbeddings",
                    "n_frequencies": 72,
                    "frequency_scale": 0.04,
                    "d_embedding": 32,
                    "lite": True,
                },
                "backbone": {
                    "type": "MLP",
                    "n_blocks": 2,
                    "d_block": 100,
                    "dropout": 0.15,
                },
            },
            "training": {"lr": 0.003, "weight_decay": 0.02},
            "general": {},
        }
    }
    opt_space = {
        "Beta": {
            "model": {
                "arch_type": "tabm",
                "k": 3,
                "num_embeddings": {
                    "type": "PLREmbeddings",
                    "n_frequencies": 77,
                    "frequency_scale": 0.04431360576139521,
                    "d_embedding": 34,
                    "lite": True,
                },
                "backbone": {
                    "type": "MLP",
                    "n_blocks": 2,
                    "d_block": 256,
                    "dropout": 0.1,
                },
                "temperature": 1,
                "sample_rate": 0.5,
            },
            "training": {"lr": 0.01, "weight_decay": 1e-5},
            "general": {},
        }
    }
    args.config = default_para[args.model_type]

    args.seed = 0
    set_seeds(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    pprint(vars(args))

    args.config["training"]["n_bins"] = args.n_bins
    return args, default_para, opt_space


def show_results_classical(args, info, metric_name, results_list, time_list):
    """Show the results for classical models.

    :args: argparse.Namespace, arguments
    :info: dict, information about the dataset
    :metric_name: list, names of the metrics
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


def tune_hyper_parameters(args, opt_space, train_val_data, info):
    """Tune hyper-parameters.

    :args: argparse.Namespace, arguments
    :opt_space: dict, search space
    :train_val_data: tuple, training and validation data
    :info: dict, information about the dataset
    :return: argparse.Namespace, arguments
    """
    import optuna
    import optuna.samplers
    import optuna.trial

    def objective(trial):
        config = {}
        try:
            opt_space[args.model_type]["training"]["n_bins"] = ["int", 2, 256]
        except:
            opt_space[args.model_type]["fit"]["n_bins"] = ["int", 2, 256]
        merge_sampled_parameters(
            config, sample_parameters(trial, opt_space[args.model_type], config)
        )
        if args.model_type == "xgboost" and torch.cuda.is_available():
            config["model"]["tree_method"] = "gpu_hist"
            config["model"]["gpu_id"] = args.gpu
            config["fit"]["verbose"] = False
        elif args.model_type == "catboost" and torch.cuda.is_available():
            config["fit"]["logging_level"] = "Silent"

        elif args.model_type == "RandomForest":
            config["model"]["max_depth"] = 12

        if args.model_type in ["resnet"]:
            config["model"]["activation"] = "relu"
            config["model"]["normalization"] = "batchnorm"

        if args.model_type in ["ftt"]:
            config["model"].setdefault("prenormalization", False)
            config["model"].setdefault("initialization", "xavier")
            config["model"].setdefault("activation", "reglu")
            config["model"].setdefault("n_heads", 8)
            config["model"].setdefault("d_token", 64)
            config["model"].setdefault("token_bias", True)
            config["model"].setdefault("kv_compression", None)
            config["model"].setdefault("kv_compression_sharing", None)

        if args.model_type in ["excelformer"]:
            config["model"].setdefault("prenormalization", False)
            config["model"].setdefault("kv_compression", None)
            config["model"].setdefault("kv_compression_sharing", None)
            config["model"].setdefault("token_bias", True)
            config["model"].setdefault("init_scale", 0.01)
            config["model"].setdefault("n_heads", 8)

        if args.model_type in ["node"]:
            config["model"].setdefault("choice_function", "sparsemax")
            config["model"].setdefault("bin_function", "sparsemoid")

        if args.model_type in ["tabr"]:
            config["model"]["num_embeddings"].setdefault("type", "PLREmbeddings")
            config["model"]["num_embeddings"].setdefault("lite", True)
            config["model"].setdefault("d_multiplier", 2.0)
            config["model"].setdefault("mixer_normalization", "auto")
            config["model"].setdefault("dropout1", 0.0)
            config["model"].setdefault("normalization", "LayerNorm")
            config["model"].setdefault("activation", "ReLU")

        if args.model_type in ["mlp_plr"]:
            config["model"]["num_embeddings"].setdefault("type", "PLREmbeddings")
            config["model"]["num_embeddings"].setdefault("lite", True)

        if args.model_type in ["ptarl"]:
            config["model"]["n_clusters"] = 20
            config["model"]["regularize"] = "True"
            config["general"]["diversity"] = "True"
            config["general"]["ot_weight"] = 0.25
            config["general"]["diversity_weight"] = 0.25
            config["general"]["r_weight"] = 0.25

        if args.model_type in ["modernNCA", "tabm"]:
            config["model"]["num_embeddings"].setdefault("type", "PLREmbeddings")
            config["model"]["num_embeddings"].setdefault("lite", True)

        if args.model_type in ["tabm"]:
            config["model"]["backbone"].setdefault("type", "MLP")
            config["model"].setdefault("arch_type", "tabm")
            config["model"].setdefault("k", 16)

        if args.model_type in ["danets"]:
            config["general"]["k"] = 5
            config["general"]["virtual_batch_size"] = 256

        if args.model_type in ["dcn2"]:
            config["model"]["stacked"] = False

        if args.model_type in ["grownet"]:
            config["ensemble_model"]["lr"] = 1.0
            config["model"]["sparse"] = False
            config["training"]["lr_scaler"] = 3

        if args.model_type in ["autoint"]:
            config["model"].setdefault("prenormalization", False)
            config["model"].setdefault("initialization", "xavier")
            config["model"].setdefault("activation", "relu")
            config["model"].setdefault("n_heads", 8)
            config["model"].setdefault("d_token", 64)
            config["model"].setdefault("kv_compression", None)
            config["model"].setdefault("kv_compression_sharing", None)

        if args.model_type in ["protogate"]:
            config["training"].setdefault("lam", 1e-3)
            config["training"].setdefault("pred_coef", 1)
            config["training"].setdefault("sorting_tau", 16)
            config["training"].setdefault("feature_selection", True)
            config["model"].setdefault("a", 1)
            config["model"].setdefault("sigma", 0.5)

        if args.model_type in ["grande"]:
            config["model"].setdefault("from_logits", True)
            config["model"].setdefault("use_class_weights", True)
            config["model"].setdefault("bootstrap", False)

        if args.model_type in ["amformer"]:
            config["model"].setdefault("heads", 8)
            config["model"].setdefault("groups", [54, 54, 54, 54])
            config["model"].setdefault("sum_num_per_group", [32, 16, 8, 4])
            config["model"].setdefault("prod_num_per_group", [6, 6, 6, 6])
            config["model"].setdefault("cluster", True)
            config["model"].setdefault("target_mode", "mix")
            config["model"].setdefault("token_descent", False)

        if config.get("config_type") == "trv4":
            if config["model"]["activation"].endswith("glu"):
                # This adjustment is needed to keep the number of parameters roughly in the
                # same range as for non-glu activations
                config["model"]["d_ffn_factor"] *= 2 / 3

        trial_configs.append(config)
        # method.fit(train_val_data, info, train=True, config=config)
        # run with this config
        try:
            method.fit(train_val_data, info, train=True, config=config)
            return method.trlog["best_res"]
        except Exception as e:
            print(e)
            return 1e9 if info["task_type"] == "regression" else 0.0

    if (
        osp.exists(osp.join(args.save_path, f"{args.model_type}-tuned.json"))
        and not args.retune
    ):
        with open(
            osp.join(args.save_path, f"{args.model_type}-tuned.json"), "rb"
        ) as fp:
            args.config = json.load(fp)
    else:
        # get data property
        if info["task_type"] == "regression":
            direction = "minimize"
            for key in opt_space[args.model_type]["model"]:
                if (
                    "dropout" in key
                    and "?" not in opt_space[args.model_type]["model"][key][0]
                ):
                    opt_space[args.model_type]["model"][key][0] = (
                        "?" + opt_space[args.model_type]["model"][key][0]
                    )
                    opt_space[args.model_type]["model"][key].insert(1, 0.0)
        else:
            direction = "maximize"

        method = get_method(args.model_type)(args, info["task_type"] == "regression")

        trial_configs = []
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=0),
        )
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=True,
        )
        # get best configs
        best_trial_id = study.best_trial.number
        # update config files
        print("Best Hyper-Parameters")
        print(trial_configs[best_trial_id])
        args.config = trial_configs[best_trial_id]
        with open(osp.join(args.save_path, f"{args.model_type}-tuned.json"), "w") as fp:
            json.dump(args.config, fp, sort_keys=True, indent=4)
    return args


def get_method(model):
    """Get the method class.

    :model: str, model name
    :return: class, method class
    """
    if model == "Beta":
        from tabrepo.benchmark.models.ag.beta.talent_beta_method import BetaMethod

        return BetaMethod
    raise ValueError(f"Unknown model: {model}. Please check the model name.")
