from __future__ import annotations

import copy
import math
import time

import category_encoders as ce
import numpy as np
import pandas as pd
import sklearn
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.common.utils.try_import import try_import_torch
from autogluon.core.models import AbstractModel
from autogluon.features import LabelEncoderFeatureGenerator

from tabarena.benchmark.models.ag.grande.grande_model_internal import (
    EarlyStopper,
    FocalLoss,
    GRANDE_Module,
    PiecewiseLinearEmbeddings,
    RobustScaleSmoothClipTransform,
    compute_bins,
    embed_data,
    evaluate,
    evaluate_regression,
    make_datasets_and_loaders,
    set_seed,
)


# TODO:
#   - Install GRANDE from Repo https://github.com/s-marton/GRANDE instead of inside of TabArena code
#       - move more "model" code to grande_model_internal
#       - also move preprocessing code as needed to make GRANDE a standalone model
#   -?avoid bins being build using val data to avoid overfitting?
# Change Log:
#   - initial split of TabArena Wrapper and standalone model
#   - formatting / linting
#   - add time-based early stopping
#   - update to newest TabArena random seed logic, and change TabRepo -> TabArena
#   - avoid y being returned from preprocessing
class GRANDEModel(AbstractModel):
    """GRANDE is a novel ensemble method for hard, axis-aligned decision trees learned end-to-end with gradient descent.

    Codebase: https://github.com/s-marton/GRANDE
    Paper: https://openreview.net/forum?id=XEFWBxi075
    License: MIT license
    """

    ag_key = "GRANDE"
    ag_name = "GRANDE"
    seed_name = "random_seed"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None
        self._indicator_columns = None
        self._features_bool = None
        self._bool_to_cat = None

        # y mean/std for regression
        self.mean = None
        self.std = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        verbosity: int = 0,
        **kwargs,
    ):
        start_time = time.monotonic()

        # Torch Lazy Import
        try_import_torch()
        import torch
        from torch import nn
        from torch.nn import functional as F

        device = "cpu" if num_gpus == 0 else "cuda:0"
        if (device == "cuda:0") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        # -- Set parameters
        hyp = self._get_model_params()
        self.random_seed = hyp["random_seed"]
        self.data_subset_fraction = hyp["data_subset_fraction"]
        self.use_category_embeddings = hyp["use_category_embeddings"]
        self.use_numeric_embeddings = hyp["use_numeric_embeddings"]
        self.embedding_threshold = hyp["embedding_threshold"]
        self.embedding_dim_cat = hyp["embedding_dim_cat"]
        self.embedding_dim_num = hyp["embedding_dim_num"]
        self.learning_rate_embedding = hyp["learning_rate_embedding"]
        self.loo_cardinality = hyp["loo_cardinality"]
        self.use_robust_scale_smoothing = hyp["use_robust_scale_smoothing"]
        self.optimizer = hyp["optimizer"]
        self.reduce_on_plateau_scheduler = hyp["reduce_on_plateau_scheduler"]
        self.verbosity = hyp["verbose"]
        hyp["device"] = device
        self.device = device
        hyp["objective"] = self.problem_type
        self.objective = hyp["objective"]
        self.missing_values = hyp["missing_values"]

        # -- Handle data
        X_train, y_train = X, y
        if X_val is None:
            from autogluon.core.utils import generate_train_test_split

            X_train, X_val, y_train, y_val = generate_train_test_split(
                X=X_train,
                y=y_train,
                problem_type=self.problem_type,
                test_size=0.2,
                random_state=0,
            )

        if isinstance(y_train, pd.Series):
            try:
                y_train = y_train.values.codes.astype(np.float32)
            except:
                y_train = y_train.values.astype(np.float32)
        X_train = self.preprocess(X_train, y=y_train, is_train=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        # -- Class weights
        hyp["number_of_variables"] = X_train.shape[1]
        if hyp["objective"] == "regression":
            hyp["use_class_weights"] = False
        if hyp["use_class_weights"]:
            if hyp["objective"] == "multiclass" or hyp["objective"] == "binary":
                hyp["number_of_classes"] = (
                    1 if hyp["objective"] == "binary" else len(np.unique(y_train))
                )
                counts = np.bincount(
                    y_train.astype(int), minlength=hyp["number_of_classes"]
                ).astype(np.float64)
                counts[counts == 0] = 1.0
                inv = 1.0 / counts
                class_weights = inv * (hyp["number_of_classes"] / inv.sum())
            else:
                hyp["number_of_classes"] = 1
                class_weights = np.ones_like(np.unique(y_train))
        else:
            hyp["number_of_classes"] = (
                1
                if (hyp["objective"] == "binary" or hyp["objective"] == "regression")
                else len(np.unique(y_train))
            )
            class_weights = np.ones_like(np.unique(y_train))
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

        # -- Embedding setup
        if self.use_category_embeddings or self.use_numeric_embeddings:
            all_indices = list(range(hyp["number_of_variables"]))
            self.categorical_features_raw_indices = [
                i
                for i in all_indices
                if i not in self.num_columns_indices
                and i not in self.encoded_columns_indices
            ]

            if (
                self.use_numeric_embeddings
                and self.num_columns_indices is not None
                and len(self.num_columns_indices) > 0
            ):
                # Compute all other indices (non-numeric features)

                # embeddings = LinearEmbeddings(len(self.num_columns_indices), self.embedding_dim)
                # FIXME: this might increase overtuning as it uses X_val to build the bins.
                bins = compute_bins(
                    torch.from_numpy(
                        np.concatenate([X_train, X_val], axis=0)[
                            :, self.num_columns_indices
                        ]
                    ),
                    n_bins=hyp["num_emb_n_bins"],
                    # NOTE: requires scikit-learn>=1.0 to be installed.
                    tree_kwargs={"min_samples_leaf": 64, "min_impurity_decrease": 1e-4},
                    y=torch.from_numpy(np.concatenate([y_train, y_val], axis=0)),
                    regression=self.objective == "regression",
                )

                set_seed(self.random_seed)
                numeric_embeddings = PiecewiseLinearEmbeddings(
                    bins=bins,
                    d_embedding=self.embedding_dim_num,
                    activation=False,
                    version="B",
                )
                set_seed(self.random_seed)
                # Get the output shape without batch dimensions:
                assert numeric_embeddings.get_output_shape() == (
                    len(self.num_columns_indices),
                    self.embedding_dim_num,
                )
                # Get the total output size after flattening:
                assert (
                    numeric_embeddings.get_output_shape().numel()
                    == len(self.num_columns_indices) * self.embedding_dim_num
                )

            else:
                numeric_embeddings = None
                swa_numeric_embeddings = None

            if (
                self.use_category_embeddings
                and self.categorical_features_raw_indices is not None
                and len(self.categorical_features_raw_indices) > 0
            ):
                categories = [
                    len(np.unique(X_train[:, idx])) + 1
                    for idx in self.categorical_features_raw_indices
                ]

                # category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
                # self.register_buffer('category_offsets', category_offsets)
                set_seed(self.random_seed)
                category_embeddings = nn.Embedding(
                    sum(categories), self.embedding_dim_cat
                )
                set_seed(self.random_seed)
                nn.init.kaiming_uniform_(category_embeddings.weight, a=math.sqrt(5))
                set_seed(self.random_seed)

            else:
                category_embeddings = None
                swa_category_embeddings = None

            if self.use_category_embeddings or self.use_numeric_embeddings:
                embedding_dim_total = 0
                feature_dim_total = 0

                num_features_total = len(self.num_columns_indices)
                cat_features_raw_total = len(self.categorical_features_raw_indices)
                cat_features_encoded_total = len(self.encoded_columns_indices)

                if self.use_category_embeddings:
                    embedding_dim_total += (
                        self.embedding_dim_cat * cat_features_raw_total
                    )
                else:
                    feature_dim_total += cat_features_raw_total

                if self.use_numeric_embeddings:
                    embedding_dim_total += self.embedding_dim_num * num_features_total
                else:
                    feature_dim_total += num_features_total

                feature_dim_total += cat_features_encoded_total

                hyp["number_of_variables"] = feature_dim_total + embedding_dim_total
        else:
            numeric_embeddings = None
            category_embeddings = None
            swa_numeric_embeddings = None
            swa_category_embeddings = None
            self.categorical_features_raw_indices = []

        # -- Model Setup
        model = GRANDE_Module(params=hyp)
        model = model.to(device)
        model = torch.compile(
            model, mode="reduce-overhead", fullgraph=False, dynamic=False
        )

        if (
            self.use_numeric_embeddings
            and self.num_columns_indices is not None
            and len(self.num_columns_indices) > 0
        ):
            numeric_embeddings = numeric_embeddings.to(device)
            # numeric_embeddings = torch.compile(numeric_embeddings)#, mode="reduce-overhead", fullgraph=False, dynamic=True)

        if (
            self.use_category_embeddings
            and self.categorical_features_raw_indices is not None
            and len(self.categorical_features_raw_indices) > 0
        ):
            category_embeddings = category_embeddings.to(device)
            # category_embeddings = torch.compile(category_embeddings)#, mode="reduce-overhead", fullgraph=False, dynamic=True)

        if hyp["swa"]:
            swa_model = torch.optim.swa_utils.AveragedModel(
                model
            )  # .to(device) #, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
            # self.swa_model = torch.compile(self.swa_model, mode="reduce-overhead", fullgraph=False, dynamic=True)
            if numeric_embeddings is not None:
                swa_numeric_embeddings = torch.optim.swa_utils.AveragedModel(
                    numeric_embeddings
                )
            if category_embeddings is not None:
                swa_category_embeddings = torch.optim.swa_utils.AveragedModel(
                    category_embeddings
                )
        else:
            swa_model = None
            swa_numeric_embeddings = None
            swa_category_embeddings = None

        self.model = model
        self.numeric_embeddings = numeric_embeddings
        self.category_embeddings = category_embeddings
        self.swa_model = swa_model
        self.swa_numeric_embeddings = swa_numeric_embeddings
        self.swa_category_embeddings = swa_category_embeddings

        if self.numeric_embeddings is not None:
            print("Use Numeric Embeddings:", self.numeric_embeddings, flush=True)
        if self.category_embeddings is not None:
            print("Use Category Embeddings:", self.category_embeddings, flush=True)

        # -- Training loop setup
        hyp["batch_size"] = min(hyp["batch_size"], X_train.shape[0] // 2)
        train_loader, val_loader = make_datasets_and_loaders(
            X_train,
            y_train,
            X_val,
            y_val,
            shuffle_train=(self.data_subset_fraction == 1.0),
            batch_size=hyp["batch_size"],
            task_type=hyp["objective"],
        )

        if val_loader is not None:
            early_stopper = EarlyStopper(
                patience=hyp["early_stopping_epochs"],
                min_delta=0.001,
                mode="min",  # was "max"
                # get_state=lambda: {"clf": self.model.state_dict()},
                # set_state=lambda st: self.model.load_state_dict(st["clf"]),
                name="TRAINING",
                verbose=(self.verbosity >= 2),
            )
        else:
            early_stopper = None

        # -----------------------------------------------------------------------------
        # Optimizer(s) and per-estimator learning rate scaling
        # -----------------------------------------------------------------------------
        #
        # We still use a single optimizer with one base LR per parameter type, but
        # if `use_multi_estimator_lr` is enabled we rescale the gradients for each
        # estimator so that the *effective* LR per estimator is different.
        #
        # This avoids slicing `nn.Parameter`s (which would create non-leaf tensors)
        # while still achieving different step sizes across estimators.

        n_estimators = getattr(self.model, "n_estimators", None)
        use_multi_estimator_lr = (
            getattr(self, "use_multi_estimator_lr", False)
            and n_estimators is not None
            and n_estimators > 1
        )

        # Define LR factors for multi-estimator LR scheme (used for both estimators and embeddings)
        # Allow user to customize, with sensible default
        if use_multi_estimator_lr:
            lr_factors = getattr(self, "estimator_lr_factors", None)
            if lr_factors is None:
                lr_factors = [0.1, 0.3, 1.0, 3.0, 10.0]
            elif not isinstance(lr_factors, (list, tuple)) or len(lr_factors) < 2:
                raise ValueError(
                    f"estimator_lr_factors must be a list/tuple with at least 2 values, got: {lr_factors}"
                )
            lr_factors = list(lr_factors)  # Ensure it's a list
            n_groups = len(lr_factors)

            if self.verbosity >= 1:
                print(
                    f"[MULTI-LR] Using {n_groups} LR groups with factors: {lr_factors}",
                    flush=True,
                )
                if n_estimators < n_groups:
                    print(
                        f"[MULTI-LR] Warning: n_estimators ({n_estimators}) < n_groups ({n_groups}). "
                        f"Some groups will be empty.",
                        flush=True,
                    )
        else:
            lr_factors = []
            n_groups = 0

        if use_multi_estimator_lr:
            # Assign each estimator to one of the factors as evenly as possible.
            base_group_size = max(1, n_estimators // n_groups)
            remainder = n_estimators % n_groups
            factors_per_estimator = []
            start = 0
            for g in range(n_groups):
                extra = 1 if g < remainder else 0
                end = min(n_estimators, start + base_group_size + extra)
                if start < end:
                    factors_per_estimator.extend([lr_factors[g]] * (end - start))
                start = end
            # In case of any rounding issues, truncate/extend to exact length
            factors_per_estimator = factors_per_estimator[:n_estimators]
            if len(factors_per_estimator) < n_estimators:
                factors_per_estimator.extend(
                    [lr_factors[-1]] * (n_estimators - len(factors_per_estimator))
                )

            factors_tensor = torch.tensor(
                factors_per_estimator, dtype=torch.float32, device=self.device
            )

            # Register gradient hooks to scale grads along estimator dimension.
            # Shapes:
            #   split_values:        [E, I, V]
            #   split_index_array:   [E, I, V]
            #   estimator_weights:   [E, L]
            #   leaf_classes_array:  [E, L] or [E, L, C]

            factors_eiv = factors_tensor.view(-1, 1, 1)
            factors_el = factors_tensor.view(-1, 1)

            def _scale_grad_split_values(grad):
                return grad * factors_eiv

            def _scale_grad_split_index(grad):
                return grad * factors_eiv

            def _scale_grad_estimator_weights(grad):
                return grad * factors_el

            def _scale_grad_leaf_classes(grad):
                # Works for [E, L] or [E, L, C] via broadcasting.
                view_shape = [factors_el.size(0), factors_el.size(1)] + [1] * (
                    grad.dim() - 2
                )
                return grad * factors_el.view(*view_shape)

            self.model.split_values.register_hook(_scale_grad_split_values)
            self.model.split_index_array.register_hook(_scale_grad_split_index)
            self.model.estimator_weights.register_hook(_scale_grad_estimator_weights)
            self.model.leaf_classes_array.register_hook(_scale_grad_leaf_classes)

        if self.optimizer.lower() == "adamw":
            param_group = [
                {
                    "params": self.model.split_values,
                    "lr": hyp["learning_rate_values"],
                    "betas": (0.9, 0.95),
                },
                {
                    "params": self.model.estimator_weights,
                    "lr": hyp["learning_rate_weights"],
                    "betas": (0.9, 0.95),
                },
            ]

            opt = torch.optim.Adam(param_group)

            param_group_adamW = [
                {
                    "params": self.model.split_index_array,
                    "lr": hyp["learning_rate_index"],
                    "betas": (0.9, 0.95),
                },
                {
                    "params": self.model.leaf_classes_array,
                    "lr": hyp["learning_rate_leaf"],
                    "betas": (0.9, 0.95),
                },
            ]

            opt_adamW = torch.optim.AdamW(param_group_adamW)  # , weight_decay=5e-4

        else:
            param_group = [
                {
                    "params": self.model.split_values,
                    "lr": hyp["learning_rate_values"],
                    "betas": (0.9, 0.95),
                },
                {
                    "params": self.model.split_index_array,
                    "lr": hyp["learning_rate_index"],
                    "betas": (0.9, 0.95),
                },
                {
                    "params": self.model.estimator_weights,
                    "lr": hyp["learning_rate_weights"],
                    "betas": (0.9, 0.95),
                },
                {
                    "params": self.model.leaf_classes_array,
                    "lr": hyp["learning_rate_leaf"],
                    "betas": (0.9, 0.95),
                },
            ]

            if self.optimizer.lower() == "nadam":
                opt = torch.optim.NAdam(param_group)
            elif self.optimizer.lower() == "radam":
                opt = torch.optim.RAdam(param_group)
            else:
                opt = torch.optim.Adam(param_group)

        if self.numeric_embeddings is not None or self.category_embeddings is not None:
            param_group_emb = []
            if self.numeric_embeddings is not None:
                param_group_emb += [
                    {
                        "params": self.numeric_embeddings.parameters(),
                        "lr": self.learning_rate_embedding,
                        "betas": (0.9, 0.95),
                    },
                ]
            elif self.category_embeddings is not None:
                param_group_emb += [
                    {
                        "params": self.category_embeddings.parameters(),
                        "lr": self.learning_rate_embedding,
                        "betas": (0.9, 0.95),
                    },
                ]
            if self.optimizer.lower() == "adamw":
                opt_emb = torch.optim.AdamW(param_group_emb)  # , weight_decay=5e-4
            elif self.optimizer.lower() == "nadam":
                opt_emb = torch.optim.NAdam(param_group_emb)
            elif self.optimizer.lower() == "radam":
                opt_emb = torch.optim.RAdam(param_group_emb)
            else:
                opt_emb = torch.optim.Adam(param_group_emb)

        if self.reduce_on_plateau_scheduler:
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",  # 'min' for loss, 'max' for accuracy, etc.
                factor=0.2,  # multiply LR by this factor
                patience=15,  # epochs with no improvement before reducing
                threshold=0.001,  # minimum change to qualify as improvement
            )
            if self.optimizer.lower() == "adamw":
                plateau_adamW = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt_adamW,
                    mode="min",  # 'min' for loss, 'max' for accuracy, etc.
                    factor=0.2,  # multiply LR by this factor
                    patience=15,  # epochs with no improvement before reducing
                    threshold=0.001,  # minimum change to qualify as improvement
                )
            if (
                self.numeric_embeddings is not None
                or self.category_embeddings is not None
            ):
                plateau_emb = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt_emb,
                    mode="min",  # 'min' for loss, 'max' for accuracy, etc.
                    factor=0.2,  # multiply LR by this factor
                    patience=15,  # epochs with no improvement before reducing
                    threshold=0.001,  # minimum change to qualify as improvement
                )

        if hyp["cosine_decay_restarts"]:
            print("Using CosineAnnealingLR", flush=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=10, T_mult=2, eta_min=1e-5
            )
            if self.optimizer.lower() == "adamw":
                scheduler_adamW = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt_adamW, T_0=10, T_mult=2, eta_min=1e-5
                )

            if (
                self.numeric_embeddings is not None
                or self.category_embeddings is not None
            ):
                scheduler_emb = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt_emb, T_0=10, T_mult=2, eta_min=1e-5
                )
        best_val_score = float("inf")
        best_state = None  # copy.deepcopy(self.model.state_dict())
        best_numeric_embeddings_state = None
        best_category_embeddings_state = None

        use_per_estimator_logits = False
        set_seed(self.random_seed)
        for epoch in range(1, hyp["epochs"] + 1):
            t0 = time.time()
            total, n = 0.0, 0
            self.model.train()
            if self.numeric_embeddings is not None:
                self.numeric_embeddings.train()
            if self.category_embeddings is not None:
                self.category_embeddings.train()
            for _batch_idx, (xb, yb) in enumerate(train_loader):
                xb = xb.to(device)
                yb = yb.to(device)
                xb = embed_data(
                    xb,
                    self.num_columns_indices,
                    self.categorical_features_raw_indices,
                    self.encoded_columns_indices,
                    self.numeric_embeddings,
                    self.category_embeddings,
                )
                logits, per_estimator_logits = self.model(
                    xb, return_per_estimator_logits=True
                )
                if use_per_estimator_logits:
                    loss = 0.0
                    # iterate over estimators (dim=1)
                    for est_logits in per_estimator_logits.unbind(
                        dim=1
                    ):  # (batch, num_classes)
                        loss += F.cross_entropy(
                            est_logits, yb, reduction="mean", weight=class_weights
                        )
                    loss = loss / hyp["n_estimators"]  # average
                elif hyp["objective"] == "binary" or hyp["objective"] == "multiclass":
                    if hyp["focal_loss"]:
                        if hyp["objective"] == "binary":
                            alpha = (
                                class_weights[1] / (class_weights[0] + class_weights[1])
                                if class_weights is not None
                                else 0.5
                            )
                            criterion = FocalLoss(
                                gamma=2,
                                alpha=alpha,
                                task_type="binary",
                                reduction="mean",
                            )
                            loss = criterion(logits[:, 1], yb)
                        else:
                            alpha = class_weights / class_weights.sum()
                            criterion = FocalLoss(
                                gamma=2,
                                alpha=alpha,
                                task_type="multi-class",
                                reduction="mean",
                                num_classes=hyp["number_of_classes"],
                            )
                            loss = criterion(logits, yb)
                    else:
                        loss = F.cross_entropy(
                            logits,
                            yb,
                            reduction="mean",
                            weight=class_weights,
                            label_smoothing=hyp.get("label_smoothing", 0.0),
                        )
                elif hyp["objective"] == "regression":
                    preds = logits
                    # preds = preds * self.std + self.mean
                    true = yb.float()
                    true = (true - self.mean) / self.std
                    loss = F.mse_loss(
                        preds,
                        true,
                        reduction="mean",
                    )
                opt.zero_grad(set_to_none=True)
                if self.optimizer.lower() == "adamw":
                    opt_adamW.zero_grad(set_to_none=True)
                if (
                    self.numeric_embeddings is not None
                    or self.category_embeddings is not None
                ):
                    opt_emb.zero_grad(set_to_none=True)

                loss.backward()

                opt.step()
                if self.optimizer.lower() == "adamw":
                    opt_adamW.step()

                if (
                    self.numeric_embeddings is not None
                    or self.category_embeddings is not None
                ):
                    opt_emb.step()

                total += loss.item() * xb.size(0)
                n += xb.size(0)
            if hyp["swa"] and epoch > 10:
                self.swa_model.update_parameters(self.model)
                if self.numeric_embeddings is not None:
                    self.swa_numeric_embeddings.update_parameters(
                        self.numeric_embeddings
                    )
                if self.category_embeddings is not None:
                    self.swa_category_embeddings.update_parameters(
                        self.category_embeddings
                    )
            if val_loader is not None:
                # compute val metrics
                if hyp["swa"] and epoch > 10:
                    numeric_embeddings_eval = self.swa_numeric_embeddings
                else:
                    numeric_embeddings_eval = self.numeric_embeddings

                if hyp["swa"] and epoch > 10:
                    category_embeddings_eval = self.swa_category_embeddings
                else:
                    category_embeddings_eval = self.category_embeddings

                model_eval = self.swa_model if hyp["swa"] and epoch > 10 else self.model

                if not (hyp["swa"] and epoch > 10):
                    model_eval.eval()
                    if self.numeric_embeddings is not None:
                        numeric_embeddings_eval.eval()
                    if self.category_embeddings is not None:
                        category_embeddings_eval.eval()

                if hyp["es_metric"]:
                    if hyp["objective"] == "binary" or hyp["objective"] == "multiclass":
                        acc, auc, f1 = evaluate(
                            model=model_eval,
                            loader=val_loader,
                            device=device,
                            numeric_embeddings=numeric_embeddings_eval,
                            category_embeddings=category_embeddings_eval,
                            num_features=self.num_columns_indices,
                            categorical_features_raw_indices=self.categorical_features_raw_indices,
                            encoded_columns_indices=self.encoded_columns_indices,
                        )
                    elif hyp["objective"] == "regression":
                        mse, mae, r2 = evaluate_regression(
                            model=model_eval,
                            loader=val_loader,
                            device=device,
                            std=self.std,
                            mean=self.mean,
                            numeric_embeddings=numeric_embeddings_eval,
                            category_embeddings=category_embeddings_eval,
                            num_features=self.num_columns_indices,
                            categorical_features_raw_indices=self.categorical_features_raw_indices,
                            encoded_columns_indices=self.encoded_columns_indices,
                        )

                # compute val loss
                with torch.no_grad():
                    vtotal, vn = 0.0, 0
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)

                        xb = embed_data(
                            xb,
                            self.num_columns_indices,
                            self.categorical_features_raw_indices,
                            self.encoded_columns_indices,
                            numeric_embeddings_eval,
                            category_embeddings_eval,
                        )
                        logits = model_eval(xb)

                        if (
                            hyp["objective"] == "binary"
                            or hyp["objective"] == "multiclass"
                        ):
                            if hyp["focal_loss"]:
                                if hyp["objective"] == "binary":
                                    alpha = (
                                        class_weights[1]
                                        / (class_weights[0] + class_weights[1])
                                        if class_weights is not None
                                        else 0.5
                                    )
                                    criterion = FocalLoss(
                                        gamma=2,
                                        alpha=alpha,
                                        task_type="binary",
                                        reduction="sum",
                                    )
                                    vloss = criterion(logits[:, 1], yb)
                                else:
                                    alpha = class_weights / class_weights.sum()
                                    criterion = FocalLoss(
                                        gamma=2,
                                        alpha=alpha,
                                        task_type="multi-class",
                                        reduction="sum",
                                        num_classes=hyp["number_of_classes"],
                                    )
                                    vloss = criterion(logits, yb)

                            else:
                                vloss = F.cross_entropy(
                                    logits,
                                    yb,
                                    reduction="sum",
                                    weight=class_weights,
                                    label_smoothing=hyp.get("label_smoothing", 0.0),
                                )
                        elif hyp["objective"] == "regression":
                            preds = logits
                            # preds = preds * self.std + self.mean
                            true = yb.float()
                            true = (true - self.mean) / self.std

                            vloss = F.mse_loss(
                                preds,
                                true,
                                reduction="sum",
                            )
                        vtotal += float(vloss.item())
                        vn += xb.size(0)
                    val_loss = vtotal / max(vn, 1)
                # val_score = (val_loss if not hyp['es_metric'] else (auc if hyp['objective'] in ['binary', 'multiclass'] else r2))
                val_score = (
                    val_loss
                    if not hyp["es_metric"]
                    else (
                        -auc
                        if hyp["objective"] == "binary"
                        else val_loss
                        if hyp["objective"] == "multiclass"
                        else mse
                    )
                )
                if val_score < best_val_score:
                    best_val_score = val_score
                    if hyp["swa"] and epoch > 10:
                        best_state = copy.deepcopy(self.swa_model.module.state_dict())
                        if self.numeric_embeddings is not None:
                            best_numeric_embeddings_state = copy.deepcopy(
                                self.swa_numeric_embeddings.module.state_dict()
                            )
                        if self.category_embeddings is not None:
                            best_category_embeddings_state = copy.deepcopy(
                                self.swa_category_embeddings.module.state_dict()
                            )

                    else:
                        best_state = copy.deepcopy(self.model.state_dict())
                        if self.numeric_embeddings is not None:
                            best_numeric_embeddings_state = copy.deepcopy(
                                self.numeric_embeddings.state_dict()
                            )
                        if self.category_embeddings is not None:
                            best_category_embeddings_state = copy.deepcopy(
                                self.category_embeddings.state_dict()
                            )

                epoch_sec = time.time() - t0
                if self.verbosity >= 1:
                    if hyp["es_metric"]:
                        if (
                            hyp["objective"] == "binary"
                            or hyp["objective"] == "multiclass"
                        ):
                            print(
                                f"Epoch {epoch:03d} | TrainLoss: {total / n:.4f} | ValLoss: {val_loss:.4f} | ValAcc: {acc:.4f} | ValAUC: {auc:.4f} | ValF1: {f1:.4f} | Time: {epoch_sec:.2f}s",
                                flush=True,
                            )
                        else:
                            print(
                                f"Epoch {epoch:03d} | TrainLoss: {total / n:.4f} | ValLoss: {val_loss:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | Time: {epoch_sec:.2f}s",
                                flush=True,
                            )
                    else:
                        print(
                            f"Epoch {epoch:03d} | TrainLoss: {total / n:.4f} | ValLoss: {val_loss:.4f} | Time: {epoch_sec:.2f}s",
                            flush=True,
                        )
                # Early stopping on val loss
                if early_stopper is not None and early_stopper.step(val_score):
                    break
            else:
                epoch_sec = time.time() - t0
                if self.verbosity >= 1:
                    print(
                        f"Epoch {epoch:03d} | TrainLoss: {total / n:.4f} | Time: {epoch_sec:.2f}s",
                        flush=True,
                    )
            if hyp["cosine_decay_restarts"]:
                scheduler.step()
                if self.optimizer.lower() == "adamw":
                    scheduler_adamW.step()
                if (
                    self.numeric_embeddings is not None
                    or self.category_embeddings is not None
                ):
                    scheduler_emb.step()

                if val_loader is not None and self.reduce_on_plateau_scheduler:
                    plateau.step(val_score)
                    if self.optimizer.lower() == "adamw":
                        plateau_adamW.step(val_score)
                    if (
                        self.numeric_embeddings is not None
                        or self.category_embeddings is not None
                    ):
                        plateau_emb.step(val_score)

            if time_limit is not None:
                elapsed_time = time.monotonic() - start_time
                if elapsed_time >= time_limit:
                    if self.verbosity >= 1:
                        print(
                            f"[EarlyStop due to time limit {elapsed_time / time_limit}).",
                            flush=True,
                        )
                    break

        # Restore best weights if we tracked validation
        if val_loader is not None and best_state is not None:
            self.model.load_state_dict(best_state)
            del best_state
            if self.numeric_embeddings is not None:
                self.numeric_embeddings.load_state_dict(best_numeric_embeddings_state)
                del best_numeric_embeddings_state
            if self.category_embeddings is not None:
                self.category_embeddings.load_state_dict(best_category_embeddings_state)
                del best_category_embeddings_state

        if hyp["swa"] and epoch > 10:
            # self.model = self.swa_model
            del self.swa_model
            del self.swa_numeric_embeddings
            del self.swa_category_embeddings

    # Move to model.predict / model.predict_proba
    def _predict_proba(self, X, **kwargs):
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, TensorDataset

        with torch.no_grad():
            self.model.eval()
            X = self.preprocess(X)

            ds = TensorDataset(torch.from_numpy(X))

            loader = DataLoader(
                ds,
                batch_size=256,
                shuffle=False,
                drop_last=False,
            )

            ps = []
            for (xb,) in loader:
                xb = xb.to(self.device)

                xb = embed_data(
                    xb,
                    self.num_columns_indices,
                    self.categorical_features_raw_indices,
                    self.encoded_columns_indices,
                    self.numeric_embeddings,
                    self.category_embeddings,
                )

                logits = self.model(xb)
                if self.objective != "regression":
                    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
                else:
                    probs = logits.detach().cpu().numpy() * self.std + self.mean
                ps.append(probs)
            y_prob = np.concatenate(ps)
            return y_prob[:, 1] if self.objective == "binary" else y_prob

    # TODO: Move missing indicator + mean fill to a generic preprocess flag available to all models
    def _preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        is_train: bool = False,
        bool_to_cat: bool = True,
        impute_bool: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Imputes missing values via the mean and adds indicator columns for numerical features.
        Converts indicator columns to categorical features to avoid them being treated as numerical by RealMLP.
        """
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._bool_to_cat = bool_to_cat
            self._features_bool = self._feature_metadata.get_features(
                required_special_types=["bool"]
            )

        if self._bool_to_cat and self._features_bool:
            # FIXME: Use CategoryFeatureGenerator? Or tell the model which is category
            X = X.copy()
            X[self._features_bool] = X[self._features_bool].astype("category")

        if y is not None and isinstance(y, pd.Series):
            try:
                y = y.values.codes.astype(np.float32)
            except:
                y = y.values.astype(np.float32)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if is_train and y is not None:
            self.mean = np.mean(y)
            self.std = np.std(y)

        if is_train:
            self._cat_features = None
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(
                X=X
            )
            if self._cat_features is None:
                self._cat_features = self._feature_generator.features_in[:]

        if not self.use_category_embeddings and not self.use_numeric_embeddings:
            if is_train:
                self.binary_indices = []
                self.low_cardinality_indices = []
                self.high_cardinality_indices = []
                self.num_columns = []

                for column_index, column in enumerate(X.columns):
                    col_data = X.iloc[:, column_index]
                    n_unique = col_data.nunique(dropna=True)

                    if self._cat_features is not None and column in self._cat_features:
                        if n_unique <= 2:
                            self.binary_indices.append(column)
                        elif n_unique < 10:
                            self.low_cardinality_indices.append(column)
                        else:
                            self.high_cardinality_indices.append(column)
                    else:
                        self.num_columns.append(column)

                self.cat_columns = [
                    col for col in X.columns if col not in self.num_columns
                ]

                if not self.missing_values:
                    if len(self.num_columns) > 0:
                        self.mean_train_num = X[self.num_columns].mean(axis=0).iloc[0]
                        X[self.num_columns] = X[self.num_columns].fillna(
                            self.mean_train_num
                        )
                    if len(self.cat_columns) > 0:
                        self.mode_train_cat = X[self.cat_columns].mode(axis=0).iloc[0]
                        X[self.cat_columns] = X[self.cat_columns].fillna(
                            self.mode_train_cat
                        )

            if is_train:
                self.encoder_ordinal = ce.OrdinalEncoder(
                    cols=self.binary_indices, handle_missing="return_nan"
                )
                self.encoder_ordinal.fit(X)

                self.encoder_loo = ce.LeaveOneOutEncoder(
                    cols=self.high_cardinality_indices, handle_missing="return_nan"
                )
                if self.problem_type == "regression":
                    self.encoder_loo.fit(X, (y - self.mean) / self.std)
                else:
                    self.encoder_loo.fit(X, y)

                self.encoder_ohe = ce.OneHotEncoder(
                    cols=self.low_cardinality_indices, handle_missing="return_nan"
                )
                self.encoder_ohe.fit(X)

            if not self.missing_values:
                if len(self.num_columns) > 0:
                    X[self.num_columns] = X[self.num_columns].fillna(
                        self.mean_train_num
                    )
                if len(self.cat_columns) > 0:
                    X[self.cat_columns] = X[self.cat_columns].fillna(
                        self.mode_train_cat
                    )

            X = self.encoder_ordinal.transform(X)
            X = self.encoder_loo.transform(X)
            X = self.encoder_ohe.transform(X)

            if is_train:
                self.num_columns_indices = [
                    X.columns.get_loc(col) for col in self.num_columns
                ]
                self.column_names_dataframe = X.columns.tolist()

                self.encoded_columns = [
                    col for col in X.columns if col not in self.num_columns
                ]
                self.encoded_columns_indices = [
                    X.columns.get_loc(col) for col in self.encoded_columns
                ]

                self.not_encoded_columns = []

            X = X.astype(np.float32)
            if is_train:
                if self.use_robust_scale_smoothing:
                    self.normalizer = RobustScaleSmoothClipTransform()
                    self.normalizer.fit(X.values.astype(np.float32))
                else:
                    quantile_noise = 1e-4
                    quantile_train = np.copy(X.values).astype(np.float32)
                    np.random.seed(42)
                    stds = np.nanstd(quantile_train, axis=0, keepdims=True)
                    noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                    quantile_train += noise_std * np.random.randn(*quantile_train.shape)

                    quantile_train = pd.DataFrame(
                        quantile_train, columns=X.columns, index=X.index
                    )

                    self.normalizer = sklearn.preprocessing.QuantileTransformer(
                        n_quantiles=min(quantile_train.shape[0], 1000),
                        output_distribution="normal",
                    )

                    self.normalizer.fit(quantile_train.values.astype(np.float32))

            X = self.normalizer.transform(X.values.astype(np.float32))

        elif self.use_category_embeddings:
            if is_train:
                self.binary_indices = []
                self.low_cardinality_indices = []
                self.high_cardinality_indices = []
                self.columns_to_encode = []
                self.num_columns = []
                self.cat_columns = []

                for column_index, column in enumerate(X.columns):
                    col_data = X.iloc[:, column_index]
                    n_unique = col_data.nunique(dropna=True)

                    if self._cat_features is not None and column in self._cat_features:
                        if n_unique <= 2:
                            self.binary_indices.append(column)
                            self.columns_to_encode.append(column)
                        elif n_unique < self.embedding_threshold:
                            self.low_cardinality_indices.append(column)
                            self.columns_to_encode.append(column)
                        else:
                            self.high_cardinality_indices.append(column)
                        self.cat_columns.append(column)
                    else:
                        self.num_columns.append(column)

                if not self.missing_values:
                    if len(self.num_columns) > 0:
                        self.mean_train_num = X[self.num_columns].mean(axis=0).iloc[0]
                    if len(self.cat_columns) > 0:
                        self.mode_train_cat = X[self.cat_columns].mode(axis=0).iloc[0]

            if not self.missing_values:
                if len(self.num_columns) > 0:
                    X[self.num_columns] = X[self.num_columns].fillna(
                        self.mean_train_num
                    )
                if len(self.cat_columns) > 0:
                    X[self.cat_columns] = X[self.cat_columns].fillna(
                        self.mode_train_cat
                    )

            if is_train:
                self.not_encoded_columns = [
                    col for col in X.columns if col not in self.columns_to_encode
                ]
                # self.columns_to_encode = [col for col in X.columns if col in self.not_encoded_columns and col not in self.num_columns]

                self.encoder_ordinal = ce.OrdinalEncoder(
                    cols=self.binary_indices, handle_missing="return_nan"
                )
                self.encoder_ordinal.fit(X)

                if self.embedding_threshold > 1:
                    self.encoder_ohe = ce.OneHotEncoder(
                        cols=self.low_cardinality_indices, handle_missing="return_nan"
                    )
                    self.encoder_ohe.fit(X)

            X = self.encoder_ordinal.transform(X)
            if self.embedding_threshold > 1:
                X = self.encoder_ohe.transform(X)

            if is_train:
                self.encoder_ordinal_emb = ce.OrdinalEncoder(
                    cols=self.high_cardinality_indices, handle_missing="return_nan"
                )
                self.encoder_ordinal_emb.fit(X)

            X = self.encoder_ordinal_emb.transform(X)

            # replace unknowns (-1) with max+1 per column
            self.max_vals = {}
            for col in self.cat_columns:
                max_val = X[col].max(skipna=True)
                self.max_vals[col] = max_val
                X.loc[X[col] == -1, col] = max_val + 1

            X = X.astype(np.float32)

            if is_train:
                self.encoded_columns = [
                    col
                    for col in X.columns
                    if col not in self.not_encoded_columns
                    and col not in self.num_columns
                ]
                self.encoded_columns_indices = [
                    X.columns.get_loc(col) for col in self.encoded_columns
                ]

                self.num_columns_indices = [
                    X.columns.get_loc(col) for col in self.num_columns
                ]
                self.column_names_dataframe = X.columns.tolist()

            if len(self.num_columns_indices) + len(self.encoded_columns_indices) > 0:
                if is_train:
                    if self.use_robust_scale_smoothing:
                        self.normalizer = RobustScaleSmoothClipTransform()
                        self.normalizer.fit(
                            X.iloc[
                                :,
                                self.num_columns_indices + self.encoded_columns_indices,
                            ].values.astype(np.float32)
                        )
                    else:
                        # --- prepare training block for quantile fit (only numeric cols) ---
                        quantile_noise = 1e-4
                        np.random.seed(42)

                        # extract numeric block as float32

                        quantile_train_num = X.iloc[
                            :, self.num_columns_indices + self.encoded_columns_indices
                        ].to_numpy(dtype=np.float32, copy=True)

                        # add tiny noise to avoid duplicate values (per column)
                        stds = np.nanstd(quantile_train_num, axis=0, keepdims=True)
                        noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                        quantile_train_num += noise_std * np.random.randn(
                            *quantile_train_num.shape
                        ).astype(np.float32)

                        # --- fit on numeric block only ---
                        from sklearn.preprocessing import QuantileTransformer

                        self.normalizer = QuantileTransformer(
                            n_quantiles=min(quantile_train_num.shape[0], 1000),
                            output_distribution="normal",
                        )

                        self.normalizer.fit(quantile_train_num)

                # --- transform train/val numeric blocks ---
                X_num = self.normalizer.transform(
                    X.iloc[
                        :, self.num_columns_indices + self.encoded_columns_indices
                    ].to_numpy(dtype=np.float32)
                )

                # --- write back the transformed blocks by position ---
                X = X.copy()
                X.iloc[:, self.num_columns_indices + self.encoded_columns_indices] = (
                    X_num
                )

            X = X.values.astype(np.float32)

        elif not self.use_category_embeddings and self.use_numeric_embeddings:
            if is_train:
                self.binary_indices = []
                self.low_cardinality_indices = []
                self.high_cardinality_indices = []
                self.num_columns = []

                for column_index, column in enumerate(X.columns):
                    col_data = X.iloc[:, column_index]
                    n_unique = col_data.nunique(dropna=True)

                    if self._cat_features is not None and column in self._cat_features:
                        if n_unique <= 2:
                            self.binary_indices.append(column)
                        elif n_unique < self.loo_cardinality:
                            self.low_cardinality_indices.append(column)
                        else:
                            self.high_cardinality_indices.append(column)
                    else:
                        self.num_columns.append(column)

                self.cat_columns = [
                    col for col in X.columns if col not in self.num_columns
                ]

                if not self.missing_values:
                    if len(self.num_columns) > 0:
                        self.mean_train_num = X[self.num_columns].mean(axis=0).iloc[0]
                        X[self.num_columns] = X[self.num_columns].fillna(
                            self.mean_train_num
                        )
                    if len(self.cat_columns) > 0:
                        self.mode_train_cat = X[self.cat_columns].mode(axis=0).iloc[0]
                        X[self.cat_columns] = X[self.cat_columns].fillna(
                            self.mode_train_cat
                        )

                self.encoder_ordinal = ce.OrdinalEncoder(
                    cols=self.binary_indices, handle_missing="return_nan"
                )
                self.encoder_ordinal.fit(X)

                self.encoder_loo = ce.LeaveOneOutEncoder(
                    cols=self.high_cardinality_indices, handle_missing="return_nan"
                )
                if self.problem_type == "regression":
                    self.encoder_loo.fit(X, (y - self.mean) / self.std)
                else:
                    self.encoder_loo.fit(X, y)

                self.encoder_ohe = ce.OneHotEncoder(
                    cols=self.low_cardinality_indices, handle_missing="return_nan"
                )
                self.encoder_ohe.fit(X)

            if not self.missing_values:
                if len(self.num_columns) > 0:
                    X[self.num_columns] = X[self.num_columns].fillna(
                        self.mean_train_num
                    )
                if len(self.cat_columns) > 0:
                    X[self.cat_columns] = X[self.cat_columns].fillna(
                        self.mode_train_cat
                    )

            X = self.encoder_ordinal.transform(X)
            X = self.encoder_loo.transform(X)
            X = self.encoder_ohe.transform(X)

            if is_train:
                self.encoded_columns = [
                    col for col in X.columns if col not in self.num_columns
                ]
                self.encoded_columns_indices = [
                    X.columns.get_loc(col) for col in self.encoded_columns
                ]
                self.num_columns_indices = [
                    X.columns.get_loc(col) for col in self.num_columns
                ]

                self.column_names_dataframe = X.columns.tolist()

                self.not_encoded_columns = []

            X = X.astype(np.float32)
            if is_train:
                if self.use_robust_scale_smoothing:
                    self.normalizer = RobustScaleSmoothClipTransform()
                    self.normalizer.fit(X.values.astype(np.float32))
                else:
                    quantile_noise = 1e-4
                    quantile_train = np.copy(X.values).astype(np.float32)
                    np.random.seed(42)
                    stds = np.nanstd(quantile_train, axis=0, keepdims=True)
                    noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                    quantile_train += noise_std * np.random.randn(*quantile_train.shape)

                    quantile_train = pd.DataFrame(
                        quantile_train, columns=X.columns, index=X.index
                    )

                    self.normalizer = sklearn.preprocessing.QuantileTransformer(
                        n_quantiles=min(quantile_train.shape[0], 1000),
                        output_distribution="normal",
                    )

                    self.normalizer.fit(quantile_train.values.astype(np.float32))

            X = self.normalizer.transform(X.values.astype(np.float32))

        return X

    def _set_default_params(self):
        default_params = {
            "depth": 5,
            "n_estimators": 1024,
            "learning_rate_weights": 0.005,
            "learning_rate_index": 0.01,
            "learning_rate_values": 0.01,
            "learning_rate_leaf": 0.01,
            "temperature": 0.0,
            "use_class_weights": False,
            "dropout": 0.2,
            "selected_variables": 0.8,
            "data_subset_fraction": 1.0,
            "bootstrap": False,
            "verbose": 0,
            "batch_size": 256,
            "early_stopping_epochs": 10,
            "epochs": 200,
            "focal_loss": False,
            "es_metric": False,  # if True use AUC for binary, MSE for regression, val_loss for multiclass
            "missing_values": False,
            "swa": False,
            "cosine_decay_restarts": False,
            "learning_rate_embedding": 0.001,
            "use_category_embeddings": False,
            "use_numeric_embeddings": True,
            "embedding_threshold": 1,
            "embedding_dim_cat": 8,
            "embedding_dim_num": 4,
            "label_smoothing": False,
            "use_robust_scale_smoothing": True,
            "loo_cardinality": 10,
            "optimizer": "adam",
            "reduce_on_plateau_scheduler": False,
            "num_emb_n_bins": 8,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_stopping_metric(self):
        return self.eval_metric

    def _get_default_resources(self) -> tuple[int, int]:
        # Use only physical cores for better performance based on benchmarks
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)

        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))

        return num_cpus, num_gpus

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=hyperparameters,
            **kwargs,
        )

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        # initial version, can be improved
        cat_sizes = []
        for col in X.select_dtypes(include=["category", "object"]):
            if isinstance(X[col], pd.CategoricalDtype):
                # Use .cat.codes for category dtype
                unique_codes = X[col].cat.codes.unique()
            else:
                # For object dtype, treat unique strings as codes
                unique_codes = X[col].astype("category").cat.codes.unique()
            cat_sizes.append(len(unique_codes))

        n_numerical = len(X.select_dtypes(include=["number"]).columns)
        embedding_dim = hyperparameters.get("embedding_dim", 16)
        numeric_embedding_dim = hyperparameters.get("numeric_embedding_dim", 16)

        embedding_memory_cat = (
            sum(cat_sizes)
            * embedding_dim
            * int(hyperparameters["use_category_embeddings"])
            * 32
        )
        embedding_memory_num = (
            n_numerical
            * numeric_embedding_dim
            * int(hyperparameters["use_numeric_embeddings"])
            * 32
        )
        embedding_memory = embedding_memory_cat + embedding_memory_num
        grande_max_memory = (
            hyperparameters["n_estimators"]
            * hyperparameters["depth"] ** 2
            * hyperparameters["depth"]
            * hyperparameters["batch_size"]
            * 32
        )
        return (
            embedding_memory + grande_max_memory + 1_000_000_000
        )  # 1GB fixed overhead

    def _validate_fit_memory_usage(self, mem_error_threshold: float = 1, **kwargs):
        return super()._validate_fit_memory_usage(
            mem_error_threshold=mem_error_threshold, **kwargs
        )

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        # TODO: Need to add train params support, track best epoch
        return {"can_refit_full": False}
