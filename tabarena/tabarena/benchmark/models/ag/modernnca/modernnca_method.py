# from TALENT.model.methods.base import Method
from __future__ import annotations

import os.path as osp
import time

import numpy as np
import torch
from autogluon.core.metrics import compute_metric
from tqdm import tqdm

from tabarena.benchmark.models.ag.modernnca.base import Method, check_softmax
from tabarena.benchmark.models.ag.modernnca.data import (
    Dataset,
    data_enc_process,
    data_label_process,
    data_loader_process,
    data_nan_process,
    data_norm_process,
)
from tabarena.benchmark.models.ag.modernnca.modernNCA import ModernNCA
from tabarena.benchmark.models.ag.modernnca.utils import Averager

# from https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/methods/modernNCA.py


def make_random_batches(
    train_size: int,
    batch_size: int,
    device: torch.device | None = None,
    drop_last: bool = True,
):
    permutation = torch.randperm(train_size, device=device)
    batches = permutation.split(batch_size)
    # this function is borrowed from tabr
    # Below, we check that we do not face this issue:
    # https://github.com/pytorch/vision/issues/3816
    # This is still noticeably faster than running randperm on CPU.
    # UPDATE: after thousands of experiments, we faced the issue zero times,
    # so maybe we should remove the assert.
    assert torch.equal(
        torch.arange(train_size, device=device),
        permutation.sort().values,
    )

    # Drop the last batch if it does not have enough samples
    # -> without this, the training code will crash if the last batch has only one sample!
    if drop_last and (len(batches[0]) != len(batches[-1])):
        batches = batches[:-1]

    return batches  # type: ignore[code]


class ModernNCAMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert args.cat_policy == "tabr_ohe"
        # tabr_ohe is the cat_policy doing one-hot encoding for categorical features, but do not concatenate the one-hot encoded features with the numerical features
        # we reuse it from tabr repo, and do not change it
        assert args.num_policy == "none"

    def construct_model(self, model_config=None):
        if model_config is None:
            model_config = self.args.config["model"]
        self.model = ModernNCA(
            d_in=self.n_num_features + self.C_features,
            d_num=self.n_num_features,
            d_out=self.d_out,
            **model_config,
        ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()

    def data_format(self, is_train=True, N=None, C=None, y=None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(
                self.N,
                self.C,
                self.args.num_nan_policy,
                self.args.cat_nan_policy,
            )
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(
                self.N,
                self.C,
                self.args.cat_policy,
            )
            self.n_num_features = self.N["train"].shape[1] if self.N is not None else 0
            self.n_cat_features = self.C["train"].shape[1] if self.C is not None else 0
            self.N, self.normalizer = data_norm_process(self.N, self.args.normalization, self.args.seed)

            if self.is_regression:
                self.d_out = 1
            else:
                self.d_out = len(np.unique(self.y["train"]))
            self.C_features = self.C["train"].shape[1] if self.C is not None else 0
            self.N, self.C, self.y, self.train_loader, self.val_loader, self.criterion = data_loader_process(
                self.is_regression,
                (self.N, self.C),
                self.y,
                self.y_info,
                self.args.device,
                self.args.batch_size,
                is_train=True,
                is_float=self.args.use_float,
            )
            if not self.D.is_regression:
                self.criterion = torch.nn.functional.nll_loss
                return None
            return None
        N_test, C_test, _, _, _ = data_nan_process(
            N,
            C,
            self.args.num_nan_policy,
            self.args.cat_nan_policy,
            self.num_new_value,
            self.imputer,
            self.cat_new_value,
        )
        y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
        N_test, C_test, _, _, _ = data_enc_process(
            N_test,
            C_test,
            self.args.cat_policy,
            None,
            self.ord_encoder,
            self.mode_values,
            self.cat_encoder,
        )
        N_test, _ = data_norm_process(N_test, self.args.normalization, self.args.seed, self.normalizer)
        _, _, _, test_loader, _ = data_loader_process(
            self.is_regression,
            (N_test, C_test),
            y_test,
            self.y_info,
            self.args.device,
            self.args.batch_size,
            is_train=False,
            is_float=self.args.use_float,
        )
        return test_loader

    def early_stop_due_to_timelimit(self, iteration: int) -> bool:
        if iteration > 0 and self.args.time_to_fit_in_seconds is not None:
            pred_time_after_next_epoch = (iteration + 1) / iteration * (time.time() - self._start_time)
            if pred_time_after_next_epoch >= self.args.time_to_fit_in_seconds:
                return True

        return False

    def fit(self, data, info, train=True, config=None):
        N, C, y = data
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        if self.D is None:
            self.D = Dataset(N, C, y, info)
            self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
            self.is_binclass, self.is_multiclass, self.is_regression = (
                self.D.is_binclass,
                self.D.is_multiclass,
                self.D.is_regression,
            )
            self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features

            self.data_format(is_train=True)
        if config is not None:
            self.reset_stats_withconfig(config)
        self.construct_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.config["training"]["lr"],
            weight_decay=self.args.config["training"]["weight_decay"],
        )
        self.train_size = self.N["train"].shape[0] if self.N is not None else self.C["train"].shape[0]
        self.train_indices = torch.arange(self.train_size, device=self.args.device)
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return None

        self._start_time = time.time()

        time_cost = 0
        for epoch in range(self.args.max_epoch):
            # check time limit
            if self.early_stop_due_to_timelimit(iteration=epoch):
                break

            tic = time.time()
            self.train_epoch(epoch)
            self.validate(epoch)
            elapsed = time.time() - tic
            time_cost += elapsed
            print(f"Epoch: {epoch}, Time cost per epoch: {elapsed}")
            if not self.continue_training:
                break
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, f"epoch-last-{self.args.seed!s}.pth"),
        )

        # new: restore to best val model
        self.model.load_state_dict(
            torch.load(osp.join(self.args.save_path, f"best-val-{self.args.seed!s}.pth"))["params"],
        )
        del self.train_loader, self.val_loader
        return time_cost

    def predict(self, data, info, model_name):
        N, C, y = data
        # self.model.load_state_dict(
        #     torch.load(osp.join(self.args.save_path, model_name + '-{}.pth'.format(str(self.args.seed))))['params'])
        print("best epoch {}, best val res={:.4f}".format(self.trlog["best_epoch"], self.trlog["best_res"]))
        ## Evaluation Stage
        self.model.eval()

        test_loader = self.data_format(False, N, C, y)
        candidate_x_num = self.N["train"] if self.N is not None else None
        candidate_x_cat = self.C["train"] if self.C is not None else None
        candidate_y = self.y["train"]
        if self.args.use_float:
            candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
            candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
            if self.is_regression:
                candidate_y = candidate_y.float()

        has_cat = self.C is not None
        has_num = self.N is not None
        if has_cat and has_num:
            candidate_x = torch.cat([candidate_x_num, candidate_x_cat], dim=1)
        elif has_cat and (not has_num):
            candidate_x = candidate_x_cat
        else:
            candidate_x = candidate_x_num

        inverse_scale_pred = self.is_regression and (self.y_info["policy"] == "mean_std")
        std = self.y_info["std"] if inverse_scale_pred else None
        mean = self.y_info["mean"] if inverse_scale_pred else None
        candidate_x = candidate_x.to(self.args.device)

        test_logit = []
        st_time = time.time()
        with torch.no_grad():
            for _i, (X, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
                if has_cat and has_num:
                    X_num, X_cat = X[0], X[1]
                elif has_cat and (not has_num):
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None

                if self.args.use_float:
                    X_num = X_num.float() if X_num is not None else None
                    X_cat = X_cat.float() if X_cat is not None else None

                if has_cat and has_num:
                    x = torch.cat([X_num, X_cat], dim=1)
                elif has_cat and (not has_num):
                    x = X_cat
                else:
                    x = X_num
                x = x.to(self.args.device)

                pred = self.model(
                    x=x,
                    y=None,
                    candidate_x=candidate_x,
                    candidate_y=candidate_y,
                    is_train=False,
                ).squeeze(-1)

                if not pred.shape:
                    print("Try unsqueeze")
                    pred = pred.unsqueeze(0)

                if inverse_scale_pred:
                    print("Try scale")
                    pred = (pred * std) + mean

                print("try to append")
                test_logit.append(pred)

                # -> This hangs as well
                # print("Trying test data")
                # del x, X_num, X_cat

        print("Trying to delete the test_loader")
        del test_loader
        print("Predict time: ", time.time() - st_time)
        return torch.cat(test_logit, 0)

    def train_epoch(self, epoch):
        self.model.train()
        tl = Averager()
        i = 0
        for batch_idx in make_random_batches(self.train_size, self.args.batch_size, self.args.device):
            if self.early_stop_due_to_timelimit(iteration=i):
                self.continue_training = False
                break
            self.train_step = self.train_step + 1

            X_num = self.N["train"][batch_idx] if self.N is not None else None
            X_cat = self.C["train"][batch_idx] if self.C is not None else None
            y = self.y["train"][batch_idx]

            candidate_indices = self.train_indices
            candidate_indices = candidate_indices[~torch.isin(candidate_indices, batch_idx)]

            candidate_x_num = self.N["train"][candidate_indices] if self.N is not None else None
            candidate_x_cat = self.C["train"][candidate_indices] if self.C is not None else None
            candidate_y = self.y["train"][candidate_indices]
            if self.args.use_float:
                X_num = X_num.float() if X_num is not None else None
                X_cat = X_cat.float() if X_cat is not None else None
                candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
                candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
                if self.is_regression:
                    candidate_y = candidate_y.float()
                    y = y.float()
            if X_cat is None and X_num is not None:
                x, candidate_x = X_num, candidate_x_num
            elif X_cat is not None and X_num is None:
                x, candidate_x = X_cat, candidate_x_cat
            else:
                x, candidate_x = torch.cat([X_num, X_cat], dim=1), torch.cat([candidate_x_num, candidate_x_cat], dim=1)

            pred = self.model(
                x=x,
                y=y,
                candidate_x=candidate_x,
                candidate_y=candidate_y,
                is_train=True,
            ).squeeze(-1)

            loss = self.criterion(pred, y)

            tl.add(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i - 1) % 50 == 0 or i == len(self.train_loader):
                print(
                    "epoch {}, train {}/{}, loss={:.4f} lr={:.4g}".format(
                        epoch,
                        i,
                        len(self.train_loader),
                        loss.item(),
                        self.optimizer.param_groups[0]["lr"],
                    ),
                )
            del loss
            i += 1

        tl = tl.item()
        self.trlog["train_loss"].append(tl)

    def validate(self, epoch):
        print("best epoch {}, best val res={:.4f}".format(self.trlog["best_epoch"], self.trlog["best_res"]))

        ## Evaluation Stage
        self.model.eval()
        test_logit, test_label = [], []
        with torch.no_grad():
            for _i, (X, y) in enumerate(self.val_loader):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None

                candidate_x_num = self.N["train"] if self.N is not None else None
                candidate_x_cat = self.C["train"] if self.C is not None else None
                candidate_y = self.y["train"]
                if self.args.use_float:
                    X_num = X_num.float() if X_num is not None else None
                    X_cat = X_cat.float() if X_cat is not None else None
                    candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
                    candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
                    if self.is_regression:
                        candidate_y = candidate_y.float()
                if X_cat is None and X_num is not None:
                    x, candidate_x = X_num, candidate_x_num
                elif X_cat is not None and X_num is None:
                    x, candidate_x = X_cat, candidate_x_cat
                else:
                    x, candidate_x = (
                        torch.cat([X_num, X_cat], dim=1),
                        torch.cat([candidate_x_num, candidate_x_cat], dim=1),
                    )

                pred = self.model(
                    x=x,
                    y=None,
                    candidate_x=candidate_x,
                    candidate_y=candidate_y,
                    is_train=False,
                ).squeeze(-1)

                if not pred.shape:
                    pred = pred.unsqueeze(0)

                test_logit.append(pred)
                test_label.append(y)

        y_pred = torch.cat(test_logit, 0).cpu().numpy()
        labels = torch.cat(test_label, 0).cpu().numpy()

        if not self.is_regression:
            y_pred = check_softmax(y_pred)

        validation_score = compute_metric(
            y=labels,
            metric=self.args.early_stopping_metric,
            y_pred=y_pred if self.is_regression else y_pred.argmax(axis=-1),
            y_pred_proba=y_pred[:, 1] if self.is_binclass else y_pred,
            silent=True,
        )

        if validation_score > self.trlog["best_res"] or epoch == 0:
            self.trlog["best_res"] = validation_score
            self.trlog["best_epoch"] = epoch
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, f"best-val-{self.args.seed!s}.pth"),
            )
            self.val_count = 0
        else:
            self.val_count += 1
            if self.val_count > 20:
                self.continue_training = False
