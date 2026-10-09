"""Dense numeric encoding for PrismBoost.

PrismBoost's estimators take a dense, finite ``float64`` matrix, so the wrapper owns the
encoding. Two choices in here are load-bearing rather than cosmetic, and both are
hyperparameters (``numeric_scaler``, ``categorical_encoder``) instead of constants:

* **Feature scale.** A SEFR node weights feature ``j`` by
  ``(avg_pos_j - avg_neg_j) / (avg_pos_j + avg_neg_j + 1e-7)``, a ratio rather than a
  difference, so a shift of the column changes the weight and not just its scale. Which
  scaling suits a dataset is therefore an open question per dataset, and measuring it says the
  same: over TabArena-Lite no one of the five wins, and the spread between best and worst
  reaches 2x on a single dataset. PrismBoost's own PMLB study reached that conclusion first and
  searched the five per dataset (its optima split 30/25/24/22/20 over 121 datasets).
* **Categorical encoding.** One-hot gives a SEFR hyperplane one axis per level, which stops
  carrying signal once a column has hundreds of levels. Cross-fitted target encoding keeps
  one ordered axis instead, which is the trade the PMLB study searched and what CatBoost's
  ordered target statistics do natively.

Everything is fit on training rows only. ``TargetEncoder`` is applied through
``fit_transform`` so the training rows get its internal cross-fitted encoding rather than the
full-data one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.compose import ColumnTransformer

#: Scalers for the numeric block, the same five the PMLB study searched. ``"minmax"`` and
#: ``"quantile-uniform"`` keep features non-negative; the other three are signed.
NUMERIC_SCALERS = ("minmax", "quantile-uniform", "quantile-normal", "standard", "robust")

#: Encoders for categorical columns with more than ``ohe_max_cardinality`` levels.
CATEGORICAL_ENCODERS = ("onehot", "target")

_MISSING_LEVEL = "__missing__"
_NA_STRINGS = ("nan", "None", "<NA>", "NaN", "")


def _make_scaler(name: str, n_samples: int):
    from sklearn.preprocessing import (
        MinMaxScaler,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    if name == "minmax":
        return MinMaxScaler(clip=True)
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name in ("quantile-uniform", "quantile-normal"):
        n_quantiles = min(1000, max(n_samples, 2))
        return QuantileTransformer(
            output_distribution="uniform" if name == "quantile-uniform" else "normal",
            n_quantiles=n_quantiles,
            subsample=max(10_000, n_quantiles),
            random_state=0,
        )
    raise ValueError(f"numeric_scaler must be one of {NUMERIC_SCALERS}, got {name!r}.")


class PrismBoostPreprocessor:
    """Stateful encoder from an AutoGluon feature frame to a dense ``float64`` matrix."""

    def __init__(
        self,
        *,
        problem_type: str,
        numeric_scaler: str = "quantile-uniform",
        categorical_encoder: str = "target",
        ohe_max_cardinality: int = 32,
        target_encoder_cv: int = 5,
    ) -> None:
        self.problem_type = problem_type
        self.numeric_scaler = numeric_scaler
        self.categorical_encoder = categorical_encoder
        self.ohe_max_cardinality = ohe_max_cardinality
        self.target_encoder_cv = target_encoder_cv
        self._column_transformer: ColumnTransformer | None = None
        self.numeric_columns_: list[str] = []
        self.onehot_columns_: list[str] = []
        self.encoded_columns_: list[str] = []

    # --- fit / transform ---------------------------------------------------------------

    def fit_transform(self, X: pd.DataFrame, y) -> np.ndarray:
        if self.problem_type == "regression":
            # TabArena hands over integer regression targets (Food_Delivery_Time is uint8), and
            # TargetEncoder's fused-type Cython kernel only dispatches on float32 / float64.
            y = np.asarray(y, dtype=np.float64)
        self._split_columns(X)
        X = self._clean(X)
        self._column_transformer = self._build(n_samples=int(X.shape[0]))
        # fit_transform, not fit-then-transform: TargetEncoder only cross-fits in fit_transform,
        # and the training rows must not see their own target through it.
        return self._finalize(self._column_transformer.fit_transform(X, y))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self._column_transformer is None:
            raise RuntimeError("PrismBoostPreprocessor.transform called before fit_transform.")
        return self._finalize(self._column_transformer.transform(self._clean(X)))

    # --- internals ---------------------------------------------------------------------

    def _split_columns(self, X: pd.DataFrame) -> None:
        categorical = list(X.select_dtypes(include=["category", "object", "string"]).columns)
        self.numeric_columns_ = [c for c in X.columns if c not in categorical]
        wide = self.categorical_encoder == "onehot"
        self.onehot_columns_ = [
            c for c in categorical if wide or X[c].nunique(dropna=False) <= self.ohe_max_cardinality
        ]
        self.encoded_columns_ = [c for c in categorical if c not in self.onehot_columns_]

    def _clean(self, X: pd.DataFrame) -> pd.DataFrame:
        """Send infinities to NaN and give every categorical an explicit missing level."""
        X = X.replace([np.inf, -np.inf], np.nan)
        categorical = self.onehot_columns_ + self.encoded_columns_
        if categorical:
            X = X.copy()
            for col in categorical:
                levels = X[col].astype("string")
                X[col] = levels.fillna(_MISSING_LEVEL).replace(
                    dict.fromkeys(_NA_STRINGS, _MISSING_LEVEL),
                )
        return X

    def _build(self, n_samples: int) -> ColumnTransformer:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import MissingIndicator, SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

        transformers: list[tuple] = []
        if self.numeric_columns_:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                            ("scale", _make_scaler(self.numeric_scaler, n_samples)),
                        ],
                    ),
                    self.numeric_columns_,
                ),
            )
            # Missingness as its own 0/1 block. Kept out of the numeric pipeline so the scaler
            # fits feature values only and never the indicators. `error_on_new=False` because a
            # bagged fold can hold the only missing value in a column: the default raises at
            # transform time, and widening the matrix there would not match the fitted model
            # anyway, so a column that was complete in training stays uninstrumented.
            transformers.append(
                (
                    "missing",
                    MissingIndicator(features="missing-only", error_on_new=False),
                    self.numeric_columns_,
                ),
            )
        if self.onehot_columns_:
            transformers.append(
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        max_categories=max(self.ohe_max_cardinality, 2),
                        sparse_output=False,
                        dtype=np.float64,
                    ),
                    self.onehot_columns_,
                ),
            )
        if self.encoded_columns_:
            target_type = {
                "binary": "binary",
                "multiclass": "multiclass",
                "regression": "continuous",
            }[self.problem_type]
            transformers.append(
                (
                    "target",
                    Pipeline(
                        [
                            ("encode", TargetEncoder(target_type=target_type, cv=self.target_encoder_cv)),
                            # Target-encoded columns come out on the target's scale; SEFR compares
                            # feature magnitudes across columns, so put them back on a common one.
                            ("scale", StandardScaler()),
                        ],
                    ),
                    self.encoded_columns_,
                ),
            )
        if not transformers:
            raise ValueError("PrismBoost received a frame with no numeric or categorical columns.")
        return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)

    @staticmethod
    def _finalize(Xt) -> np.ndarray:
        Xt = np.asarray(Xt, dtype=np.float64)
        # PrismBoost validates for finiteness; a scaler can still emit +/-inf on a degenerate
        # column, so clamp here rather than fail the fit.
        return np.nan_to_num(Xt, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def estimate_n_columns(
        X: pd.DataFrame,
        *,
        ohe_max_cardinality: int = 32,
        num_classes: int = 1,
    ) -> int:
        """Upper bound on the encoded width, for the memory estimate.

        A categorical column costs at most ``ohe_max_cardinality`` one-hot columns, or one
        target-encoded column per class, whichever is wider.
        """
        n_categorical = int(X.select_dtypes(include=["category", "object", "string"]).shape[1])
        n_numeric = max(int(X.shape[1]) - n_categorical, 0)
        per_categorical = max(ohe_max_cardinality, num_classes, 2)
        return n_numeric * 2 + n_categorical * per_categorical
