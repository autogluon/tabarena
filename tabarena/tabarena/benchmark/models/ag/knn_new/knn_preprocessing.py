import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    QuantileTransformer,
)
from typing import List, Literal, Optional


class KNNPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical_features: List[str],
        cat_threshold: int = 10,
        handle_unknown: str = "ignore",
        remainder: Literal["passthrough", "drop"] = "passthrough",
        numeric_strategy: Optional[Literal["standard", "quantile"]] = None,
    ):
        """
        Parameters
        ----------
        categorical_features : list[str]
            Names of categorical columns in the input DataFrame.
        cat_threshold : int, default=10
            For categorical columns:
            - <= threshold -> OneHot
            - > threshold  -> Ordinal
            If 0, categorical columns are dropped.
        handle_unknown : {"ignore","use_encoded_value","error"}, default="ignore"
            For OneHotEncoder, "ignore" is valid.
            For OrdinalEncoder, "ignore" is emulated with unknown_value=np.nan.
        remainder : {"passthrough","drop"}, default="passthrough"
            Whether to keep non-listed (non-categorical) columns.
        numeric_strategy : {"standard","quantile", None}, default=None
            How to scale the final transformed matrix:
            - "standard": StandardScaler
            - "quantile": QuantileTransformer (normal output distribution)
            - None: no scaling
        """
        self.categorical_features = list(categorical_features)
        self.cat_threshold = int(cat_threshold)
        self.handle_unknown = handle_unknown
        self.remainder = remainder
        self.numeric_strategy = numeric_strategy

        # fitted attributes
        self.feature_names_in_ = None
        self.non_categorical_features_ = None
        self.encoders_ = {}
        self.low_card_cols_ = []
        self.high_card_cols_ = []
        self.passthrough_cats_ = []
        self.final_scaler_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        missing = [c for c in self.categorical_features if c not in X.columns]
        if missing:
            raise KeyError(f"categorical_features not found in X: {missing}")
        if self.cat_threshold < 0:
            raise ValueError("cat_threshold must be >= 0.")
        if self.remainder not in ("passthrough", "drop"):
            raise ValueError("remainder must be 'passthrough' or 'drop'.")

        self.feature_names_in_ = list(X.columns)
        self.non_categorical_features_ = [
            c for c in self.feature_names_in_ if c not in self.categorical_features
        ]

        if len(self.categorical_features) == len(X.columns) and self.cat_threshold == 0:
            print("Warning: All features are categorical and cat_threshold=0. Fallback to ordinal encoding is used.")
            self.cat_threshold = 1  # fallback to ordinal encoding

        # reset
        self.encoders_.clear()
        self.low_card_cols_.clear()
        self.high_card_cols_.clear()
        self.passthrough_cats_.clear()

        # fit categorical encoders
        for col in self.categorical_features:
            if pd.api.types.is_numeric_dtype(X[col]):
                # numeric column flagged as categorical → passthrough
                self.passthrough_cats_.append(col)
                continue
            if self.cat_threshold == 0:
                continue

            n_unique = X[col].nunique(dropna=True)
            if n_unique <= self.cat_threshold:
                try:
                    enc = OneHotEncoder(
                        handle_unknown=self.handle_unknown,
                        sparse_output=False,
                        drop=None,
                    )
                except TypeError:  # older sklearn
                    enc = OneHotEncoder(
                        handle_unknown=self.handle_unknown,
                        sparse=False,
                        drop=None,
                    )
                enc.fit(X[[col]])
                self.encoders_[col] = enc
                self.low_card_cols_.append(col)
            else:
                if self.handle_unknown == "ignore":
                    enc = OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=np.nan,
                        encoded_missing_value=np.nan,
                    )
                else:
                    enc = OrdinalEncoder(
                        handle_unknown=self.handle_unknown,
                        encoded_missing_value=np.nan,
                    )
                enc.fit(X[[col]])
                self.encoders_[col] = enc
                self.high_card_cols_.append(col)

        # build transformed matrix for scaler
        Xt = self._transform_no_scale(X)

        if self.numeric_strategy is not None and Xt.shape[1] > 0:
            if self.numeric_strategy == "standard":
                self.final_scaler_ = StandardScaler()
            elif self.numeric_strategy == "quantile":
                self.final_scaler_ = QuantileTransformer(output_distribution="normal")
            else:
                raise ValueError("numeric_strategy must be 'standard','quantile', or None.")
            self.final_scaler_.fit(Xt)

        return self

    def _transform_no_scale(self, X):
        outputs = []

        # remainder
        if self.remainder == "passthrough" and self.non_categorical_features_:
            outputs.append(X[self.non_categorical_features_].reset_index(drop=True))

        # passthrough numeric categoricals
        if self.passthrough_cats_:
            outputs.append(X[self.passthrough_cats_].reset_index(drop=True))

        # encoded categoricals
        for col, enc in self.encoders_.items():
            Xt = enc.transform(X[[col]])
            if isinstance(enc, OneHotEncoder):
                cols = enc.get_feature_names_out([col])
                outputs.append(pd.DataFrame(Xt, columns=cols).reset_index(drop=True))
            else:
                outputs.append(pd.DataFrame(Xt, columns=[col]).reset_index(drop=True))

        if not outputs:
            return pd.DataFrame(index=X.index)

        out = pd.concat(outputs, axis=1)
        out.index = X.index
        return out

    def transform(self, X):
        if self.feature_names_in_ is None:
            raise AttributeError("Not fitted yet. Call fit before transform.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        Xt = self._transform_no_scale(X)

        if self.final_scaler_ is not None and Xt.shape[1] > 0:
            Xt_scaled = self.final_scaler_.transform(Xt)
            Xt = pd.DataFrame(Xt_scaled, columns=Xt.columns, index=X.index)

        return Xt

    def get_feature_names_out(self, input_features=None):
        names = []
        if self.remainder == "passthrough":
            names.extend(self.non_categorical_features_)
        names.extend(self.passthrough_cats_)
        for col, enc in self.encoders_.items():
            if isinstance(enc, OneHotEncoder):
                names.extend(enc.get_feature_names_out([col]))
            else:
                names.append(col)
        return np.array(names, dtype=object)
