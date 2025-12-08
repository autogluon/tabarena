import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from skrub import SquashingScaler

from typing import List, Literal

class CustomModel(BaseEstimator):
    def __init__(self, 
                 target_type,
                 scaler='squashing',
                 standardize_target=True,
                 cat_method:Literal['ohe','oof-te']='ohe',
                 scale_binary:bool=True,
                 random_state:int=42,
                 **kwargs
                 ):

        self.target_type = target_type
        self.standardize_target = standardize_target
        self.cat_method = cat_method
        self.scale_binary = scale_binary
        self.random_state = random_state
        
        if self.target_type == 'regression' and standardize_target:
            self.target_scaler = StandardScaler()
        else:
            self.standardize_target = False
            self.target_scaler = None
        
        self.pipeline = None
        self.model: BaseEstimator = None
        if self.cat_method == 'oof-te':
            from autogluon.features import OOFTargetEncodingFeatureGenerator
            self.oof_te = OOFTargetEncodingFeatureGenerator(target_type=self.target_type, keep_original=False, alpha=10., random_state=self.random_state)
        else:
            self.oof_te = None

        # --------------- feature scaling ------------ #
        if scaler == 'standard':
            self.scaler = StandardScaler()
        elif scaler == 'quantile-normal':
            from sklearn.preprocessing import QuantileTransformer
            self.scaler = QuantileTransformer(output_distribution='normal', random_state=self.random_state)
        elif scaler == 'quantile-uniform':
            from sklearn.preprocessing import QuantileTransformer
            self.scaler = QuantileTransformer(output_distribution='uniform', random_state=self.random_state)
        elif scaler == 'squashing':
            self.scaler = SquashingScaler()
        elif scaler == 'squashing':
            self.scaler = SquashingScaler()
        elif scaler is None:
            self.scaler = 'passthrough'
        else:
            raise ValueError("scaler must be 'standard', 'quantile-normal', 'quantile-uniform', 'squashing', or None")

    def _fit_preprocessor(self, X, y, **kwargs):
        if self.cat_method == 'oof-te':
            X = self.oof_te.fit_transform(X, y)

        if not self.scale_binary:
            X.loc[:, X.nunique() == 2] = X.loc[:, X.nunique() == 2].astype('object')

        # Determine which columns are categorical or numerical
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns

        # Define transformers for preprocessing
        transformers = [
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', self.scaler)
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=100))
                # ('pass', 'passthrough')
            ]), categorical_cols)
        ]

        # Combine transformations in a ColumnTransformer
        self.pipeline = ColumnTransformer(transformers)

        # Fit the transformers and transform the data
        self.pipeline.fit(X)

        if self.target_type == 'regression' and self.standardize_target:
            self.target_scaler.fit(y.values.reshape(-1, 1))


    def preprocess(self, X, y=None, is_train=False, **kwargs):
        if self.cat_method == 'oof-te':
            X = self.oof_te._transform(X, is_train=is_train)

        if not self.scale_binary:
            X.loc[:, X.nunique() == 2] = X.loc[:, X.nunique() == 2].astype('object')

        X = self.pipeline.transform(X)
        if y is not None and self.target_type == 'regression' and self.standardize_target:
            y = self.target_scaler.transform(y.values.reshape(-1, 1)).flatten()
        return X, y

    def fit(self, X_in, y_in, **kwargs):
        X = X_in.copy()
        y = y_in.copy()
        
        self._fit_preprocessor(X, y)            
        X, y = self.preprocess(X, y, is_train=True) 
        
        self.model.fit(X, y)

        return self

    def predict(self, X, is_train=False, **kwargs):
        # Transform the features using the fitted pipeline
        X_transformed, _ = self.preprocess(X, is_train=is_train)

        # Predict based on the model type
        if self.target_type == 'regression':
            y_pred = self.model.predict(X_transformed)
            if self.standardize_target:
                y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            return y_pred
        elif self.target_type == 'binary':
            return self.model.predict_proba(X_transformed)[:, 1]
        elif self.target_type == 'multiclass':
            return self.model.predict_proba(X_transformed)
        else:
            raise ValueError("target_type must be 'binary', 'regression' or 'classification'")
        
    def decision_function(self, X, is_train=False, **kwargs):
        # FIXME: Think what to do for regression
        # Transform the features using the fitted pipeline
        if self.cat_method == 'oof-te':
            X = self.oof_te._transform(X, is_train=is_train)
        X_transformed = self.pipeline.transform(X)

        if self.target_type == 'binary':
            return self.model.decision_function(X_transformed)
        elif self.target_type == 'multiclass':
            return self.model.decision_function(X_transformed)
        else:
            raise ValueError("target_type must be 'binary' or 'classification'")

from sklearn.model_selection import KFold, StratifiedKFold
class OOFCustomModel:
    def __init__(self, 
                 target_type,
                 base_model_cls: BaseEstimator,
                 base_model_kwargs: None,
                 n_splits:int=5,
                 random_state:int=42,
                 ):
        assert target_type in {"regression","binary","multiclass"}
        self.target_type = target_type
        self.n_splits = n_splits
        self.random_state = random_state

        if base_model_kwargs is None:
            base_model_kwargs = {}

        if self.target_type == "regression":
            self.kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            self.kf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)

        self.fold_models_: List[BaseEstimator] = [base_model_cls(**base_model_kwargs) for _ in range(self.n_splits)]
        self.full_model: BaseEstimator = base_model_cls(**base_model_kwargs)

    # -----------------------------------------------------------
    def fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()

        oof_preds = []
        for num, (tr, val) in enumerate(self.kf.split(X, y)):
            X_tr = X.iloc[tr]
            y_tr = y.iloc[tr]   
            X_val = X.iloc[val]

            self.fold_models_[num].fit(X_tr, y_tr)
            if self.target_type == "multiclass":
                oof_preds.append(pd.DataFrame(self.fold_models_[num].predict(X_val), index=X.index[val]))
            else:
                oof_preds.append(pd.Series(self.fold_models_[num].predict(X_val), index=X.index[val]))

        self.train_preds_ = pd.concat(oof_preds, axis=0).loc[X.index]
        self.train_decisions_ = pd.concat(oof_preds, axis=0).loc[X.index].values

        self.full_model.fit(X, y)

        return self

    # -----------------------------------------------------------
    def predict(self, X_in, is_train:bool=False, **kwargs):
        X = X_in.copy()

        # TODO: Make sure that the order of samples in X matches training data. 
        if is_train:
            # return stored OOF train encodings
            # rather than recomputing
            assert hasattr(self,"train_preds_"), "fit() not called"
            return self.train_preds_.copy()

        return self.full_model.predict(X, is_train=is_train, **kwargs)
    
    def decision_function(self, X_in, is_train:bool=False, **kwargs):
        X = X_in.copy()

        # TODO: Make sure that the order of samples in X matches training data. 
        if is_train:
            # return stored OOF train encodings
            # rather than recomputing
            assert hasattr(self,"train_decisions_"), "fit() not called"
            return self.train_decisions_.copy()

        return self.full_model.decision_function(X, is_train=is_train, **kwargs)

class CustomLinearModel(CustomModel):
    def __init__(self, 
                 target_type,
                 scaler='squashing',
                 standardize_target=True,
                 cat_method='ohe',
                 scale_binary=True,
                 linear_model_type='lasso',    # linear, lasso, ridge
                 lambda_: Literal['low', 'medium', 'high', float] = 'medium', # single regularization strength param
                 max_degree: int = 1,
                 random_state:int=42,
                 ):
        super().__init__(target_type=target_type, scaler=scaler, standardize_target=standardize_target, cat_method=cat_method, scale_binary=scale_binary, random_state=random_state)

        self.linear_model_type = linear_model_type
        self.max_degree = max_degree
        self.random_state = random_state

        if isinstance(lambda_, str):
            self.lambda_ = self.set_lambda_from_category(cat=lambda_)
        else:
            self.lambda_ = lambda_

        if self.target_type == 'regression':
            if self.linear_model_type == 'linear':
                self.model = LinearRegression(random_state=self.random_state)
            elif self.linear_model_type == 'lasso':
                self.model = Lasso(alpha=self.lambda_, random_state=self.random_state)   # λ ↑ ⇒ more reg
            elif self.linear_model_type == 'ridge':
                self.model = Ridge(alpha=self.lambda_, random_state=self.random_state)   # λ ↑ ⇒ more reg
            else:
                raise ValueError("linear_model_type must be 'linear', 'lasso', or 'ridge'")

        # --------------- classification ------------- #
        elif self.target_type in ['binary', 'multiclass']:

            # linear => no regularization
            if self.linear_model_type == 'linear':
                penalty = 'l2'
                C = 1e9           # effectively infinite C = no reg
                solver = 'lbfgs'

            else:
                # map lasso/ridge
                penalty = 'l1' if self.linear_model_type == 'lasso' else 'l2'

                # λ direction must match regression:
                # λ ↑ ⇒ stronger regularization
                # if self.lambda_ <= 0:
                #     C = 1e9
                # else:
                #     C = 1.0 / self.lambda_

                # auto solver
                solver = 'saga' if penalty == 'l1' else 'lbfgs'

            self.model = LogisticRegression(
                penalty=penalty,
                C=self.lambda_,
                solver=solver,
                max_iter=2000,
                random_state=self.random_state,
            )
            self.target_scaler = None
        else:
            raise ValueError("target_type must be 'binary', 'multiclass', or 'regression'")
        
        if self.max_degree > 1:
            self.model = Pipeline([
                ("poly", PolynomialFeatures(degree=self.max_degree, include_bias=False)),
                ("ridge", self.model)
            ])

    def set_lambda_from_category(self, cat: str) -> float:
        """
        Set self.lambda_ to a single value chosen for the given category
        ('low' | 'medium' | 'high') for this (target_type, linear_model_type).
        Values are picked on a log scale from realistic ranges.
        """
        if cat not in {"low", "medium", "high"}:
            raise ValueError("cat must be one of {'low','medium','high'}")

        table = {
            "regression": {
                "linear":   {"low": 0.0,      "medium": 0.0,     "high": 0.0},
                "ridge":    {"low": 0.1,   "medium": 1.,  "high": 10.},
                # "lasso":    {"low": 6.3e-6,   "medium": 1.0e-4,  "high": 1.6e-3},
                "lasso":    {"low": 1e-2,   "medium": 1e-3,  "high": 1e-4},
            },
            "binary": {
                # 'linear' maps to L1 in your setup
                "linear":   {"low": 4.0e-6,   "medium": 3.2e-5,  "high": 2.5e-4},
                "ridge":    {"low": 10.,   "medium": 1.,  "high": 0.1},
                "lasso":    {"low": 10.,   "medium": 1,  "high": 0.1},
            },
            "multiclass": {
                # slightly stronger than binary for stability
                "linear":   {"low": 1.0e-5,   "medium": 1.0e-4,  "high": 7.5e-4},
                "ridge":    {"low": 2.0e-4,   "medium": 3.0e-3,  "high": 5.0e-2},
                # "lasso":    {"low": 1.0e-5,   "medium": 1.0e-4,  "high": 7.5e-4},
                "lasso":    {"low": 10,   "medium": 1,  "high": 0.1},
            },
        }

        try:
            lambda_ = table[self.target_type][self.linear_model_type][cat]
        except KeyError:
            raise ValueError(f"Unsupported combination target_type={self.target_type!r}, linear_model_type={self.linear_model_type!r}")
        return lambda_

class OOFCustomLinearModel(OOFCustomModel):
    def __init__(self, 
                 target_type,
                 scaler='squashing',
                 standardize_target=True,
                 cat_method='ohe',
                 scale_binary=True,
                 n_splits:int=5,
                 linear_model_type='lasso',    # linear, lasso, ridge
                 lambda_: Literal['low', 'medium', 'high', float] = 'medium',                    # single regularization strength param
                 max_degree: int = 1,
                 random_state:int=42,
                 ):
        base_model_kwargs = {
            'target_type': target_type,
            'scaler': scaler,
            'standardize_target': standardize_target,
            'cat_method': cat_method,
            'scale_binary': scale_binary,
            'linear_model_type': linear_model_type,
            'lambda_': lambda_,
            'max_degree': max_degree,
            'random_state': random_state,
        }
        super().__init__(
            target_type=target_type,
            base_model_cls=CustomLinearModel,
            base_model_kwargs=base_model_kwargs,
            n_splits=n_splits,
            random_state=random_state
        )

class GroupedCustomLinearModel:
    def __init__(self, target_type, 
                 group_col='std', 
                 min_samples=200, 
                 base_model_kwargs: dict=None,
                 random_state:int=42):
        self.target_type = target_type
        self.group_col   = group_col
        self.min_samples = min_samples
        self.random_state = random_state
        self.base_model_kwargs = base_model_kwargs

        if base_model_kwargs is None:
            base_model_kwargs = {}

    def fit(self, X, y):
        # --- global model ---
        self.global_model_ = CustomLinearModel(target_type=self.target_type, random_state=self.random_state, **self.base_model_kwargs)
        self.global_model_.fit(X, y)

        if self.group_col == 'cardinality':
            self.group_col = X.select_dtypes(include=['object', 'category']).nunique().idxmax()
        elif self.group_col == 'std':
            cat_cols = X.select_dtypes(exclude=['number']).columns
            self.group_col = cat_cols[np.argmax([y.groupby(X[col]).mean().std() for col in cat_cols])]
        # --- group models ---
        self.group_models_ = {}
        for g in X[self.group_col].unique():
            X_use = X[X[self.group_col]==g]
            y_use = y[X[self.group_col]==g]
            # print(f'Category {g}: {len(X_use)} samples')
            if len(X_use)>20:
                self.group_models_[g] = CustomLinearModel(target_type=self.target_type, random_state=self.random_state, **self.base_model_kwargs).fit(X_use, y_use)
            else:
                # not enough data for group-specific model
                self.group_models_[g] = self.global_model_

        return self

    def predict(self, X):
        # baseline = global model
        out = pd.Series(
            self.global_model_.predict(X),
            index=X.index
        )

        # override on groups with enough data
        for g, m in self.group_models_.items():
            idx = (X[self.group_col] == g)
            if idx.any():
                out.loc[idx] = m.predict(X.loc[idx])

        return out
    
    def decision_function(self, X):
        # baseline = global model
        out = pd.Series(
            self.global_model_.decision_function(X),
            index=X.index
        )

        # override on groups with enough data
        for g, m in self.group_models_.items():
            idx = (X[self.group_col] == g)
            if idx.any():
                out.loc[idx] = m.decision_function(X.loc[idx])

        return out

class OOFLinearInitScore(OOFCustomLinearModel):
    def __init__(self, target_type: str, init_kwargs=dict(), random_state=None, **lin_kwargs):
        super().__init__(target_type=target_type, random_state=random_state, **init_kwargs, **lin_kwargs)

    def init_score(self, X_in, is_train=False, **kwargs):
        X = X_in.copy()

        if self.target_type == "regression":
            return self.predict(X, is_train=is_train)
        else:
            return self.decision_function(X, is_train=is_train)
        
class LinearInitScore(CustomLinearModel):
    def __init__(self, target_type: str, init_kwargs=dict(), random_state=None, **lin_kwargs):
        super().__init__(target_type=target_type, random_state=random_state, **init_kwargs, **lin_kwargs)

    def init_score(self, X_in, **kwargs):
        X = X_in.copy()

        if self.model is None:
            raise RuntimeError("call .fit(...) first")

        if self.target_type == "regression":
            return self.predict(X)
        else:
            return self.decision_function(X)

class GroupedLinearInitScore(GroupedCustomLinearModel):
    def __init__(self, target_type: str, random_state=None, **lin_kwargs):
        super().__init__(target_type=target_type, random_state=random_state, **lin_kwargs)
    
    def init_score(self, X_in, **kwargs):
        X = X_in.copy()

        if self.target_type == "regression":
            return self.predict(X)

        raw = self.decision_function(X)
        return raw
