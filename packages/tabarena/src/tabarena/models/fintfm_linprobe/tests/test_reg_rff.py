from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabarena.models.fintfm_linprobe.rff import RFFRidgeCV

ALPHAS = np.logspace(-3, 5, 17)


def nonlinear(n: int, seed: int):
    """A target with a sine and an interaction that no linear map captures."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = np.sin(2.0 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.5 * X[:, 3] + rng.normal(scale=0.05, size=n)
    return X, y


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_rff_beats_ridge_on_nonlinear_target():
    X, y = nonlinear(1500, 0)
    Xt, yt = nonlinear(500, 1)
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X, y)
    rff = make_pipeline(StandardScaler(), RFFRidgeCV(alphas=ALPHAS, n_features=400, random_state=0)).fit(X, y)
    assert rmse(rff.predict(Xt), yt) < 0.6 * rmse(ridge.predict(Xt), yt)


def test_rff_not_worse_on_linear_target():
    rng = np.random.default_rng(2)
    coef = np.array([1.0, -2.0, 0.5, 0.0, 1.5])
    X = rng.normal(size=(600, 5))
    y = X @ coef + rng.normal(scale=0.1, size=600)
    Xt = rng.normal(size=(300, 5))
    yt = Xt @ coef
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X, y)
    rff = make_pipeline(StandardScaler(), RFFRidgeCV(alphas=ALPHAS, n_features=200, random_state=0)).fit(X, y)
    assert rmse(rff.predict(Xt), yt) < 1.5 * rmse(ridge.predict(Xt), yt) + 0.02


def test_rff_respects_seed():
    X, y = nonlinear(400, 3)
    a = RFFRidgeCV(alphas=ALPHAS, n_features=100, random_state=5).fit(X, y).predict(X)
    b = RFFRidgeCV(alphas=ALPHAS, n_features=100, random_state=5).fit(X, y).predict(X)
    c = RFFRidgeCV(alphas=ALPHAS, n_features=100, random_state=6).fit(X, y).predict(X)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_rff_caps_features_by_rows():
    X, y = nonlinear(40, 4)
    est = RFFRidgeCV(alphas=ALPHAS, n_features=1000, random_state=0).fit(X, y)
    assert est._d_rff == 20
    assert np.isfinite(est.predict(X)).all()


def test_reg_readout_validation():
    pytest.importorskip("fintfm")
    from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel

    assert FinTFMLinearProbeModel._check_reg_readout(None, source="x") == "ridge"
    assert FinTFMLinearProbeModel._check_reg_readout("rff", source="x") == "rff"
    with pytest.raises(ValueError, match="reg_readout"):
        FinTFMLinearProbeModel._check_reg_readout("svm", source="reg_readout")


def test_regression_entry_default_is_rff_and_ridge_is_explicit(tmp_path_factory, monkeypatch):
    pytest.importorskip("fintfm")
    torch = pytest.importorskip("torch")
    from fintfm.modeling.model import FinancialTFM, ModelConfig

    from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel

    monkeypatch.setenv("FINTFM_DEVICE", "cpu")
    monkeypatch.delenv("FINTFM_REG_READOUT", raising=False)
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    cfg = ModelConfig(
        max_features=8, max_classes=10, d_cell=8, d_model=16, n_layers=1, n_col_layers=1, n_heads=2, d_ff=32
    )
    FinancialTFM(cfg).save(str(path))
    Xn, yn = nonlinear(260, 7)
    X = pd.DataFrame(Xn, columns=list("abcd"))
    y = pd.Series(yn)

    def fit(tag: str, **hp):
        params = {
            "checkpoint_regression": str(path),
            "max_context": 60,
            "feature_transform": "squash",
            "max_features_strategy": "select",
            "partition_regression": None,
            **hp,
        }
        m = FinTFMLinearProbeModel(
            path=str(tmp_path_factory.mktemp(tag)),
            name=tag,
            problem_type="regression",
            eval_metric="rmse",
            hyperparameters=params,
        )
        m._init_params()
        m.fit(X=X.iloc[:220].reset_index(drop=True), y=y.iloc[:220].reset_index(drop=True))
        return m

    base, ridge, rff = fit("a"), fit("b", reg_readout="ridge"), fit("c", reg_readout="rff")
    Xt = X.iloc[220:]
    # The regression entry reads ``rff`` (cap 5,000 rows, above this 220-row fit), so the default is rff.
    assert base._reg_readout_eff == "rff"
    assert ridge._reg_readout_eff == "ridge"
    assert np.array_equal(base.predict(Xt), rff.predict(Xt))
    p = base.predict(Xt)
    assert np.isfinite(p).all()
    assert not np.array_equal(p, ridge.predict(Xt))
