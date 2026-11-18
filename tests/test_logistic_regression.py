"""Logistic Regression module: pipeline structure and probabilities (synthetic data)."""

import numpy as np
import pandas as pd

from src.common.schema import FEATURES
from src.evaluation.metrics import positive_proba
from src.models.model_factory import get_model


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    X.loc[::5, "heading_diff_deg"] = np.nan
    y = (X.dist_m < -0.5).astype(int).to_numpy()
    return X, y


def test_pipeline_scales_and_returns_probabilities():
    X, y = synthetic()
    model = get_model("logistic_regression").fit(X, y)
    assert list(model.named_steps) == ["to_float", "impute", "scale", "model"]
    p = positive_proba(model, X)
    assert p.shape == (len(X),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert p[y == 1].mean() > p[y == 0].mean()


def test_parameters_are_configurable_and_seeded():
    model = get_model("logistic_regression", {"C": 0.1}, seed=7)
    assert model.named_steps["model"].C == 0.1
    assert model.named_steps["model"].random_state == 7
