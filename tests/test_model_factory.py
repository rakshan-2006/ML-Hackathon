"""All implemented models load through the shared factory; their results are comparable."""

import numpy as np
import pandas as pd
import pytest

from src.common.schema import FEATURES
from src.evaluation.compare_models import comparison_table
from src.evaluation.metrics import positive_proba
from src.models import gradient_boosting
from src.models.model_factory import get_model
from src.training.cross_validation import load_folds

MODELS = ["logistic_regression", "decision_tree", "gradient_boosting"]
STEPS = {
    "logistic_regression": ["to_float", "impute", "scale", "model"],
    "decision_tree": ["to_float", "impute", "model"],
    "gradient_boosting": ["to_float", "model"],  # native NaN handling, no imputation
}


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    X.loc[::5, "heading_diff_deg"] = np.nan
    y = (X.dist_m < -0.3).astype(int).to_numpy()
    return X, y


@pytest.mark.parametrize("name", MODELS)
def test_factory_builds_fits_and_scores(name):
    X, y = synthetic()
    model = get_model(name).fit(X, y)
    assert list(model.named_steps) == STEPS[name]
    p = positive_proba(model, X)
    assert p.shape == (len(X),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_gradient_boosting_factory_matches_its_own_pipeline():
    X, y = synthetic()
    params = {"max_iter": 20}
    factory = get_model("gradient_boosting", params, seed=7).fit(X, y)
    own = gradient_boosting.build_model(params, seed=7).fit(X, y)
    assert factory.named_steps["model"].get_params() == own.named_steps["estimator"].get_params()
    assert np.array_equal(positive_proba(factory, X), positive_proba(own, X))


def test_committed_results_are_comparable_across_models():
    table = comparison_table(MODELS).set_index("model")
    for name in MODELS:
        cv = pd.read_csv(f"results/models/{name}/cv_metrics.csv", dtype={"fold": str})
        val = cv[cv.fold.str.isdigit() & (cv.split == "validation")]
        assert val.fold.tolist() == ["0", "1", "2", "3", "4"]
        # Every model was evaluated on the same trusted observations per fold.
        assert val.n_trusted_obs.astype(int).tolist() == [32959, 37391, 30444, 31534, 34856]
        assert table.loc[name, "cv_top1_accuracy"] == pytest.approx(val.top1_accuracy.mean())


def test_single_shared_fold_table():
    folds = load_folds()
    assert len(folds) == 80 and sorted(folds.fold.unique()) == [0, 1, 2, 3, 4]
