"""Shared training components: metrics, folds, preprocessing, factory (synthetic data)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.common.schema import FEATURES
from src.evaluation.metrics import candidate_metrics, observation_top1, positive_proba
from src.models.model_factory import get_model, model_params
from src.training.cross_validation import FOLDS_PATH, assign_folds, load_folds
from src.training.preprocessing import make_pipeline, to_float64


def table(rows, columns):
    return pd.DataFrame(rows, columns=columns)


def test_top1_policy():
    scored = table([
        ("t", 0, 10, 0, 0.6), ("t", 0, 11, 1, 0.6),   # tie -> lowest cand_rank (arc 10)
        ("t", 1, 20, 0, 0.2), ("t", 1, 21, 1, 0.9),   # picks 21, reference is 20 -> wrong
        ("t", 2, 30, 0, 0.9),                          # reference 31 not a candidate -> miss
        ("t", 4, 50, 0, 0.9),                          # untrusted observation -> excluded
        ("t", 5, 60, 0, 0.9),                          # invalid observation -> excluded
    ], ["trip_id", "obs_idx", "arc_id", "cand_rank", "p_model"])
    obs = table([("t", i, i != 5) for i in range(6)], ["trip_id", "obs_idx", "is_valid"])
    align = table([("t", 0, "ok", 10), ("t", 1, "ok", 20), ("t", 2, "ok", 31),
                   ("t", 3, "ok", 40), ("t", 4, "ambiguous", 50), ("t", 5, "ok", 60)],
                  ["trip_id", "obs_idx", "align_status", "ref_arc_id"])
    res = observation_top1(scored, obs, align)
    # Trusted: obs 0-3 (obs 3 has zero candidates and counts as wrong).
    assert res["n_trusted_obs"] == 4
    assert res["n_top1_correct"] == 1
    assert res["top1_accuracy"] == 0.25
    assert res["n_reference_misses"] == 2
    assert res["n_zero_candidate_trusted"] == 1
    assert res["top1_upper_bound"] == 0.5


def test_candidate_metrics_known_values():
    m = candidate_metrics([1, 1, 0, 0], [0.9, 0.2, 0.6, 0.1])
    assert m["candidate_accuracy"] == 0.5
    assert m["precision_pos"] == 0.5 and m["recall_pos"] == 0.5
    assert m["f1_macro"] == 0.5
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert np.isfinite(m["log_loss"])


def test_positive_proba_uses_classes_mapping():
    class Reversed:
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return np.array([[0.8, 0.2]])
    assert positive_proba(Reversed(), None)[0] == 0.8
    model = LogisticRegression().fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    assert positive_proba(model, [[3.0]])[0] > 0.5


def test_folds_are_grouped_and_deterministic():
    groups = pd.Series(np.repeat([f"{i:08d}" for i in range(10)], 7))
    a, b = assign_folds(groups), assign_folds(groups)
    assert a.equals(b)
    assert not a.trip_id.duplicated().any() and set(a.fold) == set(range(5))


@pytest.mark.skipif(not FOLDS_PATH.is_file(), reason="shared fold file not generated")
def test_saved_folds_cover_80_train_trips():
    folds = load_folds()
    assert len(folds) == 80 and set(folds.fold) == set(range(5))


def test_preprocessing_uses_feature_order_and_training_medians():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, len(FEATURES))), columns=FEATURES)
    X.iloc[::3, 3] = np.nan
    shuffled = X[FEATURES[::-1]]
    assert np.array_equal(to_float64(shuffled), to_float64(X), equal_nan=True)
    pipe = make_pipeline(LogisticRegression(), scale=True).fit(X, (X.iloc[:, 0] > 0).astype(int))
    assert np.allclose(pipe.named_steps["impute"].statistics_, np.nanmedian(X.to_numpy(), axis=0))
    with pytest.raises(ValueError):
        to_float64(X.drop(columns=FEATURES[0]))


def test_factory_rejects_unknown_models_and_params():
    with pytest.raises(ValueError):
        get_model("no_such_model")
    with pytest.raises(ValueError):
        get_model("../x")
