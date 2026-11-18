"""Deployment bundle and Predictor contract (synthetic data)."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.common.schema import FEATURES, SCHEMA_VERSION
from src.evaluation.metrics import positive_proba
from src.inference.predictor import (CARD_FILE, LAYOUT, MODEL_FILE, PREPROCESSING_FILE,
                                     Predictor, file_sha256)
from src.models.model_factory import get_model
from src.training.export_bundle import split_pipeline


def candidates(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    X["is_oneway"] = rng.random(n) < 0.5
    X.loc[rng.random(n) < 0.1, "heading_diff_deg"] = np.nan
    y = (X.dist_m + 0.3 * rng.normal(size=n) < 0).astype(int).to_numpy()
    return X, y


def write_bundle(path, pipeline, **card_overrides):
    preprocessing, estimator = split_pipeline(pipeline)
    joblib.dump(estimator, path / MODEL_FILE)
    joblib.dump(preprocessing, path / PREPROCESSING_FILE)
    card = {"artifact_layout": LAYOUT, "schema_version": SCHEMA_VERSION,
            "ordered_features": FEATURES, "model_name": "m", "model_version": "m-v1",
            "status": "final",
            "artifact_sha256": {MODEL_FILE: file_sha256(path / MODEL_FILE),
                                PREPROCESSING_FILE: file_sha256(path / PREPROCESSING_FILE)}}
    card.update(card_overrides)
    (path / CARD_FILE).write_text(json.dumps(card))
    return path


@pytest.mark.parametrize("model", ["logistic_regression", "decision_tree", "gradient_boosting"])
def test_split_bundle_matches_full_pipeline(tmp_path, model):
    X, y = candidates()
    pipeline = get_model(model, {"max_iter": 20} if model == "gradient_boosting" else None).fit(X, y)
    predictor = Predictor.load(write_bundle(tmp_path, pipeline))
    extra = X.assign(trip_id="00000001", label=1 - y, arc_id=7)   # non-features are ignored
    p = predictor.score_candidates(extra)
    assert p.dtype == np.float64 and p.shape == (len(X),)
    np.testing.assert_allclose(p, positive_proba(pipeline, X), rtol=0, atol=1e-12)
    # Row order is preserved: scoring a permutation permutes the scores.
    order = np.random.default_rng(1).permutation(len(X))
    np.testing.assert_array_equal(predictor.score_candidates(X.iloc[order]), p[order])


def test_empty_input_and_missing_features(tmp_path):
    X, y = candidates()
    predictor = Predictor.load(write_bundle(tmp_path, get_model("decision_tree").fit(X, y)))
    empty = predictor.score_candidates(X.iloc[:0])
    assert empty.dtype == np.float64 and empty.shape == (0,)
    with pytest.raises(ValueError, match="missing candidate features"):
        predictor.score_candidates(X.drop(columns=["next_dist_m"]))


def test_checksum_schema_and_layout_are_enforced(tmp_path):
    X, y = candidates()
    pipeline = get_model("decision_tree").fit(X, y)
    for name, overrides, message in [
        ("layout", {"artifact_layout": "full_pipeline"}, "artifact_layout"),
        ("schema", {"schema_version": "0"}, "schema"),
        ("features", {"ordered_features": FEATURES[::-1]}, "feature order"),
    ]:
        folder = tmp_path / name
        folder.mkdir()
        with pytest.raises(ValueError, match=message):
            Predictor.load(write_bundle(folder, pipeline, **overrides))
    folder = write_bundle(tmp_path, pipeline)
    joblib.dump(get_model("decision_tree", {"max_depth": 1}).fit(X, y)[-1], folder / MODEL_FILE)
    with pytest.raises(ValueError, match="checksum"):
        Predictor.load(folder)


class ReversedClasses:
    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.8), np.full(len(X), 0.2)])


def test_positive_class_is_located_through_classes(tmp_path):
    X, _ = candidates(5)
    preprocessing, _ = split_pipeline(get_model("decision_tree").fit(*candidates()))
    predictor = Predictor(ReversedClasses(), preprocessing,
                          {"artifact_layout": LAYOUT, "schema_version": SCHEMA_VERSION,
                           "ordered_features": FEATURES})
    np.testing.assert_array_equal(predictor.score_candidates(X), np.full(5, 0.8))
