"""Synthetic and schema tests for the Member 2 Random Forest implementation."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from src.common.schema import FEATURES, SCHEMA_VERSION
from src.models.random_forest import RandomForestPredictor
from src.models.train_random_forest import (
    candidate_metrics,
    make_pipeline,
    verify_group_separation,
)


def synthetic_frame(rows: int = 100) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    rng = np.random.default_rng(2026)
    X = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    X.iloc[::11, 0] = np.nan
    y = np.array([0, 1] * (rows // 2) + ([0] if rows % 2 else []), dtype=np.int8)
    groups = pd.Series([f"{i // 10:08d}" for i in range(rows)])
    return X, y, groups


def test_training_on_synthetic_fixture() -> None:
    X, y, _ = synthetic_frame()
    pipeline = make_pipeline({
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
    })
    pipeline.fit(X, y)
    probabilities = pipeline.predict_proba(X)[:, 1]
    assert probabilities.dtype == np.float64
    assert probabilities.shape == (len(X),)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_grouped_cv_rejects_trip_overlap() -> None:
    _, _, groups = synthetic_frame()
    train = np.arange(0, 50)
    valid = np.arange(45, 100)
    with pytest.raises(RuntimeError, match="Trip overlap"):
        verify_group_separation(groups, train, valid)


def test_grouped_cv_accepts_separate_trip_groups() -> None:
    _, _, groups = synthetic_frame()
    train = np.flatnonzero(groups.to_numpy() != "00000000")
    valid = np.flatnonzero(groups.to_numpy() == "00000000")
    verify_group_separation(groups, train, valid)


def test_predictor_missing_feature() -> None:
    artifact_dir = Path("unused-artifacts")
    with pytest.raises(FileNotFoundError):
        RandomForestPredictor(artifact_dir)


def test_predictor_serialization_shape_range_and_order(tmp_path: Path) -> None:
    X, y, _ = synthetic_frame()
    pipeline = make_pipeline({
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
    })
    pipeline.fit(X, y)
    joblib.dump(pipeline, tmp_path / "random_forest.joblib")
    (tmp_path / "features.json").write_text(
        json.dumps({
            "schema_version": SCHEMA_VERSION,
            "features": FEATURES,
        }),
        encoding="utf-8",
    )
    (tmp_path / "model_card.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION}),
        encoding="utf-8",
    )

    predictor = RandomForestPredictor(tmp_path)
    original = X.iloc[[7, 2, 91]].copy()
    probabilities = predictor.score_candidates(original)
    assert probabilities.shape == (3,)
    assert probabilities.dtype == np.float64
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()

    with pytest.raises(ValueError, match="Missing required candidate features"):
        predictor.score_candidates(original.drop(columns=[FEATURES[0]]))


def test_feature_schema_is_official_and_stable() -> None:
    assert FEATURES == [
        "dist_m",
        "cand_rank",
        "dist_minus_min",
        "heading_diff_deg",
        "speed_mps",
        "arc_len_m",
        "proj_frac",
        "n_candidates",
        "is_oneway",
        "source_degree",
        "target_degree",
        "prev_dist_m",
        "next_dist_m",
    ]
