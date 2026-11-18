"""Random Forest model definition and saved-pipeline candidate scoring adapter."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.common.schema import FEATURES, SCHEMA_VERSION
from src.evaluation.metrics import positive_proba

NAME = "random_forest"
FAMILY = "Ensemble"
SCALE_FEATURES = False
DEFAULT_PARAMS = {
    "n_estimators": 80,
    "max_depth": 24,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": 0.5,
    "class_weight": "balanced",
    "n_jobs": 4,
}
CLASS_IMBALANCE = (
    "class_weight='balanced' weights training classes inversely to their frequency. "
    "Scores are not calibrated probabilities at the natural class prevalence."
)
LIMITATIONS = [
    "Defaults were selected by the original training-only three-fold search; "
    "the integrated run evaluates them on the canonical five folds.",
    "Class-weighted scores are uncalibrated; evaluate log loss before HMM integration.",
    "No Bayesian optimisation or HMM route reconstruction is included.",
]


def build_estimator(params, seed):
    return RandomForestClassifier(random_state=seed, **params)


class RandomForestPredictor:
    """Load the saved pipeline and score candidate rows in input order."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        feature_path = self.artifact_dir / "features.json"
        card_path = self.artifact_dir / "model_card.json"
        if not card_path.is_file():
            raise FileNotFoundError(f"Missing model card: {card_path}")

        card = json.loads(card_path.read_text(encoding="utf-8"))
        # Shared-run cards embed the feature order; original bundles use features.json.
        if feature_path.is_file():
            feature_document = json.loads(feature_path.read_text(encoding="utf-8"))
            if feature_document.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Incompatible feature schema version")
            saved_features = feature_document.get("features")
        else:
            saved_features = card.get("ordered_features")
        card_features = card.get("ordered_features", card.get("feature_names"))
        if card_features is not None and card_features != saved_features:
            raise ValueError("Feature schema and model card disagree")
        saved_schema = card.get("schema_version")
        if saved_schema != SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible schema: saved {saved_schema!r}, "
                f"expected {SCHEMA_VERSION!r}"
            )
        if saved_features != FEATURES:
            raise ValueError(
                "Saved feature order is incompatible with the official M1 schema."
            )
        self.features = list(saved_features)
        pipeline_path = self.artifact_dir / "pipeline.joblib"
        if not pipeline_path.is_file():
            pipeline_path = self.artifact_dir / "random_forest.joblib"
        self.pipeline: Pipeline = joblib.load(pipeline_path)
        if set(self.pipeline.classes_) != {0, 1}:
            raise ValueError("Random Forest requires fitted classes 0 and 1")

    def score_candidates(self, cand_df: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities in the input candidate order."""
        missing = [name for name in self.features if name not in cand_df.columns]
        if missing:
            raise ValueError(
                "Missing required candidate features: " + ", ".join(missing)
            )
        X = cand_df.loc[:, self.features]
        if X.empty:
            return np.empty(0, dtype=np.float64)
        probabilities = np.asarray(positive_proba(self.pipeline, X), dtype=np.float64)
        if len(probabilities) != len(cand_df):
            raise RuntimeError("Probability output length does not match input.")
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Model returned nonfinite probabilities.")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise RuntimeError("Model returned probabilities outside [0, 1].")
        return probabilities


def score_candidates(
    cand_df: pd.DataFrame, artifact_dir: str | Path
) -> np.ndarray:
    """Module-level integration adapter."""
    return RandomForestPredictor(artifact_dir).score_candidates(cand_df)
