"""Random Forest serialization and candidate scoring adapter."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.common.schema import FEATURES, SCHEMA_VERSION


class RandomForestPredictor:
    """Load the saved pipeline and score candidate rows in input order."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        feature_path = self.artifact_dir / "features.json"
        card_path = self.artifact_dir / "model_card.json"
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing feature schema: {feature_path}")
        if not card_path.is_file():
            raise FileNotFoundError(f"Missing model card: {card_path}")

        feature_document = json.loads(feature_path.read_text(encoding="utf-8"))
        card = json.loads(card_path.read_text(encoding="utf-8"))
        saved_features = feature_document.get("features")
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
        self.pipeline: Pipeline = joblib.load(
            self.artifact_dir / "random_forest.joblib"
        )

    def score_candidates(self, cand_df: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities in the input candidate order."""
        missing = [name for name in self.features if name not in cand_df.columns]
        if missing:
            raise ValueError(
                "Missing required candidate features: " + ", ".join(missing)
            )
        X = cand_df.loc[:, self.features]
        probabilities = np.asarray(
            self.pipeline.predict_proba(X)[:, 1], dtype=np.float64
        )
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
