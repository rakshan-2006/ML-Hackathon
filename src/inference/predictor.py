"""Public candidate scorer for the selected deployment bundle.

    predictor = Predictor.load("models")
    p = predictor.score_candidates(cand_df)   # float64, one per row, input order

Bundle layout (artifact_layout = 'split_estimator_preprocessor'):
  preprocessing.joblib  fitted transform steps, applied exactly once
  best_model.joblib     the bare fitted estimator
  model_card.json       schema/features/version/metrics and the files' SHA-256
The two joblib files must match the card's checksums before they are loaded.
Only schema.FEATURES are read from cand_df; IDs, labels and other columns are ignored.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common.schema import FEATURES, SCHEMA_VERSION

LAYOUT = "split_estimator_preprocessor"
MODEL_FILE = "best_model.joblib"
PREPROCESSING_FILE = "preprocessing.joblib"
CARD_FILE = "model_card.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Predictor:
    def __init__(self, estimator, preprocessing, card):
        if card.get("artifact_layout") != LAYOUT:
            raise ValueError(f"unsupported artifact_layout {card.get('artifact_layout')!r}")
        if card.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema {card.get('schema_version')!r} != {SCHEMA_VERSION!r}")
        if card.get("ordered_features") != FEATURES:
            raise ValueError("model card feature order differs from schema.FEATURES")
        classes = [int(c) for c in getattr(estimator, "classes_", [])]
        if 1 not in classes:
            raise ValueError(f"estimator has no positive class 1 (classes {classes})")
        self.estimator, self.preprocessing, self.card = estimator, preprocessing, card
        self.positive_column = classes.index(1)
        self.model_name = card.get("model_name")
        self.model_version = card.get("model_version")
        self.status = card.get("status")

    @classmethod
    def load(cls, models_dir):
        models_dir = Path(models_dir)
        card = json.loads((models_dir / CARD_FILE).read_text(encoding="utf-8"))
        expected = card.get("artifact_sha256", {})
        for name in (MODEL_FILE, PREPROCESSING_FILE):
            if name not in expected:
                raise ValueError(f"model card has no checksum for {name}")
            actual = file_sha256(models_dir / name)
            if actual != expected[name]:
                raise ValueError(f"{name} checksum {actual} does not match the model card")
        return cls(joblib.load(models_dir / MODEL_FILE),
                   joblib.load(models_dir / PREPROCESSING_FILE), card)

    def score_candidates(self, cand_df: pd.DataFrame) -> np.ndarray:
        missing = [f for f in FEATURES if f not in cand_df.columns]
        if missing:
            raise ValueError(f"missing candidate features: {missing}")
        if len(cand_df) == 0:
            return np.empty(0, dtype=np.float64)
        transformed = self.preprocessing.transform(cand_df[FEATURES])
        p = np.asarray(self.estimator.predict_proba(transformed)[:, self.positive_column],
                       dtype=np.float64)
        if p.shape != (len(cand_df),):
            raise RuntimeError("scorer returned the wrong number of probabilities")
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise RuntimeError("scorer returned probabilities outside [0, 1] or non-finite")
        return p
