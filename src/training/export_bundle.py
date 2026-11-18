"""Split a trained full pipeline into the deployment bundle read by Predictor.

    python -m src.training.export_bundle --name gradient_boosting_tuned --status final --output models
    python -m src.training.export_bundle --name random_forest --status development_baseline \
        --output models/development

Reads artifacts/<name>/<pipeline> and results/models/<name>/model_card.json and
writes preprocessing.joblib (every fitted step before the estimator; the shared
ordered float64 feature transform when the pipeline has none), best_model.joblib
(the bare estimator) and model_card.json with artifact_layout and checksums. The
bundle is accepted only if it reproduces the full pipeline's student_test scores.
"""

import argparse
import json
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.common.paths import PROJECT_ROOT
from src.common.schema import FEATURES, SCHEMA_VERSION
from src.evaluation.metrics import positive_proba
from src.inference.predictor import (CARD_FILE, LAYOUT, MODEL_FILE, PREPROCESSING_FILE,
                                     Predictor, file_sha256)
from src.training.preprocessing import to_float64
from src.training.train_models import ARTIFACTS, DATA, RESULTS

STATUSES = ("development_baseline", "final")
PIPELINE_FILES = ("pipeline.joblib", "gradient_boosting_pipeline.joblib")


def split_pipeline(pipeline):
    """(preprocessing, estimator) such that estimator(preprocessing(X)) == pipeline(X)."""
    *steps, (_, estimator) = pipeline.steps
    preprocessing = Pipeline(steps) if steps else Pipeline(
        [("to_float", FunctionTransformer(to_float64))])
    return preprocessing, estimator


def git_revision():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PROJECT_ROOT, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export(name, status, output):
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    source = ARTIFACTS / name
    pipeline_path = next((source / f for f in PIPELINE_FILES if (source / f).is_file()), None)
    if pipeline_path is None:
        raise FileNotFoundError(f"no fitted pipeline in {source}")
    card = json.loads((RESULTS / name / "model_card.json").read_text(encoding="utf-8"))
    if card.get("ordered_features", card.get("features")) != FEATURES:
        raise ValueError("source model card feature order differs from schema.FEATURES")
    if card.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("source model card schema version differs")

    pipeline = joblib.load(pipeline_path)
    preprocessing, estimator = split_pipeline(pipeline)
    output.mkdir(parents=True, exist_ok=True)
    stale = [p.name for p in output.iterdir() if p.name in (MODEL_FILE, PREPROCESSING_FILE, CARD_FILE)]
    if stale:
        raise FileExistsError(f"{output} already holds a bundle ({stale}); choose an empty directory")
    joblib.dump(estimator, output / MODEL_FILE, compress=3)
    joblib.dump(preprocessing, output / PREPROCESSING_FILE)

    bundle_card = {
        **card,
        "status": status,
        "artifact_layout": LAYOUT,
        "ordered_features": FEATURES,
        "estimator_class": type(estimator).__name__,
        "class_mapping": {"classes_": [int(c) for c in estimator.classes_],
                          "positive_column": [int(c) for c in estimator.classes_].index(1)},
        "preprocessing_steps": [s for s, _ in preprocessing.steps],
        "artifact_semantics": "Predictor applies preprocessing.joblib exactly once, then "
                              "best_model.joblib.predict_proba; class 1 is located via classes_.",
        "feature_policy": "Features are the 13 ordered physical features only; IDs, coordinates, "
                          "alignment, reference membership and labels are never inputs. "
                          "next_dist_m uses the next GPS point (whole-trip, not streaming).",
        "source_pipeline": {"file": pipeline_path.name, "sha256": file_sha256(pipeline_path)},
        "artifact_sha256": {MODEL_FILE: file_sha256(output / MODEL_FILE),
                            PREPROCESSING_FILE: file_sha256(output / PREPROCESSING_FILE)},
        "export_environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                               "numpy": np.__version__, "pandas": pd.__version__,
                               "joblib": joblib.__version__},
        "code_revision": git_revision(),
    }
    (output / CARD_FILE).write_text(json.dumps(bundle_card, indent=2, default=str), encoding="utf-8")

    # Parity: the reloaded split bundle must reproduce the full pipeline on student_test.
    cand = pd.read_parquet(DATA / "student_test_candidates.parquet")
    full = positive_proba(pipeline, cand[FEATURES])
    split = Predictor.load(output).score_candidates(cand)
    max_diff = float(np.abs(full - split).max())
    if max_diff > 1e-12:
        raise AssertionError(f"split bundle differs from the full pipeline by {max_diff}")
    print(json.dumps({"output": str(output), "status": status, "model_version": card["model_version"],
                      "parity_rows": int(len(cand)), "parity_max_abs_diff": max_diff,
                      "artifact_sha256": bundle_card["artifact_sha256"]}, indent=2))
    return bundle_card


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="results/artifacts folder of the trained model")
    parser.add_argument("--status", required=True, choices=STATUSES)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "models"))
    args = parser.parse_args(argv)
    from pathlib import Path
    export(args.name, args.status, Path(args.output))


if __name__ == "__main__":
    main()
