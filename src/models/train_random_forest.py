"""Reproducible Random Forest training for official M1 candidate data."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import psutil
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.common.schema import FEATURES, SCHEMA_VERSION
from src.data.model_data import load_model_data


RANDOM_SEED = 2026


def make_pipeline(params: dict[str, Any]) -> Pipeline:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    classifier = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("imputer", imputer),
            ("classifier", classifier),
        ]
    )


def verify_group_separation(
    groups: pd.Series, train_indices: np.ndarray, valid_indices: np.ndarray
) -> None:
    train_trips = set(groups.iloc[train_indices].tolist())
    valid_trips = set(groups.iloc[valid_indices].tolist())
    overlap = train_trips & valid_trips
    if overlap:
        raise RuntimeError(f"Trip overlap across folds: {sorted(overlap)}")


def candidate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": float(matrix[0, 0]),
        "fp": float(matrix[0, 1]),
        "fn": float(matrix[1, 0]),
        "tp": float(matrix[1, 1]),
    }


def candidate_top1_accuracy(
    candidates: pd.DataFrame,
    probabilities: np.ndarray,
    alignment: pd.DataFrame,
) -> dict[str, float]:
    """Calculate candidate top-1 on confident observations without inference-time route reads."""
    scored = candidates[["trip_id", "obs_idx", "arc_id"]].copy()
    scored["probability"] = probabilities
    top = scored.sort_values(
        ["trip_id", "obs_idx", "probability", "arc_id"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["trip_id", "obs_idx"])
    reference = alignment.copy()
    reference["ref_hit"] = (
        (reference.align_status == "ok") & reference.ref_in_candidates
    )
    merged = top.merge(
        reference[["trip_id", "obs_idx", "ref_arc_id", "ref_hit"]],
        on=["trip_id", "obs_idx"],
        how="left",
        validate="one_to_one",
    )
    eligible = merged["ref_hit"] == True
    if eligible.any():
        top1 = float(
            np.mean(merged.loc[eligible, "arc_id"].to_numpy()
                     == merged.loc[eligible, "ref_arc_id"].to_numpy())
        )
    else:
        top1 = 0.0
    return {
        "top1_accuracy_eligible": top1,
        "eligible_reference_hits": float(eligible.sum()),
        "all_observation_denominator": float(len(reference)),
        "missed_or_unavailable_denominator": float(
            (~eligible).sum()
        ),
    }


def evaluate_partition(
    artifact_dir: Path,
    partition: str,
    pipeline: Pipeline,
    processed_root: Path,
) -> dict[str, Any]:
    candidates = pd.read_parquet(processed_root / f"{partition}_candidates.parquet")
    alignment = pd.read_parquet(processed_root / f"{partition}_alignment.parquet")
    # Hold-out evaluation may score all candidates, not only non-null labels.
    start = time.perf_counter()
    probabilities = pipeline.predict_proba(candidates[FEATURES])[:, 1]
    inference_seconds = time.perf_counter() - start
    labelled = candidates[candidates["label"].notna()].copy()
    labelled_positions = candidates.index.isin(labelled.index)
    labelled_probabilities = probabilities[labelled_positions]
    predictions = (labelled_probabilities >= 0.5).astype(np.int8)
    metrics = candidate_metrics(
        labelled["label"].to_numpy(dtype=np.int8),
        predictions,
    )
    top1 = candidate_top1_accuracy(
        candidates,
        probabilities,
        alignment,
    )
    return {
        "partition": partition,
        "candidate_row_count": int(len(candidates)),
        "labelled_candidate_count": int(len(labelled)),
        "positive_label_count": int(labelled["label"].eq(1).sum()),
        "inference_seconds": float(inference_seconds),
        "candidate_metrics": metrics,
        "observation_metrics": top1,
    }


def cross_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    configs: list[dict[str, Any]],
    folds: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    splitter = GroupKFold(n_splits=folds)
    cv = list(splitter.split(X, y, groups))
    for train_indices, valid_indices in cv:
        verify_group_separation(groups, train_indices, valid_indices)

    results: list[dict[str, Any]] = []
    for config in configs:
        fold_records = []
        started = time.perf_counter()
        for fold_number, (train_indices, valid_indices) in enumerate(cv, start=1):
            pipeline = make_pipeline(config)
            pipeline.fit(X.iloc[train_indices], y[train_indices])
            probabilities = pipeline.predict_proba(X.iloc[valid_indices])[:, 1]
            predictions = (probabilities >= 0.5).astype(np.int8)
            metrics = candidate_metrics(y[valid_indices], predictions)
            fold_records.append(
                {
                    "fold": fold_number,
                    "validation_trips": int(len(set(groups.iloc[valid_indices]))),
                    "candidate_rows": int(len(valid_indices)),
                    **metrics,
                }
            )
        elapsed = time.perf_counter() - started
        aggregate = {
            "mean_precision": float(np.mean([r["precision"] for r in fold_records]))
        }
        aggregate.update(
            {
                "mean_recall": float(np.mean([r["recall"] for r in fold_records]))
            }
        )
        aggregate.update(
            {
                "mean_f1": float(np.mean([r["f1"] for r in fold_records]))
            }
        )
        aggregate.update(
            {
                "mean_macro_f1": float(
                    np.mean([r["macro_f1"] for r in fold_records])
                )
            }
        )
        results.append(
            {
                **config,
                "cv_seconds": elapsed,
                **aggregate,
                "folds": fold_records,
            }
        )
    return results, {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--artifact-dir", default="models/random_forest_v1")
    parser.add_argument("--results-dir", default="experiments/random_forest")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    artifact_dir = Path(args.artifact_dir)
    results_dir = Path(args.results_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups, keys = load_model_data(processed_root, "train")
    if list(X.columns) != FEATURES:
        raise ValueError("Loaded features do not match the official M1 feature order.")
    y_array = y.to_numpy(dtype=np.int8)

    configs = [
        {
            "n_estimators": 80,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
        {
            "n_estimators": 80,
            "max_depth": 16,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
        {
            "n_estimators": 80,
            "max_depth": 16,
            "min_samples_split": 4,
            "min_samples_leaf": 5,
            "max_features": 0.5,
            "class_weight": "balanced_subsample",
        },
        {
            "n_estimators": 80,
            "max_depth": 24,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": 0.5,
            "class_weight": "balanced",
        },
    ]

    cv_results, _ = cross_validate(
        X, y_array, groups, configs, args.folds
    )
    cv_frame = pd.DataFrame(
        [
            {k: v for k, v in result.items() if k != "folds"}
            for result in cv_results
        ]
    )
    cv_frame.to_csv(results_dir / "random_forest_cv.csv", index=False)
    fold_rows = [
        {"config_index": config_index, **fold}
        for config_index, result in enumerate(cv_results)
        for fold in result["folds"]
    ]
    pd.DataFrame(fold_rows).to_csv(
        results_dir / "random_forest_per_fold.csv", index=False
    )

    best_index = int(cv_frame["mean_macro_f1"].idxmax())
    best = cv_results[best_index]
    best_params = {
        key: best[key]
        for key in (
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "class_weight",
        )
    }

    final_pipeline = make_pipeline(best_params)
    process = psutil.Process()
    memory_before = process.memory_info().rss
    started = time.perf_counter()
    final_pipeline.fit(X, y_array)
    training_seconds = time.perf_counter() - started
    memory_after = process.memory_info().rss

    joblib.dump(final_pipeline, artifact_dir / "random_forest.joblib")
    (artifact_dir / "features.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "features": FEATURES,
                "label_column": "label",
                "group_column": "trip_id",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "model_config.json").write_text(
        json.dumps(
            {
                "model_name": "RandomForestClassifier",
                "model_version": "m2-rf-v1",
                "schema_version": SCHEMA_VERSION,
                "features": FEATURES,
                "hyperparameters": best_params,
                "random_seed": RANDOM_SEED,
                "group_cv": {
                    "folds": args.folds,
                    "splitter": "GroupKFold",
                    "group_column": "trip_id",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    holdout = evaluate_partition(
        artifact_dir, "student_test", final_pipeline, processed_root
    )
    classifier = final_pipeline.named_steps["classifier"]
    feature_importance = dict(
        sorted(
            zip(FEATURES, classifier.feature_importances_.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    pd.DataFrame(
        [
            {"feature": name, "importance": importance}
            for name, importance in feature_importance.items()
        ]
    ).to_csv(results_dir / "random_forest_feature_importance.csv", index=False)

    model_card = {
        "model_name": "RandomForestClassifier",
        "model_version": "m2-rf-v1",
        "schema_version": SCHEMA_VERSION,
        "feature_names": FEATURES,
        "hyperparameters": best_params,
        "random_seed": RANDOM_SEED,
        "cross_validation": {
            "strategy": "GroupKFold",
            "fold_count": args.folds,
            "group_column": "trip_id",
            "train_trip_count": int(groups.nunique()),
            "candidate_row_count": int(len(X)),
            "results": cv_results,
            "selected_config_index": best_index,
        },
        "training": {
            "eligible_training_rows": int(len(X)),
            "training_seconds": float(training_seconds),
            "rss_delta_bytes": int(max(0, memory_after - memory_before)),
            "platform": platform.platform(),
        },
        "held_out_student_test": holdout,
        "feature_importance": feature_importance,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_release": "m1-preprocessed-2026-11-18",
        "data_release_commit": "7f613e0",
    }
    (artifact_dir / "model_card.json").write_text(
        json.dumps(model_card, indent=2),
        encoding="utf-8",
    )
    (results_dir / "random_forest_results.json").write_text(
        json.dumps(model_card, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "artifact_dir": str(artifact_dir),
        "best_config": best_params,
        "mean_macro_f1": cv_frame.iloc[best_index]["mean_macro_f1"],
        "training_seconds": training_seconds,
        "held_out_student_test": holdout,
    }, indent=2))


if __name__ == "__main__":
    main()
