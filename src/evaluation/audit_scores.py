"""Independently re-verify a trained model's downloaded artifacts and saved metrics.

    python -m src.evaluation.audit_scores random_forest
    python -m src.evaluation.audit_scores logistic_regression decision_tree gradient_boosting

For each model this checks, without retraining: the OOF/student-test score
exports cover every C2 candidate in source order with unique keys and finite
float64 values in [0, 1]; every saved fold-validation row and the student-test
row are reproduced by the shared metrics from those exports with the canonical
fold mapping; and the saved pipeline re-scores the student_test candidates.
Writes results/validation/<model>_audit.json.
"""

import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT
from src.common.schema import FEATURES
from src.evaluation.metrics import positive_proba
from src.evaluation.visualizations import load_metrics
from src.training.cross_validation import N_SPLITS, fold_of_rows, folds_sha256, load_folds
from src.training.train_models import (ARTIFACTS, DATA, KEY, THRESHOLD, candidate_metrics,
                                       load_partition, top1)

OUT = PROJECT_ROOT / "results" / "validation"
# Score export file names differ between the shared runner and the original GB release.
EXPORTS = {
    "gradient_boosting": ("scores_train_oof.parquet", "scores_student_test.parquet",
                          "gradient_boosting_pipeline.joblib"),
}
DEFAULT_EXPORTS = ("oof_probabilities.parquet", "student_test_probabilities.parquet",
                   "pipeline.joblib")
COMPARED = ["n_labelled", "candidate_accuracy", "precision_pos", "recall_pos", "f1_pos",
            "precision_macro", "recall_macro", "f1_macro", "log_loss", "tn", "fp", "fn", "tp",
            "n_trusted_obs", "n_top1_correct", "top1_accuracy", "n_reference_misses",
            "n_zero_candidate_trusted", "top1_upper_bound"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_export(saved, cand, name):
    if saved[KEY].duplicated().any():
        raise AssertionError(f"{name}: duplicate (trip_id, obs_idx, arc_id) keys")
    if not saved[KEY].reset_index(drop=True).equals(cand[KEY].reset_index(drop=True)):
        raise AssertionError(f"{name}: keys/order differ from C2")
    p = saved.p_model.to_numpy()
    if p.dtype != np.float64 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise AssertionError(f"{name}: scores must be finite float64 in [0, 1]")
    return p


def compare(expected, got, where):
    mismatches = {k: (float(expected[k]), float(got[k])) for k in COMPARED
                  if not np.isclose(float(expected[k]), float(got[k]), rtol=1e-9, atol=1e-12,
                                    equal_nan=True)}
    if mismatches:
        raise AssertionError(f"{where}: saved vs recomputed differ: {mismatches}")


def audit(name, train, test, folds):
    oof_file, test_file, pipe_file = EXPORTS.get(name, DEFAULT_EXPORTS)
    art = ARTIFACTS / name
    cv, test_saved = load_metrics(name)
    report = {"model": name, "artifact_sha256": {f: sha256(art / f) for f in
                                                 (oof_file, test_file, pipe_file)}}

    cand = train["cand"]
    oof = pd.read_parquet(art / oof_file)
    p = check_export(oof, cand, oof_file)
    fold_all = fold_of_rows(cand.trip_id, folds)
    if "fold" in oof.columns and not np.array_equal(oof.fold.to_numpy(), fold_all):
        raise AssertionError(f"{oof_file}: exported fold IDs differ from the canonical folds")
    y_all = np.full(len(cand), -1)
    y_all[train["labelled"]] = train["y"]
    fold_rows = []
    for k in range(N_SPLITS):
        va = fold_all == k
        lab = va & train["labelled"]
        got = {**candidate_metrics(y_all[lab], p[lab], THRESHOLD),
               **top1(cand, p[va], va, train["obs"], train["align"],
                      trip_ids=folds.loc[folds.fold == k, "trip_id"])}
        expected = cv[(cv.fold.astype(str) == str(k)) & (cv.split == "validation")].iloc[0]
        compare(expected, got, f"{name} fold {k}")
        fold_rows.append({"fold": k, "f1_macro": got["f1_macro"],
                          "top1_accuracy": got["top1_accuracy"], "log_loss": got["log_loss"]})
    pooled = candidate_metrics(y_all[train["labelled"]], p[train["labelled"]], THRESHOLD)

    tcand = test["cand"]
    st = pd.read_parquet(art / test_file)
    p_test = check_export(st, tcand, test_file)
    got = {**candidate_metrics(test["y"], p_test[test["labelled"]], THRESHOLD),
           **top1(tcand, p_test, np.ones(len(tcand), bool), test["obs"], test["align"])}
    compare(test_saved, got, f"{name} student_test")

    pipeline = joblib.load(art / pipe_file)
    rescored = positive_proba(pipeline, tcand[FEATURES])
    max_diff = float(np.abs(rescored - p_test).max())
    if max_diff > 1e-9:
        raise AssertionError(f"{name}: pipeline re-scores student_test with max diff {max_diff}")

    report.update({
        "folds_sha256": folds_sha256(),
        "dataset_manifest_sha256": hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
        "pipeline_steps": list(pipeline.named_steps),
        "estimator": type(pipeline.steps[-1][1]).__name__,
        "classes": [int(c) for c in pipeline.classes_],
        "train_oof_rows": int(len(oof)), "student_test_rows": int(len(st)),
        "keys_unique_and_in_c2_order": True,
        "scores_finite_float64_unit_interval": True,
        "all_fold_validation_rows_recomputed": True,
        "student_test_row_recomputed": True,
        "student_test_rescore_max_abs_diff": max_diff,
        "per_fold": fold_rows,
        "pooled_oof_f1_macro": pooled["f1_macro"],
        "pooled_oof_log_loss": pooled["log_loss"],
    })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+")
    args = parser.parse_args(argv)
    train, test, folds = load_partition("train"), load_partition("student_test"), load_folds()
    if set(folds.trip_id) != set(train["cand"].trip_id.astype(str)):
        raise AssertionError("canonical folds do not cover exactly the train trips")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        report = audit(name, train, test, folds)
        (OUT / f"{name}_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[{name}] verified: {report['pipeline_steps']}, "
              f"student-test rescore max diff {report['student_test_rescore_max_abs_diff']:.2e}")


if __name__ == "__main__":
    main()
