"""Train and evaluate one model with the shared folds, label policy and metrics.

    python -m src.training.train_models --model logistic_regression
    python -m src.training.train_models --model decision_tree --smoke-only

Label policy (M1): models are fitted only on trusted labelled rows (C2 label not
null). Probabilities are produced for ALL candidate rows: out-of-fold for the 80
train trips (each trip scored by the fold model that never saw it) and from the
refitted model for student_test. student_test is never used for fitting.
"""

import argparse
import hashlib
import json
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.common.paths import PROJECT_ROOT
from src.common.schema import FEATURES, SCHEMA_VERSION
from src.data.model_data import load_model_data
from src.evaluation.metrics import candidate_metrics, observation_top1, positive_proba
from src.models.model_factory import get_model, model_module, model_params
from src.training.cross_validation import (FOLDS_PATH, N_SPLITS, fold_of_rows,
                                           folds_sha256, load_folds)
from src.training.preprocessing import PREPROCESSING_POLICY

DATA = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results" / "models"
ARTIFACTS = PROJECT_ROOT / "artifacts"
SEED = 2026
THRESHOLD = 0.5
DATASET_RELEASE = "m1-preprocessed-2026-11-18"
KEY = ["trip_id", "obs_idx", "arc_id"]
PROB_COLUMNS = KEY + ["p_model", "model_name", "model_version"]


def load_partition(partition):
    X, y, groups, _ = load_model_data(DATA, partition)
    cand = pd.read_parquet(DATA / f"{partition}_candidates.parquet")
    labelled = cand.label.notna().to_numpy()
    # The shared loader returns exactly the labelled C2 rows, in C2 order.
    pd.testing.assert_frame_equal(cand.loc[labelled, FEATURES].reset_index(drop=True), X)
    obs = pd.read_parquet(DATA / f"{partition}_observations.parquet",
                          columns=["trip_id", "obs_idx", "is_valid"])
    align = pd.read_parquet(DATA / f"{partition}_alignment.parquet",
                            columns=["trip_id", "obs_idx", "align_status", "ref_arc_id"])
    return {"cand": cand, "labelled": labelled, "y": y.to_numpy(dtype=int),
            "obs": obs, "align": align, "n_trips": int(groups.nunique())}


def top1(cand, p, mask, obs, align, trip_ids=None):
    scored = cand.loc[mask, ["trip_id", "obs_idx", "arc_id", "cand_rank"]].assign(p_model=p)
    # Select evaluated trips explicitly, never infer them from available candidates:
    # an entirely zero-candidate trip still contributes trusted failures.
    if trip_ids is not None:
        obs = obs[obs.trip_id.isin(trip_ids)]
        align = align[align.trip_id.isin(trip_ids)]
    return observation_top1(scored, obs, align)


def check_probabilities(p, n):
    if p.shape != (n,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be finite, in [0, 1], one per row")


def smoke_test(name, params, data, folds_all):
    rng = np.random.default_rng(SEED)
    cand, lab = data["cand"], data["labelled"]
    train_rows = np.flatnonzero(lab & (folds_all != 0))
    val_rows = np.flatnonzero(lab & (folds_all == 0))
    tr = rng.choice(train_rows, 20000, replace=False)
    va = rng.choice(val_rows, 5000, replace=False)
    model = get_model(name, params, SEED)
    started = time.perf_counter()
    model.fit(cand.loc[tr, FEATURES], cand.label.iloc[tr].to_numpy(dtype=int))
    p = positive_proba(model, cand.loc[va, FEATURES])
    check_probabilities(p, len(va))
    return {"train_rows": 20000, "val_rows": 5000, "seconds": round(time.perf_counter() - started, 3)}


def run(name, params, smoke_only=False):
    module = model_module(name)
    params = model_params(name, params)
    config_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    version = f"{name}-v1-{config_hash}"
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    artifact_out = ARTIFACTS / name
    artifact_out.mkdir(parents=True, exist_ok=True)

    train, test = load_partition("train"), load_partition("student_test")
    folds = load_folds()
    cand, lab, y_lab = train["cand"], train["labelled"], train["y"]
    fold_all = fold_of_rows(cand.trip_id, folds)
    if train["n_trips"] != len(folds):
        raise ValueError("fold table does not cover exactly the labelled train trips")

    smoke = smoke_test(name, params, train, fold_all)
    print(f"[{name}] smoke test ok: {smoke}", flush=True)
    if smoke_only:
        return

    y_all = np.full(len(cand), -1)
    y_all[lab] = y_lab
    oof = np.full(len(cand), np.nan)
    rows, timings = [], {"smoke_test": smoke, "folds": []}
    for k in range(N_SPLITS):
        tr = lab & (fold_all != k)
        va_all = fold_all == k
        overlap = set(cand.trip_id[tr]) & set(cand.trip_id[va_all])
        if overlap:
            raise AssertionError(f"fold {k}: trips in both train and validation: {overlap}")
        model = get_model(name, params, SEED)
        X_tr = cand.loc[tr, FEATURES]
        print(f"[{name}] fold {k}: fitting {len(X_tr):,} labelled rows", flush=True)
        started = time.perf_counter()
        model.fit(X_tr, y_all[tr])
        fit_s = time.perf_counter() - started

        # Imputer statistics must come from the training fold only.
        if "impute" in model.named_steps:
            fold_medians = np.nanmedian(X_tr.to_numpy(dtype=np.float64), axis=0)
            if not np.allclose(model.named_steps["impute"].statistics_, fold_medians, equal_nan=True):
                raise AssertionError(f"fold {k}: imputer not fitted on the training fold")
        elif getattr(module, 'IMPUTE_MISSING', True):
            raise AssertionError('Model must declare how it handles missing features')

        p_tr = positive_proba(model, X_tr)
        started = time.perf_counter()
        p_va = positive_proba(model, cand.loc[va_all, FEATURES])
        predict_s = time.perf_counter() - started
        check_probabilities(p_va, int(va_all.sum()))
        oof[va_all] = p_va

        va_lab = va_all & lab
        val = {**candidate_metrics(y_all[va_lab], oof[va_lab], THRESHOLD),
               **top1(cand, p_va, va_all, train["obs"], train["align"],
                      trip_ids=folds.loc[folds.fold == k, "trip_id"])}
        trn = candidate_metrics(y_all[tr], p_tr, THRESHOLD)
        n_val_trips = int(cand.loc[va_all, "trip_id"].nunique())
        rows.append({"fold": k, "split": "train", "n_trips": int(cand.loc[tr, "trip_id"].nunique()), **trn})
        rows.append({"fold": k, "split": "validation", "n_trips": n_val_trips, **val})
        timings["folds"].append({"fold": k, "fit_s": round(fit_s, 3), "predict_s": round(predict_s, 3),
                                 "fit_rows": int(tr.sum()), "predict_rows": int(va_all.sum())})
        print(f"[{name}] fold {k}: val acc={val['candidate_accuracy']:.4f} "
              f"macroF1={val['f1_macro']:.4f} logloss={val['log_loss']:.4f} "
              f"top1={val['top1_accuracy']:.4f} fit={fit_s:.1f}s", flush=True)
    check_probabilities(oof, len(cand))

    cv = pd.DataFrame(rows)
    val_rows = cv[cv.split == "validation"].drop(columns=["fold", "split"])
    summary = [{"fold": stat, "split": "validation", **val_rows.agg(stat).to_dict()}
               for stat in ("mean", "std")]
    cv = pd.concat([cv, pd.DataFrame(summary)], ignore_index=True)
    cv.to_csv(out / "cv_metrics.csv", index=False)

    # Refit on all labelled training rows; evaluate once on student_test.
    final = get_model(name, params, SEED)
    print(f"[{name}] refitting on all {len(y_lab):,} labelled training rows", flush=True)
    started = time.perf_counter()
    final.fit(cand.loc[lab, FEATURES], y_lab)
    refit_s = time.perf_counter() - started
    tcand, tlab = test["cand"], test["labelled"]
    started = time.perf_counter()
    p_test = positive_proba(final, tcand[FEATURES])
    test_predict_s = time.perf_counter() - started
    check_probabilities(p_test, len(tcand))
    test_metrics = {"n_trips": test["n_trips"],
                    **candidate_metrics(test["y"], p_test[tlab], THRESHOLD),
                    **top1(tcand, p_test, np.ones(len(tcand), bool), test["obs"], test["align"])}
    pd.DataFrame([test_metrics]).to_csv(out / "student_test_metrics.csv", index=False)

    joblib.dump(final, artifact_out / "pipeline.joblib")
    reloaded = joblib.load(artifact_out / "pipeline.joblib")
    # Parallel tree probability reduction can change the last floating-point bits.
    if not np.allclose(positive_proba(reloaded, tcand[FEATURES]), p_test,
                       rtol=0, atol=1e-12):
        raise AssertionError("reloaded pipeline does not reproduce student_test predictions")

    for filename, frame, p in (("oof_probabilities.parquet", cand, oof),
                               ("student_test_probabilities.parquet", tcand, p_test)):
        frame[KEY].assign(p_model=p, model_name=name, model_version=version)[PROB_COLUMNS] \
            .to_parquet(artifact_out / filename, index=False)

    verify_saved_metrics(out, train, test, cv, test_metrics, artifact_out)

    timings.update({"cv_total_fit_s": round(sum(f["fit_s"] for f in timings["folds"]), 3),
                    "refit_s": round(refit_s, 3), "student_test_predict_s": round(test_predict_s, 3),
                    "student_test_predict_rows": int(len(tcand)),
                    "student_test_predict_us_per_row": round(1e6 * test_predict_s / len(tcand), 3)})
    (out / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")

    config = {"model_name": name, "model_version": version, "family": module.FAMILY,
              "estimator": type(final.named_steps["model"]).__name__, "params": params,
              "seed": SEED, "decision_threshold": THRESHOLD,
              "class_imbalance": module.CLASS_IMBALANCE,
              "preprocessing": ({"missing_values": "Native estimator handling; no fitted imputer",
                                  "scaled": False, "feature_order": "src.common.schema.FEATURES"}
                                if not getattr(module, 'IMPUTE_MISSING', True)
                                else {**PREPROCESSING_POLICY, "scaled": module.SCALE_FEATURES}),
              "folds_file": str(FOLDS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
              "folds_sha256": folds_sha256(), "n_splits": N_SPLITS,
              "command": f"python -m src.training.train_models --model {name}"}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    cv_mean = cv[(cv.fold == "mean")].iloc[0]
    manifest = json.loads((DATA / "manifest.json").read_text())
    card = {
        **{k: config[k] for k in ("model_name", "model_version", "family", "estimator",
                                  "params", "seed", "decision_threshold", "class_imbalance",
                                  "preprocessing", "folds_file", "folds_sha256", "n_splits")},
        "schema_version": SCHEMA_VERSION,
        "ordered_features": FEATURES,
        "dataset_release": DATASET_RELEASE,
        "dataset_manifest_sha256": hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
        "dataset_schema_version": manifest["schema_version"],
        "label_policy": "Fit on trusted labelled C2 rows only (label not null; ambiguous/unaligned "
                        "observations excluded). Scores produced for all candidate rows.",
        "row_counts": {"train_labelled": int(lab.sum()), "train_candidates": int(len(cand)),
                       "train_trips": train["n_trips"], "student_test_labelled": int(tlab.sum()),
                       "student_test_candidates": int(len(tcand)), "student_test_trips": test["n_trips"]},
        "probability_coverage": {"oof_rows_scored": int(np.isfinite(oof).sum()),
                                 "student_test_rows_scored": int(len(p_test))},
        "results": {"cv_mean": {k: float(cv_mean[k]) for k in
                                ("candidate_accuracy", "precision_pos", "recall_pos", "f1_macro",
                                 "log_loss", "top1_accuracy")},
                    "student_test": {k: float(test_metrics[k]) for k in
                                     ("candidate_accuracy", "precision_pos", "recall_pos", "f1_macro",
                                      "log_loss", "top1_accuracy")}},
        "metric_definitions": "src/evaluation/metrics.py (candidate metrics at threshold 0.5; "
                              "precision/recall *_pos = positive class, f1_macro = unweighted "
                              "mean over both classes; Top-1 over trusted observations)",
        "library_versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__},
        "limitations": module.LIMITATIONS + [
            "Labels come from M1's derived reference alignment, not verified ground truth.",
            "Candidate/observation metrics only; no HMM or route-level result.",
        ],
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    from src.evaluation.visualizations import plot_model_accuracy
    plot_model_accuracy(name)
    print(f"[{name}] done -> {out}", flush=True)


def verify_saved_metrics(out, train, test, cv, test_metrics, artifact_dir=None):
    """Recompute the student-test and fold metrics from the saved probability files."""
    for filename, data, expected in (("student_test_probabilities.parquet", test, [test_metrics]),
                                     ("oof_probabilities.parquet", train, None)):
        saved = pd.read_parquet((artifact_dir or out) / filename)
        cand = data["cand"]
        if not saved[KEY].reset_index(drop=True).equals(cand[KEY].reset_index(drop=True)):
            raise AssertionError(f"{filename}: keys/order differ from C2")
        p = saved.p_model.to_numpy()
        if expected is not None:
            got = candidate_metrics(data["y"], p[data["labelled"]], THRESHOLD)
            got.update(top1(cand, p, np.ones(len(cand), bool), data["obs"], data["align"]))
        else:
            folds = fold_of_rows(cand.trip_id, load_folds())
            mask = (folds == 0)
            y = np.full(len(cand), -1)
            y[data["labelled"]] = data["y"]
            got = candidate_metrics(y[mask & data["labelled"]], p[mask & data["labelled"]], THRESHOLD)
            got.update(top1(cand, p[mask], mask, data["obs"], data["align"],
                            trip_ids=load_folds().query('fold == 0').trip_id))
            expected = [cv[(cv.fold == 0) & (cv.split == "validation")].iloc[0].to_dict()]
        for key, value in got.items():
            if not np.isclose(value, float(expected[0][key]), equal_nan=True):
                raise AssertionError(f"{filename}: saved {key}={expected[0][key]} vs recomputed {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--params", default="{}", help="JSON overrides of DEFAULT_PARAMS")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args(argv)
    run(args.model, json.loads(args.params), smoke_only=args.smoke_only)


if __name__ == "__main__":
    sys.exit(main())
