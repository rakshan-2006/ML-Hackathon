"""Shared candidate-level and observation-level metrics for all PathSnap models.

Candidate level (labelled C2 rows only, decision threshold 0.5):
  accuracy, precision/recall/F1 of the positive class ("binary") and their
  macro averages over both classes, log loss and the confusion matrix.
Observation level (Top-1), following M1's coverage policy (data/README.md):
  denominator = trusted observations = valid C1 rows whose C3 align_status is
  'ok'; the prediction is the highest-probability candidate (ties -> lowest
  cand_rank). Trusted observations whose reference is not among the candidates,
  including zero-candidate observations, count as failures.
These are candidate/point metrics, not route metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss,
                             precision_recall_fscore_support)


def positive_proba(model, X):
    """P(label == 1), located through the fitted estimator's classes_."""
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"class 1 not among fitted classes {classes}")
    return model.predict_proba(X)[:, classes.index(1)]


def candidate_metrics(y_true, p, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=np.float64)
    if y_true.shape != p.shape:
        raise ValueError("y_true and p must have the same shape")
    y_pred = (p >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    mprec, mrec, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_labelled": int(len(y_true)),
        "candidate_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_pos": float(prec), "recall_pos": float(rec), "f1_pos": float(f1),
        "precision_macro": float(mprec), "recall_macro": float(mrec), "f1_macro": float(mf1),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def observation_top1(scored, observations, alignment):
    """Top-1 accuracy over trusted observations.

    scored      : ALL candidate rows of the evaluated trips with columns
                  trip_id, obs_idx, arc_id, cand_rank, p_model
    observations: C1 rows (trip_id, obs_idx, is_valid) of the same trips
    alignment   : C3 rows (trip_id, obs_idx, align_status, ref_arc_id)
    """
    key = ["trip_id", "obs_idx"]
    obs = observations[key + ["is_valid"]].merge(
        alignment[key + ["align_status", "ref_arc_id"]], on=key, how="left", validate="1:1")
    trusted = obs[obs.is_valid & (obs.align_status == "ok")]

    ranked = scored.sort_values(key + ["p_model", "cand_rank"],
                                ascending=[True, True, False, True], kind="mergesort")
    best = ranked.drop_duplicates(key)[key + ["arc_id"]]
    judged = trusted.merge(best, on=key, how="left", validate="1:1")
    correct = judged.arc_id.notna() & (judged.arc_id == judged.ref_arc_id)
    in_cands = trusted.merge(scored[key + ["arc_id"]].rename(columns={"arc_id": "ref_arc_id"}),
                             on=key + ["ref_arc_id"], how="inner")
    n = len(trusted)
    return {
        "n_trusted_obs": int(n),
        "n_top1_correct": int(correct.sum()),
        "top1_accuracy": float(correct.sum() / n) if n else float("nan"),
        "n_reference_misses": int(n - len(in_cands)),
        "n_zero_candidate_trusted": int(judged.arc_id.isna().sum()),
        "top1_upper_bound": float(len(in_cands) / n) if n else float("nan"),
    }
