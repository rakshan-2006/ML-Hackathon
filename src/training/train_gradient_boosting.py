"""Train full-data gradient boosting with trip-grouped OOF and held-out evaluation.

Run from repository root: python -m src.training.train_gradient_boosting
"""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss,
                             precision_recall_fscore_support)
from threadpoolctl import threadpool_limits

from src.common.schema import FEATURES, SCHEMA_VERSION
from src.models.gradient_boosting import build_model, default_params
from src.training.cross_validation import assign_folds, load_folds

MODEL_NAME = 'gradient_boosting'
KEYS = ['trip_id', 'obs_idx', 'arc_id']


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf8')


def positive_probability(model, X, batch_size=100000):
    """Preserve input order, with an explicit class mapping and bounded batches."""
    classes = np.asarray(model.classes_)
    column = np.flatnonzero(classes == 1)
    if len(column) != 1:
        raise ValueError('Estimator must contain exactly one positive class 1')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    scores = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        scores[start:stop] = model.predict_proba(X.iloc[start:stop])[:, column[0]]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid probabilities')
    return scores


def make_folds(groups, path, seed=2026):
    """Persist one deterministic trip assignment for reuse by all five models.

    GroupKFold operates on labelled candidate rows (the actual fitting dataset).
    A pre-existing assignment is used verbatim after coverage validation.
    """
    path = Path(path)
    trips = sorted(groups.unique().tolist())
    if len(trips) < 5:
        raise ValueError('Five-fold CV needs at least five distinct trips')
    if path.exists():
        if path.suffix == '.csv':
            table = load_folds(path)
            spec = dict(schema_version=SCHEMA_VERSION,
                        trip_to_fold=dict(zip(table.trip_id.astype(str), table.fold.astype(int))))
        else:
            # Accept earlier downloaded JSON fold maps explicitly supplied by users.
            spec = json.loads(path.read_text(encoding='utf8'))
        mapping = spec['trip_to_fold']
        if set(mapping) != set(trips) or set(mapping.values()) != set(range(5)):
            raise ValueError('Saved fold assignment does not cover these train trips')
    else:
        table = assign_folds(groups)
        mapping = dict(zip(table.trip_id.astype(str), table.fold.astype(int)))
        spec = dict(scheme='GroupKFold', n_splits=5, shuffle=False,
                    seed=seed, seed_note='Recorded; nonshuffled GroupKFold uses no RNG',
                    group_column='trip_id', partition='train',
                    grouping_rows='candidates with non-null label',
                    schema_version=SCHEMA_VERSION, trip_to_fold=dict(sorted(mapping.items())))
        if path.suffix == '.csv':
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False)
        else:
            write_json(path, spec)
    if spec.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Fold/schema version mismatch')
    assignment = groups.map(mapping).to_numpy(dtype=int)
    for fold in range(5):
        train_trips = set(groups[assignment != fold])
        valid_trips = set(groups[assignment == fold])
        if not valid_trips or train_trips & valid_trips:
            raise ValueError('Invalid or leaking group split')
    return mapping, assignment, hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(candidates, scores, observations, alignment):
    """Candidate metrics on trusted labels; Top-1 on ALL trusted valid GPS rows.

    Argmax is over every candidate, including candidates with nullable targets.
    Reference misses and zero-candidate observations count as Top-1 failures.
    """
    if len(scores) != len(candidates) or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and aligned with candidates')
    labelled = candidates.label.notna().to_numpy()
    y = candidates.loc[labelled, 'label'].to_numpy(dtype=np.int8)
    prob = np.asarray(scores)[labelled]
    if not len(y):
        raise ValueError('No labelled candidates for evaluation')
    pred = (prob >= 0.5).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=[0, 1], average='macro', zero_division=0)
    p1, r1, f1_positive, _ = precision_recall_fscore_support(
        y, pred, average='binary', zero_division=0)
    ranked = candidates[KEYS + ['cand_rank']].copy()
    ranked['p_model'] = scores
    best = ranked.sort_values(
        ['trip_id', 'obs_idx', 'p_model', 'cand_rank'],
        ascending=[True, True, False, True], kind='stable').drop_duplicates(['trip_id', 'obs_idx'])
    eligible = alignment.merge(observations[['trip_id', 'obs_idx', 'is_valid']],
                               on=['trip_id', 'obs_idx'], validate='one_to_one')
    eligible = eligible.loc[(eligible.align_status == 'ok') & eligible.is_valid].copy()
    eligible = eligible.merge(best[KEYS], on=['trip_id', 'obs_idx'], how='left',
                               validate='one_to_one')
    eligible['correct'] = eligible.arc_id.eq(eligible.ref_arc_id)
    per_trip = eligible.groupby('trip_id', as_index=False).agg(
        top1_correct=('correct', 'sum'), top1_denominator=('correct', 'size'))
    per_trip['top1_accuracy'] = per_trip.top1_correct / per_trip.top1_denominator
    top1_n = len(eligible)
    result = dict(labelled_candidates=len(y), positive_candidates=int(y.sum()),
                  candidate_accuracy=float(accuracy_score(y, pred)),
                  precision_macro=float(precision), recall_macro=float(recall),
                  macro_f1=float(f1), precision_positive=float(p1),
                  recall_positive=float(r1), f1_positive=float(f1_positive),
                  log_loss=float(log_loss(y, prob, labels=[0, 1])),
                  confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                  top1_correct=int(eligible.correct.sum()), top1_denominator=top1_n,
                  top1_accuracy=float(eligible.correct.sum()/top1_n) if top1_n else None)
    return result, per_trip


def load_partition(root, partition):
    return tuple(pd.read_parquet(root/f'{partition}_{kind}.parquet')
                 for kind in ['candidates', 'observations', 'alignment'])


def score_export(candidates, scores, version, fold_ids=None):
    out = candidates[KEYS].copy()
    out['p_model'] = scores
    out['model_name'] = MODEL_NAME
    out['model_version'] = version
    if fold_ids is not None:
        out['fold'] = fold_ids
    return out


def run(args):
    started = perf_counter()
    root, results, artifacts = map(Path, [args.processed_root, args.results_root, args.artifacts_root])
    if (results/'model_card.json').exists() and not args.overwrite:
        raise FileExistsError('Completed run already exists; specify --overwrite')
    results.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    (results/'model_card.json').unlink(missing_ok=True)
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Dataset/schema mismatch')
    train, obs, align = load_partition(root, 'train')
    trusted = train.label.notna().to_numpy()
    X = train.loc[trusted, FEATURES].reset_index(drop=True)
    y = train.loc[trusted, 'label'].astype('int8').reset_index(drop=True)
    groups = train.loc[trusted, 'trip_id'].reset_index(drop=True)
    mapping, assignment, fold_hash = make_folds(groups, args.folds)
    all_assignment = train.trip_id.map(mapping).to_numpy(dtype=int)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    version = f'hgb-baseline-{revision[:8]}'
    config = dict(model_name=MODEL_NAME, estimator='HistGradientBoostingClassifier',
                  params=default_params(), seed=args.seed, threads=args.threads,
                  prediction_batch_size=args.batch_size, folds=str(args.folds),
                  fold_assignment_sha256=fold_hash, features=FEATURES,
                  preprocessing='Native NaN handling; no scaling or fitted imputation',
                  class_imbalance='Natural prevalence, class_weight=None; report macro and positive-class metrics',
                  early_stopping='Disabled; no random row-level holdout',
                  code_revision=revision, code_dirty=dirty,
                  dataset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    write_json(results/'config.json', config)
    print(f'Loaded {len(X):,} labelled training rows, {groups.nunique()} trips, {len(FEATURES)} features', flush=True)
    # Cheap wiring check before the full folds; never substitutes for full-data fit.
    smoke_idx = np.concatenate([np.flatnonzero(y.to_numpy() == c)[:100] for c in [0, 1]])
    smoke = build_model({'max_iter': 2}, args.seed)
    smoke.fit(X.iloc[smoke_idx], y.iloc[smoke_idx])
    positive_probability(smoke, X.iloc[smoke_idx], args.batch_size)
    oof = np.full(len(train), np.nan)
    fold_metrics, timing_rows = [], []
    for fold in range(5):
        fit_rows, valid_rows = assignment != fold, all_assignment == fold
        model = build_model(seed=args.seed)
        print(f'Fold {fold+1}/5: fitting {int(fit_rows.sum()):,} labelled rows', flush=True)
        t = perf_counter()
        model.fit(X.loc[fit_rows], y.loc[fit_rows])
        fit_s = perf_counter()-t
        t = perf_counter()
        oof[valid_rows] = positive_probability(model, train.loc[valid_rows, FEATURES], args.batch_size)
        predict_s = perf_counter()-t
        valid_trips = set(train.loc[valid_rows, 'trip_id'])
        metrics, _ = evaluate(train.loc[valid_rows], oof[valid_rows],
                              obs[obs.trip_id.isin(valid_trips)], align[align.trip_id.isin(valid_trips)])
        fold_metrics.append(dict(fold=fold, train_rows=int(fit_rows.sum()),
                                 validation_trips=len(valid_trips), fit_s=fit_s,
                                 predict_s=predict_s, **metrics))
        timing_rows.append(dict(stage=f'fold_{fold}', fit_s=fit_s, predict_s=predict_s))
        pd.DataFrame(fold_metrics).to_csv(results/'cv_metrics.csv', index=False)
        print(f'Fold {fold+1}: macro-F1={metrics["macro_f1"]:.4f}, Top-1={metrics["top1_accuracy"]:.4f}, fit={fit_s:.1f}s', flush=True)
    if not np.isfinite(oof).all():
        raise ValueError('OOF coverage incomplete')
    oof_metrics, oof_trips = evaluate(train, oof, obs, align)
    oof_trips.to_csv(results/'train_oof_per_trip.csv', index=False)
    score_export(train, oof, version, all_assignment).to_parquet(artifacts/'scores_train_oof.parquet', index=False)
    print('Refitting final model on every labelled train row', flush=True)
    final = build_model(seed=args.seed)
    t = perf_counter()
    final.fit(X, y)
    final_fit_s = perf_counter()-t
    joblib.dump(final, artifacts/'gradient_boosting_pipeline.joblib', compress=3)
    reloaded = joblib.load(artifacts/'gradient_boosting_pipeline.joblib')
    check = X.iloc[np.linspace(0, len(X)-1, min(2048, len(X)), dtype=int)]
    if not np.array_equal(positive_probability(final, check), positive_probability(reloaded, check)):
        raise ValueError('Artifact reload prediction mismatch')
    test, test_obs, test_align = load_partition(root, 'student_test')
    t = perf_counter()
    test_scores = positive_probability(reloaded, test[FEATURES], args.batch_size)
    test_predict_s = perf_counter()-t
    test_metrics, test_trips = evaluate(test, test_scores, test_obs, test_align)
    pd.DataFrame([test_metrics]).to_csv(results/'student_test_metrics.csv', index=False)
    test_trips.to_csv(results/'student_test_per_trip.csv', index=False)
    score_export(test, test_scores, version).to_parquet(artifacts/'scores_student_test.parquet', index=False)
    # OOF scores include ALL candidate rows; fitting still excludes nullable targets.
    timing_rows.append(dict(stage='final', fit_s=final_fit_s, predict_s=test_predict_s))
    versions = dict(python=platform.python_version(), sklearn=sklearn.__version__,
                    numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__)
    cv_mean = {name: float(np.mean([r[name] for r in fold_metrics]))
               for name in ['macro_f1', 'log_loss', 'top1_accuracy']}
    elapsed = perf_counter()-started
    write_json(results/'timings.json', dict(stages=timing_rows, total_s=elapsed, threads=args.threads))
    card = dict(**config, model_version=version, schema_version=SCHEMA_VERSION,
                dataset_release='m1-preprocessed-2026-11-18', versions=versions,
                train_trips=int(groups.nunique()), train_rows=len(X),
                train_all_candidate_rows=len(train), student_test_all_candidate_rows=len(test),
                cv_fold_mean=cv_mean, train_oof=oof_metrics, student_test=test_metrics,
                artifact='gradient_boosting_pipeline.joblib', artifact_reload_identical=True,
                feature_contract='Whole-trip; next_dist_m uses next GPS observation',
                limitations=['Derived alignment is approximate; ambiguous/unaligned labels excluded',
                             'Baseline only; no Bayesian optimisation or calibration',
                             'No independent route/HMM continuity or sequence metrics',
                             'CV fold mapping newly published; teammates must reuse it for comparisons'],
                total_s=elapsed)
    write_json(results/'model_card.json', card)
    write_json(artifacts/'model_card.json', card)
    write_json(artifacts/'schema.json', dict(schema_version=SCHEMA_VERSION, features=FEATURES))
    write_json(artifacts/'cv_folds.json', dict(schema_version=SCHEMA_VERSION,
               source_file=str(args.folds), source_sha256=fold_hash, trip_to_fold=mapping))
    hashes = {p.name: dict(size_bytes=p.stat().st_size,
                           sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in artifacts.iterdir() if p.is_file() and p.name != 'artifact_manifest.json'}
    write_json(artifacts/'artifact_manifest.json', hashes)
    print(json.dumps(dict(status='complete', total_s=elapsed, cv_fold_mean=cv_mean,
                          train_oof=oof_metrics, student_test=test_metrics), indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-root', default='data/processed')
    p.add_argument('--results-root', default='results/models/gradient_boosting')
    p.add_argument('--artifacts-root', default='artifacts/gradient_boosting')
    p.add_argument('--folds', default='configs/cv_folds.csv')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=100000)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        p.error('threads and batch-size must be positive')
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == '__main__':
    main()
