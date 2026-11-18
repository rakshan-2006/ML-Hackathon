"""Validate metric denominators, class mapping, grouped folds and persistence."""
import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.models.gradient_boosting import build_model
from src.training.train_gradient_boosting import evaluate, make_folds, positive_probability


def test_top1_counts_miss_and_zero_candidates_and_breaks_ties_by_rank():
    cand = pd.DataFrame(dict(trip_id=['00000001']*4, obs_idx=[0, 0, 1, 3],
                             arc_id=[8, 7, 5, 1], cand_rank=[1, 0, 0, 0],
                             label=pd.array([0, 1, 0, pd.NA], dtype='Int8')))
    obs = pd.DataFrame(dict(trip_id=['00000001']*4, obs_idx=range(4), is_valid=True))
    align = pd.DataFrame(dict(trip_id=['00000001']*4, obs_idx=range(4),
                              ref_arc_id=[7, 6, 9, 1], align_status=['ok', 'ok', 'ok', 'ambiguous']))
    metrics, _ = evaluate(cand, np.array([0.7, 0.7, 0.9, 0.8]), obs, align)
    assert metrics['top1_denominator'] == 3
    assert metrics['top1_correct'] == 1
    assert metrics['labelled_candidates'] == 3
    assert metrics['top1_accuracy'] == pytest.approx(1/3)


def test_probability_mapping_does_not_assume_positive_class_column():
    class ReversedClasses:
        classes_ = np.array([1, 0])
        def predict_proba(self, X):
            return np.column_stack([X['p'], 1-X['p']])
    X = pd.DataFrame({'p': [0.2, 0.8, 0.5]})
    assert np.array_equal(positive_probability(ReversedClasses(), X, 2), X.p)
    assert positive_probability(ReversedClasses(), X.iloc[:0]).shape == (0,)


def test_folds_persist_and_keep_trips_together(tmp_path):
    groups = pd.Series(np.repeat([f'{i:08d}' for i in range(10)], 3))
    mapping, assignments, digest = make_folds(groups, tmp_path/'folds.json')
    for fold in range(5):
        assert not set(groups[assignments == fold]) & set(groups[assignments != fold])
    assert make_folds(groups, tmp_path/'folds.json')[2] == digest
    with pytest.raises(ValueError, match='cover'):
        make_folds(groups.iloc[3:], tmp_path/'folds.json')


def test_pipeline_handles_nans_and_reloads_identically(tmp_path):
    X = pd.DataFrame({'distance': [0., 1., np.nan, 2., 8., 9., 10., np.nan]*10})
    y = np.tile([1, 1, 1, 1, 0, 0, 0, 0], 10)
    with threadpool_limits(limits=2):
        model = build_model({'max_iter': 5, 'min_samples_leaf': 2})
        model.fit(X, y)
        original = positive_probability(model, X)
        joblib.dump(model, tmp_path/'pipeline.joblib')
        assert np.array_equal(original, positive_probability(joblib.load(tmp_path/'pipeline.joblib'), X))
        shuffled = X.sample(frac=1, random_state=42)
        assert np.array_equal(original[shuffled.index], positive_probability(model, shuffled))
