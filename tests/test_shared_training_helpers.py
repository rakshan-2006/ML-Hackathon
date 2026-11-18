import numpy as np
import pandas as pd
import pytest
import json
from src.training.cross_validation import get_cv_splits, get_stratified_cv_splits, load_folds
from src.training.preprocessing import load_training_data
from src.common.schema import FEATURES, SCHEMA_VERSION


def test_canonical_split_membership_is_unchanged():
    folds = load_folds()
    groups = np.repeat(folds.trip_id.to_numpy(), 2)
    X, y = np.zeros((len(groups), 1)), np.tile([0, 1], len(folds))
    splits = get_cv_splits(X, y, groups)
    for k, (tr, va) in enumerate(splits):
        assert set(groups[va]) == set(folds.loc[folds.fold == k, 'trip_id'])
        assert not set(groups[tr]) & set(groups[va])
    assert sorted(np.concatenate([va for _, va in splits])) == list(range(len(groups)))
    with pytest.raises(ValueError, match='five folds'):
        get_cv_splits(X, y, groups, n_splits=3)


def test_stratification_is_explicit_deterministic_and_accepts_numpy_groups():
    groups = np.repeat(np.arange(10), 10)
    X, y = np.zeros((100, 1)), np.tile([0, 1], 50)
    a, b = get_stratified_cv_splits(X, y, groups), get_stratified_cv_splits(X, y, groups)
    for (tr, va), (tr2, va2) in zip(a, b):
        assert np.array_equal(tr, tr2) and np.array_equal(va, va2)
        assert not set(groups[tr]) & set(groups[va])
    with pytest.raises(ValueError, match='equal lengths'):
        get_stratified_cv_splits(X, y[:-1], groups)
    with pytest.raises(ValueError, match='not be null'):
        get_stratified_cv_splits(X, y, [None] * 100)


def test_loader_uses_requested_root_and_excludes_uncertain_labels(tmp_path):
    rows = pd.DataFrame({name: [1., 2., 3.] for name in FEATURES})
    rows['trip_id'], rows['obs_idx'], rows['arc_id'] = 't', [0, 1, 2], [0, 1, 2]
    rows['label'] = pd.array([1, 0, None], dtype='Int8')
    rows['cand_rank'] = 0
    (tmp_path / 'manifest.json').write_text(json.dumps({'schema_version': SCHEMA_VERSION}))
    rows.to_parquet(tmp_path / 'train_candidates.parquet')
    X, y, groups = load_training_data(tmp_path)
    assert list(X.columns) == FEATURES and len(X) == 2
    assert y.tolist() == [1, 0] and groups.tolist() == ['t', 't']
