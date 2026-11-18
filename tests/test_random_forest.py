"""Random Forest factory, artifact compatibility, scoring and training regressions."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.common.schema import FEATURES, SCHEMA_VERSION
from src.evaluation.metrics import positive_proba
from src.models.model_factory import get_model
from src.models.random_forest import RandomForestPredictor
from src.models.train_random_forest import main


def synthetic_frame(rows=100):
    rng = np.random.default_rng(2026)
    X = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    X.iloc[::11, 0] = np.nan
    y = np.arange(rows) % 2
    return X, y


@pytest.fixture
def fitted():
    X, y = synthetic_frame()
    pipeline = get_model('random_forest', {'n_estimators': 5, 'max_depth': 5, 'n_jobs': 1})
    pipeline.fit(X, y)
    return X, pipeline


def test_factory_fits_fold_local_preprocessing(fitted):
    X, pipeline = fitted
    assert list(pipeline.named_steps) == ['to_float', 'impute', 'model']
    assert np.allclose(pipeline.named_steps['impute'].statistics_,
                       np.nanmedian(X.to_numpy(), axis=0))
    assert pipeline.named_steps['model'].random_state == 2026


@pytest.mark.parametrize('legacy', [False, True])
def test_saved_predictor_preserves_order_and_empty_input(tmp_path, fitted, legacy):
    X, pipeline = fitted
    joblib.dump(pipeline, tmp_path / ('random_forest.joblib' if legacy else 'pipeline.joblib'))
    card = {'schema_version': SCHEMA_VERSION}
    if legacy:
        (tmp_path / 'features.json').write_text(json.dumps({
            'schema_version': SCHEMA_VERSION, 'features': FEATURES}))
    else:
        card['ordered_features'] = FEATURES
    (tmp_path / 'model_card.json').write_text(json.dumps(card))
    predictor = RandomForestPredictor(tmp_path)
    # Shuffled observations and feature columns must preserve row correspondence.
    sample = X.iloc[[7, 2, 91]][FEATURES[::-1]]
    assert np.array_equal(predictor.score_candidates(sample), positive_proba(pipeline, sample))
    assert predictor.score_candidates(X.iloc[:0]).shape == (0,)
    with pytest.raises(ValueError, match='Missing required candidate features'):
        predictor.score_candidates(sample.drop(columns=FEATURES[0]))


def test_predictor_rejects_schema_and_feature_mismatch(tmp_path):
    card = tmp_path / 'model_card.json'
    card.write_text(json.dumps({'schema_version': 'wrong', 'ordered_features': FEATURES}))
    with pytest.raises(ValueError, match='Incompatible schema'):
        RandomForestPredictor(tmp_path)
    card.write_text(json.dumps({'schema_version': SCHEMA_VERSION,
                               'ordered_features': FEATURES[::-1]}))
    with pytest.raises(ValueError, match='feature order'):
        RandomForestPredictor(tmp_path)
    with pytest.raises(FileNotFoundError):
        RandomForestPredictor(tmp_path / 'absent')


def test_legacy_command_rejects_incomparable_three_fold_run():
    with pytest.raises(SystemExit) as exc:
        main(['--folds', '3'])
    assert exc.value.code == 2
