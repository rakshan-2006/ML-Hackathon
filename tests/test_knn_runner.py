"""Verify exact shortcut against normal queries, including conflicting duplicates."""
import numpy as np
import pandas as pd
import pytest
from src.common.schema import FEATURES
from src.models.model_factory import get_model
from src.evaluation.metrics import positive_proba
from src.training.train_knn import batched_proba


@pytest.mark.parametrize('weights', ['distance', 'uniform'])
@pytest.mark.parametrize('mixed', [False, True])
def test_training_diagnostics_match_normal_queries(weights, mixed):
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(40, len(FEATURES))), columns=FEATURES)
    X.iloc[1:8] = X.iloc[0].to_numpy()
    X.loc[::9, 'heading_diff_deg'] = np.nan
    y = np.arange(40) % 2
    y[:8] = np.arange(8) % 2 if mixed else 1
    model = get_model('knn', {'n_neighbors':3, 'weights':weights, 'n_jobs':1}).fit(X,y)
    assert np.allclose(batched_proba(model,X),positive_proba(model,X),rtol=0,atol=1e-12)
    held = X.iloc[[20,10,30]].copy()
    assert np.allclose(batched_proba(model,held),positive_proba(model,held),rtol=0,atol=1e-12)


def test_runner_restores_shared_predictor(monkeypatch):
    from src.training import train_knn, train_models
    original = train_models.positive_proba
    monkeypatch.setattr(train_models, 'run', lambda *a, **kw: None)
    train_knn.main(['--smoke-only','--jobs','1'])
    assert train_models.positive_proba is original
