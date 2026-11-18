"""KNN regression tests exercise the real 13-feature model contract."""
import importlib
import joblib
import numpy as np
import pandas as pd
import pytest
from src.common.schema import FEATURES
from src.models.knn import build_knn_model
from src.models.model_factory import get_model
from src.evaluation.metrics import positive_proba


def fixture():
    rng = np.random.default_rng(2026)
    X = pd.DataFrame(rng.normal(size=(80, len(FEATURES))), columns=FEATURES)
    X.loc[::7, 'heading_diff_deg'] = np.nan
    y = (X.dist_m > 0).astype(int)
    return X, y


def test_knn_factory_scaling_reload_and_feature_order(tmp_path):
    X, y = fixture()
    model = get_model('knn', {'n_neighbors': 3}).fit(X, y)
    assert list(model.named_steps) == ['to_float', 'impute', 'scale', 'model']
    assert model.named_steps['model'].algorithm == 'kd_tree'
    assert np.allclose(model.named_steps['impute'].statistics_, np.nanmedian(X, axis=0))
    sample = X.iloc[[5, 1, 70]][FEATURES[::-1]].assign(label=[0, 0, 0])
    p = positive_proba(model, sample)
    assert np.allclose(p, positive_proba(model, X.iloc[[5, 1, 70]]))
    assert p.shape == (3,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    joblib.dump(model, tmp_path/'pipeline.joblib', compress=3)
    assert np.array_equal(positive_proba(joblib.load(tmp_path/'pipeline.joblib'), sample), p)
    with pytest.raises(ValueError, match='missing feature'):
        model.predict(X.drop(columns=FEATURES[0]))


def test_compatibility_constructor_uses_shared_pipeline():
    X, y = fixture()
    model = build_knn_model(n_neighbors=3).fit(X, y)
    assert model.named_steps['model'].n_neighbors == 3
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1)


@pytest.mark.parametrize('module', ['evaluate_knn', 'save_knn', 'benchmark_knn'])
def test_commands_are_import_safe_and_delegate(monkeypatch, module):
    from src.training import train_models
    calls = []
    monkeypatch.setattr(train_models, 'main', lambda argv: calls.append(argv))
    command = importlib.import_module('experiments.' + module)
    assert calls == []
    command.main([])
    expected = ['--model', 'knn']
    if module == 'benchmark_knn':
        expected += ['--smoke-only']
    assert calls == [expected]
