
import numpy as np
import pandas as pd

from src.models.knn import build_knn_model


def test_knn_pipeline_fit_and_predict():
    X = pd.DataFrame({
        "dist_m": [1.0, 2.0, 8.0, 9.0, 3.0, 7.0],
        "speed_mps": [10.0, 12.0, 30.0, 32.0, np.nan, 28.0],
    })
    y = np.array([1, 1, 0, 0, 1, 0])

    model = build_knn_model(n_neighbors=3)
    model.fit(X, y)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    assert len(predictions) == len(y)
    assert probabilities.shape == (len(y), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(predictions).issubset({0, 1})
