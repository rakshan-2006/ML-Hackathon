"""Model-side preprocessing; always fitted inside a training fold as part of a Pipeline.

M1 delivers raw physical features with explicit NaNs and no fitted scaling
(data/README.md). Imputation uses training-fold medians plus missing-value
indicators, because a missing heading or speed is itself informative.
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.common.schema import FEATURES


def to_float64(X):
    """Ordered FEATURES as a float64 array (the bool is_oneway becomes 0/1)."""
    missing = [f for f in FEATURES if f not in X.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return X[FEATURES].to_numpy(dtype=np.float64)


def preprocessing_steps(scale, impute=True):
    steps = [("to_float", FunctionTransformer(to_float64))]
    if impute:
        steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True)))
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


def make_pipeline(estimator, scale, impute=True):
    return Pipeline(preprocessing_steps(scale, impute) + [("model", estimator)])


PREPROCESSING_POLICY = {
    "feature_order": "src.common.schema.FEATURES (exact order)",
    "missing_values": "SimpleImputer(strategy='median', add_indicator=True), fitted on training-fold rows only "
                      "(models with IMPUTE_MISSING = False pass NaNs to an estimator with native support)",
    "scaling": "StandardScaler fitted on training-fold rows (only models that request scaling)",
}


TARGET = 'label'
GROUP = 'trip_id'


def load_training_data(processed_root=None):
    """Load all trusted training candidates through the canonical M1 loader.

    No imputation or scaling is fitted here; those remain inside model pipelines.
    """
    from src.common.paths import PROJECT_ROOT
    from src.data.model_data import load_model_data
    root = PROJECT_ROOT / 'data' / 'processed' if processed_root is None else processed_root
    X, y, groups, _ = load_model_data(root, 'train')
    return X, y, groups
