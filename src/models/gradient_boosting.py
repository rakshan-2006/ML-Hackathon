"""Bounded full-data gradient boosting baseline with native missing-value support."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

DEFAULT_PARAMS = dict(
    learning_rate=0.1, max_iter=100, max_leaf_nodes=15,
    min_samples_leaf=30, l2_regularization=1.0, max_bins=127,
    early_stopping=False, class_weight=None,
)


def build_model(params=None, seed=2026):
    """Return a complete pipeline; estimator handles NaNs without fitted imputation.

    No scaling is needed for the tree estimator. Early stopping is disabled so
    no random row-level internal validation split crosses trip boundaries.
    Natural class proportions retain the empirical probability prior.
    """
    effective = DEFAULT_PARAMS | (params or {})
    return Pipeline([('estimator', HistGradientBoostingClassifier(
        **effective, random_state=seed))])


def default_params():
    return DEFAULT_PARAMS.copy()
