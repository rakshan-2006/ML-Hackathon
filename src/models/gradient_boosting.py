"""Bounded full-data gradient boosting baseline with native missing-value support."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

NAME = 'gradient_boosting'
FAMILY = 'Boosting'
SCALE_FEATURES = False   # tree splits are invariant to monotone scaling
IMPUTE_MISSING = False   # HistGradientBoosting routes NaNs natively; no fitted imputation
DEFAULT_PARAMS = dict(
    learning_rate=0.1, max_iter=100, max_leaf_nodes=15,
    min_samples_leaf=30, l2_regularization=1.0, max_bins=127,
    early_stopping=False, class_weight=None,
)
CLASS_IMBALANCE = ('Natural prevalence, class_weight=None; report macro and '
                   'positive-class metrics')
LIMITATIONS = [
    'Baseline only; no Bayesian optimisation or calibration.',
    'Early stopping disabled so no random row-level holdout crosses trip boundaries.',
]


def build_estimator(params, seed):
    """Shared-factory contract: the bare estimator (preprocessing added by the factory)."""
    return HistGradientBoostingClassifier(**params, random_state=seed)


def build_model(params=None, seed=2026):
    """Return a complete pipeline; estimator handles NaNs without fitted imputation.

    No scaling is needed for the tree estimator. Early stopping is disabled so
    no random row-level internal validation split crosses trip boundaries.
    Natural class proportions retain the empirical probability prior.
    """
    effective = DEFAULT_PARAMS | (params or {})
    return Pipeline([('estimator', build_estimator(effective, seed))])


def default_params():
    return DEFAULT_PARAMS.copy()
