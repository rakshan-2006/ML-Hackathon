"""Distance-weighted KNN using the shared, fold-fitted feature pipeline."""
from sklearn.neighbors import KNeighborsClassifier

NAME = 'knn'
FAMILY = 'Instance-based'
SCALE_FEATURES = True
DEFAULT_PARAMS = {
    'n_neighbors': 5, 'weights': 'distance', 'metric': 'minkowski', 'p': 2,
    'algorithm': 'kd_tree', 'leaf_size': 40, 'n_jobs': 4,
}
CLASS_IMBALANCE = (
    'No class weighting or resampling; report positive-class recall and macro-F1 '
    'alongside candidate and observation accuracy.'
)
LIMITATIONS = [
    'Prediction latency and stored training data can be large on the full dataset.',
    'KD-tree search avoids a full query-by-training distance matrix; performance '
    'still depends on the feature distribution and query count.',
    'Untuned baseline; no Bayesian optimisation, calibration or HMM integration.',
]


def build_estimator(params, seed):
    # KNN has no stochastic fit; seed remains part of the shared model interface.
    return KNeighborsClassifier(**params)


def build_knn_model(n_neighbors=5, weights='distance', metric='minkowski', p=2):
    """Compatibility constructor; uses the same 13-feature pipeline as the factory."""
    from src.models.model_factory import get_model
    return get_model(NAME, dict(n_neighbors=n_neighbors, weights=weights, metric=metric, p=p))
