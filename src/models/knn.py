"""Distance-weighted KNN using the shared, fold-fitted feature pipeline."""
from sklearn.neighbors import KNeighborsClassifier

NAME = 'knn'
FAMILY = 'Instance-based'
SCALE_FEATURES = True
DEFAULT_PARAMS = {
    'n_neighbors': 5, 'weights': 'distance', 'metric': 'minkowski', 'p': 2,
    'algorithm': 'kd_tree', 'leaf_size': 40, 'n_jobs': 4,
}
# Every training row is its own nearest neighbour, so train-split metrics are
# near-perfect by construction; score a fixed per-fold sample instead of ~0.86M
# rows (each KD-tree query costs ~2 ms, measured on fold 0).
TRAIN_METRICS_SAMPLE = 50000
CLASS_IMBALANCE = (
    'No class weighting or resampling; report positive-class recall and macro-F1 '
    'alongside candidate and observation accuracy.'
)
LIMITATIONS = [
    'Prediction latency and stored training data can be large on the full dataset.',
    'KD-tree search avoids a full query-by-training distance matrix; performance '
    'still depends on the feature distribution and query count.',
    'Train-split metrics use a fixed 50,000-row sample per fold and are optimistic '
    'by construction (each training row is its own neighbour).',
    'Distance-weighted votes of five neighbours give coarse, often exactly 0/1 '
    'probabilities, which inflates log loss.',
    'Untuned baseline; no Bayesian optimisation, calibration or HMM integration.',
]


def build_estimator(params, seed):
    # KNN has no stochastic fit; seed remains part of the shared model interface.
    return KNeighborsClassifier(**params)


def build_knn_model(n_neighbors=5, weights='distance', metric='minkowski', p=2):
    """Compatibility constructor; uses the same 13-feature pipeline as the factory."""
    from src.models.model_factory import get_model
    return get_model(NAME, dict(n_neighbors=n_neighbors, weights=weights, metric=metric, p=p))
