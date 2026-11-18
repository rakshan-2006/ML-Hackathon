"""Logistic Regression candidate scorer (linear / probabilistic family)."""

from sklearn.linear_model import LogisticRegression

NAME = "logistic_regression"
FAMILY = "Linear and probabilistic"
SCALE_FEATURES = True
DEFAULT_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "tol": 1e-4,
    "class_weight": None,
}
CLASS_IMBALANCE = (
    "No reweighting or resampling (class_weight=None): keeps predicted probabilities "
    "calibrated for log loss and for use as HMM emissions. Imbalance is addressed in "
    "evaluation through positive-class precision/recall, macro-F1 and observation Top-1."
)
LIMITATIONS = [
    "Linear decision boundary in the imputed, standardised features; feature "
    "interactions (e.g. distance x heading) are not modelled.",
    "Decision threshold 0.5 is not tuned; positive-class recall depends on it.",
]


def build_estimator(params, seed):
    return LogisticRegression(random_state=seed, **params)
