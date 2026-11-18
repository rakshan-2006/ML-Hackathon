"""Decision Tree candidate scorer (tree-based family)."""

from sklearn.tree import DecisionTreeClassifier

NAME = "decision_tree"
FAMILY = "Tree-based"
SCALE_FEATURES = False  # splits are invariant to monotone feature scaling
DEFAULT_PARAMS = {
    "criterion": "gini",
    "max_depth": 12,
    "min_samples_leaf": 50,
    "min_samples_split": 2,
    "max_features": None,
    "class_weight": None,
}
CLASS_IMBALANCE = (
    "No reweighting or resampling (class_weight=None), so leaf probabilities stay "
    "empirical class frequencies. Imbalance is addressed in evaluation through "
    "positive-class precision/recall, macro-F1 and observation Top-1."
)
LIMITATIONS = [
    "Probabilities are piecewise constant leaf frequencies (coarse ranking, many ties "
    "within an observation; ties resolved by lowest cand_rank).",
    "max_depth=12 / min_samples_leaf=50 are untuned baseline limits against overfitting.",
    "Decision threshold 0.5 is not tuned; positive-class recall depends on it.",
]


def build_estimator(params, seed):
    return DecisionTreeClassifier(random_state=seed, **params)
