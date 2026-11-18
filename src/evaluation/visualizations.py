"""Accuracy plots drawn only from a model's saved metrics CSVs."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.paths import PROJECT_ROOT  # noqa: E402

RESULTS = PROJECT_ROOT / "results" / "models"


def load_metrics(name):
    folder = RESULTS / name
    paths = [folder / "cv_metrics.csv", folder / "student_test_metrics.csv"]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing measured results: {missing}")
    cv = pd.read_csv(paths[0], dtype={"fold": "string"})
    test = pd.read_csv(paths[1]).iloc[0]
    if 'split' not in cv.columns:
        # Earlier GB evidence has one row per validation fold and different names.
        # Adapt at the read boundary; never rewrite historical measured results.
        rename = {'macro_f1': 'f1_macro', 'precision_positive': 'precision_pos',
                  'recall_positive': 'recall_pos', 'f1_positive': 'f1_pos'}
        cv = cv.rename(columns=rename)
        test = test.rename(index=rename)
        numeric = cv.select_dtypes('number')
        summary = [dict(fold=stat, split='validation', **numeric.agg(stat).to_dict())
                   for stat in ['mean', 'std']]
        cv['split'] = 'validation'
        cv = pd.concat([cv, pd.DataFrame(summary)], ignore_index=True)
    return cv, test


def _bars(ax, positions, values, colors, errors=None):
    bars = ax.bar(positions, values, color=colors, yerr=errors, capsize=4)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")


def plot_model_accuracy(name):
    cv, test = load_metrics(name)
    folds = cv[cv.fold.str.isdigit()]
    train = folds[folds.split == "train"].set_index("fold")
    val = folds[folds.split == "validation"].set_index("fold")
    mean = cv[cv.fold == "mean"].iloc[0]
    std = cv[cv.fold == "std"].iloc[0]
    labels = [f"fold {f}" for f in val.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    x = range(len(labels))
    width = 0.38
    if not train.empty:
        ax.bar([i - width / 2 for i in x], train.candidate_accuracy, width, label="train fold", color="#9bb7d4")
    ax.bar([i + width / 2 for i in x], val.candidate_accuracy, width, label="validation fold", color="#2f6db5")
    for i, (a, b) in enumerate(zip(train.candidate_accuracy, val.candidate_accuracy)):
        ax.annotate(f"{a:.4f}", (i - width / 2, a), ha="center", va="bottom", fontsize=7)
        ax.annotate(f"{b:.4f}", (i + width / 2, b), ha="center", va="bottom", fontsize=7)
    n = len(labels)
    _bars(ax, [n, n + 1],
          [mean.candidate_accuracy, test.candidate_accuracy], ["#1b3f73", "#d9822b"],
          errors=[std.candidate_accuracy, 0])
    ax.set_xticks(list(x) + [n, n + 1], labels + ["CV mean (val)", "student test"], rotation=20)
    ax.set_ylabel("Candidate accuracy (labelled candidates, threshold 0.5)")
    ax.set_title(f"{name}: candidate-level accuracy")
    low = min(val.candidate_accuracy.min(), test.candidate_accuracy)
    ax.set_ylim(max(0.0, low - 0.05), 1.0)
    ax.legend(loc="lower right")

    ax = axes[1]
    _bars(ax, list(range(n + 2)),
          list(val.top1_accuracy) + [mean.top1_accuracy, test.top1_accuracy],
          ["#2f6db5"] * n + ["#1b3f73", "#d9822b"],
          errors=[0] * n + [std.top1_accuracy, 0])
    ax.set_xticks(list(range(n + 2)), labels + ["CV mean (val)", "student test"], rotation=20)
    ax.set_ylabel("Observation Top-1 accuracy (trusted observations)")
    ax.set_title(f"{name}: observation-level Top-1 (misses count as wrong)")
    ax.set_ylim(0, 1.0)

    fig.suptitle("Measured results; CV = 5 trip-grouped folds of the 80 train trips; "
                 "student test = 10 held-out trips", fontsize=10)
    fig.tight_layout()
    path = RESULTS / name / "accuracy.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
