"""Compare models from their saved metrics CSVs only; refuses to plot missing results.

    python -m src.evaluation.compare_models logistic_regression decision_tree gradient_boosting

Writes results/models/comparison/<a>_vs_<b>.../ with comparison_metrics.csv, a
combined comparison.png and one PNG per metric.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.visualizations import RESULTS, load_metrics  # noqa: E402

CANDIDATE = "Candidate accuracy (labelled candidates, threshold 0.5)"
TOP1 = "Observation Top-1 accuracy (trusted observations; misses count as wrong)"
CV = "Mean over 5 trip-grouped validation folds of the 80 train trips (error bar = std)"
TEST = "10 held-out student_test trips"
# column, title, y-axis label, data description
PANELS = [
    ("cv_candidate_accuracy", "Mean 5-fold CV candidate accuracy", CANDIDATE, CV),
    ("student_test_candidate_accuracy", "Student-test candidate accuracy", CANDIDATE, TEST),
    ("cv_top1_accuracy", "Mean 5-fold CV observation Top-1", TOP1, CV),
    ("student_test_top1_accuracy", "Student-test observation Top-1", TOP1, TEST),
]
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "gradient_boosting": "HistGradientBoosting",
    "random_forest": "Random Forest",
    "knn": "KNN",
}
COLORS = ["#2f6db5", "#d9822b", "#3a9a5b", "#8e5bb5", "#b54a4a"]


def comparison_table(names):
    rows = []
    for name in names:
        cv, test = load_metrics(name)  # raises if a model has no measured results
        mean = cv[cv.fold == "mean"].iloc[0]
        std = cv[cv.fold == "std"].iloc[0]
        rows.append({"model": name,
                     "cv_candidate_accuracy": mean.candidate_accuracy,
                     "cv_candidate_accuracy_std": std.candidate_accuracy,
                     "student_test_candidate_accuracy": test.candidate_accuracy,
                     "cv_top1_accuracy": mean.top1_accuracy,
                     "cv_top1_accuracy_std": std.top1_accuracy,
                     "student_test_top1_accuracy": test.top1_accuracy,
                     "cv_f1_macro": mean.f1_macro, "student_test_f1_macro": test.f1_macro,
                     "cv_log_loss": mean.log_loss, "student_test_log_loss": test.log_loss})
    return pd.DataFrame(rows)


def _bars(ax, table, column):
    labels = [DISPLAY_NAMES.get(m, m) for m in table.model]
    errors = table.get(f"{column}_std")
    bars = ax.bar(labels, table[column], color=COLORS[:len(table)],
                  yerr=errors if errors is not None else None, capsize=4)
    tops = table[column] + (errors if errors is not None else 0)
    for bar, value, top in zip(bars, table[column], tops):
        ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, top),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Model")


def plot_comparison(names):
    table = comparison_table(names)
    out = RESULTS / "comparison" / "_vs_".join(names)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison_metrics.csv", index=False)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.5 * len(PANELS), 5))
    for ax, (column, title, _, _) in zip(axes, PANELS):
        _bars(ax, table, column)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=15)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Candidate accuracy = labelled candidates at threshold 0.5; "
                 "Top-1 = trusted observations (reference misses count as wrong)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=130)
    plt.close(fig)

    for column, title, ylabel, description in PANELS:
        fig, ax = plt.subplots(figsize=(7, 5))
        _bars(ax, table, column)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{title}\n{description}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"{column}.png", dpi=130)
        plt.close(fig)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+")
    print(f"wrote {plot_comparison(parser.parse_args(argv).models)}")


if __name__ == "__main__":
    main()
