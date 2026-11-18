"""Five-model comparison table (Deliverables 3.1) and required figures from saved results.

    python -m src.evaluation.model_selection
    python -m src.evaluation.model_selection --models logistic_regression ... --out five_model

Reads only results/models/<name>/ (cv_metrics, student_test_metrics, config,
timings) and, when present, results/models/comparison/classifier_only_continuity.csv
(classifier-only route continuity computed with Person B's shared
src.hmm.route_metrics.route_continuity). Nothing is retrained. Fold means are
kept separate from pooled metrics; every column states its partition.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.visualizations import RESULTS, load_metrics  # noqa: E402
from src.models.model_factory import model_module  # noqa: E402

MODELS = ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting", "knn"]
DISPLAY = {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree",
           "random_forest": "Random Forest", "gradient_boosting": "HistGradientBoosting",
           "knn": "KNN", "gradient_boosting_tuned": "HistGB (tuned)"}
# Fixed categorical order (validated palette); a model keeps its colour in every figure.
COLOURS = {"logistic_regression": "#2a78d6", "decision_tree": "#eb6834",
           "random_forest": "#1baf7a", "gradient_boosting": "#eda100", "knn": "#e87ba4",
           "gradient_boosting_tuned": "#6250d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CONTINUITY = RESULTS / "comparison" / "classifier_only_continuity.csv"
KEY_PARAMS = {
    "logistic_regression": ["C", "class_weight", "max_iter"],
    "decision_tree": ["max_depth", "min_samples_leaf", "class_weight"],
    "random_forest": ["n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight"],
    "gradient_boosting": ["learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf",
                          "l2_regularization"],
    "knn": ["n_neighbors", "weights", "metric", "p", "algorithm"],
}


def module_name(name):
    config = json.loads((RESULTS / name / "config.json").read_text(encoding="utf-8"))
    return config.get("model_module", name), config


def timing(name):
    """Student-test inference cost per candidate row and total CV fitting time."""
    t = json.loads((RESULTS / name / "timings.json").read_text(encoding="utf-8"))
    if "stages" in t:   # original GB runner layout
        final = next(s for s in t["stages"] if s["stage"] == "final")
        card = json.loads((RESULTS / name / "model_card.json").read_text(encoding="utf-8"))
        rows = card["student_test_all_candidate_rows"]
        folds = [s for s in t["stages"] if s["stage"].startswith("fold_")]
        return {"inference_us_per_row": 1e6 * final["predict_s"] / rows,
                "cv_fit_s": sum(s["fit_s"] for s in folds), "refit_s": final["fit_s"],
                "threads": t.get("threads")}
    return {"inference_us_per_row": t["student_test_predict_us_per_row"],
            "cv_fit_s": t["cv_total_fit_s"], "refit_s": t["refit_s"], "threads": None}


def key_config(name):
    module, config = module_name(name)
    params = config.get("params", {})
    keys = KEY_PARAMS.get(module, sorted(params))
    return ", ".join(f"{k}={params[k]}" for k in keys if k in params)


def continuity():
    if not CONTINUITY.is_file():
        return None
    return pd.read_csv(CONTINUITY).set_index(["model", "partition"])


def full_table(names):
    cont = continuity()
    rows = []
    for name in names:
        cv, test = load_metrics(name)
        mean = cv[cv.fold == "mean"].iloc[0]
        std = cv[cv.fold == "std"].iloc[0]
        module, _ = module_name(name)
        row = {"model": name, "display": DISPLAY.get(name, name),
               "family": model_module(module).FAMILY, "key_config": key_config(name)}
        for metric in ("f1_macro", "top1_accuracy", "log_loss", "candidate_accuracy",
                       "precision_pos", "recall_pos", "f1_pos", "precision_macro", "recall_macro"):
            row[f"cv_mean_{metric}"] = mean[metric]
            row[f"cv_std_{metric}"] = std[metric]
            row[f"student_test_{metric}"] = test[metric]
        row["cv_mean_candidate_miss_rate"] = 1 - mean["top1_upper_bound"]
        row["student_test_candidate_miss_rate"] = 1 - test["top1_upper_bound"]
        for k in ("tn", "fp", "fn", "tp", "n_trusted_obs", "n_top1_correct", "n_labelled"):
            row[f"student_test_{k}"] = int(test[k])
        row.update(timing(name))
        for partition in ("train_oof", "student_test"):
            key = (name, partition)
            row[f"{partition}_route_continuity"] = (cont.loc[key, "continuity"]
                                                    if cont is not None and key in cont.index else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def deliverables_table(full):
    """Columns exactly as Deliverables 3.1, with the partition stated in each header."""
    def continuity_cell(r):
        if np.isfinite(r.student_test_route_continuity):
            return f"{r.student_test_route_continuity:.4f}"
        return "pending: computed with Person B's route_continuity"
    return pd.DataFrame({
        "Model": full.display,
        "Algorithm Family": full.family,
        "Key Configuration": full.key_config,
        "CV Macro F1-score (mean of 5 folds)": full.cv_mean_f1_macro.map("{:.4f}".format)
            + " ± " + full.cv_std_f1_macro.map("{:.4f}".format),
        "Top-1 Accuracy (student-test observations)": full.student_test_top1_accuracy.map("{:.4f}".format),
        "Route continuity (student-test, classifier-only)": full.apply(continuity_cell, axis=1),
        "Student-Test Macro F1-score": full.student_test_f1_macro.map("{:.4f}".format),
        "Log Loss (student-test)": full.student_test_log_loss.map("{:.4f}".format),
        "Inference Time (µs per candidate, student-test)": full.inference_us_per_row.map("{:.2f}".format),
    })


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def _label(ax, bars, values, fmt="{:.4f}", at=None):
    """Value labels above each bar, or at height `at` (inside the bar) when error bars sit on top."""
    for bar, value in zip(bars, values):
        y = bar.get_height() if at is None else at
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, y),
                    ha="center", va="bottom", fontsize=8, color=INK,
                    xytext=(0, 2), textcoords="offset points",
                    bbox=None if at is None else dict(boxstyle="round,pad=0.15", fc="white",
                                                      ec="none", alpha=0.9))


def plot_cv_f1(full, names, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, full.cv_mean_f1_macro, 0.6, color=[COLOURS[n] for n in names],
                  yerr=full.cv_std_f1_macro, capsize=4, ecolor=MUTED)
    for i, name in enumerate(names):
        cv, _ = load_metrics(name)
        folds = cv[cv.fold.str.isdigit() & (cv.split == "validation")]
        ax.scatter(np.full(len(folds), i) + np.linspace(-0.18, 0.18, len(folds)), folds.f1_macro,
                   s=16, color=INK, zorder=3, label="individual fold" if i == 0 else None)
    _label(ax, bars, full.cv_mean_f1_macro, at=0.752)
    ax.set_xticks(x, full.display)
    ax.set_ylim(0.75, 0.95)
    ax.set_ylabel("Candidate macro-F1 (threshold 0.5)")
    ax.set_title("5-fold trip-grouped CV macro-F1 on the 80 train trips\n"
                 "bar = mean of folds, error bar = std, dots = folds (y-axis starts at 0.75)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "cv_macro_f1.png", dpi=150)
    plt.close(fig)


def plot_student_test(full, names, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(names))
    for ax, column, title, low in (
            (axes[0], "student_test_top1_accuracy", "Observation Top-1 accuracy\n"
             "(16,073 trusted observations; reference misses count as wrong)", 0.75),
            (axes[1], "student_test_f1_macro", "Candidate macro-F1\n(106,730 labelled candidates, "
             "threshold 0.5)", 0.80)):
        bars = ax.bar(x, full[column], 0.6, color=[COLOURS[n] for n in names])
        _label(ax, bars, full[column])
        ax.set_xticks(x, full.display, rotation=15)
        ax.set_ylim(low, 1.0)
        ax.set_title(title, fontsize=10)
        _style(ax)
    fig.suptitle(f"10 held-out student_test trips (y-axes truncated)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "student_test_top1_and_macro_f1.png", dpi=150)
    plt.close(fig)


def plot_log_loss(full, names, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    width = 0.38
    cv = ax.bar(x - width / 2, full.cv_mean_log_loss, width - 0.02, color=[COLOURS[n] for n in names],
                yerr=full.cv_std_log_loss, capsize=3, ecolor=MUTED)
    test = ax.bar(x + width / 2, full.student_test_log_loss, width - 0.02,
                  color=[COLOURS[n] for n in names], hatch="//", edgecolor="white")
    floor = 0.8 * min(full.cv_mean_log_loss.min(), full.student_test_log_loss.min())
    _label(ax, cv, full.cv_mean_log_loss, "{:.3f}", at=floor * 1.03)
    _label(ax, test, full.student_test_log_loss, "{:.3f}", at=floor * 1.03)
    ax.set_ylim(bottom=floor)
    ax.bar(0, 0, color=MUTED, label="CV mean of 5 folds (± std)")
    ax.bar(0, 0, color=MUTED, hatch="//", edgecolor="white", label="student_test")
    ax.set_xticks(x, full.display)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Log loss (lower is better, log scale)")
    ax.set_title("Probability quality: candidate log loss", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "log_loss.png", dpi=150)
    plt.close(fig)


def plot_confusion(full, names, out):
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.6))
    for ax, (_, r) in zip(np.atleast_1d(axes), full.iterrows()):
        m = np.array([[r.student_test_tn, r.student_test_fp], [r.student_test_fn, r.student_test_tp]])
        share = m / m.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]:,}\n({share[i, j]:.1%})", ha="center", va="center",
                        fontsize=8, color="white" if share[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(r.display, fontsize=9)
    fig.suptitle("student_test confusion matrices (labelled candidates, threshold 0.5; "
                 "row-normalised shares)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "student_test_confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_fold_stability(names, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name in names:
        cv, _ = load_metrics(name)
        folds = cv[cv.fold.str.isdigit() & (cv.split == "validation")]
        for ax, metric in zip(axes, ("f1_macro", "top1_accuracy")):
            ax.plot(folds.fold.astype(int), folds[metric], marker="o", markersize=5, linewidth=2,
                    color=COLOURS[name], label=DISPLAY.get(name, name))
    for ax, title in zip(axes, ("Candidate macro-F1", "Observation Top-1 accuracy")):
        ax.set_xticks(range(5))
        ax.set_xlabel("Validation fold (canonical configs/cv_folds.csv)")
        ax.set_title(title, fontsize=10)
        _style(ax)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Per-fold validation results on the 80 train trips "
                 "(fold 4 contains the trips with most reference misses)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fold_stability.png", dpi=150)
    plt.close(fig)


def plot_timing(full, names, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(names))
    for ax, column, title, fmt in (
            (axes[0], "cv_fit_s", "Total 5-fold CV fitting time (s)", "{:.0f}"),
            (axes[1], "inference_us_per_row", "student_test inference (µs per candidate row)", "{:.2f}")):
        bars = ax.bar(x, full[column], 0.6, color=[COLOURS[n] for n in names])
        _label(ax, bars, full[column], fmt)
        ax.set_yscale("log")
        ax.set_xticks(x, full.display, rotation=15)
        ax.set_title(title + ", log scale", fontsize=10)
        _style(ax)
    fig.suptitle("Measured computational cost (same laptop; thread counts in each config)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "timing.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--out", default="five_model")
    args = parser.parse_args(argv)
    out = RESULTS / "comparison" / args.out
    out.mkdir(parents=True, exist_ok=True)
    full = full_table(args.models)
    full.to_csv(out / "all_metrics.csv", index=False)
    table = deliverables_table(full)
    table.to_csv(out / "deliverables_table.csv", index=False)
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False)]
    (out / "deliverables_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_cv_f1(full, args.models, out)
    plot_student_test(full, args.models, out)
    plot_log_loss(full, args.models, out)
    plot_confusion(full, args.models, out)
    plot_fold_stability(args.models, out)
    plot_timing(full, args.models, out)
    print(table.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
