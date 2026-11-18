"""Generate report-ready PNG/SVG graphs from saved GB results and probabilities.

No model fitting occurs. All probability metrics use trusted labelled candidates;
Top-1 uses the saved all-trusted-observation denominator, including misses.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, log_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

BLUE, TEAL, ORANGE = '#3867d6', '#008577', '#de7b28'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', default='results/models/gradient_boosting')
    p.add_argument('--processed-root', default='data/processed')
    p.add_argument('--artifacts-root', default='artifacts/gradient_boosting')
    args = p.parse_args()
    root, data, artifacts = map(Path, [args.results_root, args.processed_root, args.artifacts_root])
    out = root/'figures'
    out.mkdir(parents=True, exist_ok=True)
    card = json.loads((root/'model_card.json').read_text())
    cv = pd.read_csv(root/'cv_metrics.csv')
    trips = {name: pd.read_csv(root/f'{name}_per_trip.csv', dtype={'trip_id': str})
             for name in ['train_oof', 'student_test']}
    timings = json.loads((root/'timings.json').read_text())
    sources = [root/'model_card.json', root/'cv_metrics.csv', root/'timings.json']
    sources += [root/f'{name}_per_trip.csv' for name in trips]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.facecolor': 'white',
                         'axes.axisbelow': True, 'svg.hashsalt': 'pathsnap-gb'})
    outputs = []

    def save(fig, name, subtitle):
        fig.suptitle(subtitle, fontsize=15, fontweight='bold', y=0.995)
        fig.text(0.5, 0.012, 'PathSnap | Gradient Boosting baseline | Derived reference labels; no HMM decoding',
                 ha='center', fontsize=8, color='#555555')
        fig.tight_layout(rect=(0, 0.035, 1, 0.95))
        for suffix in ['png', 'svg']:
            dest = out/f'{name}.{suffix}'
            fig.savefig(dest, dpi=180, bbox_inches='tight',
                        metadata={'Date': None} if suffix == 'svg' else None)
            outputs.append(dest)
        plt.close(fig)

    def percentage(ax):
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=0.2)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    xs = np.arange(1, 6)
    ax.bar(xs-.18, cv.macro_f1, .36, label='Candidate macro-F1', color=BLUE)
    ax.bar(xs+.18, cv.top1_accuracy, .36, label='Observation Top-1', color=TEAL)
    for x, value in zip(xs+.18, cv.top1_accuracy):
        ax.text(x, value+.018, f'{value:.1%}', ha='center', fontsize=8)
    ax.set(title='Five-fold train validation: whole trips held out', xlabel='Fold')
    ax.set_xticks(xs)
    percentage(ax)
    ax.legend(fontsize=8, loc='lower left')
    ax = axes[0, 1]
    metric_names = ['macro_f1', 'precision_macro', 'recall_macro', 'top1_accuracy']
    xx = np.arange(4)
    for shift, name, color, label in [(-.18, 'train_oof', BLUE, 'Pooled train OOF'),
                                    (.18, 'student_test', TEAL, 'Student-test')]:
        values = [card[name][m] for m in metric_names]
        ax.bar(xx+shift, values, .36, label=label, color=color)
        for x, value in zip(xx+shift, values):
            ax.text(x, value+.015, f'{value:.1%}', ha='center', fontsize=8)
    ax.set_xticks(xx, ['Macro-F1', 'Macro precision', 'Macro recall', 'Top-1'])
    ax.set_title('Pooled out-of-fold vs held-out results')
    percentage(ax)
    ax.legend(fontsize=8, loc='lower left')
    ax = axes[1, 0]
    test_trips = trips['student_test'].sort_values('trip_id')
    ax.bar(np.arange(len(test_trips)), test_trips.top1_accuracy, color=TEAL)
    ax.axhline(card['student_test']['top1_accuracy'], color=ORANGE, linestyle='--',
               label=f"Micro overall: {card['student_test']['top1_accuracy']:.2%}")
    ax.set_xticks(np.arange(len(test_trips)), test_trips.trip_id, rotation=45, ha='right')
    ax.set(title='Student-test Top-1 by trip', xlabel='Trip ID')
    percentage(ax)
    ax.legend(fontsize=8, loc='lower left')
    ax = axes[1, 1]
    losses = [card['cv_fold_mean']['log_loss'], card['train_oof']['log_loss'],
              card['student_test']['log_loss']]
    ax.bar(['CV fold mean', 'Pooled train OOF', 'Student-test'], losses, color=[BLUE, BLUE, TEAL])
    for x, v in enumerate(losses):
        ax.text(x, v+.003, f'{v:.4f}', ha='center')
    ax.set(title='Candidate log loss — lower is better', ylabel='Log loss', ylim=(0, max(losses)*1.25))
    ax.grid(axis='y', alpha=.2)
    save(fig, '01_performance_overview', 'Gradient Boosting | Full-data baseline results')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, name, label in zip(axes, ['train_oof', 'student_test'], ['Train out-of-fold', 'Student-test']):
        counts = np.array(card[name]['confusion_matrix'])
        normalized = counts/counts.sum(axis=1, keepdims=True)
        image = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{counts[i,j]:,}\n{normalized[i,j]:.1%} of true class',
                        ha='center', va='center', color='white' if normalized[i,j]>.5 else '#172b4d')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Other arc (0)', 'Reference arc (1)'],
               yticklabels=['Other arc (0)', 'Reference arc (1)'], xlabel='Predicted class',
               ylabel='True derived class', title=f'{label}\nn = {counts.sum():,} labelled candidates')
        fig.colorbar(image, ax=ax, fraction=.045, label='Row-normalized proportion')
    save(fig, '02_confusion_matrices', 'Candidate classification | Threshold p ≥ 0.5')

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    for ax, name, color, label in zip(axes, trips, [BLUE, TEAL], ['Train OOF — 80 trips', 'Student-test — 10 trips']):
        table = trips[name].sort_values('top1_accuracy')
        ax.bar(np.arange(len(table)), table.top1_accuracy, color=color)
        ax.axhline(card[name]['top1_accuracy'], color=ORANGE, ls='--',
                   label=f'Micro overall: {card[name]["top1_accuracy"]:.2%}')
        ax.set_xticks(np.arange(len(table)), table.trip_id, rotation=90,
                      fontsize=6 if len(table)>20 else 9)
        ax.set(title=label, xlabel='Trip ID (sorted by Top-1)', ylabel='Top-1 agreement')
        percentage(ax)
        ax.legend(loc='lower right')
    save(fig, '03_per_trip_top1', 'Trip-level variation | Reference misses and zero-candidate points count as failures')

    fig, ax = plt.subplots(figsize=(11, 5))
    stages = timings['stages']
    xx = np.arange(len(stages))
    for offset, field, color, label in [(-.18, 'fit_s', BLUE, 'Fit'), (.18, 'predict_s', TEAL, 'Predict')]:
        values = [row[field] for row in stages]
        bars = ax.bar(xx+offset, values, .36, label=label, color=color)
        ax.bar_label(bars, fmt='%.1f s', padding=3, fontsize=9)
    ax.set_xticks(xx, ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Final refit'])
    ax.set(ylabel='Wall-clock seconds', title='Final prediction timing covers all student-test candidates')
    ax.set_ylim(0, max(s['fit_s'] for s in stages)*1.2)
    ax.grid(axis='y', alpha=.2)
    ax.legend()
    save(fig, '04_runtime', f'Training and prediction time | 4 threads | Total workflow {timings["total_s"]:.1f} s')

    curves, probability_metrics = {}, {}
    for partition, score_name, key, color, label in [
            ('train', 'scores_train_oof.parquet', 'train_oof', BLUE, 'Train OOF'),
            ('student_test', 'scores_student_test.parquet', 'student_test', TEAL, 'Student-test')]:
        scores_path, cand_path = artifacts/score_name, data/f'{partition}_candidates.parquet'
        sources.extend([scores_path, cand_path])
        scores = pd.read_parquet(scores_path)
        cand = pd.read_parquet(cand_path, columns=['trip_id', 'obs_idx', 'arc_id', 'label'])
        if not cand[['trip_id','obs_idx','arc_id']].equals(scores[['trip_id','obs_idx','arc_id']]):
            raise ValueError(f'{partition}: score keys/order mismatch')
        trusted = cand.label.notna().to_numpy()
        y = cand.loc[trusted, 'label'].to_numpy(dtype=np.int8)
        probs = scores.loc[trusted, 'p_model'].to_numpy()
        if not np.isclose(log_loss(y, probs), card[key]['log_loss'], atol=1e-12):
            raise ValueError('Saved model card and probability log loss disagree')
        fpr, tpr, _ = roc_curve(y, probs)
        precision, recall, _ = precision_recall_curve(y, probs)
        fraction, mean = calibration_curve(y, probs, n_bins=10, strategy='uniform')
        metrics = dict(roc_auc=float(roc_auc_score(y, probs)),
                       average_precision=float(average_precision_score(y, probs)),
                       positive_prevalence=float(y.mean()), labelled_candidates=len(y))
        probability_metrics[key] = metrics
        curves[key] = dict(fpr=fpr, tpr=tpr, precision=precision, recall=recall,
                           fraction=fraction, mean=mean, probs=probs, color=color, label=label)
        pd.DataFrame({'mean_probability': mean, 'positive_fraction': fraction}).to_csv(
            out/f'{key}_calibration.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for key, curve in curves.items():
        m = probability_metrics[key]
        # Render deterministic subsets of the full empirical curves to keep SVGs small.
        for ax, x, y, score, score_name in [
                (axes[0], curve['fpr'], curve['tpr'], m['roc_auc'], 'AUC'),
                (axes[1], curve['recall'], curve['precision'], m['average_precision'], 'AP')]:
            idx = np.unique(np.linspace(0, len(x)-1, min(2000, len(x)), dtype=int))
            ax.plot(x[idx], y[idx], color=curve['color'], lw=2,
                    label=f'{curve["label"]}: {score_name} = {score:.4f}')
    axes[0].plot([0,1],[0,1], '--', color='gray', lw=1, label='Random ranking')
    axes[0].set(title='ROC curve', xlabel='False-positive rate', ylabel='True-positive rate')
    axes[1].axhline(probability_metrics['student_test']['positive_prevalence'], ls='--',
                    color='gray', lw=1, label='Student-test prevalence')
    axes[1].set(title='Precision–recall curve', xlabel='Recall (positive class)', ylabel='Precision (positive class)')
    for ax in axes:
        ax.set(xlim=(0,1), ylim=(0,1.03))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9, loc='lower right' if ax == axes[0] else 'lower left')
    save(fig, '05_roc_precision_recall', 'Candidate probability ranking | Metrics use every trusted candidate')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].plot([0,1],[0,1], '--', color='gray', label='Perfect calibration')
    for curve in curves.values():
        axes[0].plot(curve['mean'], curve['fraction'], 'o-', color=curve['color'], label=curve['label'])
        axes[1].hist(curve['probs'], bins=np.linspace(0,1,21), histtype='step',
                     density=True, linewidth=2, color=curve['color'], label=curve['label'])
    axes[0].set(title='Reliability diagram — 10 equal-width bins',
                xlabel='Mean predicted probability', ylabel='Observed positive fraction', xlim=(0,1), ylim=(0,1))
    axes[1].set(title='Predicted probability distribution', xlabel='P(candidate is reference arc)', ylabel='Density (log scale)', yscale='log')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    save(fig, '06_probability_calibration', 'Probability quality | Baseline probabilities; no post-hoc calibration fitted')

    (out/'probability_metrics.json').write_text(json.dumps(probability_metrics, indent=2), encoding='utf8')
    manifest = dict(generator='python -m src.evaluation.plot_gradient_boosting',
                    model_version=card['model_version'],
                    note='Empirical ROC/PR displayed with at most 2000 points; AUC/AP use full data. No invented epoch/loss training history.',
                    source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                    outputs=[str(path) for path in outputs])
    (out/'figure_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps(dict(figures=len(outputs)//2, formats=['png','svg'], output=str(out),
                         probability_metrics=probability_metrics), indent=2))


if __name__ == '__main__':
    main()
