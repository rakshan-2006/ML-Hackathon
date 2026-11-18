# Gradient boosting baseline

## Graphs

Generate graphs from the saved results and probability exports, without retraining:

```powershell
python -m src.evaluation.plot_gradient_boosting
```

Install the documented dependencies and download both dataset and model artifacts
first if working from a clean clone. Each figure is saved as PNG and SVG under
`results/models/gradient_boosting/figures/`:

1. `01_performance_overview`: fold macro-F1/Top-1, pooled OOF vs held-out metrics,
   student-test per-trip Top-1 and log loss.
2. `02_confusion_matrices`: counts and row-normalized percentages, threshold 0.5.
3. `03_per_trip_top1`: every train OOF and student-test trip, sorted by agreement.
4. `04_runtime`: measured fit/predict time for five folds and final refit.
5. `05_roc_precision_recall`: empirical candidate ranking curves with full-data
   ROC AUC and average precision.
6. `06_probability_calibration`: reliability curves and probability distributions.

The generator checks score keys/order and log loss against the saved model card.
Its figure manifest records source checksums. ROC/PR rendering uses at most 2000
points per line to keep vector files small; summary metrics use all trusted rows.
Top-1 and candidate classification have different denominators, as documented
below. No artificial epoch/training-loss curves are generated.

This model uses scikit-learn HistGradientBoostingClassifier, the histogram-based
implementation of the team's Gradient Boosting family. It trains on all trusted
labelled candidates in the published M1 dataset. It does not sample the training
dataset. Histograms, 15 leaves, 100 boosting iterations, and four CPU threads bound
the runtime. Hyperparameters are fixed before student-test evaluation.

## Run from the repository root (Python 3.11)

```powershell
py -3.11 -m venv .venv-gb
.venv-gb/Scripts/python -m pip install -r requirements.txt
.venv-gb/Scripts/python -m src.data.download_dataset
.venv-gb/Scripts/python -m src.data.validate_dataset
.venv-gb/Scripts/python -m src.training.train_gradient_boosting --threads 4
.venv-gb/Scripts/python -m pytest -q
```

At the original baseline run, the model branch contained empty scaffolding. The
integrated repository now has a shared runner and a single `configs/cv_folds.csv`
for all five models. The dedicated GB baseline runner remains available.
Five-fold GroupKFold balances counts of labelled candidate rows and keeps whole
trips together. Existing fold assignments are validated and reused. There is no
shuffle; the recorded seed controls the estimator rather than the fold splitter.

Preprocessing PR #2 and model PRs #3–#5 have been integrated into main.

## Model and supervision

Use `build_model(params=None, seed=2026)` from `src.models.gradient_boosting`.
The returned sklearn Pipeline contains the estimator, with native missing-value
handling and no scaling. No fitted global imputer, scaler or random row-level
early-stopping holdout is used. Class weights remain unset to retain natural
candidate prevalence; macro-F1 and positive-class precision/recall/F1 expose the
imbalance. This is a baseline, not a calibrated or Bayesian-optimised final model.

Label `1` means the candidate equals a confident derived reference arc; null
labels are excluded from fitting. Derived labels are approximate. Model inputs
contain the 13 declared physical features, including next-observation distance
under the agreed whole-trip contract. No alignment information is a feature.

## Evaluation and artifacts

Every fold scores ALL candidates of its held-out trips, including rows with null
labels. OOF export keys preserve the original candidate order and contain an
extra `fold` column. Fitting includes only labelled candidates. Candidate metrics
use only labelled rows; Top-1 chooses the highest class-1 probability over every
candidate per observation, breaking ties by lowest candidate rank. The Top-1
denominator is every valid observation with alignment status `ok`, including
reference misses and zero-candidate observations as failures. This metric is
agreement with derived alignment, not independently verified route accuracy.

Small evidence lives in `results/models/gradient_boosting/`. Fitted pipeline,
model card, schema, fold map, OOF and student-test probabilities, and artifact
checksums live in ignored `artifacts/gradient_boosting/` for Release distribution.
The final pipeline is fitted only on train; student-test is evaluated once at the
end. Reloaded pipeline probabilities must match the in-memory pipeline exactly.

Use the complete exported pipeline directly:

```python
import joblib
from src.common.schema import FEATURES
from src.training.train_gradient_boosting import positive_probability

pipeline = joblib.load('artifacts/gradient_boosting/gradient_boosting_pipeline.joblib')
scores = positive_probability(pipeline, candidate_table[FEATURES])
```

Do not apply a second preprocessor or rename this baseline to the shared selected
`best_model.joblib` before comparing all five models. No production decoder,
Predictor adapter, route continuity, or sequence metrics are supplied here.


## Measured baseline results

| Metric | Five-fold mean | Pooled train OOF | Held-out student-test |
| --- | ---: | ---: | ---: |
| Candidate macro-F1 | 0.886149 | 0.886669 | 0.903897 |
| Candidate log loss | 0.132879 | 0.132908 | 0.117970 |
| Observation Top-1 | 0.767965 | 0.764218 | 0.835687 |

Top-1 student-test denominator: 16,073; correct: 13,432.
Total workflow runtime: 211.1 seconds (four CPU threads).
Evidence: results/models/gradient_boosting/model_card.json, cv_metrics.csv,
student_test_metrics.csv, per-trip CSVs and timings.json. All 35 repository tests
passed after adding the four model tests. Fold results vary materially; compare
per-trip results and identically assigned folds rather than student-test alone.

[Download the pipeline and probability exports](https://github.com/rakshan-2006/ML-Hackathon/releases/tag/gradient-boosting-baseline-2026-11-18). From the model
branch after installing the documented dependencies, download with:

```powershell
python -m src.data.download_dataset --metadata data/releases/gradient_boosting.json --output-root artifacts/gradient_boosting
```

This verifies every asset's SHA-256/size. The pipeline is a trusted project
artifact; load it only in the documented matching environment.
