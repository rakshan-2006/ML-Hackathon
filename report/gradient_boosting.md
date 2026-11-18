# Gradient boosting baseline

This model uses scikit-learn HistGradientBoostingClassifier, the histogram-based
implementation of the team's Gradient Boosting family. It trains on all trusted
labelled candidates in the published M1 dataset. It does not sample the training
dataset. Histograms, 15 leaves, 100 boosting iterations, and four CPU threads bound
the runtime. Hyperparameters are fixed before student-test evaluation.

## Run from the repository root (Python 3.11)

```powershell
py -3.11 -m venv .venv-gb
.venv-gb/Scripts/python -m pip install -r data/requirements-m1.txt -r data/requirements-gradient-boosting.txt
.venv-gb/Scripts/python -m src.data.download_dataset
.venv-gb/Scripts/python -m src.data.validate_dataset
.venv-gb/Scripts/python -m src.training.train_gradient_boosting --threads 4
.venv-gb/Scripts/python -m pytest -q
```

The shared model branch currently contains empty scaffolding, so no executable
shared runner or fold file was available. This implementation isolates the GB
runner and publishes `experiments/cv_folds.json` for all five models to reuse.
Five-fold GroupKFold balances counts of labelled candidate rows and keeps whole
trips together. Existing fold assignments are validated and reused. There is no
shuffle; the recorded seed controls the estimator rather than the fold splitter.

The code depends on the M1 preprocessing PR (#2). Until that PR is merged, this
model branch includes its commits; reviewers should merge the preprocessing
dependency first so the GB PR shows only model-specific changes.

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
