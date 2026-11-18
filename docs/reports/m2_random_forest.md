# Random Forest candidate classifier

Random Forest implements the shared model contract in `src/models/random_forest.py`.
It uses the same 13 ordered features, `configs/cv_folds.csv`, training-only fitted
preprocessing and evaluation functions as the other models.

## Train and evaluate

```powershell
python -m src.training.train_models --model random_forest
python -m pytest tests/test_random_forest.py tests/test_project_layout.py
```

The compatibility command `python -m src.models.train_random_forest` invokes the
same trainer. Only the canonical five folds are accepted; the old `--folds 3`
workflow and independent output-directory options have been retired.

The pipeline converts the declared features to float64, fits median imputation
with missing-value indicators inside each training fold, and fits a Random Forest.
No scaling is applied. Classifier defaults are 80 trees, maximum depth 24,
minimum split size 4, minimum leaf size 2, `max_features=0.5`,
`class_weight='balanced'`, seed 2026, and four worker threads. Class weighting
changes score interpretation; the probabilities have not been calibrated.

The original contribution selected the tree settings using training-only
three-fold macro-F1. The integrated run freezes those settings and evaluates
them on the repository's existing five folds. This is model development evidence,
not a nested-CV estimate of the entire hyperparameter-selection procedure.
Student-test data is never used for fitting or selecting these settings.

## Current outputs and scoring

Small measured results go to `results/models/random_forest/`; fitted artifacts
and all-candidate probability exports go to ignored `artifacts/random_forest/`:

- `pipeline.joblib`: fitted preprocessing and classifier together.
- `oof_probabilities.parquet`: scores for all training candidates, each from a
  fold model that did not train on that trip.
- `student_test_probabilities.parquet`: all held-out candidate scores from the
  final model refitted on every trusted labelled training row.

`RandomForestPredictor(artifact_dir).score_candidates(candidates)` loads the
pipeline plus `model_card.json`, checks schema/feature compatibility, and returns
float64 class-1 scores in input order. Empty candidate frames return empty scores.
The original `random_forest.joblib` plus `features.json` bundle is also supported.
The adapter does not read reference routes. API/HMM integration remains separate.

Candidate metrics use non-null labels and threshold 0.5. Observation Top-1 uses
all trusted valid observations, including failures where the reference arc is
missing from candidates or there are no candidates. Ties use `cand_rank`.
Log loss, precision/recall, macro-F1, confusion matrices and timings are recorded
alongside accuracy. These are classifier results, not HMM or route-level results.

## Original run preserved as provenance

[The legacy directory](../../results/models/random_forest/legacy_run/) preserves
three original metadata files unchanged. Its model binary was not included in
the contribution. Its Top-1 value of 0.8388767550702028 uses only the 16,025
student-test observations whose reference arc was present among candidates;
it excludes 48 trusted misses. It must not be compared directly with the other
models' full trusted-observation denominator of 16,073. The original diagnostic
`missed_or_unavailable_denominator` also omitted zero-candidate observations
because it counted rows after a candidate-led join. It is historical output,
not a corrected coverage statistic.

The integrated trainer replaces that metric with the shared implementation and
never rewrites the historical run to imply it was trained or scored differently.
See [training documentation](../training.md) for dependencies, metrics and downloads.
