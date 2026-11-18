# Member 2: Random Forest map-matching candidate classifier

## Data source

This implementation consumes the official M1 release
`m1-preprocessed-2026-11-18` at commit `7f613e0`. It uses M1's
`src.data.model_data.load_model_data` loader and the ordered schema in
`src/common/schema.py`. The model development partition is the 80-trip
`train` partition. The 10-trip `student_test` partition is held out of
cross-validation and hyperparameter selection.

## Preprocessing

The pipeline fits `SimpleImputer(strategy="median")` inside grouped
cross-validation training folds. No scaling is used. NaN values remain explicit
in M1 data and are not imputed before the fold split. The official 13 features
are preserved in their declared order. Candidate keys and labels are never
included as features.

## Training design

`RandomForestClassifier` configurations are compared with three-fold
`GroupKFold` by `trip_id`. Every fold verifies that no trip appears in both
training and validation partitions. Candidate-level precision, recall,
F1, macro-F1, accuracy and confusion matrix values are measured. The final
pipeline is trained only after CV on all eligible, non-null-labelled training
candidates.

Student-test candidate and observation evaluation occurs only after the CV
configuration is selected. Candidate metrics are not presented as final route
accuracy. Observation top-1 is a separate denominator that can account for
reference misses and observations with no candidates according to M1 alignment
coverage.

## Execution

```powershell
python -m src.data.download_dataset
python -m src.data.validate_dataset
python -m src.models.train_random_forest
python -m pytest tests/test_random_forest.py
```

Actual configuration comparisons, CV metrics, training/inference timings,
feature importance and confusion matrices are written by the training command
to `experiments/random_forest/`. The measured full-data run is summarized below.

## Measured results

The selected configuration, chosen only by mean grouped-CV macro-F1, was:

- 80 estimators
- Maximum depth 24
- Minimum samples split 4
- Minimum samples leaf 2
- Maximum features 0.5
- `class_weight="balanced"`

Three-fold training-only CV mean metrics:

| Metric | Mean |
| --- | ---: |
| Precision | 0.7580673371776547 |
| Recall | 0.8588536603017266 |
| F1 | 0.805207198493196 |
| Macro-F1 | 0.8843373553189404 |

Final training used 1,070,140 eligible labelled candidates and took
620.2527945 seconds. Student-test holdout candidate metrics were:

| Metric | Result |
| --- | ---: |
| Precision | 0.7623504387131083 |
| Recall | 0.8946021840873635 |
| F1 | 0.8231983921906403 |
| Macro-F1 | 0.894362449748122 |
| Accuracy | 0.9423030075892439 |

The holdout confusion matrix was TN=86236, FP=4469, FN=1689, TP=14336.
The observation-level top-1 result over 16,025 eligible reference hits was
0.8388767550702028. It is reported separately from the 19,160-observation
denominator.

Student-test inference over 130,919 candidate rows took 1.2099231 seconds.

## Artifacts

The dedicated artifact directory is `models/random_forest_v1/`:

- `random_forest.joblib`: fitted imputer plus Random Forest pipeline.
- `features.json`: exact feature order and M1 schema version.
- `model_config.json`: hyperparameters and CV strategy.
- `model_card.json`: reproducibility metadata, CV results, holdout results,
  timings and feature importance.

The predictor adapter is `src/models/random_forest.py` and exposes
`score_candidates(cand_df)`, returning float64 positive-class probabilities
in input row order.

## Limitations

The RF candidate probabilities are classifier scores, not HMM route confidence.
The data release's labels are derived from offline alignment and remain
conservative in ambiguous cases. Candidate-level metrics must not be interpreted
as route reconstruction accuracy. The final M3/HMM integration still needs the
team's predictor calling convention and observation-level candidate joining.
