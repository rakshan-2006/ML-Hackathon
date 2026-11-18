# KNN evaluation status and handoff

## Status at handoff

The full evaluation is running. Four of five canonical trip-grouped validation
folds have completed. Code fold 4 (the fifth fold) is active. The process retains
the completed scores in memory. Do not start a duplicate run or stop this one.

Final aggregate CV metrics and student-test results are pending. No final KNN
accuracy or completed end-to-end system result is claimed here.

## Observed preliminary results

These values are rounded readings from completed fold logs, not final averages.

| Code fold | Candidate accuracy | Macro-F1 | Observation Top-1 |
| --- | ---: | ---: | ---: |
| 0 | 92.20% | 84.60% | 74.76% |
| 1 | 91.62% | 85.33% | 76.40% |
| 2 | 94.04% | 87.56% | 79.63% |
| 3 | 95.40% | 90.60% | 83.49% |
| 4 | Pending | Pending | Pending |

All held-out candidate rows are queried normally. Fit data includes all trusted
labelled rows from the other training trips in each fold. The final fit will use
all 1,070,140 eligible labelled rows across the 80 training trips, followed by
all-candidate scoring on the ten student-test trips.

## Reproduction

```powershell
python -m src.training.train_knn --jobs 8
```

This command uses the shared trainer and frozen folds, queries in batches of
10,000, and prints progress. For training-only diagnostics, distance-weighted
KNN assigns exactly the stored label when all zero-distance training matches
share that label. Mixed-label duplicates use normal queries; 256 ordinary
queries per fold verify the shortcut. Uniform weighting uses normal queries.
Validation and student-test predictions never use this shortcut.

Outputs after successful completion:

- results/models/knn/cv_metrics.csv
- results/models/knn/student_test_metrics.csv
- results/models/knn/model_card.json
- artifacts/knn/pipeline.joblib
- artifacts/knn/oof_probabilities.parquet
- artifacts/knn/student_test_probabilities.parquet

## Person A can continue

Continue HMM/Viterbi route reconstruction and model/API integration on a separate
branch or worktree. Preserve the shared 13-feature schema, candidate-to-arc keys,
canonical fold membership and reference-read boundary. Use existing verified
model outputs to develop the interface. Do not restart KNN or change this run's
inputs, settings or output files. KNN final metrics will be added once verified.

## Runner verification

The committed runner and repository suite passed 81 tests. The complete KNN
accuracy evaluation remains a separate long-running computation; its pending
results are not inferred from the test count.
