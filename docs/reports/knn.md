# KNN evaluation status and handoff

## Final measured results

The full five-fold run and the student-test evaluation are complete
(`results/models/knn/`, audited in `results/validation/knn_audit.json`).

| Code fold | Candidate accuracy | Macro-F1 | Observation Top-1 |
| --- | ---: | ---: | ---: |
| 0 | 92.20% | 84.60% | 74.76% |
| 1 | 91.62% | 85.33% | 76.40% |
| 2 | 94.04% | 87.56% | 79.63% |
| 3 | 95.40% | 90.60% | 83.49% |
| 4 | 91.49% | 80.87% | 53.59% |
| **Mean** | **92.95%** | **85.79%** | **73.58%** |

Student-test (10 trips): candidate accuracy 94.18%, macro-F1 0.8848, Top-1 81.02%,
log loss 0.7103. KNN has the lowest CV macro-F1 of the five models; distance-weighted
votes of five neighbours give coarse probabilities, hence the high log loss. See the
five-model comparison in `results/models/comparison/five_model/`.

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

## Runner verification

The committed runner and repository tests pass. The measured results above come
from the completed evaluation run, not from the test suite.
