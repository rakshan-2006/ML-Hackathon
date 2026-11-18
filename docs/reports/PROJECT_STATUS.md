# PathSnap technical progress report

## Problem and data

PathSnap maps noisy GPS observations to directed road segments, then aims to
reconstruct a connected vehicle route. The supplied student data has 80 training
trips (208,417 GPS observations) and ten student-test trips (19,160 observations).
Each trip supplies track coordinates/timestamps, road nodes, directed arcs, and
a reference route. Reference routes require ordered alignment and are not direct
one-label-per-GPS files. Derived labels are approximate rather than independently
verified physical ground truth.

Preprocessing generates up to eight directed candidates within 100 metres and
13 physical/context features. All 1,070,140 trusted labelled training candidate
rows are available for fitting. Ambiguous labels remain excluded from fitting.
The same persisted five trip-grouped folds are used for comparable experiments.
Learned imputation and scaling are fitted inside each training fold.

## Verified student-test baseline results

| Model | Candidate accuracy | Observation Top-1 |
| --- | ---: | ---: |
| Logistic Regression | 94.44% | 82.79% |
| Decision Tree | 94.88% | 83.16% |
| Histogram Gradient Boosting | 95.16% | 83.57% |
| Random Forest | 94.24% | 83.77% |
| KNN | Pending | Pending |

Candidate accuracy measures binary candidate classification. Top-1 chooses one
road per trusted valid observation; missing reference candidates and zero-candidate
observations count as failures. These are classifier metrics, not route accuracy.

Five model implementations are integrated. Four have completed measured baseline
reports. The full KNN run is active; [KNN status](knn.md) records its preliminary
fold results and execution method. It has not yet completed student-test scoring.

## Implemented and remaining work

Completed: dataset preparation, candidate features, common model factory, frozen
trip folds, five classifier implementations, four baseline evaluations, saved
artifact workflows, API/UI foundation, and automated integration checks.

Pending: final KNN evaluation, Bayesian hyperparameter optimization, HMM/Viterbi
route reconstruction, classifier-versus-decoder comparison, real inference adapter,
and full two-laptop end-to-end verification. The API currently remains a stub
foundation. No finished route-reconstruction or deployment result is claimed.

## Evidence and limitations

Measured results live in results/models; preprocessing evidence lives in
results/eda. See [training documentation](../training.md),
[Random Forest validation](m2_random_forest.md), and
[Gradient Boosting report](gradient_boosting.md). Class imbalance makes raw
candidate accuracy insufficient; compare Top-1, macro-F1 and probability quality.
No evaluator-test data was used for training or selecting model settings.

This is an honest progress report, not a declaration that all deliverables are
finished. Replace pending sections with measured outcomes after their checks pass.
