# PathSnap: comparison of the three implemented models

Layout note: the numeric comparisons below describe the original baseline runs.
PR #6 subsequently standardized the shared factory/metric interfaces without
changing their measured values. Current runners all read `configs/cv_folds.csv`;
historical model cards retain their original CSV/JSON file paths and hashes.

**Recommendation: Histogram Gradient Boosting is the strongest current baseline.**
It leads Logistic Regression and Decision Tree on five-fold validation Macro-F1,
observation Top-1, and log loss, and it also leads on all three student-test metrics.
Its main tradeoff is slower recorded prediction time. This is a recommendation
among the three completed baselines, not a final selection across all five planned
models or a claim about HMM-decoded route quality.

## 1. Models and comparison setup

| Model | Implemented estimator | Configuration | Preprocessing |
|---|---|---|---|
| Logistic Regression | `LogisticRegression` | C=1, lbfgs, max_iter=1000, class_weight=None | Fold-fitted median imputation with missingness indicators, followed by scaling |
| Decision Tree | `DecisionTreeClassifier` | max_depth=12, min_samples_leaf=50, Gini criterion | Fold-fitted median imputation with missingness indicators; no scaling |
| Gradient Boosting | `HistGradientBoostingClassifier` | 100 iterations, learning_rate=0.1, 15 leaves, min_samples_leaf=30, L2=1, 127 bins | Native NaN handling; no scaling or fitted imputation; early stopping disabled |

The Gradient Boosting implementation is **histogram-based gradient boosting**, not
the classic sklearn `GradientBoostingClassifier`.

Comparison checks confirmed that all three runs used:

- The same released preprocessing dataset and manifest fingerprint.
- The same schema and exact order of 13 physical features.
- 1,070,140 non-null labelled training candidates from 80 train trips.
- The same trip-to-fold assignments for five-fold grouped CV.
- The same student-test partition: 106,730 labelled candidate rows from 10 trips.
- All 130,919 student-test candidates for observation-level ranking.
- A 0.5 probability threshold for candidate-level classification metrics.
- Trusted valid observations as the Top-1 denominator, counting reference misses
  and zero-candidate observations as failures; probability ties use candidate rank.

Logistic Regression/Decision Tree store their folds in CSV, whereas Gradient
Boosting stores them in JSON. Their file hashes naturally differ, but the actual
80-trip mappings were compared and are identical. The runners use different column
names for some metrics; these were normalized before comparison.

## 2. Main results

Higher Macro-F1 and Top-1 are better. **Lower log loss is better.**

| Metric | Logistic Regression | Decision Tree | Gradient Boosting |
|---|---:|---:|---:|
| Mean five-fold CV Macro-F1 | 0.867249 | 0.875897 | **0.886149** |
| Mean five-fold CV Top-1 | 75.920% | 75.999% | **76.797%** |
| Mean five-fold CV log loss | 0.167834 | 0.190576 | **0.132879** |
| Student-test Macro-F1 | 0.889451 | 0.898298 | **0.903897** |
| Student-test Top-1 | 82.791% | 83.164% | **83.569%** |
| Student-test log loss | 0.152641 | 0.135127 | **0.117970** |
| Student-test candidate accuracy | 94.444% | 94.883% | **95.162%** |
| Student-test positive-class precision | 82.710% | 84.144% | **85.042%** |
| Student-test positive-class recall | 79.644% | 81.229% | **82.240%** |
| Correct student-test Top-1 observations / 16,073 | 13,307 | 13,367 | **13,432** |

CV entries are the **unweighted mean of the five validation folds**, recomputed
from the saved individual fold rows. They are not pooled OOF metrics. In
particular, Gradient Boosting's pooled OOF Macro-F1 (0.886669) and pooled OOF Top-1
(76.422%) must not be substituted into the mean-CV comparison above.

The normalized results are saved in
[comparison_metrics.csv](../../results/models/comparison/logistic_regression_vs_decision_tree_vs_gradient_boosting/comparison_metrics.csv).

## 3. Why Gradient Boosting currently wins

**It wins on training-fold validation, before relying on student-test results.**
Its mean CV Macro-F1 is 0.010252 higher than Decision Tree and 0.018900 higher than
Logistic Regression. Its CV log loss is also lower than both alternatives. The
student-test results support the same ranking on these primary measures.

**It selects more correct road candidates.** Student-test Top-1 is 0.404 percentage
points higher than Decision Tree and 0.778 points higher than Logistic Regression.
On the same 16,073 trusted observations, that means 65 and 125 additional correct
selections, respectively. The improvement is useful but modest; it is not a
dramatic increase in observation-level accuracy.

**Its probability predictions perform better under log loss.** This matters because
the later HMM decoder will use candidate probabilities as evidence. However,
lower log loss alone does not prove calibration or guarantee better decoded
routes; those require separate calibration and decoder evaluation.

Gradient Boosting leads both other models in **each of the five validation folds**
on Macro-F1, Top-1 and log loss, not just on their averages. This supports the
current recommendation but is not a formal statistical-significance result.

## 4. What the other models still offer

### Logistic Regression: fastest recorded fitting and a useful linear baseline

Logistic Regression gives the lowest recorded CV fit time and remains useful for
understanding what a linear relationship among the supplied features can achieve.
Its CV log loss is better than Decision Tree's, despite lower Macro-F1 and Top-1.
It is therefore a useful alternative when fit cost, simplicity, or CV probability
quality is important. It is not the current accuracy leader.

### Decision Tree: strong classifications and fast recorded inference

Decision Tree improves on Logistic Regression in CV/student-test Macro-F1 and
student-test Top-1, and has the fastest recorded student-test prediction time.
It can represent nonlinear rules and interactions through explicit splits.

Its mean CV log loss, however, is **worse than Logistic Regression's**. A tree can
rank or classify candidates reasonably while giving poor confidence to some
errors. Coarse leaf probabilities are one possible explanation; a calibration
analysis would be needed to establish the cause. The worse log loss should not
be concealed by reporting candidate accuracy alone.

Overall: Decision Tree is generally second on classification/ranking measures;
Logistic Regression is preferable to it on mean CV log loss and recorded fit cost.
There is no universal second-place model across every criterion.

## 5. Recorded computational cost

| Timing | Logistic Regression | Decision Tree | Gradient Boosting |
|---|---:|---:|---:|
| Sum of five CV fit times | **24.254 s** | 121.958 s | 113.823 s |
| Final fit on all labelled train rows | **6.784 s** | 32.514 s | 35.314 s |
| Prediction on all 130,919 student-test candidates | 0.093 s | **0.064 s** | 2.381 s |

These are saved run timings, not a newly controlled benchmark. The runners differ
in batching, thread controls and execution conditions; Gradient Boosting records
four threads. Treat exact speed ratios cautiously. These timings exclude the full
application's data loading, candidate generation, HMM decoding and API/UI overhead.

**Practical tradeoff:** Gradient Boosting has the best measured quality but costs
more prediction time. Its recorded 2.381 seconds for this candidate table does
not by itself establish whether the team's end-to-end latency requirement is met.

## 6. Fold variability and dataset limitations

| Validation measure | Logistic Regression | Decision Tree | Gradient Boosting |
|---|---:|---:|---:|
| Macro-F1: fold mean +/- sample SD | 0.867249 +/- 0.033092 | 0.875897 +/- 0.039792 | **0.886149 +/- 0.031155** |
| Top-1: fold mean +/- sample SD | 75.920% +/- 12.168 pp | 75.999% +/- 12.185 pp | **76.797% +/- 12.271 pp** |

All standard deviations were recomputed with `ddof=1` for consistency. SD describes
variation across these folds; it is not a confidence interval for future accuracy.
Gradient Boosting has the lowest Macro-F1 SD of the three, but Top-1 remains highly
variable for every model.

For example, Gradient Boosting's fold Top-1 ranges from 55.775% to 86.957%. Fold 4
has a reference-availability ceiling of approximately 74.007% because many trusted
reference arcs are absent from its candidates. Candidate coverage therefore limits
performance on that fold, although it does not explain every prediction error.

All metrics use M1's **derived alignment labels**, not independently verified
physical ground truth. Trusted alignment covers approximately 80.22% of train
observations and 83.89% of student-test observations. Results on this subset must
not be described as accuracy over every raw GPS observation. No model here has
been compared on final HMM continuity, complete route reconstruction, evaluator
trips, or an independently verified label set.

## 7. Recommendation for the team

Use **Histogram Gradient Boosting as the leading baseline for subsequent tuning
and decoder experiments**. Keep Logistic Regression as the simple/fast baseline
and Decision Tree as the interpretable nonlinear baseline.

Before selecting the team's final model:

1. Complete KNN and Random Forest using the same data, exact fold mapping and
   metric denominators. This comparison currently covers three of five models.
2. Tune promising models using train-only grouped CV, retaining the trial records.
   Do not repeatedly choose hyperparameters from student-test results.
3. Check probability calibration and compare downstream HMM results using OOF
   train probabilities. Do not assume the best standalone classifier produces
   the best complete route.
4. Measure end-to-end latency under the same deployment conditions if inference
   speed becomes a selection constraint.

No final shared predictor was replaced and no new model was trained for this
comparison. The recommendation is based on the completed, saved baseline runs.

## 8. Evidence and verification

Source snapshot: `origin/main` commit
`a0fb173eb2c8da3b3ce7f32e34997f2e5db8bb7e`.
The local root checkout was older, so the analysis read the committed files from
this fetched remote revision rather than assuming leftover local files were current.

Original metric tables, cards and timings:

- [Logistic Regression evidence](https://github.com/rakshan-2006/ML-Hackathon/tree/a0fb173eb2c8da3b3ce7f32e34997f2e5db8bb7e/results/models/logistic_regression)
- [Decision Tree evidence](https://github.com/rakshan-2006/ML-Hackathon/tree/a0fb173eb2c8da3b3ce7f32e34997f2e5db8bb7e/results/models/decision_tree)
- [Gradient Boosting evidence](https://github.com/rakshan-2006/ML-Hackathon/tree/a0fb173eb2c8da3b3ce7f32e34997f2e5db8bb7e/results/models/gradient_boosting)

New comparison evidence:

- [Normalized comparison table](../../results/models/comparison/logistic_regression_vs_decision_tree_vs_gradient_boosting/comparison_metrics.csv)
- [Per-model fold metrics and cards](../../results/models/)
- [Canonical trip-to-fold assignment](../../configs/cv_folds.csv)

Checks performed: exact 80-trip fold-map equality, matching dataset manifest,
matching schema and ordered features, CV means recomputed from fold rows and
checked against model cards, and consistent aggregation of mean CV versus held-out
metrics. No new tests, training runs, or statistical significance claims are implied.
