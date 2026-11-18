# PathSnap: data, models and optimisation (Person A contribution)

Draft section text for the consolidated PDF (Deliverables sections 2.2, 3 and 4).
Every number below is copied from a committed CSV/JSON file named next to it.
Partition labels: **train** = 80 trips (fitting and five-fold trip-grouped CV),
**student_test** = 10 held-out trips (evaluation only, never fitted). The evaluator
archive was not opened, listed or used for anything in this document.


## 2.2 Dataset understanding and exploratory data analysis

### Preprocessing (how the data was prepared)

Source: `src/data/` (M1 pipeline, released as `m1-preprocessed-2026-11-18`,
manifest SHA-256 `fff85ca4…f6bd9`), settings frozen in `configs/data.yaml`.

1. Each trip's raw `.track` (lon, lat, t), `.nodes`, `.arcs` (directed) and `.route`
   are parsed strictly; malformed non-empty rows raise errors rather than being skipped.
2. Coordinates are projected to metres with one WGS84 azimuthal-equidistant
   projection centred on each trip's road network. Measured distortion on 5,760 node
   pairs: median absolute error 1.9e-5 m, maximum 0.12 m, maximum relative error
   6.1 ppm (`results/eda/projection_errors.csv`).
3. All raw GPS points are kept (no smoothing). dt, speed and heading are derived
   between consecutive points. Gaps longer than 30 s, non-positive dt and stationary
   steps (< 1 m) disable the movement features they would corrupt.
4. Candidate generation: every directed arc within 100 m of a valid point,
   ordered by distance, at most 8 per point. These values were selected on **train
   only** from a 3×3 radius/K sweep. The rule was the smallest mean table within
   0.5 points of the best trusted-reference recall
   (`results/eda/candidate_sweep.csv`, `selection.json`). R = 100 m and K = 8 give
   93.7% recall with 6.5 candidates per point, against 94.0% recall with 10.8
   candidates for K = 16.
5. Labels: the reference route is aligned to the GPS points to obtain a reference
   arc per point (`align_status` ok / ambiguous / unaligned). Candidate label = 1 if
   it is the aligned reference arc and 0 otherwise. Labels are **null** for points
   whose alignment is ambiguous or unaligned; these rows are excluded from fitting
   and from candidate metrics, never turned into negatives. The reference is never
   inserted into the candidate set.

### Patterns observed (measured)

| Quantity | train | student_test | Source |
|---|---:|---:|---|
| Trips / GPS observations | 80 / 208,417 | 10 / 19,160 | `label_and_sampling_summary.csv` |
| Candidate rows (all / labelled) | 1,358,450 / 1,070,140 | 130,919 / 106,730 | same |
| Positive rate among labelled candidates | 14.6% | 15.0% | same |
| Trusted observations (alignment ok) | 167,184 (80.2% of valid) | 16,073 (83.9%) | `coverage_by_partition.csv` |
| Reference arc not among candidates (trusted) | 10,503 (6.28%) | 48 (0.30%) | same |
| Valid observations with zero candidates | 1,417 | 0 | same |
| Median sampling interval / gaps > 30 s | 1 s / 133 | 1 s / 2 | `inventory.csv` |
| Largest single time gap | 153,064 s (trip 00000090) | 210 s | `label_and_sampling_summary.csv` |
| Trips whose reference route has a connectivity break | 10 | 1 | `inventory.csv` |
| Directed arcs / weakly connected components (sum) | 3,876,904 / 3,121 | 185,041 / 227 | `graph_stats.csv` |
| Invalid coordinates / timestamps / duplicate points | 0 / 0 / 0 | 0 / 0 / 0 | `inventory.csv` |

Observations:

* The task is imbalanced, with about one positive per 6–7 candidates. Accuracy alone
  is therefore misleading, and positive-class and macro metrics are reported.
* **Candidate coverage limits every model.** 6.3% of trusted train observations
  cannot be matched correctly by any classifier, because the reference arc was not
  generated. This upper bound is reported as `top1_upper_bound`. Train fold 4
  contains most of these misses (9,060), which explains why every model's fold-4
  Top-1 is lowest (`fold_stability.png`).
* Missing values are rare and structural, not random (`missing_values.csv`,
  `figures/eda/missing_values.png`). heading_diff_deg is missing on 0.78% of train
  candidate rows (stationary or first points); speed_mps and prev_dist_m on 0.12%
  (first point of a trip or after a gap); next_dist_m on 0.12% (last point).
  Missingness rates are similar for positive and negative candidates. Missing-value
  indicators are still kept, because "no heading available" changes how much the
  distance can be trusted.
* By class on train (`train_features_by_label.csv`, `figures/eda/features_by_label.png`),
  positives are much closer to the GPS point: median dist_m 4.75 m against 25.25 m.
  They are better aligned with the direction of travel (median heading difference
  23.7° against 100.6°) and lie on longer arcs (median 120 m against 54 m). Only
  57.4% of positives are the nearest candidate (cand_rank 0), so "pick the nearest
  arc" is a weak rule and the other features matter. prev_dist_m and next_dist_m
  behave like dist_m (median 5.3 m against 26.7 m), because a correct arc is
  usually also close to the neighbouring points.

### Features and why

The 13 ordered features (`src/common/schema.py`) are all physical and contain
no identifiers or labels: dist_m, cand_rank, dist_minus_min, heading_diff_deg,
speed_mps, arc_len_m, proj_frac, n_candidates, is_oneway, source_degree,
target_degree, prev_dist_m and next_dist_m. Distance and rank describe geometric
closeness. Heading difference uses the direction of travel to separate the two
directed twins of a road. Speed, arc length, one-way flag and node degrees describe
road and motion context, for example short connector arcs near junctions.
Neighbouring-point distances add short temporal context. `next_dist_m` needs the
next GPS point, which fits the whole-trip contract but is not a streaming feature.

### EDA figures (all generated from data, each titled with its partition)

`results/figures/eda/`: `trajectory_network.png` (GPS points, directed road network
and reference route for the median-length trip of each partition: train 00000095,
student_test 00000076), `missing_values.png`, `sampling_and_speed.png`,
`features_by_label.png` (train only), `inventory.png`, `quality.png`,
`graph_projection.png`, `alignment_coverage.png` and `candidate_sweep.png` (train
only). Person B produces the Folium maps.

## 3. Model experimentation and comparison

All five models use the same 13 ordered features, the same canonical five folds
(`configs/cv_folds.csv`, SHA-256 `17db171e…`), fit only on trusted labelled train
rows (null labels excluded), fold-fitted preprocessing, and the same metric code
(`src/evaluation/metrics.py`). Every saved fold and student-test row was recomputed
from the published score exports (`results/validation/*_audit.json`).

### 3.1 Overall comparison (`results/models/comparison/five_model/`)

| Model | Algorithm Family | Key Configuration | CV Macro F1-score (mean of 5 folds) | Top-1 Accuracy (student-test observations) | Route continuity (student-test, classifier-only) | Student-Test Macro F1-score | Log Loss (student-test) | Inference Time (µs per candidate, student-test) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear and probabilistic | C=1.0, class_weight=None, max_iter=1000 | 0.8672 ± 0.0331 | 0.8279 | 0.7446 | 0.8895 | 0.1526 | 0.71 |
| Decision Tree | Tree-based | max_depth=12, min_samples_leaf=50, class_weight=None | 0.8759 ± 0.0398 | 0.8316 | 0.7403 | 0.8983 | 0.1351 | 0.49 |
| Random Forest | Ensemble | n_estimators=80, max_depth=24, min_samples_leaf=2, max_features=0.5, class_weight=balanced | 0.8818 ± 0.0262 | 0.8377 | 0.7458 | 0.8946 | 0.1471 | 4.48 |
| HistGradientBoosting | Boosting | learning_rate=0.1, max_iter=100, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0 | 0.8861 ± 0.0312 | 0.8357 | 0.7545 | 0.9039 | 0.1180 | 18.19 |
| KNN | Instance-based | n_neighbors=5, weights=distance, metric=minkowski, p=2, algorithm=kd_tree | 0.8579 ± 0.0361 | 0.8102 | 0.7152 | 0.8848 | 0.7103 | 1872.41 |

CV macro-F1 is the mean ± std over the 5 validation folds of the 80 train trips.
The other columns are measured on the 10 student_test trips. Route continuity is
classifier-only: argmax per observation, then the share of consecutive distinct
arcs that are joined in the directed graph, using Person B's `route_continuity`
(`classifier_only_continuity.csv`). It includes train-OOF values (GB 0.794, RF
0.790, LR 0.778, DT 0.759, KNN 0.734).

The inference-time column is not fully comparable across rows:
* GB's 18.19 µs comes from its original dedicated runner, which used batched
  4-thread prediction.
* LR, DT, RF and KNN were timed by the shared runner.
* KNN's 1,872 µs per row was measured while the optimisation search and a refit
  shared the CPU. It is inflated, but KNN is still 2–3 orders of magnitude slower
  than the trees: about 2 ms per kd-tree query, uncontended, in
  `results/validation/knn_query_benchmark.json`.

**Selection metric (fixed before the search): mean train-CV macro-F1, with CV
log loss and CV Top-1 as supporting criteria.** Student-test is reported, not used
to choose. HistGradientBoosting has the highest CV macro-F1 (0.8861), the lowest CV
log loss (0.133, so the best probabilities for HMM emissions) and the highest CV
Top-1 (0.768) and train-OOF continuity. **GB is selected.** RF is close on CV
macro-F1 (0.8818) and has the best student-test Top-1 (0.8377 against 0.8357), but
it has worse log loss and its class-balanced scores are not calibrated.

### 3.2 Per-model notes (connected to the measured results)

* **Logistic Regression (linear).** Fast (0.71 µs per row) and well calibrated
  (CV log loss 0.168). Train and validation macro-F1 are nearly equal (0.870 and
  0.867), so it **underfits**: it cannot model interactions such as "close but
  heading-misaligned". Student-test: macro-F1 0.8895, Top-1 0.8279, log loss 0.153.
* **Decision Tree (max_depth 12, min_samples_leaf 50).** Captures interactions
  but its probabilities are piecewise-constant: CV log loss 0.191, the worst of the
  tree models, with the largest fold variability (±0.040). Train macro-F1 0.894
  against validation 0.876 shows mild overfitting. Student-test: macro-F1 0.8983,
  Top-1 0.8316, log loss 0.135.
* **Random Forest (80 trees, depth 24, balanced classes).** The largest
  train/validation gap (macro-F1 0.940 against 0.882, log loss 0.081 against 0.172)
  shows clear overfitting to training trips. class_weight='balanced' raises
  positive recall (0.865 CV) at the cost of precision (0.745). Feature importance is
  led by dist_minus_min (0.34), cand_rank (0.22) and heading_diff_deg (0.11).
  Student-test: macro-F1 0.8946, Top-1 0.8377, log loss 0.147.
* **Histogram Gradient Boosting (100 iterations, 15 leaves, lr 0.1).** Best CV
  macro-F1 and log loss and the lowest fold std among the strong models. Its
  original runner did not record train-split metrics. The tuned refit with the
  shared runner reports train macro-F1 0.906 against validation 0.886 (see
  `results/models/gradient_boosting_tuned/cv_metrics.csv`), a modest gap.
  Student-test: macro-F1 0.9039, Top-1 0.8357, log loss 0.118.
* **KNN (k=5, distance-weighted, kd-tree, scaled features).** Competitive on
  easier folds (fold 3 macro-F1 0.906) but weak on fold 4 (0.809). It has the
  lowest CV macro-F1 (0.858) and very poor log loss (0.71 on student-test): votes
  of 5 neighbours often give probabilities of exactly 0 or 1. Its train-split
  metrics are near-perfect by construction, because each training row is its own
  neighbour (measured on a fixed 50,000-row sample per fold), so they show
  memorisation rather than generalisation. Student-test: macro-F1 0.8848, Top-1
  0.8102. It is also the slowest model to query.

Figures in `results/models/comparison/five_model/`: `cv_macro_f1.png`,
`student_test_top1_and_macro_f1.png`, `log_loss.png`,
`student_test_confusion_matrices.png`, `fold_stability.png` and `timing.png`.
None of these models trains in epochs, so no loss curves are drawn.


## 4. Bayesian hyperparameter optimisation

Model: histogram gradient boosting, the best of the five baselines on train CV
macro-F1 (section 3). Code: `src/optimization/bayes_search.py`. Evidence:
`results/optimization/gradient_boosting_tpe_deadline/` (`trials.csv`,
`summary.json`, `search_history.png`).

**Honest budget statement.** The planned search (`configs/optimisation.yaml`) had
20 trials: 8 random startup trials and 12 adaptive ones, with max_iter up to 800.
Because of the submission deadline, the only search actually run used the reduced
`configs/optimisation_deadline.yaml`: **5 trials (3 seeded random startup + 2
adaptive TPE)**, cheaper ranges and 173 s of total trial time. This is a minimal
search. Its evidence about the hyperparameter landscape is limited.

1. **Hyperparameters and ranges:** learning_rate log-uniform [0.05, 0.3];
   max_iter {50, 75, 100, 125, 150}; max_leaf_nodes log-int [15, 63];
   min_samples_leaf log-int [20, 400]; l2_regularization log-uniform [0.001, 10].
   max_bins = 63, class_weight = None and early_stopping = False are fixed. Early
   stopping stays off so that no random row-level holdout crosses trip boundaries.
2. **Objective:** maximise the mean validation candidate macro-F1 (threshold 0.5)
   over the five canonical trip-grouped folds (`configs/cv_folds.csv`, SHA-256
   `17db171e…`). Each fold is fitted only on the labelled rows of the other four
   folds of the 80 train trips. student_test is never read by the search.
3. **Method:** Optuna 4.5.0 TPESampler (multivariate, seed 2026). Trials 0–2 were
   random startup and trials 3–4 were proposed by TPE from the earlier results.
   A median pruner stops a trial after fold 2 or later if it is below the median of
   earlier trials at the same fold. Trial 4 was pruned after 3 folds; its recorded
   value is its last fold score, not a 5-fold mean.
4. **Trials:** 5 (4 complete, 1 pruned, 0 failed).
5. **Best values (trial 0, a startup trial):** learning_rate 0.0741, max_iter 100,
   max_leaf_nodes 61, min_samples_leaf 26, l2_regularization 0.0826, max_bins 63.
   Mean CV macro-F1 0.88612.
6. **Did tuning help?** No measurable improvement on the objective. The untuned
   baseline's mean CV macro-F1 is 0.88615 (`results/models/gradient_boosting`),
   and the best searched configuration reaches 0.88612. The adaptive trial 3
   (0.88548) moved toward trial 0's region (61 leaves, small min_samples_leaf)
   but did not beat it. The best value is the maximum over trials scored on the
   same folds, so it is optimistically biased. It is not a nested-CV estimate.

**Tuned versus untuned (same folds, metrics and partitions; `results/models/comparison/tuned_vs_untuned/`):**

| Metric | Untuned GB | Tuned GB (trial 0) |
|---|---:|---:|
| CV macro-F1 (mean of 5 folds) | 0.88615 | 0.88612 |
| CV log loss | 0.1329 | 0.1327 |
| CV observation Top-1 | 0.7680 | 0.7673 |
| Student-test Top-1 | 0.8357 | 0.8374 |
| Student-test macro-F1 | 0.9039 | 0.9038 |
| Student-test log loss | 0.1180 | 0.1161 |
| Student-test inference (µs per row) | 18.19 (original runner) | 3.74 (shared runner) |

On the selection metric (train CV) the tuned configuration does not beat the
baseline. It is marginally better on CV log loss and on the held-out student-test
numbers. Those differences are within the fold-to-fold spread (CV std about 0.03),
and student-test was not used to choose. **The untuned GB therefore remains the
selected model.** The inference times come from different runners and should not
be read as a tuning effect.

