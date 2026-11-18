# PathSnap: data, models and optimisation (Person A contribution)

Draft section text for the consolidated PDF (Deliverables sections 2.2, 3 and 4).
Every number below is copied from a committed CSV/JSON file named next to it.
Partition labels: **train** = 80 trips (fitting and five-fold trip-grouped CV),
**student_test** = 10 held-out trips (evaluation only, never fitted). The evaluator
archive was not opened, listed or used for anything in this document.

<!-- KNN_AND_BO_PENDING: sections marked TODO are filled after the measured runs finish. -->

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

TODO (KNN pending)

## 4. Bayesian hyperparameter optimisation

TODO (search pending)
