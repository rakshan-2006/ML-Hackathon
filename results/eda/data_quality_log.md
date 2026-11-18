# Data quality audit

Only train and student_test were opened. All nonempty raw counts match loaded
counts; strict parsers stop on malformed nonempty records (no skipped errors).
Inventory includes per-file presence, bytes, records, blank lines and nonfinite
cell counts. Malformed references = 0 denotes successful strict range validation.

Measured trips with reference connectivity breaks: 11. The draft reported
11; confirmed.

Metric definitions: invalid observations have nonfinite/out-of-range lon/lat or
nonfinite timestamp; nonpositive dt counts finite dt <=0; large gaps use the frozen
gap threshold; duplicate coordinates/observations count repeats beyond first;
stationary steps are below stationary_m; suspicious speeds exceed configured m/s.
First-point dt/speed/heading are intentionally undefined, not missing raw data.
Coordinate bounds are raw degree minima/maxima (not wrapped extent estimates).
Graph duplicates count endpoint rows beyond first; reverse count counts directed
arcs possessing a distinct opposite arc; components include isolated nodes.
Array memory sums top-level numpy buffers, excludes Python/index/runtime overhead;
graph timing includes loading/parsing/projection/components but not candidates.

Coverage: trusted = valid C1 observations whose C3 status is ok; misses include
zero-candidate observations. miss_rate_trusted = misses/trusted;
known_misses_over_all_valid = same known misses / all valid observations (not
an estimate of unknown-reference errors); alignment_coverage_valid = trusted/valid.
All aggregate rates are micro sums; per-trip evidence is also retained. Ambiguous
and unaligned candidate labels are nullable and must be excluded by M2. Alignment
is derived, not independently verified ground truth. No reference is forced into C2.

Projection pairs: at most 64 evenly spaced node rows, paired by cyclic shift 17,
per trip. WGS84 geodesic vs AEQD planar chord; no route or candidate dependence.
No projection retuning on student_test. See projection_errors.csv for every pair.
Local policy rejects >1500 km centre radius; error evidence covers supplied trips
only. No smoothing: preserve raw candidate locations and avoid a fitted transform.
30 s gap and 1 m stationary thresholds are declared physical heuristics; train
inventory and dt/heading figures expose their impact, not a claim of optimality.

Missing/invalid values, suspicious measurements, and structural route breaks are
kept separately in inventory. No manual label repairs were applied. EDA logs are
not a substitute for independent reference-label review.
