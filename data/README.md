# Model-ready preprocessing

Run commands from the repository root using Python 3.11 (validated with 3.11.9).
Install the isolated M1 dependencies without changing the M4-owned requirements:

```powershell
python -m venv .venv-m1
.venv-m1/Scripts/python -m pip install -r data/requirements-m1.txt
```

This workstation reused installed packages in a `--system-site-packages` virtual
environment, adding missing pyproj, shapely and folium. Actual versions are in
`results/eda/runtime.json`. The earlier `.venv` attempt is unused and ignored.

Point `--raw-root` or `PATHSNAP_RAW_ROOT` at the supplied `Student folder`. Data is
read in place; no copies or symlinks into `data/raw` are needed. Expected layout:

```text
<external-root>/train/<8-digit-id>/<id>.track|nodes|arcs|route
<external-root>/student_test/<8-digit-id>/<id>.track|nodes|arcs|route
```

IDs count nonempty file records, starting at zero. Blank physical lines are
counted in the inventory; malformed nonempty rows raise with physical line number.
Track = lon, lat, t (seconds); nodes = lon, lat; arcs = directed source, target node
IDs; route = ordered arc IDs. Repeated route occurrences remain distinct. Empty
track is an error; empty routes/graphs are supported and create no trusted labels.
Eight-digit trip IDs are strings, never globally merged local arc/node IDs.

```powershell
# Frozen config already selected using train only; regenerate the final tables.
.venv-m1/Scripts/python -m src.data.build_dataset --raw-root '../Student folder'
.venv-m1/Scripts/python -m src.data.eda audit --raw-root '../Student folder'
.venv-m1/Scripts/python -m src.data.validate_dataset
.venv-m1/Scripts/python -m pytest -q
```

Default output `data/processed` is local and ignored. Six typed Parquet files:
`train_observations`, `train_candidates`, `train_alignment` and the corresponding
three `student_test` files, each ending `.parquet`. A final atomic `manifest.json`
marks success. Failed builds must not be consumed; no resume cache is used.
Existing outputs require `--overwrite`; a failed overwrite removes the success
marker. Concurrent builds must use distinct output roots. Debugging supports
`--partitions train --trips 00000001`; trip IDs must be discovered, not assumed.
Evaluator partitions are rejected. Help is available on all CLIs.

For an independent deterministic rebuild:

```powershell
.venv-m1/Scripts/python -m src.data.build_dataset --raw-root '../Student folder' --output-root tmp/rebuild
.venv-m1/Scripts/python -m src.data.validate_dataset --compare-root tmp/rebuild --evidence results/eda/reproducibility.json
```

To deliberately repeat train-only configuration selection, use
`python -m src.data.eda select --raw-root '../Student folder'`. This overwrites
the frozen config and sweep evidence, so all processed tables must then be rebuilt.
Selection uses the smallest average table within 0.5 percentage points of the
highest micro recall over trusted valid observations. It queries max R once and
derives nested radius/K sets exactly. The selection report explains shared timing.
No student_test evidence is used to select configuration.

`src/common/schema.py` is the single versioned column/dtype/feature contract. C1
keeps one row per raw observation with validity, dt, speed, heading, gaps, and
candidate counts. Additional diagnostics are explicit and not model features.
C2 contains directed candidate keys, exact segment projections, physical features,
and offline nullable Int8 labels. C3 has one ordered reference-alignment row per
observation: arc ID, occurrence position, distance, status and membership in C2.
CSV evidence must be read with `dtype={'trip_id':'string'}` to preserve leading zeros.

Projection uses one network-centred WGS84 azimuthal equidistant transformer per
trip, shared by graph and observations. Input lon/lat degrees, output x/y metres;
bearings are clockwise degrees from true north. Circular longitude mean handles
dateline wrapping; networks extending >1500 km from centre are rejected. Error
samples are recorded, not an assertion of universal distortion-free distances.

All raw observations remain. Nonfinite/out-of-range coordinates or nonfinite t
disable candidates. Nonpositive dt disables speed/heading but not spatial matches.
First point and invalid neighbors have undefined movement features. Stationary
headings are NaN. Large gaps disable speed/heading and neighboring-distance
features; gap flags are informational for M3. No smoothing, fitted scaling or
imputation occurs. Next-observation arc distance requires whole-trip inference.

Candidate generation uses a full-segment STRtree, strict metric radius, then
distance and arc-ID ordering and a K limit. Reverse arcs are distinct. Zero-length
arcs project to their source with fraction zero and undefined bearing. No nearest
fallback and no forced reference inclusion. Nodes/arcs retain original row IDs.

Offline alignment is a sparse monotone DAG over all route occurrences within a
separate fixed radius. Repeated observations may use the same route occurrence;
positions never decrease. Forward/backward path-score margins flag alternatives.
Reference breaks and gap observations are conservatively ambiguous; outliers and
skipped observations are unaligned. Alignment is derived supervision, not verified
ground truth; it cannot prove its own correctness through candidate recall.
No route-dependent function is called by `build_features(with_labels=False)`.

Trusted labels: for C3 status `ok`, label=1 iff candidate arc==reference arc, else
0. A reference miss has zero positives. For ambiguous/unaligned observations,
every candidate has a null label. Keep misses in performance denominators; do
not evaluate only reference hits. Candidate miss counts include zero candidates.
`coverage_by_partition.csv` gives micro denominators and `coverage.csv` per trip.

```python
from src.data.model_data import load_model_data
from src.common.schema import FEATURES, SCHEMA_VERSION

X, y, groups, keys = load_model_data('data/processed', partition='train')
assert list(X.columns) == FEATURES
# groups are trip_id: use GroupKFold within train, never random candidate splits.
# M2: fit imputation/scaling inside each training fold; explicit NaNs remain in X.
# The separate student_test partition is for held-out development evaluation.
```

This convenience loader materializes one partition; use Parquet row groups for
bounded-memory consumers. M2 should score *all* candidate rows during evaluation,
then join to C1/C3, preserving no-candidate observations and reference misses.
M3 consumes C1/C2 plus the directed RoadGraph; M4 imports the same preprocessing
functions directly. See `report/m1_handoff.md` for the inference example and
consumer review requests. Large tables are regenerated, never committed.
