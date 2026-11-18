# Preprocessing handoff

## Integration review against main

The M1 branch incorporates `main` at `6a17f63` (M4 API/UI foundation, PR #1).
The API now reuses `src.common.schema.SCHEMA_VERSION` through the existing
`src.common.paths` import, rather than keeping an incompatible `1.0` constant.
M2 model cards and M3 decoder cards must declare `1.0.0-m1-provisional` for this
dataset release. API tests use the same contract. Existing stub behavior remains;
this change does not supply a trained model, decoder, or real inference adapter.

After integration, `python -m pytest -q` passed all 31 tests in the M1 Python 3.11
environment with FastAPI 0.115.12, HTTPX 0.28.1 and joblib 1.5.1 installed for the
API tests. This is not a claim of testing the entire UI or every pinned package
combination. A fresh `python -m src.data.validate_dataset --evidence
tmp/pr-review/validation.json` also passed all 90 released trips. The original
independent-rebuild and release-download evidence remains in `results/eda/`.

The root README links the ready-to-use dataset release. API and historical M1
reproducibility environments remain documented separately because Folium/pytest
pins differ. Shared requirements add the numpy/pandas dependencies imported by
the shared schema. Ignore rules preserve M4 exclusions while allowing M1's
versioned config, EDA evidence, and synthetic `.route` fixture; processed data
remains excluded from Git and is downloaded from the checksummed Release assets.

Scope narrowed by the user to preprocessing ready for model training. This branch
does not implement models, the production route decoder, API, UI, deployment or
the original full phase-report/map acceptance gate. No teammate approval is claimed.

The authoritative loading/regeneration commands and data semantics are in
`data/README.md`. Schema version and the exact feature order, dtypes, meanings and
units live in `src/common/schema.py`. `FEATURE_SPEC` is deliberately machine-readable.
`results/eda/build_manifest.json` records config, discovered trip lists, source
counts/hashes, output row counts, per-trip ordered semantic hashes, timing, runtime
and code revision/dirty state. Processed Parquet stays local in `data/processed`.

## M2: train on candidates without leaking trips

```python
from src.data.model_data import load_model_data
from src.common.schema import FEATURES

X, y, groups, keys = load_model_data('data/processed', 'train')
assert list(X.columns) == FEATURES
```

`X` has only the declared city-invariant physical features. Nullable target rows
are removed; `y` is binary int8. `groups` is the eight-character trip ID for
GroupKFold within train. Whole-trip inference is assumed: next-point distance to
the same candidate arc is included. NaNs mean physically undefined measurements;
M2 must fit its imputer/scaler within each training fold. No global scaling has
been fitted. Class imbalance is retained and must be addressed explicitly.

For evaluation/scoring, load all C2 candidates directly with pandas.read_parquet;
the convenience training loader excludes untrusted labels. Keep keys beside
scores, preserve candidate row order and join predictions to C1/C3. C1 preserves
zero-candidate and invalid observations. A trusted reference absent from C2 has
zero positive candidates; do not restrict Top-1 denominators to reference hits.
`coverage.csv` and `coverage_by_partition.csv` define known misses/trusted valid
observations and known misses/all valid observations separately. Unknown-reference
observations cannot be honestly counted as known matches or known misses.

## M3: directed graph and geometry

`load_road_graph(trip_dir)` returns original float64 lon/lat and metric node arrays,
int64 source/target arrays, metre lengths, true-north bearings, reverse twins,
one-way indicators, incoming/outgoing degrees, weak/strong components and isolated
node indices. `successors[i]` lazily returns arcs leaving i's target;
`predecessors[i]` returns arcs entering i's source. Parallel arcs are kept.
Reverse twin = lowest distinct opposite-endpoint arc ID, -1 if none; duplicates
can share a twin and the mapping is not necessarily one-to-one. Self-loops need
a distinct duplicate to have a reverse twin; zero-length arcs have NaN bearings.
No production shortest-path transitions are implemented here.

C2 `proj_frac` runs source-to-target [0,1]; `proj_x/proj_y` are metres in the
graph's exact projection. Reuse `graph.projection`, never create a second trip
projection from GPS alone. C1 gap flags are informational; M3 owns segmentation.
C3 route positions refer to occurrences, not unique arc IDs. Status `ambiguous`
is excluded from supervised targets; repeated arcs and breaks are never repaired.

## M4: identical route-free preprocessing at inference

```python
from src.data.config import load_config
from src.data.dataset_loader import load_track
from src.data.road_graph import load_road_graph
from src.data.track_cleaning import clean_track
from src.data.candidate_generation import generate_candidates
from src.data.feature_engineering import build_features

cfg = load_config('configs/data.yaml')
graph = load_road_graph(trip_dir)
obs = clean_track(load_track(trip_dir), graph.projection, graph.trip_id,
                  partition='student_test', config=cfg)
cand = generate_candidates(obs, graph, cfg['radius_m'], cfg['max_candidates'])
features = build_features(obs, cand, graph, with_labels=False)
assert 'label' not in features
```

These imports and functions never load `.route`; the leakage test guards file
reads with a route present, altered and removed. In offline/server-side reference
evaluation only, explicitly call `load_route(trip_dir, len(graph.arc_from))`, then
`align_reference(obs, graph, route, cfg)` and `reference_membership(alignment,cand)`.
The evaluation function is a separate import and must never enter prediction.

## Algorithms and limits for the viva

The spatial tree indexes bounding boxes of complete directed segments. Query
radius boxes give a superset; exact clamped point-to-segment projection then
filters by radius before deterministic top-K selection. A long arc cannot be lost
because its midpoint is distant. No fallback changes the stated contract.

Reference alignment builds a sparse DAG of all nearby observation/route-occurrence
pairs within a separate fixed alignment radius, with nondecreasing route position
and increasing observation index. A Fenwick prefix-maximum structure computes
the best path, allowing repeated route positions and skipping outliers. Reward
uses distance and heading; skipping route positions incurs a small penalty.
Backward scores expose alternative near-best occurrence assignments; those plus
gap and break-neighbor positions are ambiguous. This does not model acceleration,
road travel time or full path uncertainty. Sparse search is spatially pruned but
has no top-K occurrence cap; dense revisited loops can increase memory/time.
It is an approximate source of supervision, not independent truth. Conservative
null labels trade sample quantity for caution; human route verification remains
a limitation. Do not claim high candidate recall validates alignment correctness.

Projection uses WGS84 network-centred AEQD and enforces local extent limits. The
frozen candidate config was selected only on train. No smoother was fitted; raw
GPS locations remain intact. Invalid points survive C1. No observation or graph
row has been silently renumbered or dropped.

Pending consumer review: M2/M3/M4 for schema/whole-trip feature contract; M3 for
directed graph and derived alignment; M2 for label/miss denominators and feature
preprocessing; M4 for minimal shared .gitignore additions and eventual dependency
consolidation. These requests do not block isolated-branch preprocessing delivery.
