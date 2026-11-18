"""Connect Member 1's tables and RoadGraph to the HMM decoder.

Inputs follow src/common/schema.py: C1 observations (one row per raw `.track` row,
obs_idx 0..n-1), C2 candidates (trip-local arc IDs ranked by cand_rank) and
src.data.road_graph.RoadGraph. Scores are Member 2 probabilities aligned with the
C2 row order they were computed on. Nothing here reads `.route`.
"""

from dataclasses import fields

import numpy as np

from src.hmm.route_decoder import DecoderConfig, classifier_only, decode_trip
from src.hmm.transition import RoadGraph as DecoderGraph

_CONFIG_METADATA = {'schema_version','version','notes','status','model_version',
                    'data_config_sha256','tuning_provenance'}


def decoder_config_from_dict(document):
    """DecoderConfig from decoder_config.json; metadata keys are ignored."""
    known = {f.name for f in fields(DecoderConfig)}
    unknown = set(document) - known - _CONFIG_METADATA
    if unknown:
        raise ValueError(f"Unknown decoder config keys: {sorted(unknown)}")
    return DecoderConfig(**{k: v for k, v in document.items() if k in known})


def decoder_graph(m1_graph):
    """Directed arrays of M1's RoadGraph in the decoder's graph structure."""
    return DecoderGraph(m1_graph.arc_from, m1_graph.arc_to, m1_graph.arc_len_m)


def decoder_inputs(observations, candidates, scores):
    """Group C2 rows and their scores per C1 observation, ordered by cand_rank.

    Returns (candidates, probs, proj_frac, times, xy) as per-observation lists in
    obs_idx order; observations without candidates get empty lists.
    """
    obs = observations.sort_values("obs_idx")
    n = len(obs)
    if not np.array_equal(obs.obs_idx.to_numpy(), np.arange(n)):
        raise ValueError("observations must contain obs_idx 0..n-1 exactly once")
    if obs.trip_id.nunique() > 1:
        raise ValueError("observations must belong to a single trip")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(candidates),):
        raise ValueError(f"expected {len(candidates)} scores, got shape {scores.shape}")
    if len(candidates) and not candidates.trip_id.isin(obs.trip_id.unique()).all():
        raise ValueError("candidates reference a different trip")
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('scores must be finite probabilities')
    if candidates.duplicated(['trip_id','obs_idx','arc_id']).any():
        raise ValueError('duplicate candidate keys')

    obs_idx = candidates.obs_idx.to_numpy(dtype=np.int64)
    order = np.lexsort((candidates.cand_rank.to_numpy(), obs_idx))
    obs_idx = obs_idx[order]
    if len(obs_idx) and (obs_idx[0] < 0 or obs_idx[-1] >= n):
        raise ValueError("candidate obs_idx outside the observation range")
    arcs = candidates.arc_id.to_numpy(dtype=np.int64)[order]
    fracs = candidates.proj_frac.to_numpy(dtype=np.float64)[order]
    probs = scores[order]
    bounds = np.searchsorted(obs_idx, np.arange(n + 1))

    def per_obs(values):
        return [values[bounds[t]:bounds[t + 1]].tolist() for t in range(n)]

    return (per_obs(arcs), per_obs(probs), per_obs(fracs),
            obs.t.to_numpy(dtype=np.float64).tolist(),
            obs[["x", "y"]].to_numpy(dtype=np.float64).tolist())


def decode_tables(observations, candidates, scores, m1_graph, config=None, graph=None):
    """Run classifier-plus-HMM and classifier-only decoding on M1's tables."""
    cands, probs, fracs, times, xy = decoder_inputs(observations, candidates, scores)
    graph = graph or decoder_graph(m1_graph)
    hmm = decode_trip(cands, probs, graph, times=times, xy=xy, proj_frac=fracs,
                      config=config)
    baseline = classifier_only(cands, probs)
    # Common segmentation for fair point-sequence continuity (not HMM restarts).
    cfg = config or DecoderConfig()
    segments, current, previous = [-1]*len(times), 0, None
    for i, arcs in enumerate(cands):
        if not arcs:
            previous = None
            current += 1
            continue
        if previous is not None:
            dt = times[i]-times[previous]
            if not np.isfinite(dt) or dt<=0 or dt>cfg.gap_split_s:
                current += 1
        segments[i], previous = current, i
    for result in [hmm,baseline]:
        result['metric_segment_ids'] = segments
    baseline['segment_ids'] = segments
    baseline['route_segments'] = []
    last_segment = None
    for arc, segment in zip(baseline['pred_arc_ids'],segments):
        if arc < 0:
            continue
        if segment != last_segment:
            baseline['route_segments'].append([])
        route = baseline['route_segments'][-1]
        if not route or route[-1]!=arc:
            route.append(arc)
        last_segment = segment
    baseline['n_segments'] = len(baseline['route_segments'])
    return hmm, baseline


def route_polylines(route_segments, m1_graph):
    """Arc sequences -> polylines of (lat, lon) taken from M1's node_lonlat.

    A new polyline starts wherever consecutive arcs are not joined head-to-tail.
    """
    lines = []
    for segment in route_segments:
        current = None  # never join across decoder segments
        for arc in segment:
            u, v = int(m1_graph.arc_from[arc]), int(m1_graph.arc_to[arc])
            if current is not None and current[-1] == u:
                current.append(v)
            else:
                current = [u, v]
                lines.append(current)
    lonlat = m1_graph.node_lonlat
    return [[(float(lonlat[n, 1]), float(lonlat[n, 0])) for n in line] for line in lines]
