"""Public entry point: candidate probabilities -> coherent route (HMM + Viterbi).

PROVISIONAL INTERFACE (see TEAM_IMPLEMENTATION_PLAN.md section 10). Inputs are plain
per-observation lists so that M1's candidate table and M2's scores can be grouped by
(trip_id, obs_idx) and passed in without any adapter layer:

    candidates[t] : arc IDs of observation t's candidates, ordered by cand_rank
    probs[t]      : M2 probabilities, same order as candidates[t]
    proj_frac[t]  : position (0-1) of the projected GPS point along each candidate arc
    times[t]      : observation timestamp in seconds
    xy[t]         : projected GPS position in metres
"""

import math
from dataclasses import dataclass

import numpy as np

from src.hmm.emission import log_emissions
from src.hmm.transition import log_transition_matrix
from src.hmm.viterbi import viterbi


@dataclass
class DecoderConfig:
    # Provisional defaults; to be tuned on permitted data (train OOF / student_test).
    emission_eps: float = 1e-6
    emission_weight: float = 1.0
    beta_m: float = 10.0               # transition distance-mismatch scale
    max_speed_mps: float = 50.0        # bounds the path search per time step
    dist_slack_m: float = 50.0
    max_route_dist_m: float = 2000.0   # search bound when timestamps are missing
    same_arc_backtrack_m: float = 15.0  # backwards GPS jitter tolerated on one arc
    gap_split_s: float = 300.0         # larger time gaps start a new segment
    fill_route_gaps: bool = True       # insert shortest-path arcs between predictions


def _validate(candidates, probs, graph, times, xy, proj_frac):
    n = len(candidates)
    if len(probs) != n:
        raise ValueError("candidates and probs must have one entry per observation")
    for name, seq in (("times", times), ("xy", xy), ("proj_frac", proj_frac)):
        if seq is not None and len(seq) != n:
            raise ValueError(f"{name} must have one entry per observation")
    for t in range(n):
        if len(candidates[t]) != len(probs[t]):
            raise ValueError(f"observation {t}: candidates and probs differ in length")
        if proj_frac is not None and len(proj_frac[t]) != len(candidates[t]):
            raise ValueError(f"observation {t}: proj_frac and candidates differ in length")
        for arc in candidates[t]:
            if not 0 <= int(arc) < graph.n_arcs:
                raise ValueError(f"observation {t}: arc_id {arc} not in road graph")


def _result(n):
    return {
        "status": "empty",
        "warnings": [],
        "pred_arc_ids": [-1] * n,       # one per observation, -1 = no prediction
        "pred_cand_index": [-1] * n,    # index into candidates[t]
        "segment_ids": [-1] * n,
        "route": [],                    # ordered arc sequence (all segments)
        "route_segments": [],           # one ordered arc list per decoded segment
        "log_score": 0.0,
        "n_segments": 0,
        "n_breaks_gap": 0,
        "n_breaks_no_transition": 0,
        "n_no_candidates": 0,
    }


def decode_trip(candidates, probs, graph, times=None, xy=None, proj_frac=None,
                config=None):
    """Reconstruct the most likely arc sequence for one trip.

    Observations without candidates are skipped (prediction -1). The chain is split
    into independent segments at time gaps > gap_split_s and wherever no directed
    path connects any candidate pair (no valid transition).
    """
    config = config or DecoderConfig()
    _validate(candidates, probs, graph, times, xy, proj_frac)
    n = len(candidates)
    out = _result(n)

    active = [t for t in range(n) if len(candidates[t]) > 0]
    out["n_no_candidates"] = n - len(active)
    if not active:
        out["warnings"].append("no observation has candidates")
        return out

    fracs = proj_frac if proj_frac is not None else [[0.5] * len(c) for c in candidates]
    log_em = {t: log_emissions(probs[t], config.emission_eps, config.emission_weight)
              for t in active}

    # Split the active observations into chains at large time gaps.
    chains = [[active[0]]]
    for prev, cur in zip(active, active[1:]):
        dt = None if times is None else float(times[cur]) - float(times[prev])
        if dt is not None and dt > config.gap_split_s:
            chains.append([cur])
            out["n_breaks_gap"] += 1
        else:
            chains[-1].append(cur)

    for chain in chains:
        log_tr, paths = [], []
        for prev, cur in zip(chain, chain[1:]):
            dt = None if times is None else float(times[cur]) - float(times[prev])
            gc = None if xy is None else float(np.hypot(*(np.asarray(xy[cur], float)
                                                          - np.asarray(xy[prev], float))))
            la, pa = log_transition_matrix(graph, candidates[prev], fracs[prev],
                                           candidates[cur], fracs[cur], dt, gc, config)
            log_tr.append(la)
            paths.append(pa)

        start = 0
        while start < len(chain):
            path, score, brk = viterbi([log_em[t] for t in chain[start:]], log_tr[start:])
            end = start + brk
            _record_segment(out, candidates, chain, paths, start, end, path, score, config)
            if end < len(chain):
                out["n_breaks_no_transition"] += 1
            start = end

    out["route"] = [arc for seg in out["route_segments"] for arc in seg]
    clean = out["n_no_candidates"] == 0 and out["n_segments"] == 1
    out["status"] = "ok" if clean else "partial"
    if out["n_no_candidates"]:
        out["warnings"].append(f"{out['n_no_candidates']} observation(s) without candidates")
    if out["n_breaks_gap"]:
        out["warnings"].append(f"route split at {out['n_breaks_gap']} time gap(s)")
    if out["n_breaks_no_transition"]:
        out["warnings"].append(f"route split at {out['n_breaks_no_transition']} "
                               "point(s) with no valid transition")
    return out


def _record_segment(out, candidates, chain, paths, start, end, path, score, config):
    seg_id = out["n_segments"]
    out["n_segments"] += 1
    out["log_score"] += score
    route = []
    for k, t in enumerate(chain[start:end]):
        idx = path[k]
        arc = int(candidates[t][idx])
        out["pred_cand_index"][t] = idx
        out["pred_arc_ids"][t] = arc
        out["segment_ids"][t] = seg_id
        if k > 0 and config.fill_route_gaps:
            route.extend(int(a) for a in paths[start + k - 1][(path[k - 1], idx)])
        if not route or route[-1] != arc:
            route.append(arc)
    out["route_segments"].append(route)


def classifier_only(candidates, probs):
    """Baseline: independent argmax per observation (ties -> lower candidate index)."""
    n = len(candidates)
    out = _result(n)
    for t in range(n):
        if len(candidates[t]) == 0:
            out["n_no_candidates"] += 1
            continue
        if len(probs[t]) != len(candidates[t]):
            raise ValueError(f"observation {t}: candidates and probs differ in length")
        idx = int(np.argmax(np.asarray(probs[t], dtype=np.float64)))
        out["pred_cand_index"][t] = idx
        out["pred_arc_ids"][t] = int(candidates[t][idx])
        if not out["route"] or out["route"][-1] != out["pred_arc_ids"][t]:
            out["route"].append(out["pred_arc_ids"][t])
    out["status"] = "empty" if out["n_no_candidates"] == n else "ok"
    out["log_score"] = math.nan  # not defined for the independent baseline
    return out
