"""Member 3 HMM / Viterbi tests on small synthetic road networks (no real data)."""

import itertools
import math

import numpy as np
import pytest

from src.hmm.emission import log_emissions
from src.hmm.route_decoder import DecoderConfig, classifier_only, decode_trip
from src.hmm.transition import RoadGraph, log_transition_matrix
from src.hmm.viterbi import viterbi

# Nodes lie on a line every 100 m: 0 -> 1 -> 2 -> 3 is a one-way main road.
# Arc 3 (4 -> 5) is a parallel road reachable from node 1 only via a 300 m detour (arc 5).
# Arc 4 (2 -> 1) is the reverse twin of arc 1.
#   arc: 0     1     2     3     4     5
FROM = [0, 1, 2, 4, 2, 1]
TO = [1, 2, 3, 5, 1, 4]
LEN = [100.0, 100.0, 100.0, 100.0, 100.0, 300.0]


@pytest.fixture
def graph():
    return RoadGraph(FROM, TO, LEN)


@pytest.fixture
def disconnected_graph():
    # Same main road, but the parallel road has no connection at all.
    return RoadGraph([0, 1, 2, 4], [1, 2, 3, 5], [100.0] * 4)


def along_main_road(n, spacing=100.0):
    """Timestamps (10 s apart) and GPS positions in the middle of arcs 0, 1, 2, ..."""
    times = [10.0 * t for t in range(n)]
    xy = [(50.0 + spacing * t, 0.0) for t in range(n)]
    return times, xy


def brute_force(log_em, log_tr):
    best_score, best_path = -math.inf, None
    for path in itertools.product(*[range(len(e)) for e in log_em]):
        s = log_em[0][path[0]] + sum(
            log_tr[t - 1][path[t - 1], path[t]] + log_em[t][path[t]]
            for t in range(1, len(path)))
        if s > best_score:
            best_score, best_path = s, list(path)
    return best_path, best_score


# 1. Simple connected network --------------------------------------------------

def test_connected_route_is_recovered(graph):
    times, xy = along_main_road(3)
    res = decode_trip([[0, 3], [1, 3], [2, 3]],
                      [[0.9, 0.1], [0.8, 0.2], [0.9, 0.1]],
                      graph, times=times, xy=xy)
    assert res["status"] == "ok"
    assert res["pred_arc_ids"] == [0, 1, 2]
    assert res["pred_cand_index"] == [0, 0, 0]
    assert res["route"] == [0, 1, 2]
    assert res["n_segments"] == 1
    assert math.isfinite(res["log_score"])


def test_route_fills_skipped_arcs(graph):
    # Observations on arc 0 then arc 2: arc 1 must be inserted from the network path.
    res = decode_trip([[0], [2]], [[1.0], [1.0]], graph, times=[0.0, 20.0],
                      xy=[(50.0, 0.0), (250.0, 0.0)])
    assert res["route"] == [0, 1, 2]
    assert res["pred_arc_ids"] == [0, 2]


def test_directed_arcs_are_respected(graph):
    cfg = DecoderConfig()
    # arc 1 (1->2) to arc 0 (0->1): nothing leads back to node 0, so impossible.
    la, _ = log_transition_matrix(graph, [1], [0.5], [0], [0.5], 10.0, 100.0, cfg)
    assert la[0, 0] == -np.inf
    # arc 0 -> arc 1 follows the arc direction.
    la, paths = log_transition_matrix(graph, [0], [0.5], [1], [0.5], 10.0, 100.0, cfg)
    assert np.isfinite(la[0, 0]) and paths[(0, 0)] == []


# 2. Ambiguous: global best beats locally best --------------------------------

def test_viterbi_prefers_coherent_route_over_local_argmax(graph):
    times, xy = along_main_road(3)
    cands = [[0, 3], [1, 3], [2, 3]]
    probs = [[0.9, 0.1], [0.4, 0.6], [0.9, 0.1]]   # middle point looks like the parallel road

    greedy = classifier_only(cands, probs)
    assert greedy["pred_arc_ids"] == [0, 3, 2]

    res = decode_trip(cands, probs, graph, times=times, xy=xy)
    assert res["pred_arc_ids"] == [0, 1, 2]

    # Same answer and score as exhaustive search over the decoder's own lattice.
    cfg = DecoderConfig()
    log_em = [log_emissions(p) for p in probs]
    log_tr = [log_transition_matrix(graph, cands[t - 1], [0.5, 0.5], cands[t], [0.5, 0.5],
                                    10.0, 100.0, cfg)[0] for t in (1, 2)]
    bf_path, bf_score = brute_force(log_em, log_tr)
    assert res["pred_cand_index"] == bf_path
    assert res["log_score"] == pytest.approx(bf_score)


def test_viterbi_matches_brute_force_on_random_lattices():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(200):
        T = int(rng.integers(2, 6))
        ks = rng.integers(1, 4, size=T)
        log_em = [np.log(rng.uniform(0.01, 1, size=k)) for k in ks]
        log_tr = []
        for t in range(1, T):
            m = np.log(rng.uniform(0.01, 1, size=(ks[t - 1], ks[t])))
            m[rng.uniform(size=m.shape) < 0.3] = -np.inf
            log_tr.append(m)
        path, score, brk = viterbi(log_em, log_tr)
        bf_path, bf_score = brute_force(log_em, log_tr)
        if bf_path is None:
            assert brk < T          # no complete path exists -> decoder must report a break
        else:
            assert brk == T
            assert path == bf_path
            assert score == pytest.approx(bf_score)
            checked += 1
    assert checked > 50


# 3. Disconnected roads / no valid transition ---------------------------------

def test_disconnected_roads_split_into_segments(disconnected_graph):
    times, xy = along_main_road(4)
    res = decode_trip([[0], [1], [3], [3]], [[0.9], [0.9], [0.9], [0.9]],
                      disconnected_graph, times=times, xy=xy)
    assert res["status"] == "partial"
    assert res["pred_arc_ids"] == [0, 1, 3, 3]
    assert res["segment_ids"] == [0, 0, 1, 1]
    assert res["n_breaks_no_transition"] == 1
    assert res["route_segments"] == [[0, 1], [3]]
    assert math.isfinite(res["log_score"])


def test_large_time_gap_starts_new_segment(graph):
    res = decode_trip([[0], [1]], [[1.0], [1.0]], graph, times=[0.0, 5000.0],
                      xy=[(50.0, 0.0), (150.0, 0.0)])
    assert res["n_breaks_gap"] == 1
    assert res["segment_ids"] == [0, 1]
    assert res["status"] == "partial"


# 4. Zero probabilities ---------------------------------------------------------

def test_zero_probabilities_are_handled(graph):
    em = log_emissions([0.0, 1.0])
    assert np.all(np.isfinite(em)) and em[0] < em[1]

    times, xy = along_main_road(3)
    res = decode_trip([[0, 3], [1, 3], [2, 3]],
                      [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],   # middle observation all zero
                      graph, times=times, xy=xy)
    assert res["status"] == "ok"
    assert res["pred_arc_ids"] == [0, 1, 2]
    assert math.isfinite(res["log_score"])


def test_invalid_probabilities_raise():
    with pytest.raises(ValueError):
        log_emissions([0.5, float("nan")])
    with pytest.raises(ValueError):
        log_emissions([1.5])


# 5. Empty input and missing candidates ---------------------------------------

def test_empty_input(graph):
    res = decode_trip([], [], graph)
    assert res["status"] == "empty"
    assert res["route"] == [] and res["pred_arc_ids"] == []


def test_observation_without_candidates(graph):
    res = decode_trip([[0], [], [2]], [[1.0], [], [1.0]], graph,
                      times=[0.0, 10.0, 20.0], xy=[(50.0, 0.0), (150.0, 0.0), (250.0, 0.0)])
    assert res["status"] == "partial"
    assert res["pred_arc_ids"] == [0, -1, 2]
    assert res["n_no_candidates"] == 1
    assert res["route"] == [0, 1, 2]


def test_all_observations_without_candidates(graph):
    res = decode_trip([[], []], [[], []], graph)
    assert res["status"] == "empty"
    assert res["pred_arc_ids"] == [-1, -1]


def test_unknown_arc_id_raises(graph):
    with pytest.raises(ValueError):
        decode_trip([[99]], [[1.0]], graph)
