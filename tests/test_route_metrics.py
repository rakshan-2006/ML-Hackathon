"""Route metric tests on synthetic sequences (no real data)."""

import math

import pytest

from src.hmm.route_metrics import pointwise_accuracy, route_continuity, sequence_similarity
from src.hmm.transition import RoadGraph

# Main road 0 -> 1 -> 2 -> 3 as arcs 0, 1, 2; arc 3 (4 -> 5) is disconnected;
# arc 4 (2 -> 1) is the reverse twin of arc 1.
GRAPH = RoadGraph([0, 1, 2, 4, 2], [1, 2, 3, 5, 1], [100.0] * 5)


# Pointwise accuracy -------------------------------------------------------------

def test_pointwise_perfect_and_incorrect():
    assert pointwise_accuracy([0, 0, 1, 2], [0, 0, 1, 2])["accuracy"] == 1.0
    res = pointwise_accuracy([0, 3, 1, 3], [0, 0, 1, 2])
    assert res["accuracy"] == 0.5 and res["n_correct"] == 2


def test_pointwise_missing_values():
    # Missing reference -> excluded; missing prediction -> counted wrong.
    res = pointwise_accuracy([0, -1, 1, 2], [0, 0, -1, None])
    assert res["n_evaluated"] == 2
    assert res["n_missing_ref"] == 2
    assert res["n_missing_pred"] == 1
    assert res["accuracy"] == 0.5


def test_pointwise_empty_and_length_mismatch():
    assert math.isnan(pointwise_accuracy([], [])["accuracy"])
    assert math.isnan(pointwise_accuracy([1, 2], [-1, -1])["accuracy"])
    with pytest.raises(ValueError):
        pointwise_accuracy([0, 1], [0])


# Route continuity ---------------------------------------------------------------

def test_continuity_connected_route():
    res = route_continuity([0, 0, 1, 1, 2], GRAPH)
    assert res == {"continuity": 1.0, "n_transitions": 2, "n_connected": 2, "n_breaks": 0}


def test_continuity_disconnected_and_wrong_direction():
    # 0 -> 3 jumps to a disconnected road; 3 -> 2 is also not joined.
    res = route_continuity([0, 3, 2], GRAPH)
    assert res["n_breaks"] == 2 and res["continuity"] == 0.0
    # arc 1 then arc 0 goes against the arc directions.
    assert route_continuity([1, 0], GRAPH)["continuity"] == 0.0
    # U-turn onto the reverse twin shares node 2 -> connected.
    assert route_continuity([1, 4], GRAPH)["continuity"] == 1.0


def test_continuity_missing_and_empty():
    # Missing observations are boundaries, not evidence of a connected transition.
    assert route_continuity([0, -1, 1, None, 2], GRAPH)['n_transitions'] == 0
    assert math.isnan(route_continuity([], GRAPH)["continuity"])
    assert math.isnan(route_continuity([2, 2, -1], GRAPH)["continuity"])
    with pytest.raises(ValueError):
        route_continuity([0, 99], GRAPH)


# Sequence similarity ------------------------------------------------------------

def test_sequence_identical():
    res = sequence_similarity([0, 0, 1, 2], [0, 1, 2])
    assert res["arc_f1"] == 1.0 and res["lcs_similarity"] == 1.0 and res["lcs_length"] == 3


def test_sequence_partial_and_different_lengths():
    # pred misses arc 2 and adds arc 3: common arcs {0, 1}; LCS = [0, 1].
    res = sequence_similarity([0, 3, 1], [0, 1, 2, 5])
    assert res["arc_precision"] == pytest.approx(2 / 3)
    assert res["arc_recall"] == pytest.approx(2 / 4)
    assert res["arc_f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert res["lcs_length"] == 2
    assert res["lcs_similarity"] == pytest.approx(4 / 7)


def test_sequence_order_matters_for_lcs_only():
    res = sequence_similarity([2, 1, 0], [0, 1, 2])
    assert res["arc_f1"] == 1.0
    assert res["lcs_length"] == 1 and res["lcs_similarity"] == pytest.approx(1 / 3)


def test_sequence_missing_and_empty():
    res = sequence_similarity([0, -1, None, 1], [0, 1])
    assert res["lcs_similarity"] == 1.0
    both = sequence_similarity([], [])
    assert math.isnan(both["arc_f1"]) and math.isnan(both["lcs_similarity"])
    one = sequence_similarity([], [0, 1])
    assert one["arc_f1"] == 0.0 and one["lcs_similarity"] == 0.0 and one["len_ref"] == 2
