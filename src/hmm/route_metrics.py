"""Route-reconstruction metrics shared by Member 2 (comparison table), Member 3
(classifier-only vs HMM) and Member 4 (server-side evaluation).

Definitions follow TEAM_IMPLEMENTATION_PLAN.md decision D9, which is still PENDING
TEAM APPROVAL (M3 proposes, M2 + M4 approve). Conventions:
  * Arc IDs are per-trip 0-based rows of `.arcs`; -1 or None means "missing".
  * A metric with nothing to evaluate returns NaN rather than a misleading 0 or 1.
"""

import math

MISSING = -1


def _is_missing(arc):
    return arc is None or arc == MISSING


def _collapse(arcs):
    """Drop missing values and merge consecutive repeats: [3, 3, -1, 3, 5] -> [3, 5]."""
    out = []
    for arc in arcs:
        if _is_missing(arc):
            continue
        arc = int(arc)
        if not out or out[-1] != arc:
            out.append(arc)
    return out


def pointwise_accuracy(pred_arc_ids, ref_arc_ids):
    """Observation-level Top-1 accuracy.

    Both inputs have one entry per GPS observation, in the same order.
    Observations without a reference arc are excluded from the denominator; a
    missing prediction for an observation that has a reference counts as wrong.
    """
    if len(pred_arc_ids) != len(ref_arc_ids):
        raise ValueError("pred_arc_ids and ref_arc_ids must have one entry per observation")
    n_eval = n_correct = n_missing_pred = n_missing_ref = 0
    for pred, ref in zip(pred_arc_ids, ref_arc_ids):
        if _is_missing(ref):
            n_missing_ref += 1
            continue
        n_eval += 1
        if _is_missing(pred):
            n_missing_pred += 1
        elif int(pred) == int(ref):
            n_correct += 1
    return {
        "accuracy": n_correct / n_eval if n_eval else math.nan,
        "n_correct": n_correct,
        "n_evaluated": n_eval,
        "n_missing_pred": n_missing_pred,
        "n_missing_ref": n_missing_ref,
    }


def route_continuity(route_arc_ids, graph):
    """Share of consecutive distinct arcs that are joined head-to-tail in the directed graph.

    route_arc_ids : ordered predicted arcs (per observation or already collapsed).
    graph         : any object with `arc_from` / `arc_to` arrays (e.g. hmm.transition.RoadGraph).

    Missing values are dropped and repeats merged first, so a pair (a, b) is
    connected iff arc_to[a] == arc_from[b]. A U-turn onto the reverse twin counts as
    connected because the arcs share the node.
    """
    arcs = _collapse(route_arc_ids)
    n_arcs = len(graph.arc_from)
    for arc in arcs:
        if not 0 <= arc < n_arcs:
            raise ValueError(f"arc_id {arc} not in road graph")
    n_trans = len(arcs) - 1 if arcs else 0
    n_connected = sum(1 for a, b in zip(arcs, arcs[1:])
                      if int(graph.arc_to[a]) == int(graph.arc_from[b]))
    return {
        "continuity": n_connected / n_trans if n_trans else math.nan,
        "n_transitions": n_trans,
        "n_connected": n_connected,
        "n_breaks": n_trans - n_connected,
    }


def _lcs_length(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def sequence_similarity(pred_route, ref_route):
    """Compare a predicted route with the reference route (e.g. the trip's `.route`).

    Both are ordered arc sequences; missing values are dropped and repeats merged.
      arc_precision / arc_recall / arc_f1 : overlap of the *sets* of arcs (order ignored)
      lcs_similarity : 2 * LCS / (len_pred + len_ref), the order-aware match (= LCS F1)
    Returns NaN metrics when both routes are empty.
    """
    pred = _collapse(pred_route)
    ref = _collapse(ref_route)
    lcs = _lcs_length(pred, ref)
    pred_set, ref_set = set(pred), set(ref)
    common = len(pred_set & ref_set)

    if not pred and not ref:
        precision = recall = f1 = lcs_sim = math.nan
    else:
        precision = common / len(pred_set) if pred_set else 0.0
        recall = common / len(ref_set) if ref_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        lcs_sim = 2 * lcs / (len(pred) + len(ref))
    return {
        "arc_precision": precision,
        "arc_recall": recall,
        "arc_f1": f1,
        "lcs_length": lcs,
        "lcs_similarity": lcs_sim,
        "len_pred": len(pred),
        "len_ref": len(ref),
    }
