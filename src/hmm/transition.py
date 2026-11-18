"""Transition scores between candidate arcs of consecutive observations.

Connectivity comes only from the directed arcs (from_node -> to_node); arc IDs are
never assumed to imply adjacency.

PROVISIONAL: `RoadGraph` is a minimal container for the arrays Member 1 will supply
(arc_from, arc_to, arc_len_m). Swap in / adapt to M1's real graph during integration.
"""

import heapq
import math

import numpy as np


class RoadGraph:
    """Directed road network: arc a goes from node arc_from[a] to node arc_to[a]."""

    def __init__(self, arc_from, arc_to, arc_len_m):
        self.arc_from = np.asarray(arc_from, dtype=np.int64)
        self.arc_to = np.asarray(arc_to, dtype=np.int64)
        self.arc_len_m = np.asarray(arc_len_m, dtype=np.float64)
        if not (len(self.arc_from) == len(self.arc_to) == len(self.arc_len_m)):
            raise ValueError("arc_from, arc_to and arc_len_m must have equal length")
        if (self.arc_len_m < 0).any():
            raise ValueError("arc lengths must be non-negative")
        self.out_arcs = {}
        for arc, node in enumerate(self.arc_from):
            self.out_arcs.setdefault(int(node), []).append(arc)

    @property
    def n_arcs(self):
        return len(self.arc_from)


def shortest_paths(graph, source, max_dist):
    """Bounded Dijkstra over nodes. Returns (dist, pred_arc) dicts keyed by node."""
    dist = {source: 0.0}
    pred = {}
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for arc in graph.out_arcs.get(u, ()):
            v = int(graph.arc_to[arc])
            nd = d + graph.arc_len_m[arc]
            if nd <= max_dist and nd < dist.get(v, math.inf):
                dist[v] = nd
                pred[v] = arc
                heapq.heappush(heap, (nd, v))
    return dist, pred


def route_distance(graph, arc_i, frac_i, arc_j, frac_j, sp_cache, max_dist, backtrack_m):
    """Driving distance from position frac_i on arc_i to position frac_j on arc_j.

    Returns (distance_m, intermediate_arcs); distance is inf when arc_j cannot be
    reached within max_dist following arc directions.
    """
    len_i = graph.arc_len_m[arc_i]
    len_j = graph.arc_len_m[arc_j]
    if arc_i == arc_j and (frac_j >= frac_i or (frac_i - frac_j) * len_i <= backtrack_m):
        # Same arc: forward movement, or a small backwards jitter treated as standing still.
        return max(frac_j - frac_i, 0.0) * len_i, []

    head = (1.0 - frac_i) * len_i
    tail = frac_j * len_j
    source = int(graph.arc_to[arc_i])
    target = int(graph.arc_from[arc_j])
    if source not in sp_cache:
        sp_cache[source] = shortest_paths(graph, source, max_dist)
    dist, pred = sp_cache[source]
    if target not in dist:
        return math.inf, []
    total = head + dist[target] + tail
    if total > max_dist:
        return math.inf, []

    arcs = []
    node = target
    while node != source:
        arc = pred[node]
        arcs.append(arc)
        node = int(graph.arc_from[arc])
    return total, arcs[::-1]


def log_transition_matrix(graph, prev_arcs, prev_fracs, cur_arcs, cur_fracs,
                          dt, gc_dist, config):
    """Log transition scores between the candidates of two consecutive observations.

    dt      : seconds between the observations (None if unknown).
    gc_dist : straight-line distance in metres between the two GPS points (None if unknown).

    Score = -|route_dist - gc_dist| / beta  (Newson-Krumm style), or -route_dist / beta
    without GPS positions. Unreachable pairs (no directed path within the
    speed-based distance bound) get -inf.

    Returns (log_A with shape (K_prev, K_cur), paths) where paths[(i, j)] lists the
    intermediate arcs driven between candidate i and candidate j.
    """
    if dt is not None and dt > 0:
        max_dist = config.max_speed_mps * dt + config.dist_slack_m
    else:
        max_dist = config.max_route_dist_m

    log_a = np.full((len(prev_arcs), len(cur_arcs)), -np.inf)
    paths = {}
    sp_cache = {}
    for i, (a, fa) in enumerate(zip(prev_arcs, prev_fracs)):
        for j, (b, fb) in enumerate(zip(cur_arcs, cur_fracs)):
            d, via = route_distance(graph, a, fa, b, fb, sp_cache, max_dist,
                                    config.same_arc_backtrack_m)
            if math.isinf(d):
                continue
            cost = abs(d - gc_dist) if gc_dist is not None else d
            log_a[i, j] = -cost / config.beta_m
            paths[(i, j)] = via
    return log_a, paths
