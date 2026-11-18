"""Route-free inference for one trip: M1 preprocessing -> M2 scores -> M3 decoding.

Never reads `.route`. `score_candidates` is Member 2's scoring function: it takes
the C2 feature table and returns one probability per row, in row order.
"""

from time import perf_counter

import numpy as np

from src.data.candidate_generation import generate_candidates
from src.data.dataset_loader import load_track
from src.data.feature_engineering import build_features
from src.data.road_graph import load_road_graph
from src.data.track_cleaning import clean_track
from src.hmm.adapter import decode_tables, decoder_graph, route_polylines


def run_trip(trip_dir, score_candidates, data_config, decoder_config=None):
    timings = {}

    def timed(stage, fn, *args, **kwargs):
        started = perf_counter()
        value = fn(*args, **kwargs)
        timings[stage] = round((perf_counter() - started) * 1000, 3)
        return value

    graph = timed("load_graph", load_road_graph, trip_dir)
    obs = timed("clean_track", lambda: clean_track(
        load_track(trip_dir), graph.projection, graph.trip_id,
        partition="inference", config=data_config))
    cand = timed("candidates", generate_candidates, obs, graph,
                 data_config["radius_m"], data_config["max_candidates"])
    features = timed("features", build_features, obs, cand, graph, with_labels=False)

    scores = np.asarray(timed("score", score_candidates, features), dtype=np.float64)
    if scores.shape != (len(features),):
        raise ValueError(f"scorer returned shape {scores.shape} for {len(features)} candidates")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("scorer must return finite probabilities in [0, 1]")

    dgraph = timed("decoder_graph", decoder_graph, graph)
    hmm, classifier = timed("decode", decode_tables, obs, features, scores, graph,
                            decoder_config, dgraph)
    return {
        "graph": graph,
        "observations": obs,
        "candidates": features,
        "scores": scores,
        "hmm": hmm,
        "classifier": classifier,
        "route_hmm": route_polylines(hmm["route_segments"], graph),
        "route_classifier": route_polylines([classifier["route"]], graph),
        "timings": timings,
    }
