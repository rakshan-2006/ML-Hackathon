"""Offline / server-side evaluation of one decoded trip against its reference route.

Reads the trip's `.route`: never import this module from the prediction pipeline.
Pointwise accuracy uses Member 1's derived alignment; only status 'ok' references
count, and trusted references missing from the candidate set stay in the
denominator (counted wrong), as required by data/README.md.
"""

import numpy as np

from src.data.dataset_loader import load_route
from src.data.reference_alignment import align_reference
from src.hmm.route_metrics import pointwise_accuracy, route_continuity, sequence_similarity


def evaluate_trip(trip_dir, observations, result, m1_graph, data_config):
    """Metrics for one decode_trip / classifier_only result (dict of floats)."""
    route = load_route(trip_dir, len(m1_graph.arc_from))
    alignment = align_reference(observations, m1_graph, route, data_config)
    alignment = alignment.sort_values("obs_idx")
    if len(alignment) != len(result["pred_arc_ids"]):
        raise ValueError("result must hold one prediction per observation")
    trusted = alignment.align_status.to_numpy() == "ok"
    ref = np.where(trusted, alignment.ref_arc_id.to_numpy(), -1).tolist()

    point = pointwise_accuracy(result["pred_arc_ids"], ref)
    cont = route_continuity(result["route"], m1_graph)
    seq = sequence_similarity(result["route"], route.tolist())
    return {
        "top1_accuracy": point["accuracy"],
        "n_evaluated": float(point["n_evaluated"]),
        "route_continuity": cont["continuity"],
        "n_route_breaks": float(cont["n_breaks"]),
        "arc_f1": seq["arc_f1"],
        "lcs_similarity": seq["lcs_similarity"],
    }
