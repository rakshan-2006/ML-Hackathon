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


def evaluate_trip(trip_dir, observations, result, m1_graph, data_config, alignment=None, route=None):
    """Metrics for one decode_trip / classifier_only result (dict of floats)."""
    if route is None:
        route = load_route(trip_dir, len(m1_graph.arc_from))
    if alignment is None:
        alignment = align_reference(observations, m1_graph, route, data_config)
    alignment = alignment.sort_values("obs_idx")
    observations = observations.sort_values('obs_idx')
    if not np.array_equal(alignment.obs_idx.to_numpy(),observations.obs_idx.to_numpy()):
        raise ValueError('alignment must match observation keys exactly')
    if len(alignment) != len(result["pred_arc_ids"]):
        raise ValueError("result must hold one prediction per observation")
    trusted = (alignment.align_status.to_numpy() == 'ok') & observations.is_valid.to_numpy()
    ref = np.where(trusted, alignment.ref_arc_id.to_numpy(), -1).tolist()

    point = pointwise_accuracy(result["pred_arc_ids"], ref)
    cont = route_continuity(result['pred_arc_ids'],m1_graph,result.get('metric_segment_ids'))
    assembled = [route_continuity(seg,m1_graph) for seg in result.get('route_segments',[result['route']])]
    assembled_n = sum(m['n_transitions'] for m in assembled)
    assembled_connected = sum(m['n_connected'] for m in assembled)
    seq = sequence_similarity(result["route"], route.tolist())
    return {
        "top1_accuracy": point["accuracy"],
        "n_evaluated": float(point["n_evaluated"]),
        "route_continuity": cont["continuity"],
        "n_route_breaks": float(cont["n_breaks"]),
        'n_continuity_transitions': float(cont['n_transitions']),
        'n_connected_transitions': float(cont['n_connected']),
        'assembled_continuity': assembled_connected/assembled_n if assembled_n else float('nan'),
        "arc_f1": seq["arc_f1"],
        "lcs_similarity": seq["lcs_similarity"],
        'n_correct': float(point['n_correct']),
    }
