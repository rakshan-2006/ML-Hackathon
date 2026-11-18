"""Inference-safe physical features; offline reference joining is explicit."""
import numpy as np
import pandas as pd
from src.common.schema import C2, typed_table
from .candidate_generation import project_segments
from .geo import angular_difference

def reference_membership(alignment, candidates):
    alignment = alignment.copy()
    keys = set(zip(candidates.trip_id,candidates.obs_idx.astype(int),candidates.arc_id.astype(int)))
    alignment['ref_in_candidates'] = [(str(t),int(i),int(a)) in keys for t,i,a in zip(alignment.trip_id,alignment.obs_idx,alignment.ref_arc_id)]
    return alignment

def join_labels(candidates,alignment):
    candidates = candidates.copy()
    ref = alignment.set_index(['trip_id','obs_idx']).reindex(pd.MultiIndex.from_frame(candidates[['trip_id','obs_idx']]))
    values = pd.array([pd.NA]*len(candidates),dtype='Int8')
    trusted = ref.align_status.to_numpy() == 'ok'
    values[trusted] = (candidates.arc_id.to_numpy()[trusted] == ref.ref_arc_id.to_numpy()[trusted]).astype(np.int8)
    candidates['label'] = values
    return candidates

def build_features(observations, candidates, graph, *, with_labels=False, alignment=None):
    c = candidates.copy()
    if not len(c):
        c = typed_table({},C2)
    else:
        # Join by stable identity, never assume candidate or observation input order.
        o = observations.set_index(['trip_id','obs_idx']).reindex(pd.MultiIndex.from_frame(c[['trip_id','obs_idx']]))
        ids = c.arc_id.to_numpy(dtype=int)
        c['dist_minus_min'] = c.dist_m-c.groupby(['trip_id','obs_idx']).dist_m.transform('min')
        c['heading_diff_deg'] = angular_difference(o.heading_deg.to_numpy(),graph.arc_bearing[ids])
        c['speed_mps'] = o.speed_mps.to_numpy()
        c['arc_len_m'] = graph.arc_len_m[ids]
        c['n_candidates'] = o.n_candidates.to_numpy()
        c['is_oneway'] = graph.is_oneway[ids]
        c['source_degree'] = graph.node_degree[graph.arc_from[ids]]
        c['target_degree'] = graph.node_degree[graph.arc_to[ids]]
        by_idx = observations.set_index('obs_idx')
        for offset,name in [(-1,'prev_dist_m'),(1,'next_dist_m')]:
            adjacent = by_idx.reindex(c.obs_idx.to_numpy()+offset)
            point = adjacent[['x','y']].to_numpy()
            # project_segments accepts per-candidate point arrays too.
            d,_,_ = project_segments(point,graph.node_xy[graph.arc_from[ids]],graph.node_xy[graph.arc_to[ids]])
            valid = adjacent.is_valid.astype('boolean').fillna(False).to_numpy(dtype=bool)
            boundary_gap = o.gap_flag.to_numpy() if offset==-1 else adjacent.gap_flag.astype('boolean').fillna(True).to_numpy(dtype=bool)
            d[~valid | boundary_gap] = np.nan
            c[name] = d
        c = c[list(C2)].astype(C2)
    if with_labels:
        if alignment is None:
            raise ValueError('Offline labels require explicit alignment table')
        c = join_labels(c,alignment)
    return c
