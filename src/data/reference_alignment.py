"""Offline sparse monotone alignment on route *occurrences*, independent of C2.

Each spatially admissible (observation, route-position) is a DAG vertex. A path
uses strictly increasing observation indices and nondecreasing route positions.
Reward = 2R - distance - heading_weight*heading_difference/180. Skipped route
positions cost skip_penalty. Skipping observations is allowed (zero reward).
Fenwick prefix maxima solve the DAG in O(S log L), memory O(S+L), S spatial hits.
No spatial top-K/band pruning: all occurrences within the fixed alignment radius
are searched. Forward/backward scores identify near-optimal alternative states.
This is derived supervision, not a production decoder or verified ground truth.
"""
import numpy as np
from src.common.schema import C3, typed_table
from .candidate_generation import SegmentIndex
from .geo import angular_difference

class PrefixMax:
    def __init__(self,n):
        self.scores = np.full(n+1,-np.inf)
        self.ids = np.full(n+1,-1,dtype=np.int64)
    def query(self,pos):
        pos += 1
        score,idx = 0.,-1
        while pos>0:
            if self.scores[pos]>score:
                score,idx = self.scores[pos],self.ids[pos]
            pos -= pos & -pos
        return score,idx
    def update(self,pos,score,idx):
        pos += 1
        while pos<len(self.scores):
            if score>self.scores[pos]:
                self.scores[pos],self.ids[pos] = score,idx
            pos += pos & -pos

def route_break_positions(route,graph):
    if len(route)<2:
        return np.array([],dtype=int)
    return np.flatnonzero(graph.arc_to[route[:-1]] != graph.arc_from[route[1:]])+1

def align_reference(observations, graph, route, config=None):
    cfg = config or {}
    radius = cfg.get('alignment_radius_m',150.)
    weight = cfg.get('alignment_heading_weight_m',15.)
    penalty = cfg.get('alignment_skip_penalty_m',.05)
    margin = cfg.get('alignment_ambiguity_margin_m',3.)
    route = np.asarray(route,dtype=np.int64)
    n,l = len(observations),len(route)
    rows = dict(trip_id=observations.trip_id.to_numpy(),obs_idx=observations.obs_idx.to_numpy(),ref_arc_id=np.full(n,-1),ref_route_pos=np.full(n,-1),align_dist_m=np.full(n,np.nan),align_status=['unaligned']*n,ref_in_candidates=np.zeros(n,dtype=bool))
    result = typed_table(rows,C3)
    if not l:
        return result
    if (route<0).any() or (route>=len(graph.arc_from)).any():
        raise ValueError('Reference arc out of range')
    index = SegmentIndex(graph.node_xy[graph.arc_from[route]],graph.node_xy[graph.arc_to[route]])
    positions, distances, rewards, obs_positions, groups = [],[],[],[],[]
    count = 0
    for i,row in enumerate(observations.itertuples(index=False)):
        if not row.is_valid:
            continue
        pos,d,_,_ = index.query(np.array([row.x,row.y]),radius)
        if not len(pos):
            continue
        order = np.argsort(pos,kind='stable')
        pos,d = pos[order],d[order]
        hd = angular_difference(row.heading_deg,graph.arc_bearing[route[pos]])
        reward = 2*radius-d-weight*np.nan_to_num(hd,nan=90.)/180.
        positions.extend(pos); distances.extend(d); rewards.extend(reward); obs_positions.extend([i]*len(pos))
        groups.append((count,count+len(pos)))
        count += len(pos)
    if not count:
        return result
    positions,distances,rewards,obs_positions = map(np.asarray,(positions,distances,rewards,obs_positions))
    forward,backward = np.zeros(count),np.zeros(count)
    parents = np.full(count,-1,dtype=np.int64)
    tree = PrefixMax(l)
    for start,end in groups:
        for j in range(start,end):
            prev,parent = tree.query(int(positions[j]))
            # A path may start at any route occurrence without a start penalty.
            linked = prev-penalty*positions[j]
            forward[j] = rewards[j]+max(0,linked)
            parents[j] = parent if linked>0 else -1
        for j in range(start,end):
            tree.update(int(positions[j]),forward[j]+penalty*positions[j],j)
    tree = PrefixMax(l)
    for start,end in reversed(groups):
        for j in range(start,end):
            reverse = l-1-positions[j]
            nxt,_ = tree.query(int(reverse))
            backward[j] = rewards[j]+max(0,nxt-penalty*reverse)
        for j in range(start,end):
            reverse = l-1-positions[j]
            tree.update(int(reverse),backward[j]+penalty*reverse,j)
    chosen, j = [],int(np.argmax(forward))
    best = forward[j]
    while j>=0:
        chosen.append(j)
        j = int(parents[j])
    group_for_obs = {int(obs_positions[s]):(s,e) for s,e in groups}
    breaks = route_break_positions(route,graph)
    affected = set(breaks.tolist()+np.maximum(breaks-1,0).tolist())
    for j in reversed(chosen):
        i,pos = int(obs_positions[j]),int(positions[j])
        s,e = group_for_obs[i]
        totals = forward[s:e]+backward[s:e]-rewards[s:e]
        alternative = (np.arange(s,e)!=j) & (totals>=best-margin)
        ambiguous = bool(alternative.any() or observations.gap_flag.iloc[i] or pos in affected)
        result.loc[i,['ref_arc_id','ref_route_pos','align_dist_m','align_status']] = [int(route[pos]),pos,float(distances[j]),'ambiguous' if ambiguous else 'ok']
    result = result.astype(C3)
    result.attrs.update(route_break_count=len(breaks), spatial_states=count, algorithm='sparse_monotone_DAG')
    return result
