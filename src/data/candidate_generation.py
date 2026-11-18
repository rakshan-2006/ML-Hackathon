"""Full-segment STRtree, exact radius filtering; no reference imports or fallback."""
import numpy as np
from shapely import linestrings, box
from shapely.strtree import STRtree
from src.common.schema import C2_BASE, typed_table

def project_segments(point, starts, ends):
    delta = ends-starts
    length2 = np.einsum('ij,ij->i',delta,delta)
    fraction = np.divide(np.einsum('ij,ij->i',point-starts,delta),length2,out=np.zeros(len(starts)),where=length2>0)
    fraction = np.clip(fraction,0,1)
    xy = starts+fraction[:,None]*delta
    return np.linalg.norm(xy-point,axis=1),fraction,xy

class SegmentIndex:
    def __init__(self, starts, ends):
        self.starts,self.ends = starts,ends
        self.tree = STRtree(linestrings(np.stack([starts,ends],axis=1)))
    def query(self, point, radius):
        x,y = point
        ids = self.tree.query(box(x-radius,y-radius,x+radius,y+radius))
        d,f,xy = project_segments(point,self.starts[ids],self.ends[ids])
        keep = d <= radius
        ids,d,f,xy = ids[keep],d[keep],f[keep],xy[keep]
        order = np.lexsort((ids,d))
        return ids[order],d[order],f[order],xy[order]

def generate_candidates(observations, graph, radius_m=100., max_candidates=16, index=None):
    if radius_m<=0 or max_candidates<1:
        raise ValueError('radius_m and max_candidates must be positive')
    index = index or SegmentIndex(graph.node_xy[graph.arc_from],graph.node_xy[graph.arc_to])
    rows, truncated = [],0
    counts = np.zeros(len(observations),dtype=np.int64)
    for row_pos,row in enumerate(observations.itertuples(index=False)):
        if not row.is_valid:
            continue
        ids,d,f,xy = index.query(np.array([row.x,row.y]),radius_m)
        truncated += len(ids)>max_candidates
        counts[row_pos] = min(len(ids),max_candidates)
        for rank,arc in enumerate(ids[:max_candidates]):
            rows.append((row.trip_id,row.obs_idx,int(arc),rank,d[rank],f[rank],xy[rank,0],xy[rank,1]))
    observations['n_candidates'] = counts
    df = typed_table(rows,C2_BASE) if not rows else typed_table(dict(zip(C2_BASE,zip(*rows))),C2_BASE)
    df.attrs['truncated_observations'] = int(truncated)
    return df
