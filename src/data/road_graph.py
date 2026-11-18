"""Directed arrays plus lazy arc adjacency; preserve all raw row identities."""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .dataset_loader import load_nodes, load_arcs
from .geo import TripProjection, bearing_deg

class ArcAdjacency:
    def __init__(self, endpoints, node_arc_order, indptr):
        self.endpoints, self.order, self.indptr = endpoints,node_arc_order,indptr
    def __getitem__(self, arc_id):
        node = self.endpoints[arc_id]
        return self.order[self.indptr[node]:self.indptr[node+1]]
    def __len__(self):
        return len(self.endpoints)

@dataclass
class RoadGraph:
    trip_id: str
    projection: TripProjection
    node_xy: np.ndarray
    node_lonlat: np.ndarray
    arc_from: np.ndarray
    arc_to: np.ndarray
    arc_len_m: np.ndarray
    arc_bearing: np.ndarray
    successors: ArcAdjacency
    predecessors: ArcAdjacency
    reverse_twin: np.ndarray
    is_oneway: np.ndarray
    in_degree: np.ndarray
    out_degree: np.ndarray
    weak_component: np.ndarray
    strong_component: np.ndarray
    isolated_nodes: np.ndarray
    build_time_s: float
    @property
    def node_degree(self):
        return self.in_degree+self.out_degree

def load_road_graph(trip_dir, projection=None) -> RoadGraph:
    start = perf_counter()
    ll = load_nodes(trip_dir)
    arcs = load_arcs(trip_dir,len(ll))
    if projection is None:
        projection = TripProjection.from_lonlat(ll) if len(ll) else TripProjection(0,0)
    xy = projection.project(ll)
    if not np.isfinite(xy).all():
        raise ValueError(f'{trip_dir}: invalid graph coordinates')
    u,v = arcs[:,0],arcs[:,1]
    n = len(ll)
    indeg,outdeg = np.bincount(v,minlength=n),np.bincount(u,minlength=n)
    outorder,inorder = np.argsort(u,kind='stable'),np.argsort(v,kind='stable')
    successors = ArcAdjacency(v,outorder,np.r_[0,np.cumsum(outdeg)])
    predecessors = ArcAdjacency(u,inorder,np.r_[0,np.cumsum(indeg)])
    # Lowest arc ID among reversed endpoint pairs, excluding the arc itself.
    pairs = {}
    for i,(a,b) in enumerate(arcs):
        pairs.setdefault((int(a),int(b)),[]).append(i)
    twin = np.full(len(arcs),-1,dtype=np.int64)
    for i,(a,b) in enumerate(arcs):
        for j in pairs.get((int(b),int(a)),[])[:2]:
            if j != i:
                twin[i] = j
                break
    adj = csr_matrix((np.ones(len(u),dtype=np.int8),(u,v)),shape=(n,n))
    _, weak = connected_components(adj,directed=True,connection='weak')
    _, strong = connected_components(adj,directed=True,connection='strong')
    a,b = ll[u],ll[v]
    return RoadGraph(str(Path(trip_dir).name),projection,xy,ll,u,v,np.linalg.norm(xy[v]-xy[u],axis=1),bearing_deg(a[:,0],a[:,1],b[:,0],b[:,1]),successors,predecessors,twin,twin<0,indeg,outdeg,weak,strong,np.flatnonzero(indeg+outdeg==0),perf_counter()-start)
