import numpy as np
import pytest
from src.data.geo import TripProjection, haversine_m, bearing_deg
from src.data.road_graph import load_road_graph
from src.data.track_cleaning import clean_track
from src.data.dataset_loader import load_track
from tests.test_data_loader import FIXTURE

def test_graph():
    g = load_road_graph(FIXTURE)
    assert g.successors[0].tolist() == [1,2,4]
    assert g.predecessors[2].tolist() == [0,3]
    assert g.reverse_twin.tolist() == [1,0,3,2,-1,-1]
    assert len(set(g.weak_component)) == 2
    assert g.is_oneway[4]
    assert g.arc_len_m[0] == pytest.approx(111.31949,rel=1e-5)

def test_projection():
    for ll in [np.array([[179.999,-40],[-179.999,-40]]),np.array([[20,-30],[20.001,-30]])]:
        p = TripProjection.from_lonlat(ll)
        xy = p.project(ll)
        assert np.allclose(p.inverse(xy),ll)
        assert np.linalg.norm(xy[1]-xy[0]) == pytest.approx(haversine_m(*ll[0],*ll[1]),rel=.005)
    assert bearing_deg(0,0,1,0) == pytest.approx(90)
    assert np.isnan(bearing_deg(0,0,0,0))

def test_clean_preserves_invalid():
    g = load_road_graph(FIXTURE)
    t = load_track(FIXTURE)
    t.loc[1,'lon'] = np.nan
    t.loc[2,'t'] = t.loc[0,'t']
    t.loc[3,'t'] = 999
    o = clean_track(t,g.projection,'00000001')
    assert o.obs_idx.tolist() == [0,1,2,3]
    assert not o.is_valid.iloc[1]
    assert np.isnan(o.speed_mps.iloc[2]) and o.gap_flag.iloc[3]

def test_parallel_self_loop_empty(tmp_path):
    name = tmp_path.name
    (tmp_path/f'{name}.nodes').write_text('0 0\n.001 0\n.002 0\n')
    (tmp_path/f'{name}.arcs').write_text('0 1\n0 1\n1 0\n1 1\n')
    g = load_road_graph(tmp_path)
    assert g.reverse_twin.tolist() == [2,2,0,-1]
    assert g.arc_len_m[3] == 0 and np.isnan(g.arc_bearing[3])
    assert g.isolated_nodes.tolist() == [2]
    (tmp_path/f'{name}.nodes').write_text('')
    (tmp_path/f'{name}.arcs').write_text('')
    g = load_road_graph(tmp_path)
    assert g.node_xy.shape == (0,2) and len(g.arc_from) == 0
