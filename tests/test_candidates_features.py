import builtins
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
from src.data.dataset_loader import load_track,load_route
from src.data.road_graph import load_road_graph
from src.data.track_cleaning import clean_track
from src.data.candidate_generation import generate_candidates,SegmentIndex,project_segments
from src.data.feature_engineering import build_features,reference_membership,join_labels
from src.data.reference_alignment import align_reference
from src.common.schema import FEATURES,C2,C3,typed_table,validate_tables
from tests.test_data_loader import FIXTURE

def pipeline(path=FIXTURE):
    g = load_road_graph(path)
    o = clean_track(load_track(path),g.projection,'00000001')
    c = generate_candidates(o,g,30,4)
    return g,o,c,build_features(o,c,g)

def test_candidates_features_labels():
    g,o,c,f = pipeline()
    assert c.groupby('obs_idx').arc_id.apply(list).iloc[0] == [0,1]
    assert c.dist_m.max()<30
    first = c.iloc[0]
    assert first.proj_frac == pytest.approx(.2,abs=1e-5)
    assert first.dist_m == pytest.approx(0,abs=.001)
    a = reference_membership(align_reference(o,g,load_route(FIXTURE)),c)
    f = build_features(o,c,g,with_labels=True,alignment=a)
    validate_tables(o,f,a,g,30,4)
    assert f.groupby('obs_idx').label.sum().max() <= 1
    assert not set(FEATURES)&{'lon','lat','x','y','trip_id','arc_id','obs_idx','label','ref_in_candidates'}
    shuffled = build_features(o.sample(frac=1,random_state=1),c.sample(frac=1,random_state=2),g)
    pd.testing.assert_frame_equal(f.drop(columns='label').sort_values(['obs_idx','arc_id']).reset_index(drop=True),shuffled.sort_values(['obs_idx','arc_id']).reset_index(drop=True))

def test_full_segment_zero_and_strict_radius():
    idx = SegmentIndex(np.array([[-1000.,0],[0,1],[0,0]]),np.array([[1000.,0],[0,1],[0,0]]))
    ids,d,frac,xy = idx.query(np.array([-999.,0]),1.)
    assert ids.tolist() == [0] and frac[0] == pytest.approx(.0005)
    ids,d,frac,xy = idx.query(np.array([0.,0]),.5)
    assert ids.tolist() == [0,2] and np.isfinite(frac).all()

def test_empty_and_nullable():
    g,o,_,_ = pipeline()
    o['x'] = o.x+1e6
    c = generate_candidates(o,g,1,1)
    assert not len(c) and o.n_candidates.sum()==0
    f = build_features(o,c,g)
    assert {k:str(f[k].dtype) for k in C2} == C2
    a = typed_table(dict(trip_id=['00000001'],obs_idx=[0],ref_arc_id=[99],ref_route_pos=[0],align_dist_m=[1.],align_status=['ok'],ref_in_candidates=[False]),C3)
    _,_,c,_ = pipeline()
    c = c[c.obs_idx==0]
    assert join_labels(c,a).label.sum() == 0
    a['align_status'] = 'ambiguous'
    assert join_labels(c,a).label.isna().all()

def test_alignment_order_breaks_outliers():
    g,o,c,f = pipeline()
    a = align_reference(o,g,np.array([0,1,0,2]))
    assert np.diff(a.loc[a.ref_route_pos>=0,'ref_route_pos']).min()>=0
    # Repeated occurrence ambiguity must not yield confident arbitrary labels.
    a2 = align_reference(o.iloc[:1].reset_index(drop=True),g,np.array([0,0]))
    assert a2.align_status.iloc[0] == 'ambiguous'
    broken = align_reference(o,g,np.array([0,5]))
    assert broken.attrs['route_break_count']==1
    assert 'ambiguous' in broken.align_status.tolist()
    o.loc[0,['x','y']] = [1e6,1e6]
    far = align_reference(o,g,np.array([0,2]))
    assert far.ref_arc_id.iloc[0] == -1 and far.align_status.iloc[0]=='unaligned'

def test_inference_never_reads_route(tmp_path,monkeypatch):
    trip = tmp_path/'00000001'
    trip.mkdir()
    for kind in ['nodes','arcs','track','route']:
        shutil.copyfile(FIXTURE/f'tiny_trip.{kind}',trip/f'00000001.{kind}')
    original = builtins.open
    original_path = Path.open
    def guarded(file,*args,**kwargs):
        assert not str(file).endswith('.route'), 'Route read during inference'
        return original(file,*args,**kwargs)
    def guarded_path(self,*args,**kwargs):
        assert self.suffix!='.route','Route read during inference'
        return original_path(self,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',guarded)
    monkeypatch.setattr(Path,'open',guarded_path)
    first = pipeline(trip)[-1]
    with original_path(trip/'00000001.route','w') as stream:
        stream.write('5\n')
    second = pipeline(trip)[-1]
    (trip/'00000001.route').unlink()
    third = pipeline(trip)[-1]
    pd.testing.assert_frame_equal(first,second)
    pd.testing.assert_frame_equal(first,third)
