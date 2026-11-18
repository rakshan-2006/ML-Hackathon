import json
import numpy as np
import pandas as pd
import pytest
from src.hmm.tune import keyed_scores,validate_provenance,sha256
from src.common.schema import SCHEMA_VERSION
from src.hmm.transition import RoadGraph,route_distance
from src.hmm.route_metrics import _lcs_length


def test_keyed_oof_join_preserves_shuffled_candidate_identity():
    candidates=pd.DataFrame(dict(trip_id=['a','a'],obs_idx=[0,0],arc_id=[2,1]))
    scores=candidates.assign(p_model=[.8,.2],model_version='final1',fold=0).iloc[::-1]
    folds=pd.DataFrame(dict(trip_id=['a'],fold=[0]))
    assert keyed_scores(candidates,scores,'final1',folds).tolist()==[.8,.2]
    with pytest.raises(ValueError,match='Duplicate'):
        keyed_scores(candidates,pd.concat([scores.iloc[:1]]*2),'final1',folds)
    with pytest.raises(ValueError,match='fold'):
        keyed_scores(candidates,scores.assign(fold=1),'final1',folds)
    with pytest.raises(ValueError,match='keys'):
        keyed_scores(candidates,scores.assign(arc_id=[8,9]),'final1',folds)


def test_final_oof_gate_requires_hashes_and_held_out_trip_provenance(tmp_path):
    path=tmp_path/'proof'; path.write_text('fixture')
    paths={'scores':path}; folds=pd.DataFrame(dict(trip_id=['a'],fold=[0]))
    card=dict(status='final',model_version='v1')
    manifest=dict(status='final_parameter_oof',schema_version=SCHEMA_VERSION,model_version='v1',
        partition='train',score_kind='out_of_fold',sha256={'scores':sha256(path)},fold_holdout_trip_ids={'0':['a']})
    validate_provenance(manifest,card,paths,folds)
    with pytest.raises(ValueError,match='final selected'):
        validate_provenance(manifest,{**card,'status':'development_baseline'},paths,folds)
    with pytest.raises(ValueError,match='held-out'):
        validate_provenance({**manifest,'fold_holdout_trip_ids':{}},card,paths,folds)
    path.write_text('changed')
    with pytest.raises(ValueError,match='checksum'):
        validate_provenance(manifest,card,paths,folds)


def test_distance_bound_cache_and_same_arc_speed_bound():
    graph=RoadGraph([0,1,2],[1,2,3],[100,100,100])
    cache={}
    assert np.isinf(route_distance(graph,0,1,2,0,cache,50,0)[0])
    assert route_distance(graph,0,1,2,0,cache,200,0)[0]==100
    assert np.isinf(route_distance(graph,0,0,0,1,{},50,0)[0])


def test_manifest_rejects_partial_hash_binding(tmp_path):
    from src.api.service import ArtifactLoader,ArtifactUnavailableError
    from tests.runtime_helpers import write_bundle
    write_bundle(tmp_path)
    (tmp_path/'deployment_manifest.json').write_text(json.dumps({'sha256':{}}))
    with pytest.raises(ArtifactUnavailableError,match='every required artifact'):
        ArtifactLoader(tmp_path).load()


def test_synthetic_oof_selection_writes_traceable_trials(tmp_path,monkeypatch):
    # Algorithm integration only: no real final model or development data tuning.
    import shutil
    import importlib
    tuner=importlib.import_module('src.hmm.tune')
    from src.common.paths import PROJECT_ROOT
    from src.data.config import load_config
    from src.data.reference_alignment import align_reference
    from src.data.dataset_loader import load_route
    from src.inference.pipeline import run_trip
    trip=tmp_path/'train/tiny_trip'
    shutil.copytree(PROJECT_ROOT/'tests/fixtures/tiny_trip',trip)
    config_dir=tmp_path/'configs'; config_dir.mkdir()
    shutil.copyfile(PROJECT_ROOT/'configs/data.yaml',config_dir/'data.yaml')
    folds=pd.DataFrame({'trip_id':['tiny_trip'],'fold':[0]})
    folds.to_csv(config_dir/'cv_folds.csv',index=False)
    data=load_config(config_dir/'data.yaml')
    output=run_trip(trip,lambda df:np.full(len(df),.5),data)
    reference=load_route(trip,len(output['graph'].arc_from))
    aligned=align_reference(output['observations'],output['graph'],reference,data)
    processed=tmp_path/'processed'; processed.mkdir()
    paths={}
    for name,frame in [('candidates',output['candidates']),('observations',output['observations']),('alignment',aligned)]:
        path=processed/f'train_{name}.parquet'; frame.to_parquet(path,index=False);paths[name]=path
    scores=output['candidates'][['trip_id','obs_idx','arc_id']].assign(p_model=.5,fold=0,model_version='synthetic-final')
    scores.to_parquet(tmp_path/'scores.parquet',index=False)
    card=dict(status='final',model_version='synthetic-final')
    (tmp_path/'card.json').write_text(json.dumps(card))
    paths.update(scores=tmp_path/'scores.parquet',model_card=tmp_path/'card.json',
                 folds=config_dir/'cv_folds.csv',data_config=config_dir/'data.yaml')
    manifest=dict(status='final_parameter_oof',schema_version=SCHEMA_VERSION,model_version='synthetic-final',
                  partition='train',score_kind='out_of_fold',fold_holdout_trip_ids={'0':['tiny_trip']},
                  sha256={name:sha256(path) for name,path in paths.items()})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(tuner,'PROJECT_ROOT',tmp_path)
    selected=tuner.tune(tmp_path/'train',processed,tmp_path/'scores.parquet',tmp_path/'manifest.json',
                        tmp_path/'card.json',tmp_path/'trials')
    assert selected['status']=='final_tuned'
    assert len(pd.read_csv(tmp_path/'trials/trials.csv'))==12
    assert len(pd.read_csv(tmp_path/'trials/per_trip.csv'))==24
    assert selected['tuning_provenance']['trials_sha256']==sha256(tmp_path/'trials/trials.csv')


def test_bit_parallel_lcs_matches_dynamic_programming_randomized():
    rng=np.random.default_rng(8)
    for _ in range(100):
        a=rng.integers(0,5,size=int(rng.integers(0,30))).tolist()
        b=rng.integers(0,5,size=int(rng.integers(0,30))).tolist()
        dp=[0]*(len(b)+1)
        for value in a:
            old=dp[:]
            for j,other in enumerate(b,1):
                dp[j]=old[j-1]+1 if value==other else max(old[j],dp[j-1])
        assert _lcs_length(a,b)==dp[-1]


def test_ui_result_survives_component_rerun(monkeypatch):
    import requests
    import streamlit_folium
    from streamlit.testing.v1 import AppTest
    from src.common.paths import PROJECT_ROOT
    class Response:
        def __init__(self,value): self.value=value
        def json(self): return self.value
        def raise_for_status(self): pass
    monkeypatch.setattr(requests,'get',lambda url,**kw:Response(
        dict(status='development',model_name='fixture',schema_version=SCHEMA_VERSION)
        if url.endswith('/health') else ['tiny_trip']))
    calls=[]
    def post(*args,**kwargs):
        calls.append(kwargs)
        return Response(dict(trip_id='tiny_trip',n_obs=1,is_stub=False,
            gps_coordinates=[dict(latitude=0,longitude=0)],route_hmm=[],route_classifier=[],
            metrics={'top1_accuracy_hmm':.75},diagnostics={'deployment_status':'development_baseline'},timings={}))
    monkeypatch.setattr(requests,'post',post)
    monkeypatch.setattr(streamlit_folium,'st_folium',lambda *a,**kw:None)
    app=AppTest.from_file(str(PROJECT_ROOT/'ui/app.py')).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[1].value=='75.00%'
    app.run()
    assert not app.exception and app.metric[1].value=='75.00%' and len(calls)==1
