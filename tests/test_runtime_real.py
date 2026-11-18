import builtins
from pathlib import Path
import shutil
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.common.paths import PROJECT_ROOT,Settings
from src.data.config import load_config
from src.inference.pipeline import run_trip
from src.hmm.adapter import decoder_config_from_dict
from tests.runtime_helpers import write_bundle


def settings(tmp_path,reference=False):
    trip=tmp_path/'trips/tiny_trip'
    shutil.copytree(PROJECT_ROOT/'tests/fixtures/tiny_trip',trip)
    if not reference:
        (trip/'tiny_trip.route').unlink()
    write_bundle(tmp_path/'models')
    return Settings('real',tmp_path/'trips',tmp_path/'models',tmp_path/'logs','fixture','fixture')


@pytest.mark.parametrize('reference',[False,True])
def test_actual_api_prediction_and_isolated_optional_evaluation(tmp_path,reference):
    cfg=settings(tmp_path,reference)
    app=create_app(cfg)
    with TestClient(app) as client:
        assert client.get('/health').json()['status']=='development'
        assert client.get('/trips').json()==['tiny_trip']
        response=client.post('/predict',json={'trip_id':'tiny_trip'})
        assert response.status_code==200,response.text
        result=response.json()
        assert result['is_stub'] is False and result['n_obs']==4
        assert result['route_hmm'] and len(result['gps_coordinates'])==4
        assert result['diagnostics']['reference_available'] is reference
        assert result['diagnostics']['request_id']==response.headers['X-Request-ID']
        assert 'ref_arc_id' not in response.text and 'reference_route' not in response.text
        assert result['timings']['decode']>=0
        assert ('top1_accuracy_hmm' in result['metrics']) is reference


def test_route_free_pipeline_is_invariant_and_instrumented(tmp_path,monkeypatch):
    cfg=settings(tmp_path,True)
    trip=cfg.evaluation_dir/'tiny_trip'
    data=load_config(PROJECT_ROOT/'configs/data.yaml')
    score=lambda cand: np.full(len(cand),.5)
    first=run_trip(trip,score,data)
    (trip/'tiny_trip.route').write_text('9999999\n')
    original=Path.open
    def guarded(path,*a,**kw):
        if path.suffix=='.route':
            raise AssertionError('Prediction tried reading a reference route')
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    second=run_trip(trip,score,data)
    (trip/'tiny_trip.route').unlink()
    third=run_trip(trip,score,data)
    for other in [second,third]:
        assert first['candidates'].equals(other['candidates'])
        assert first['hmm']['pred_arc_ids']==other['hmm']['pred_arc_ids']


def test_final_gate_rejects_development_bundle(tmp_path):
    cfg=settings(tmp_path)
    from dataclasses import replace
    with pytest.raises(RuntimeError,match='Final model'):
        with TestClient(create_app(replace(cfg,require_final=True))):
            pass


def test_invalid_config_and_nonfinite_probabilities_rejected():
    for config in [{'beta_m':0},{'gap_split_s':-1},{'emission_eps':2}]:
        with pytest.raises(ValueError):
            decoder_config_from_dict(config)


def test_metrics_keep_gap_and_missing_boundaries():
    from src.hmm.route_metrics import route_continuity
    from src.hmm.transition import RoadGraph
    graph=RoadGraph([0,2],[1,3],[10,10])
    assert route_continuity([0,1],graph)['n_breaks']==1
    assert route_continuity([0,1],graph,[0,1])['n_transitions']==0
    assert route_continuity([0,-1,1],graph)['n_transitions']==0
