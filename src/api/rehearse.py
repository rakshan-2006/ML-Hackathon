"""Loopback/remote HTTP rehearsal on STUDENT_TEST only, never evaluator inputs."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import platform
import requests
import pandas as pd
import numpy as np
from src.api.service import ArtifactLoader,load_scorer
from src.data.config import load_config
from src.hmm.adapter import decoder_config_from_dict
from src.inference.pipeline import run_trip
from src.visualization.folium_maps import create_trip_map
from src.common.paths import PROJECT_ROOT


def rehearse(api_url,raw_root,models_dir,output_root='results/rehearsal',timeout=1800,create_maps=False):
    raw=Path(raw_root).resolve()
    if raw.name!='student_test':
        raise ValueError('This preparation rehearsal only accepts the permitted student_test directory')
    result_root=Path(output_root); result_root.mkdir(parents=True,exist_ok=True)
    health=requests.get(f'{api_url}/health',timeout=10); health.raise_for_status()
    trips_response=requests.get(f'{api_url}/trips',timeout=10); trips_response.raise_for_status()
    ids=trips_response.json()
    expected=sorted(p.name for p in raw.iterdir() if p.is_dir() and (p/f'{p.name}.track').is_file())
    if sorted(ids)!=expected:
        raise ValueError('Server trips differ from permitted rehearsal root')
    bundle=ArtifactLoader(Path(models_dir)).load()
    scorer=load_scorer(Path(models_dir),bundle)
    decoder=decoder_config_from_dict(bundle.decoder_config)
    data=load_config(PROJECT_ROOT/'configs/data.yaml')
    track_sizes={tid:sum(1 for line in (raw/tid/f'{tid}.track').open() if line.strip()) for tid in ids}
    order=sorted(ids,key=lambda tid:track_sizes[tid])
    # Select maps by trajectory length, before looking at accuracy.
    selected={order[0],order[len(order)//2],order[-1]} if ids else set()
    rows=[]
    for tid in order:
        started=perf_counter()
        response=requests.post(f'{api_url}/predict',json={'trip_id':tid},timeout=timeout)
        response.raise_for_status(); payload=response.json()
        if payload['is_stub'] or payload['mode']!='real':
            raise ValueError('Rehearsal returned a stub')
        offline=run_trip(raw/tid,scorer,data,decoder)
        # Public geometry and row counts agree with the independently called pipeline.
        for key in ['route_hmm','route_classifier']:
            actual=[[[p['latitude'],p['longitude']] for p in line['points']] for line in payload[key]]
            expected_geometry=[[[float(lat),float(lon)] for lat,lon in line] for line in offline[key]]
            if actual!=expected_geometry:
                raise AssertionError('HTTP and offline geometry disagree')
        if payload['n_obs']!=track_sizes[tid]:
            raise AssertionError('Raw observation count mismatch')
        row=dict(trip_id=tid,n_obs=payload['n_obs'],http_status=response.status_code,
                 loopback_only=api_url.startswith('http://127.0.0.1') or api_url.startswith('http://localhost'),
                 model_version=payload['diagnostics']['model_version'],
                 decoder_version=payload['diagnostics']['decoder_version'],
                 deployment_status=payload['diagnostics']['deployment_status'],
                 offline_geometry_equal=True,request_id=response.headers['X-Request-ID'],
                 **payload['metrics'],**{f'{k}_ms':v for k,v in payload['timings'].items()},
                 verification_wall_s=perf_counter()-started)
        rows.append(row)
        pd.DataFrame(rows).to_csv(result_root/'student_test_api.csv',index=False)
        # Keep full responses with GPS coordinates local, outside committed evidence.
        response_dir=PROJECT_ROOT/'artifacts/rehearsal_responses'; response_dir.mkdir(parents=True,exist_ok=True)
        (response_dir/f'{tid}.json').write_text(json.dumps(payload,indent=2))
        if create_maps and tid in selected:
            create_trip_map(offline,raw/tid,PROJECT_ROOT/f'maps/development/{tid}.html')
        print(json.dumps({'trip_id':tid,'n_obs':payload['n_obs'],'http_ms':payload['timings']['total'],
                          'top1_classifier':payload['metrics'].get('top1_accuracy_classifier'),
                          'top1_hmm':payload['metrics'].get('top1_accuracy_hmm')}),flush=True)
    table=pd.DataFrame(rows)
    aggregate={}
    for name in ['classifier','hmm']:
        n=table[f'n_evaluated_{name}'].sum()
        aggregate[f'top1_{name}']=float(table[f'n_correct_{name}'].sum()/n) if n else None
    report=dict(status='development_baseline_rehearsal',trip_count=len(rows),partition='student_test',
                all_http_offline_geometry_equal=True,two_laptop_verified=False,
                evaluation_archive_opened=False,python=platform.python_version(),
                aggregate=aggregate,api_url=api_url,
                map_selection='Shortest, median and longest allowed trajectory before observing metrics',
                final_model_and_decoder_pending=True)
    (result_root/'summary.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--api-url',default='http://127.0.0.1:8011')
    p.add_argument('--raw-root',required=True)
    p.add_argument('--models-dir',default='models/development')
    p.add_argument('--output-root',default='results/rehearsal')
    p.add_argument('--create-maps',action='store_true')
    a=p.parse_args()
    print(json.dumps(rehearse(a.api_url,a.raw_root,a.models_dir,a.output_root,create_maps=a.create_maps),indent=2))


if __name__=='__main__':
    main()
