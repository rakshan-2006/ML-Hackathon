"""Bounded-memory, atomic offline build. Never opens disallowed partitions."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.common.schema import SCHEMA_VERSION,validate_tables
from .config import load_config
from .dataset_loader import list_trips,load_track,load_route,raw_path
from .road_graph import load_road_graph
from .track_cleaning import clean_track
from .candidate_generation import generate_candidates
from .reference_alignment import align_reference
from .feature_engineering import build_features,reference_membership

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def semantic_hash(df):
    # Column names + dtypes + ordered pandas element hashes; runtime pinned in manifest.
    h = hashlib.sha256(json.dumps([(c,str(df[c].dtype)) for c in df],separators=(',',':')).encode())
    h.update(pd.util.hash_pandas_object(df,index=False,categorize=False).to_numpy().tobytes())
    return h.hexdigest()

def runtime_info():
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':{m:importlib.metadata.version(m) for m in ['numpy','pandas','scipy','pyproj','shapely','matplotlib','folium','PyYAML','pytest','pyarrow']}}

def git_info():
    def run(*args):
        return subprocess.check_output(['git',*args],text=True,stderr=subprocess.DEVNULL).strip()
    try:
        return {'revision':run('rev-parse','HEAD'),'dirty':bool(run('status','--porcelain'))}
    except (subprocess.CalledProcessError,FileNotFoundError):
        return {'revision':None,'dirty':None}

def process_trip(path,partition,trip_id,cfg):
    graph = load_road_graph(path)
    obs = clean_track(load_track(path),graph.projection,trip_id,partition,cfg)
    cand = generate_candidates(obs,graph,cfg['radius_m'],cfg['max_candidates'])
    alignment = reference_membership(align_reference(obs,graph,load_route(path,len(graph.arc_from)),cfg),cand)
    features = build_features(obs,cand,graph,with_labels=True,alignment=alignment)
    validate_tables(obs,features,alignment,graph,cfg['radius_m'],cfg['max_candidates'])
    return obs,features,alignment,graph

def build(raw_root,output_root,config,partitions=('train','student_test'),trips=None,overwrite=False):
    cfg = load_config(config)
    discovered = list_trips(raw_root,partitions)
    if trips:
        missing = set(trips)-{x[1] for x in discovered}
        if missing:
            raise ValueError(f'Unknown trips: {missing}')
        discovered = [x for x in discovered if x[1] in trips]
    output = Path(output_root)
    output.mkdir(parents=True,exist_ok=True)
    target = output/'manifest.json'
    final_names = [f'{p}_{kind}.parquet' for p in partitions for kind in ['observations','candidates','alignment']]
    if not overwrite and (target.exists() or any((output/n).exists() for n in final_names)):
        raise FileExistsError('Outputs exist; use --overwrite (no cache/resume)')
    # A failed overwrite cannot leave an old success marker over new data.
    target.unlink(missing_ok=True)
    start = perf_counter()
    manifest = {'schema_version':SCHEMA_VERSION,'effective_config':cfg,'config_hash':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),'seed':cfg['seed'],'runtime':runtime_info(),'code':git_info(),'command':sys.argv,'partitions':{},'trips':[],'outputs':{}}
    # Scrub machine-local paths from the public manifest command.
    manifest['command'] = ['python','-m','src.data.build_dataset','--raw-root','<external-root>','--output-root','<output-root>','--config',str(Path(config).name),'--partitions',*partitions]
    if trips:
        manifest['command'] += ['--trips',*trips]
    if overwrite:
        manifest['command'] += ['--overwrite']
    writers = {}
    temporary = []
    try:
        for partition,trip_id,path in discovered:
            trip_start = perf_counter()
            obs,cand,alignment,graph = process_trip(path,partition,trip_id,cfg)
            record = {'partition':partition,'trip_id':trip_id,'source':{},'tables':{},'graph_time_s':graph.build_time_s,'nodes':len(graph.node_xy),'arcs':len(graph.arc_from)}
            for kind,count in [('track',len(obs)),('nodes',len(graph.node_xy)),('arcs',len(graph.arc_from)),('route',len(load_route(path,len(graph.arc_from))))]:
                source = raw_path(path,kind)
                with source.open() as stream:
                    raw_count = sum(bool(line.split()) for line in stream)
                if raw_count!=count:
                    raise ValueError(f'{trip_id}/{kind}: raw/loaded row mismatch')
                record['source'][kind] = {'sha256':sha256_file(source),'nonempty_records':raw_count,'loaded_rows':count,'size_bytes':source.stat().st_size}
            for kind,df in [('observations',obs),('candidates',cand),('alignment',alignment)]:
                name = f'{partition}_{kind}.parquet'
                table = pa.Table.from_pandas(df,preserve_index=False)
                if name not in writers:
                    partial = output/(name+'.partial')
                    temporary.append(partial)
                    writers[name] = pq.ParquetWriter(partial,table.schema,compression='zstd')
                    manifest['outputs'][name] = {'rows':0,'ordered_trip_semantic_hashes':[],'dtypes':{c:str(df[c].dtype) for c in df}}
                writers[name].write_table(table)
                content_hash = semantic_hash(df)
                manifest['outputs'][name]['rows'] += len(df)
                manifest['outputs'][name]['ordered_trip_semantic_hashes'].append([trip_id,content_hash])
                record['tables'][kind] = {'rows':len(df),'semantic_sha256':content_hash}
            record['elapsed_s'] = perf_counter()-trip_start
            manifest['trips'].append(record)
            manifest['partitions'].setdefault(partition,[]).append(trip_id)
            print(f'{partition}/{trip_id}: {len(obs)} obs, {len(cand)} candidates, {record["elapsed_s"]:.2f}s',flush=True)
        for writer in writers.values():
            writer.close()
        writers.clear()
        for partial in temporary:
            name = partial.name.removesuffix('.partial')
            os.replace(partial,output/name)
            manifest['outputs'][name]['file_sha256'] = sha256_file(output/name)
            semantic = manifest['outputs'][name]['ordered_trip_semantic_hashes']
            manifest['outputs'][name]['semantic_sha256'] = hashlib.sha256(json.dumps(semantic,separators=(',',':')).encode()).hexdigest()
        manifest['elapsed_s'] = perf_counter()-start
        partial_manifest = output/'manifest.json.partial'
        partial_manifest.write_text(json.dumps(manifest,indent=2),encoding='utf8')
        os.replace(partial_manifest,target)
        return manifest
    finally:
        for writer in writers.values():
            writer.close()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',default=os.environ.get('PATHSNAP_RAW_ROOT'),required=not bool(os.environ.get('PATHSNAP_RAW_ROOT')))
    p.add_argument('--output-root',default='data/processed')
    p.add_argument('--config',default='experiments/data_config.yaml')
    p.add_argument('--partitions',nargs='+',choices=['train','student_test'],default=['train','student_test'])
    p.add_argument('--trips',nargs='+')
    p.add_argument('--overwrite',action='store_true')
    a = p.parse_args()
    build(a.raw_root,a.output_root,a.config,a.partitions,a.trips,a.overwrite)

if __name__=='__main__':
    main()
