"""Profile fixed development settings on permitted extreme trips; no tuning."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import numpy as np
import pandas as pd
from src.common.paths import PROJECT_ROOT
from src.api.service import ArtifactLoader, load_scorer
from src.data.dataset_loader import list_trips, load_track
from src.data.config import load_config
from src.hmm.adapter import decoder_config_from_dict
from src.inference.pipeline import run_trip


def peak_rss_bytes():
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)] + [
                (k,ctypes.c_size_t) for k in ['PeakWorkingSetSize','WorkingSetSize',
                'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
        counters=Counters(); counters.cb=ctypes.sizeof(counters)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform=='darwin' else value*1024


def worker(trip,models):
    started=perf_counter(); bundle=ArtifactLoader(Path(models)).load()
    scorer=load_scorer(Path(models),bundle)
    startup=(perf_counter()-started)*1000
    data=load_config(PROJECT_ROOT/'configs/data.yaml')
    decoder=decoder_config_from_dict(bundle.decoder_config)
    result={}
    graph=None
    for label in ['cold_graph','warm_graph']:
        started=perf_counter()
        output=run_trip(Path(trip),scorer,data,decoder,graph=graph)
        result[label]={'wall_ms':round((perf_counter()-started)*1000,3),**output['timings']}
        graph=output['graph']
    result.update(trip_id=Path(trip).name,n_obs=len(output['observations']),n_arcs=len(graph.arc_from),
                  n_candidates=len(output['candidates']),model_startup_ms=startup,
                  peak_process_working_set_bytes=int(peak_rss_bytes()),
                  memory_method='OS process lifetime peak resident working set; fresh child per case',
                  n_breaks_time=output['hmm']['n_breaks_time'],
                  n_breaks_no_transition=output['hmm']['n_breaks_no_transition'])
    return result


def profile(raw_root,models,output):
    inventory=[]
    # list_trips restricts discovery to train/student_test, excluding evaluator inputs.
    for partition,tid,path in list_trips(raw_root):
        track=load_track(path)
        dt=np.diff(track.t.to_numpy())
        gaps=dt[np.isfinite(dt)&(dt>0)]
        with (path/f'{tid}.arcs').open() as stream:
            n_arcs=sum(bool(line.strip()) for line in stream)
        inventory.append(dict(partition=partition,trip_id=tid,n_obs=len(track),n_arcs=n_arcs,
                              max_gap_s=float(gaps.max()) if len(gaps) else 0))
    table=pd.DataFrame(inventory)
    target=Path(output); target.mkdir(parents=True,exist_ok=True)
    table.to_csv(target/'permitted_inventory.csv',index=False)
    results=[]
    for criterion in ['n_obs','n_arcs','max_gap_s']:
        row=table.loc[table[criterion].idxmax()]
        trip=Path(raw_root)/row.partition/row.trip_id
        child=subprocess.run([sys.executable,'-m','src.api.profile_development','--worker',
            str(trip),'--models-dir',str(models)],capture_output=True,text=True,check=True)
        result=json.loads(child.stdout); result.update(criterion=criterion,partition=row.partition)
        results.append(result)
        print(json.dumps(result),flush=True)
    report=dict(status='development_fixed_untuned_profile',python=platform.python_version(),
        platform=platform.platform(),processor=platform.processor(),logical_cpus=os.cpu_count(),
        omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
        cold_definition='Fresh process/model, uncached application graph; OS file cache not flushed',
        warm_definition='Same model and graph; features/candidates/decoding recomputed',
        scope='Route-free pipeline only, excludes optional reference evaluation and HTTP',results=results)
    (target/'profile.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report


def profile_sequence(raw_root,models,output):
    from src.data.dataset_loader import load_route
    from src.hmm.route_metrics import sequence_similarity
    trips=list_trips(raw_root)
    partition,tid,trip=max(trips,key=lambda item:len(load_track(item[2])))
    bundle=ArtifactLoader(Path(models)).load()
    result=run_trip(trip,load_scorer(Path(models),bundle),load_config(PROJECT_ROOT/'configs/data.yaml'),
                    decoder_config_from_dict(bundle.decoder_config))
    reference=load_route(trip,len(result['graph'].arc_from)); rows=[]
    for name in ['classifier','hmm']:
        started=perf_counter(); metrics=sequence_similarity(result[name]['route'],reference)
        rows.append(dict(method=name,sequence_metric_ms=(perf_counter()-started)*1000,**metrics))
    report=dict(status='development_baseline_sequence_profile',partition=partition,trip_id=tid,
        n_obs=len(result['observations']),selection='Longest permitted raw track from inventory',
        scope='Sequence comparison only, not final accuracy or OOF tuning',results=rows)
    target=Path(output); target.mkdir(parents=True,exist_ok=True)
    (target/'sequence_longest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root'); p.add_argument('--worker')
    p.add_argument('--models-dir',default='models/development')
    p.add_argument('--output',default='results/rehearsal/profile')
    p.add_argument('--sequence-only',action='store_true')
    a=p.parse_args()
    if a.worker:
        if Path(a.worker).parent.name not in {'train','student_test'}:
            p.error('Worker accepts permitted development trips only')
        print(json.dumps(worker(a.worker,a.models_dir)))
    else:
        if not a.raw_root: p.error('--raw-root is required')
        if a.sequence_only:
            print(json.dumps(profile_sequence(a.raw_root,a.models_dir,a.output),indent=2))
        else:
            profile(a.raw_root,a.models_dir,a.output)
