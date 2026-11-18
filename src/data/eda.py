"""Train-only candidate selection, full inventory, and model-readiness evidence.

select never touches student_test. audit uses the frozen processed outputs and
does not modify configuration. Tables, rather than individual GPS coordinates,
are retained as small public evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from .build_dataset import load_config,semantic_hash,runtime_info
from .dataset_loader import list_trips,load_track,load_route,read_records,raw_path
from .road_graph import load_road_graph
from .track_cleaning import clean_track
from .candidate_generation import SegmentIndex
from .reference_alignment import align_reference,route_break_positions
from .geo import GEOD,valid_lonlat

def select_config(raw_root,config,evidence):
    cfg = load_config(config)
    out = Path(evidence); out.mkdir(parents=True,exist_ok=True)
    radii,ks = cfg['sweep_radii_m'],cfg['sweep_k']
    records = []
    started = perf_counter()
    for partition,trip_id,path in list_trips(raw_root,['train']):
        start = perf_counter()
        g = load_road_graph(path)
        o = clean_track(load_track(path),g.projection,trip_id,partition,cfg)
        a = align_reference(o,g,load_route(path,len(g.arc_from)),cfg)
        trusted = (a.align_status=='ok').to_numpy() & o.is_valid.to_numpy()
        idx = SegmentIndex(g.node_xy[g.arc_from],g.node_xy[g.arc_to])
        # One maximum-radius query is exactly reusable for nested radii and K.
        stats = {(r,k):dict(hits=0,zero=0,truncated=0,count=0,counts=[]) for r in radii for k in ks}
        query_start = perf_counter()
        for i,row in enumerate(o.itertuples(index=False)):
            ids,d,_,_ = idx.query(np.array([row.x,row.y]),max(radii)) if row.is_valid else (np.array([],dtype=int),np.array([]),None,None)
            for r in radii:
                inside = ids[d<=r]
                for k in ks:
                    s = stats[r,k]; retained = inside[:k]
                    s['count'] += len(retained); s['counts'].append(len(retained))
                    s['zero'] += bool(row.is_valid and not len(retained))
                    s['truncated'] += bool(row.is_valid and len(inside)>k)
                    s['hits'] += bool(trusted[i] and a.ref_arc_id.iloc[i] in retained)
        query_s = perf_counter()-query_start
        for (r,k),s in stats.items():
            counts = s.pop('counts')
            records.append(dict(partition=partition,trip_id=trip_id,radius_m=r,max_candidates=k,observations=len(o),valid=int(o.is_valid.sum()),trusted=int(trusted.sum()),**s,candidates_p50=float(np.median(counts)),candidates_p95=float(np.quantile(counts,.95)),shared_query_wall_s=query_s,trip_wall_s=perf_counter()-start))
        pd.DataFrame(records).to_csv(out/'candidate_sweep_per_trip.csv',index=False)
        print(f'select train/{trip_id}: {len(o)} obs, trusted={trusted.sum()}, {perf_counter()-start:.2f}s',flush=True)
    all_rows = pd.DataFrame(records)
    sums = all_rows.groupby(['radius_m','max_candidates'],as_index=False)[['observations','valid','trusted','hits','zero','truncated','count']].sum()
    sums['reference_recall_trusted'] = sums.hits/sums.trusted
    sums['zero_fraction_valid'] = sums.zero/sums.valid
    sums['truncated_fraction_valid'] = sums.truncated/sums.valid
    sums['mean_candidates_per_observation'] = sums['count']/sums.observations
    # Predeclared selection: within 0.5 percentage points of best recall, smallest
    # mean candidate table; deterministic R then K tie-break. No student-test use.
    threshold = sums.reference_recall_trusted.max()-.005
    winning = sums[sums.reference_recall_trusted>=threshold].sort_values(['mean_candidates_per_observation','radius_m','max_candidates']).iloc[0]
    cfg['radius_m'],cfg['max_candidates'] = float(winning.radius_m),int(winning.max_candidates)
    cfg['selection_partition'] = 'train'
    cfg['frozen'] = True
    Path(config).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8')
    sums.to_csv(out/'candidate_sweep.csv',index=False)
    (out/'selection.json').write_text(json.dumps({'policy':'Smallest mean candidate count within 0.005 absolute micro trusted-reference recall of sweep maximum; ties radius then K','partition':'train','trip_ids':sorted(all_rows.trip_id.unique().tolist()),'selected_radius_m':cfg['radius_m'],'selected_max_candidates':cfg['max_candidates'],'recall_threshold':threshold,'elapsed_s':perf_counter()-started,'timing_note':'Each trip queries max radius once; shared query time covers all nested R/K statistics, not independent per-config benchmarks. Final build measures selected configuration.','config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()},indent=2),encoding='utf8')
    print(sums.to_string(index=False),flush=True)

def graph_stats(g):
    pairs = np.column_stack([g.arc_from,g.arc_to])
    return dict(nodes=len(g.node_xy),arcs=len(g.arc_from),zero_length_arcs=int((g.arc_len_m<1e-9).sum()),self_loops=int((g.arc_from==g.arc_to).sum()),duplicate_arcs=int(len(pairs)-len(np.unique(pairs,axis=0))),reverse_arc_count=int((g.reverse_twin>=0).sum()),oneway_arcs=int(g.is_oneway.sum()),isolated_nodes=len(g.isolated_nodes),weak_components=len(np.unique(g.weak_component)),strong_components=len(np.unique(g.strong_component)),largest_weak_nodes=int(np.bincount(g.weak_component).max()) if len(g.node_xy) else 0,graph_build_s=g.build_time_s,array_bytes=int(sum(a.nbytes for a in vars(g).values() if isinstance(a,np.ndarray))))

def audit(raw_root,processed_root,config,evidence,figures):
    cfg = load_config(config)
    out = Path(evidence); out.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((Path(processed_root)/'manifest.json').read_text())
    if cfg != manifest['effective_config']:
        raise ValueError('Audit config differs from build config')
    inventory,graphs,alignments,coverage,projection,quality = [],[],[],[],[],[]
    # Streaming row-group readers avoid full combined candidate-table memory.
    tables = {(p,k):pq.ParquetFile(Path(processed_root)/f'{p}_{k}.parquet') for p in manifest['partitions'] for k in ['observations','candidates','alignment']}
    # One writer row group per trip (all actual trips fit the writer row-group limit).
    for partition,trip_id,path in list_trips(raw_root,list(manifest['partitions'])):
        j = manifest['partitions'][partition].index(trip_id)
        o,c,a = [tables[partition,k].read_row_group(j).to_pandas() for k in ['observations','candidates','alignment']]
        g = load_road_graph(path)
        route = load_route(path,len(g.arc_from))
        row = dict(partition=partition,trip_id=trip_id)
        for kind,cols,integer,loaded in [('track',3,False,len(o)),('nodes',2,False,len(g.node_xy)),('arcs',2,True,len(g.arc_from)),('route',1,True,len(route))]:
            values,meta = read_records(raw_path(path,kind),cols,integer)
            assert meta['records']==loaded
            row.update({f'{kind}_{k}':v for k,v in meta.items()})
            row[f'{kind}_present'] = True
            row[f'{kind}_nonfinite_cells'] = int((~np.isfinite(values)).sum())
        dt = o.dt_prev.to_numpy(); finite_dt = dt[np.isfinite(dt)]
        row.update(invalid_observations=int((~o.is_valid).sum()),invalid_node_coordinates=int((~valid_lonlat(g.node_lonlat)).sum()),invalid_track_coordinates=int((~o.coordinate_valid).sum()),invalid_timestamps=int((~o.timestamp_valid).sum()),nonpositive_dt=int(o.nonpositive_dt.sum()),dt_p50=float(np.median(finite_dt)),dt_p95=float(np.quantile(finite_dt,.95)),dt_max_s=float(finite_dt.max()),large_gaps=int(o.gap_flag.sum()),duplicate_observations=int(o[['lon','lat','t']].duplicated().sum()),duplicate_coordinates=int(o[['lon','lat']].duplicated().sum()),stationary_steps=int((o.step_dist_m<cfg['stationary_m']).sum()),suspicious_speeds=int(o.suspicious_speed.sum()),lon_min=float(o.lon.min()),lon_max=float(o.lon.max()),lat_min=float(o.lat.min()),lat_max=float(o.lat.max()),node_lon_min=float(g.node_lonlat[:,0].min()),node_lon_max=float(g.node_lonlat[:,0].max()),node_lat_min=float(g.node_lonlat[:,1].min()),node_lat_max=float(g.node_lonlat[:,1].max()),route_break_count=len(route_break_positions(route,g)),malformed_references=0)
        inventory.append(row)
        graphs.append(dict(partition=partition,trip_id=trip_id,**graph_stats(g)))
        valid = o.is_valid.to_numpy(); trusted = (a.align_status=='ok').to_numpy()&valid
        hits = a.ref_in_candidates.to_numpy()&trusted
        misses = trusted&~hits
        al = dict(partition=partition,trip_id=trip_id,observations=len(o),ok=int((a.align_status=='ok').sum()),ambiguous=int((a.align_status=='ambiguous').sum()),unaligned=int((a.align_status=='unaligned').sum()),route_break_count=row['route_break_count'])
        for q in [.5,.9,.95,.99]:
            al[f'align_dist_p{int(q*100)}_m'] = float(a.align_dist_m.quantile(q))
        alignments.append(al)
        coverage.append(dict(partition=partition,trip_id=trip_id,observations=len(o),valid=int(valid.sum()),trusted=int(trusted.sum()),reference_hits=int(hits.sum()),reference_misses=int(misses.sum()),zero_candidates_valid=int(((o.n_candidates==0)&o.is_valid).sum()),candidates=len(c),labelled_candidates=int(c.label.notna().sum()),positive_candidates=int((c.label==1).sum()),miss_rate_trusted=float(misses.sum()/trusted.sum()) if trusted.sum() else None,known_misses_over_all_valid=float(misses.sum()/valid.sum()) if valid.sum() else None,alignment_coverage_valid=float(trusted.sum()/valid.sum()) if valid.sum() else None))
        # Deterministic evenly spaced node pairs plus a displaced permutation.
        sample = np.linspace(0,len(g.node_xy)-1,min(64,len(g.node_xy)),dtype=int)
        other = np.roll(sample,17)
        ll1,ll2 = g.node_lonlat[sample],g.node_lonlat[other]
        _,_,geodesic = GEOD.inv(ll1[:,0],ll1[:,1],ll2[:,0],ll2[:,1])
        projected = np.linalg.norm(g.node_xy[sample]-g.node_xy[other],axis=1)
        for u,v,d,dp in zip(sample,other,geodesic,projected):
            projection.append(dict(partition=partition,trip_id=trip_id,node_a=int(u),node_b=int(v),geodesic_m=float(d),projected_m=float(dp),absolute_error_m=float(abs(dp-d)),relative_error=float(abs(dp-d)/d) if d else 0))
        print(f'audit {partition}/{trip_id}: trusted {trusted.sum()}/{valid.sum()}, misses {misses.sum()}',flush=True)
    for filename,rows in [('inventory.csv',inventory),('graph_stats.csv',graphs),('alignment_stats.csv',alignments),('coverage.csv',coverage),('projection_errors.csv',projection)]:
        pd.DataFrame(rows).to_csv(out/filename,index=False)
    cv = pd.DataFrame(coverage)
    aggregate = cv.groupby('partition',as_index=False)[['observations','valid','trusted','reference_hits','reference_misses','zero_candidates_valid','candidates','labelled_candidates','positive_candidates']].sum()
    aggregate['miss_rate_trusted'] = aggregate.reference_misses/aggregate.trusted
    aggregate['known_misses_over_all_valid'] = aggregate.reference_misses/aggregate.valid
    aggregate['alignment_coverage_valid'] = aggregate.trusted/aggregate.valid
    aggregate.to_csv(out/'coverage_by_partition.csv',index=False)
    sweep = pd.read_csv(out/'candidate_sweep.csv')
    selected = sweep[(sweep.radius_m==cfg['radius_m']) & (sweep.max_candidates==cfg['max_candidates'])].iloc[0]
    train = aggregate[aggregate.partition=='train'].iloc[0]
    for final_name,sweep_name in [('trusted','trusted'),('reference_hits','hits'),('candidates','count'),('zero_candidates_valid','zero')]:
        if int(train[final_name])!=int(selected[sweep_name]):
            raise ValueError(f'Frozen build differs from train selection: {final_name}')
    inv = pd.DataFrame(inventory)
    breaks = int((inv.route_break_count>0).sum())
    text = f'''# Data quality audit

Only train and student_test were opened. All nonempty raw counts match loaded
counts; strict parsers stop on malformed nonempty records (no skipped errors).
Inventory includes per-file presence, bytes, records, blank lines and nonfinite
cell counts. Malformed references = 0 denotes successful strict range validation.

Measured trips with reference connectivity breaks: {breaks}. The draft reported
11; {'confirmed' if breaks==11 else 'discrepancy remains; see per-trip inventory'}.

Metric definitions: invalid observations have nonfinite/out-of-range lon/lat or
nonfinite timestamp; nonpositive dt counts finite dt <=0; large gaps use the frozen
gap threshold; duplicate coordinates/observations count repeats beyond first;
stationary steps are below stationary_m; suspicious speeds exceed configured m/s.
First-point dt/speed/heading are intentionally undefined, not missing raw data.
Coordinate bounds are raw degree minima/maxima (not wrapped extent estimates).
Graph duplicates count endpoint rows beyond first; reverse count counts directed
arcs possessing a distinct opposite arc; components include isolated nodes.
Array memory sums top-level numpy buffers, excludes Python/index/runtime overhead;
graph timing includes loading/parsing/projection/components but not candidates.

Coverage: trusted = valid C1 observations whose C3 status is ok; misses include
zero-candidate observations. miss_rate_trusted = misses/trusted;
known_misses_over_all_valid = same known misses / all valid observations (not
an estimate of unknown-reference errors); alignment_coverage_valid = trusted/valid.
All aggregate rates are micro sums; per-trip evidence is also retained. Ambiguous
and unaligned candidate labels are nullable and must be excluded by M2. Alignment
is derived, not independently verified ground truth. No reference is forced into C2.

Projection pairs: at most 64 evenly spaced node rows, paired by cyclic shift 17,
per trip. WGS84 geodesic vs AEQD planar chord; no route or candidate dependence.
No projection retuning on student_test. See projection_errors.csv for every pair.
Local policy rejects >1500 km centre radius; error evidence covers supplied trips
only. No smoothing: preserve raw candidate locations and avoid a fitted transform.
30 s gap and 1 m stationary thresholds are declared physical heuristics; train
inventory and dt/heading figures expose their impact, not a claim of optimality.

Missing/invalid values, suspicious measurements, and structural route breaks are
kept separately in inventory. No manual label repairs were applied. EDA logs are
not a substitute for independent reference-label review.
'''
    (out/'data_quality_log.md').write_text(text,encoding='utf8')
    make_figures(out,figures)
    # Small reproducibility evidence copy; large processed tables remain local.
    (out/'build_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    (out/'runtime.json').write_text(json.dumps(runtime_info(),indent=2),encoding='utf8')
    print(aggregate.to_string(index=False),flush=True)

def make_figures(evidence,figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(figures); out.mkdir(parents=True,exist_ok=True)
    inv = pd.read_csv(evidence/'inventory.csv')
    gs = pd.read_csv(evidence/'graph_stats.csv')
    al = pd.read_csv(evidence/'alignment_stats.csv')
    pr = pd.read_csv(evidence/'projection_errors.csv')
    cv = pd.read_csv(evidence/'coverage_by_partition.csv')
    fig,axes = plt.subplots(2,2,figsize=(11,7))
    for ax,col in zip(axes.flat,['track_records','nodes_records','arcs_records','route_records']):
        for partition,group in inv.groupby('partition'):
            ax.hist(group[col],bins=15,alpha=.55,label=partition)
        ax.set(xlabel=col.replace('_',' '),ylabel='Trips'); ax.set_xscale('log'); ax.legend()
    fig.suptitle('Permitted partitions: raw record counts'); fig.tight_layout(); fig.savefig(out/'inventory.png',dpi=150); plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(13,4))
    for p,group in inv.groupby('partition'):
        axes[0].hist(group.dt_max_s,bins=np.geomspace(1,max(2,inv.dt_max_s.max()),20),alpha=.55,label=p)
        axes[1].scatter(group.large_gaps,group.suspicious_speeds,label=p,alpha=.65)
        axes[2].bar(p,group.invalid_observations.sum(),label=p)
    axes[0].set(xscale='log',xlabel='Maximum sampling gap (s)',ylabel='Trips'); axes[0].legend()
    axes[1].set(xlabel='Gaps > frozen threshold per trip',ylabel='Suspicious speed observations'); axes[1].legend()
    axes[2].set(ylabel='Invalid observations',title='Raw quality (zero is measured)')
    fig.suptitle('train / student_test: sampling and quality'); fig.tight_layout(); fig.savefig(out/'quality.png',dpi=150); plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(13,4))
    for p,group in gs.groupby('partition'):
        axes[0].scatter(group.arcs,group.weak_components,label=p,alpha=.6)
        axes[1].hist(group.reverse_arc_count/group.arcs,bins=15,alpha=.55,label=p)
    axes[0].set(xlabel='Directed arcs',ylabel='Weak components',xscale='log'); axes[0].legend()
    axes[1].set(xlabel='Directed arcs with reverse twin (fraction)',ylabel='Trips'); axes[1].legend()
    for p,group in pr.groupby('partition'):
        axes[2].hist(group.relative_error*1e6,bins=25,alpha=.55,label=p)
    axes[2].set(xlabel='Projection distance error (ppm)',ylabel='Sample pairs'); axes[2].legend()
    fig.suptitle('train / student_test: directed structure and projection audit'); fig.tight_layout(); fig.savefig(out/'graph_projection.png',dpi=150); plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(13,4))
    status = al.groupby('partition')[['ok','ambiguous','unaligned']].sum()
    status.div(status.sum(axis=1),axis=0).plot.bar(stacked=True,ax=axes[0],rot=0)
    axes[0].set(ylabel='Observation fraction',xlabel='Partition')
    for p,group in al.groupby('partition'):
        axes[1].hist(group.align_dist_p95_m,bins=20,alpha=.55,label=p)
    axes[1].set(xlabel='Per-trip aligned distance p95 (m)',ylabel='Trips'); axes[1].legend()
    axes[2].bar(cv.partition,cv.miss_rate_trusted*100)
    axes[2].set(ylabel='Reference miss / trusted valid observations (%)',xlabel='Partition')
    fig.suptitle('Derived alignment and candidate coverage (no independent label verification)'); fig.tight_layout(); fig.savefig(out/'alignment_coverage.png',dpi=150); plt.close(fig)
    sweep = pd.read_csv(evidence/'candidate_sweep.csv')
    fig,ax = plt.subplots(figsize=(7,4))
    for k,group in sweep.groupby('max_candidates'):
        ax.plot(group.radius_m,group.reference_recall_trusted*100,'o-',label=f'K={k}')
    ax.set(xlabel='Radius (m)',ylabel='Trusted-reference recall (%)',title='train only: nested radius/K sweep'); ax.legend(); fig.tight_layout(); fig.savefig(out/'candidate_sweep.png',dpi=150); plt.close(fig)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['select','audit'])
    p.add_argument('--raw-root',required=True)
    p.add_argument('--processed-root',default='data/processed')
    p.add_argument('--config',default='experiments/data_config.yaml')
    p.add_argument('--evidence-root',default='results/eda')
    p.add_argument('--figure-root',default='results/figures/eda')
    args = p.parse_args()
    if args.mode=='select':
        select_config(args.raw_root,args.config,args.evidence_root)
    else:
        audit(args.raw_root,args.processed_root,args.config,args.evidence_root,args.figure_root)

if __name__=='__main__':
    main()
