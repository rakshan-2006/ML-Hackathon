"""Final-model OOF decoder selection. Refuses development/in-sample score exports.

No classifier fitting occurs here. Student-test and evaluator data are excluded
from selection. See docs/contracts/person_b.md for the score manifest contract.
"""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from src.common.paths import PROJECT_ROOT
from src.common.schema import SCHEMA_VERSION
from src.data.config import load_config
from src.data.road_graph import load_road_graph
from src.hmm.adapter import decode_tables
from src.hmm.route_decoder import DecoderConfig
from src.hmm.trip_evaluation import evaluate_trip


KEYS=['trip_id','obs_idx','arc_id']


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024): digest.update(chunk)
    return digest.hexdigest()


def keyed_scores(candidates,scores,model_version,folds):
    """Exact left-key join with no dropping extra, unlabeled or absent candidates."""
    required=KEYS+['p_model','model_version','fold']
    if not set(required)<=set(scores): raise ValueError('Missing OOF score columns')
    if candidates.duplicated(KEYS).any() or scores.duplicated(KEYS).any():
        raise ValueError('Duplicate candidate/score keys')
    if len(scores)!=len(candidates): raise ValueError('Score count differs from all candidate rows')
    if set(scores.model_version)!={model_version}: raise ValueError('OOF model version mismatch')
    if not np.isfinite(scores.p_model).all() or not scores.p_model.between(0,1).all():
        raise ValueError('Invalid OOF probabilities')
    joined=candidates[KEYS].merge(scores[required],on=KEYS,how='left',validate='one_to_one',sort=False)
    if joined.p_model.isna().any(): raise ValueError('Missing/extra candidate score keys')
    mapping=folds.set_index('trip_id')['fold']
    expected=joined.trip_id.map(mapping)
    if expected.isna().any() or not (joined.fold.to_numpy()==expected.to_numpy()).all():
        raise ValueError('OOF fold differs from canonical trip split')
    return joined.p_model.to_numpy(dtype=np.float64)


def validate_provenance(manifest,card,paths,folds):
    if card.get('status')!='final': raise ValueError('Person A final selected model is required')
    if manifest.get('status')!='final_parameter_oof' or manifest.get('schema_version')!=SCHEMA_VERSION:
        raise ValueError('Verified final-parameter OOF provenance is required')
    if manifest.get('model_version')!=card.get('model_version'):
        raise ValueError('Manifest/model version mismatch')
    if manifest.get('partition')!='train' or manifest.get('score_kind')!='out_of_fold':
        raise ValueError('Decoder selection accepts train OOF only')
    for name,path in paths.items():
        if manifest.get('sha256',{}).get(name)!=sha256(path):
            raise ValueError(f'OOF provenance checksum mismatch: {name}')
    exclusions=manifest.get('fold_holdout_trip_ids',{})
    for fold,group in folds.groupby('fold'):
        if sorted(exclusions.get(str(fold),[]))!=sorted(group.trip_id.tolist()):
            raise ValueError('Missing canonical held-out trip provenance')


def tune(raw_root,processed,scores_path,manifest_path,model_card,output):
    raw=Path(raw_root)
    if raw.name!='train': raise ValueError('Tuning raw root must be the permitted train partition')
    processed=Path(processed); out=Path(output)
    if out.exists(): raise FileExistsError('Use a new tuning run directory; do not overwrite evidence')
    card=json.loads(Path(model_card).read_text(encoding='utf8'))
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf8'))
    paths={name:processed/f'train_{name}.parquet' for name in ['candidates','observations','alignment']}
    paths.update(scores=Path(scores_path),model_card=Path(model_card),
                 folds=PROJECT_ROOT/'configs/cv_folds.csv',data_config=PROJECT_ROOT/'configs/data.yaml')
    folds=pd.read_csv(paths['folds'],dtype={'trip_id':str})
    validate_provenance(manifest,card,paths,folds)
    candidates=pd.read_parquet(paths['candidates'])
    observations=pd.read_parquet(paths['observations'])
    alignment=pd.read_parquet(paths['alignment'])
    scores=pd.read_parquet(scores_path)
    probabilities=keyed_scores(candidates,scores,card['model_version'],folds)
    if set(observations.trip_id)!=set(folds.trip_id) or set(candidates.trip_id)!=set(folds.trip_id):
        raise ValueError('OOF tables must cover every canonical training trip')
    out.mkdir(parents=True)
    configs=[DecoderConfig(beta_m=beta,emission_weight=weight) for beta,weight in
             itertools.product([5.,10.,20.,50.],[0.5,1.,2.])]
    plan=dict(status='selection_plan',objective='observation-weighted trusted-reference Top-1',
        tie_rule='Highest Top-1, then lowest trial index (fixed order); runtime is descriptive',
        seed=0,trial_count=len(configs),search_space={'beta_m':[5,10,20,50],'emission_weight':[.5,1,2]},
        fixed_parameters=asdict(DecoderConfig()),partition='train_oof',
        schema_version=SCHEMA_VERSION,model_version=card['model_version'],
        provenance_manifest_sha256=sha256(manifest_path),sha256={k:sha256(v) for k,v in paths.items()})
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
    rows=[]; trials=[]; data=load_config(paths['data_config'])
    for tid,obs in observations.groupby('trip_id',sort=True):
        graph=load_road_graph(raw/tid)
        indices=np.flatnonzero(candidates.trip_id.to_numpy()==tid)
        cand=candidates.iloc[indices]; p=probabilities[indices]
        aligned=alignment[alignment.trip_id==tid]
        ordered_obs=obs.sort_values('obs_idx'); ordered_align=aligned.sort_values('obs_idx')
        trusted=(ordered_align.align_status.to_numpy()=='ok')&ordered_obs.is_valid.to_numpy()
        available=set(zip(cand.obs_idx,cand.arc_id))
        misses=sum(bool(ok) and (int(i),int(a)) not in available for ok,i,a in
                   zip(trusted,ordered_align.obs_idx,ordered_align.ref_arc_id))
        n_trusted=int(trusted.sum())
        for trial,cfg in enumerate(configs):
            started=perf_counter()
            try:
                hmm,base=decode_tables(obs,cand,p,graph,cfg)
            except Exception as exc:
                pd.DataFrame(rows).to_csv(out/'per_trip.csv',index=False)
                failure=dict(status='aborted',trip_id=tid,trial=trial,config=asdict(cfg),
                             error_type=type(exc).__name__,error=str(exc))
                (out/'failure.json').write_text(json.dumps(failure,indent=2),encoding='utf8')
                raise
            for name,result in [('hmm',hmm),('classifier',base)]:
                metrics=evaluate_trip(raw/tid,obs,result,graph,data,alignment=aligned)
                rows.append(dict(trial=trial,trip_id=tid,method=name,**metrics,
                    n_segments=result['n_segments'],n_inferred_arcs=result.get('n_inferred_arcs',0),
                    n_no_candidates=result['n_no_candidates'],n_breaks_gap=result.get('n_breaks_gap',0),
                    n_breaks_time=result.get('n_breaks_time',0),
                    n_breaks_no_transition=result.get('n_breaks_no_transition',0),
                    alignment_coverage=n_trusted/len(obs) if len(obs) else None,
                    n_reference_misses=misses,candidate_miss_rate=misses/n_trusted if n_trusted else None,
                    wall_ms=(perf_counter()-started)*1000))
        pd.DataFrame(rows).to_csv(out/'per_trip.csv',index=False)
    table=pd.DataFrame(rows)
    for trial,cfg in enumerate(configs):
        part=table[(table.trial==trial)&(table.method=='hmm')]
        n=part.n_evaluated.sum()
        if not n: raise ValueError('No trusted observations for decoder tuning')
        trials.append(dict(trial=trial,top1=float(part.n_correct.sum()/n),**asdict(cfg)))
    ranked=sorted(trials,key=lambda row:(-row['top1'],row['trial']))
    pd.DataFrame(trials).to_csv(out/'trials.csv',index=False)
    best=ranked[0]; selected=asdict(configs[best['trial']])
    selected.update(schema_version=SCHEMA_VERSION,version=f"oof-{card['model_version']}",
        status='final_tuned',model_version=card['model_version'],
        data_config_sha256=sha256(paths['data_config']),
        tuning_provenance={'plan_sha256':sha256(out/'plan.json'),'trials_sha256':sha256(out/'trials.csv'),
                           'scores_manifest_sha256':sha256(manifest_path),'selected_trial':best['trial']})
    (out/'decoder_config.json').write_text(json.dumps(selected,indent=2),encoding='utf8')
    return selected


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['raw-root','processed','scores','manifest','model-card','output']:
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    print(json.dumps(tune(args.raw_root,args.processed,args.scores,args.manifest,args.model_card,args.output),indent=2))
