"""Only post-prediction server evaluation loads reference routes/alignment."""
from pathlib import Path
import numpy as np
from src.data.dataset_loader import load_route
from src.data.reference_alignment import align_reference
from src.hmm.trip_evaluation import evaluate_trip


def evaluate_prediction(trip_dir, output, data_config):
    trip_dir=Path(trip_dir)
    if not (trip_dir/f'{trip_dir.name}.route').is_file():
        return {},{'reference_available':False,'evaluation_status':'reference_unavailable'}
    graph,obs=output['graph'],output['observations']
    route=load_route(trip_dir,len(graph.arc_from))
    alignment=align_reference(obs,graph,route,data_config)
    metrics={}
    for name in ['classifier','hmm']:
        values=evaluate_trip(trip_dir,obs,output[name],graph,data_config,alignment,route)
        metrics.update({f'{key}_{name}':float(value) if np.isfinite(value) else None for key,value in values.items()})
    trusted=(alignment.align_status=='ok')&obs.is_valid
    cand_keys=set(zip(output['candidates'].obs_idx,output['candidates'].arc_id))
    hit=np.array([(int(i),int(a)) in cand_keys for i,a in zip(alignment.obs_idx,alignment.ref_arc_id)])
    n=int(trusted.sum())
    metrics['candidate_miss_rate']=float((trusted & ~hit).sum()/n) if n else None
    metrics['alignment_coverage']=float(n/len(obs)) if len(obs) else None
    return metrics,{'reference_available':True,'evaluation_status':'derived_alignment',
                    'n_trusted':n,'n_ambiguous':int((alignment.align_status=='ambiguous').sum()),
                    'n_unaligned':int((alignment.align_status=='unaligned').sum())}
