"""Explicit integration validation and independent rebuild comparison."""
import argparse
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from .build_dataset import semantic_hash
from src.common.schema import SCHEMA_VERSION,FEATURES

def validate(root,compare=None):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Schema mismatch')
    if set(manifest['partitions'])-{'train','student_test'}:
        raise ValueError('Forbidden partition')
    trip_map = {(t['partition'],t['trip_id']):t for t in manifest['trips']}
    total = 0
    coverage = []
    for partition,trip_ids in manifest['partitions'].items():
        handles = {kind:pq.ParquetFile(root/f'{partition}_{kind}.parquet') for kind in ['observations','candidates','alignment']}
        if any(h.num_row_groups!=len(trip_ids) for h in handles.values()):
            raise ValueError('Trip row-group coverage mismatch')
        for j,trip_id in enumerate(trip_ids):
            tables = {kind:h.read_row_group(j).to_pandas() for kind,h in handles.items()}
            record = trip_map[partition,trip_id]
            for kind,df in tables.items():
                if semantic_hash(df)!=record['tables'][kind]['semantic_sha256']:
                    raise ValueError(f'Content/dtype mismatch: {partition}/{trip_id}/{kind}')
                if not (df.trip_id==trip_id).all():
                    raise ValueError('Leading-zero/trip identity mismatch')
            obs,cand,al = [tables[k] for k in ['observations','candidates','alignment']]
            if cand[FEATURES].select_dtypes(include='number').isin([float('inf'),-float('inf')]).any().any():
                raise ValueError('Infinite model feature')
            if len(obs)!=record['source']['track']['nonempty_records'] or len(al)!=len(obs):
                raise ValueError('Observation identity coverage')
            valid = obs.is_valid.to_numpy()
            trusted = (al.align_status=='ok').to_numpy() & valid
            hits = al.ref_in_candidates.to_numpy() & trusted
            misses = trusted & ~hits
            coverage.append(dict(partition=partition,trip_id=trip_id,observations=len(obs),valid=int(valid.sum()),trusted=int(trusted.sum()),ambiguous=int((al.align_status=='ambiguous').sum()),unaligned=int((al.align_status=='unaligned').sum()),reference_hits=int(hits.sum()),reference_misses=int(misses.sum()),zero_candidates_valid=int(((obs.n_candidates==0)&obs.is_valid).sum()),candidates=len(cand),labelled_candidates=int(cand.label.notna().sum()),positive_candidates=int((cand.label==1).sum())))
            total += 1
    aggregate = pd.DataFrame(coverage).groupby('partition',as_index=False).sum(numeric_only=True)
    aggregate['miss_rate_trusted'] = aggregate.reference_misses/aggregate.trusted
    aggregate['known_misses_over_all_valid'] = aggregate.reference_misses/aggregate.valid
    aggregate['alignment_coverage_valid'] = aggregate.trusted/aggregate.valid
    result = {'trip_count':total,'schema_version':SCHEMA_VERSION,'content_validation':'passed','compared_to_independent_build':False,'coverage_by_partition':aggregate.to_dict(orient='records'),'coverage_per_trip':coverage}
    if compare:
        validate(compare)
        other = json.loads((Path(compare)/'manifest.json').read_text())
        if manifest['effective_config']!=other['effective_config'] or manifest['partitions']!=other['partitions']:
            raise ValueError('Rebuild config/partitions differ')
        checks = {}
        for name,output in manifest['outputs'].items():
            peer = other['outputs'][name]
            checks[name] = all(output[field]==peer[field] for field in ['rows','semantic_sha256','ordered_trip_semantic_hashes','dtypes'])
        if not all(checks.values()):
            raise ValueError(f'Rebuild mismatch: {checks}')
        source_before = [(r['partition'],r['trip_id'],r['source']) for r in manifest['trips']]
        source_after = [(r['partition'],r['trip_id'],r['source']) for r in other['trips']]
        if source_before!=source_after:
            raise ValueError('Source fingerprints changed')
        result.update(compared_to_independent_build=True,ordered_semantic_content_identical=True,table_checks=checks,first_build_s=manifest['elapsed_s'],second_build_s=other['elapsed_s'],note='Semantic hashes include ordered values and dtypes, exclude timing/revision metadata. No byte-identical Parquet claim.')
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-root',default='data/processed')
    p.add_argument('--compare-root')
    p.add_argument('--evidence',default='results/eda/validation.json')
    args = p.parse_args()
    result = validate(args.processed_root,args.compare_root)
    output = Path(args.evidence); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf8')
    pd.DataFrame(result['coverage_by_partition']).to_csv(output.parent/'coverage_by_partition.csv',index=False)
    pd.DataFrame(result['coverage_per_trip']).to_csv(output.parent/'coverage.csv',index=False)
    summary = {k:v for k,v in result.items() if k!='coverage_per_trip'}
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
