"""Stage only processed artifacts for a GitHub Release; never include raw files."""
import argparse
import json
from pathlib import Path
import shutil
from src.common.schema import SCHEMA_VERSION,FEATURES,FEATURE_SPEC,C1,C2,C3
from .build_dataset import sha256_file

def package(processed_root,asset_root,tag,repository='rakshan-2006/ML-Hackathon'):
    source = Path(processed_root)
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest['schema_version']!=SCHEMA_VERSION or set(manifest['partitions'])!={'train','student_test'}:
        raise ValueError('Expected complete two-partition matching-schema build')
    assets = Path(asset_root); assets.mkdir(parents=True,exist_ok=True)
    expected = [f'{p}_{k}.parquet' for p in ['train','student_test'] for k in ['observations','candidates','alignment']]
    if set(manifest['outputs'])!=set(expected):
        raise ValueError('Expected six processed tables')
    for name in expected:
        if sha256_file(source/name)!=manifest['outputs'][name]['file_sha256']:
            raise ValueError(f'File integrity mismatch: {name}')
        shutil.copyfile(source/name,assets/name)
    shutil.copyfile(source/'manifest.json',assets/'manifest.json')
    shutil.copyfile('experiments/data_config.yaml',assets/'data_config.yaml')
    shutil.copyfile('data/requirements-m1.txt',assets/'requirements-m1.txt')
    schema = dict(schema_version=SCHEMA_VERSION,ordered_features=FEATURES,feature_spec=FEATURE_SPEC,C1=C1,C2=C2|{'label':'Int8'},C3=C3,label_column='label',label_policy='Exclude null labels (ambiguous/unaligned). Confident reference misses retain all-zero labels.',group_column='trip_id',partitions=['train','student_test'])
    (assets/'schema.json').write_text(json.dumps(schema,indent=2),encoding='utf8')
    checksums = ''.join(f'{sha256_file(p)}  {p.name}\n' for p in sorted(assets.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt')
    (assets/'SHA256SUMS.txt').write_text(checksums,encoding='utf8')
    prefix = f'https://github.com/{repository}/releases/download/{tag}'
    metadata = dict(repository=repository,tag=tag,release_url=f'https://github.com/{repository}/releases/tag/{tag}',schema_version=SCHEMA_VERSION,build_revision=manifest['code']['revision'],ordered_features=FEATURES,label_column='label',group_column='trip_id',assets=[dict(name=p.name,url=f'{prefix}/{p.name}',size_bytes=p.stat().st_size,sha256=sha256_file(p)) for p in sorted(assets.iterdir()) if p.is_file()])
    Path('data/dataset_release.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print(json.dumps(metadata,indent=2))
    return metadata

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-root',default='data/processed')
    p.add_argument('--asset-root',default='tmp/release')
    p.add_argument('--tag',required=True)
    a = p.parse_args()
    package(a.processed_root,a.asset_root,a.tag)

if __name__=='__main__':
    main()
