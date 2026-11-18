"""Download published processed tables and verify SHA-256 before installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

def download(metadata,output_root,overwrite=False):
    release = json.loads(Path(metadata).read_text(encoding='utf8'))
    output = Path(output_root); output.mkdir(parents=True,exist_ok=True)
    # Install success marker last. Only declared basenames may be written.
    assets = sorted(release['assets'],key=lambda a:a['name']=='manifest.json')
    for asset in assets:
        name = asset['name']
        if Path(name).name!=name or '/' in name or '\\' in name:
            raise ValueError(f'Unsafe asset basename: {name}')
    changes = False
    for asset in assets:
        target = output/asset['name']
        same = target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==asset['sha256']
        if target.exists() and not same and not overwrite:
            raise FileExistsError(f'{target}: checksum mismatch; use --overwrite')
        changes |= not same
    if changes:
        (output/'manifest.json').unlink(missing_ok=True)
    for asset in assets:
        name = asset['name']
        target = output/name
        if target.exists():
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest==asset['sha256']:
                print(f'Already verified: {name}')
                continue
            if not overwrite:
                raise FileExistsError(f'{target}: checksum mismatch; use --overwrite')
        partial = output/(name+'.partial')
        request = urllib.request.Request(asset['url'],headers={'User-Agent':'PathSnap-M1-dataset-downloader'})
        digest = hashlib.sha256(); size = 0
        with urllib.request.urlopen(request,timeout=120) as source, partial.open('wb') as sink:
            while chunk := source.read(1024*1024):
                sink.write(chunk); digest.update(chunk); size += len(chunk)
        if digest.hexdigest()!=asset['sha256'] or size!=asset['size_bytes']:
            partial.unlink(missing_ok=True)
            raise ValueError(f'Download integrity mismatch: {name}')
        os.replace(partial,target)
        print(f'Verified {name}: {size:,} bytes, SHA-256 {digest.hexdigest()}',flush=True)
    return release

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',default=str(Path(__file__).parents[2]/'data/releases/preprocessing.json'))
    p.add_argument('--output-root',default='data/processed')
    p.add_argument('--overwrite',action='store_true')
    a = p.parse_args()
    download(a.metadata,a.output_root,a.overwrite)

if __name__=='__main__':
    main()
