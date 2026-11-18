"""Gated official-stage ZIP import. Development tests use synthetic archives only."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import unicodedata
import zipfile
import re
import numpy as np

from src.data.dataset_loader import load_track,load_nodes,load_arcs,load_route


def import_archive(archive,destination,*,evaluation_authorized=False,expected_trips=10,
                   max_members=100000,max_total_bytes=1500000000,max_file_bytes=256000000,
                   max_ratio=1000):
    # This check precedes even archive existence/listing/header access.
    if not evaluation_authorized:
        raise PermissionError('Official evaluation authorization is required before opening the archive')
    archive,destination=Path(archive),Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Import destination must be a new session directory')
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.pathsnap-import-',dir=destination.parent)).resolve()
    extracted,ready=stage/'extracted',stage/'ready'
    extracted.mkdir(); ready.mkdir()
    try:
        with zipfile.ZipFile(archive) as bundle:
            members=bundle.infolist()
            if len(members)>max_members:
                raise ValueError('Too many archive entries')
            seen,total=set(),0
            planned=[]
            for member in members:
                name=member.filename.replace('\\','/')
                path=PurePosixPath(name)
                if not name or name.startswith('/') or ':' in name or '..' in path.parts:
                    raise ValueError('Unsafe archive path')
                for part in path.parts:
                    stem=part.split('.')[0].upper()
                    if part.rstrip(' .')!=part or stem in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}:
                        raise ValueError('Unsafe Windows path component')
                target=(extracted/Path(*path.parts)).resolve()
                if not target.is_relative_to(extracted):
                    raise ValueError('Archive path escapes destination')
                key=unicodedata.normalize('NFKC',str(path)).casefold()
                if key in seen:
                    raise ValueError('Duplicate or case-colliding archive path')
                seen.add(key)
                mode=(member.external_attr>>16)&0xffff
                if stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR):
                    raise ValueError('Links or special files are not allowed')
                if member.flag_bits & 1:
                    raise ValueError('Encrypted archives are unsupported')
                if member.is_dir():
                    planned.append((member,target)); continue
                if path.suffix.lower() not in {'.track','.nodes','.arcs','.route'}:
                    raise ValueError('Unexpected archive file type')
                total+=member.file_size
                if member.file_size>max_file_bytes or total>max_total_bytes:
                    raise ValueError('Archive expansion exceeds size limit')
                if member.file_size/max(1,member.compress_size)>max_ratio:
                    raise ValueError('Archive expansion ratio exceeds limit')
                planned.append((member,target))
            for member,target in planned:
                if member.is_dir():
                    target.mkdir(parents=True,exist_ok=True); continue
                target.parent.mkdir(parents=True,exist_ok=True)
                written=0
                with bundle.open(member) as source,target.open('xb') as sink:
                    while chunk:=source.read(1024*1024):
                        written+=len(chunk)
                        if written>member.file_size or written>max_file_bytes:
                            raise ValueError('Archive size does not match declared size')
                        sink.write(chunk)
                if written!=member.file_size:
                    raise ValueError('Incomplete archive member')
        tracks=sorted(extracted.rglob('*.track'))
        if not tracks:
            raise ValueError('No trip tracks discovered')
        trips=[]; identifiers=set()
        for track in tracks:
            trip=track.parent; trip_id=trip.name
            if track.stem!=trip_id or not re.fullmatch(r'[A-Za-z0-9_-]+',trip_id):
                raise ValueError('Trip folder/file identifiers are inconsistent')
            if trip_id.casefold() in identifiers:
                raise ValueError('Duplicate trip identifier across archive wrappers')
            identifiers.add(trip_id.casefold())
            nodes=load_nodes(trip); arcs=load_arcs(trip,len(nodes)); observations=load_track(trip)
            if (not np.isfinite(nodes).all() or (np.abs(nodes[:,0])>180).any() or
                (np.abs(nodes[:,1])>90).any()):
                raise ValueError('Invalid road-network coordinates')
            reference=(trip/f'{trip_id}.route').is_file()
            if reference:
                load_route(trip,len(arcs))
            dest=ready/trip_id; dest.mkdir()
            for kind in ['track','nodes','arcs']+(['route'] if reference else []):
                shutil.copyfile(trip/f'{trip_id}.{kind}',dest/f'{trip_id}.{kind}')
            trips.append(dict(trip_id=trip_id,observations=len(observations),nodes=len(nodes),arcs=len(arcs),reference_available=reference))
        # Reject orphan/misnamed payloads rather than silently discarding them.
        consumed = {p.relative_to(ready) for p in ready.rglob('*') if p.is_file()}
        for path in extracted.rglob('*'):
            if path.is_file() and Path(path.parent.name,path.name) not in consumed:
                raise ValueError('Orphan or inconsistent trip file in archive')
        digest=hashlib.sha256()
        with archive.open('rb') as source:
            while chunk:=source.read(1024*1024):
                digest.update(chunk)
        manifest=dict(status='imported',archive_sha256=digest.hexdigest(),trip_count=len(trips),
                      expected_trips=expected_trips,count_matches_expectation=len(trips)==expected_trips,trips=trips)
        (ready/'import_manifest.json').write_text(json.dumps(manifest,indent=2))
        os.replace(ready,destination)
        return manifest
    finally:
        # Recursive cleanup is limited to the resolved, random staging sibling.
        if stage.parent!=destination.parent or not stage.name.startswith('.pathsnap-import-'):
            raise RuntimeError('Refusing cleanup outside the validated staging directory')
        shutil.rmtree(stage)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True)
    p.add_argument('--destination',required=True)
    p.add_argument('--evaluation-authorized',action='store_true',help='Only after explicit official-stage authorization')
    a=p.parse_args()
    print(json.dumps(import_archive(a.archive,a.destination,evaluation_authorized=a.evaluation_authorized),indent=2))


if __name__=='__main__':
    main()
