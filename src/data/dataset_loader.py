"""Strict raw parsers. IDs count nonblank records; blank physical lines are audited.

Coordinates/timestamps may be nonfinite and remain for quality flags. Graph IDs
must be lexical integers (no float truncation). No function implicitly reads route.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ALLOWED_PARTITIONS = ('train', 'student_test')

def list_trips(raw_root: str | Path, partitions=ALLOWED_PARTITIONS) -> list[tuple[str, str, Path]]:
    root = Path(raw_root)
    result = []
    for partition in partitions:
        if partition not in ALLOWED_PARTITIONS:
            raise ValueError(f'Forbidden development partition: {partition}')
        directory = root / partition
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.iterdir()):
            if path.is_dir():
                if not re.fullmatch(r'\d{8}', path.name):
                    raise ValueError(f'Expected eight-digit trip ID: {path.name}')
                result.append((partition, path.name, path))
    ids = [r[1] for r in result]
    if len(ids) != len(set(ids)):
        raise ValueError('Trip appears in multiple partitions')
    return result

def raw_path(trip_dir: str | Path, kind: str) -> Path:
    directory = Path(trip_dir)
    return directory / f'{directory.name}.{kind}'

def read_records(path: Path, columns: int, integer: bool = False) -> tuple[np.ndarray, dict]:
    rows, blanks = [], 0
    with path.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            tokens = line.split()
            if not tokens:
                blanks += 1
                continue
            reason = None
            if len(tokens) != columns:
                reason = f'expected {columns} columns, got {len(tokens)}'
            elif integer and any(not re.fullmatch(r'[+-]?\d+', t) for t in tokens):
                reason = 'expected integer IDs'
            if reason:
                raise ValueError(f'{path.parent.name}/{path.name}: physical line {line_no}: {reason}')
            try:
                rows.append([int(t) if integer else float(t) for t in tokens])
            except (ValueError, OverflowError) as exc:
                raise ValueError(f'{path.parent.name}/{path.name}: physical line {line_no}: invalid numeric record') from exc
    try:
        values = np.asarray(rows, dtype=np.int64 if integer else np.float64).reshape(-1, columns)
    except OverflowError as exc:
        raise ValueError(f'{path}: ID outside int64 range') from exc
    return values, {'records': len(rows), 'blank_lines': blanks, 'size_bytes': path.stat().st_size}

def load_track(trip_dir: str | Path) -> pd.DataFrame:
    a, audit = read_records(raw_path(trip_dir, 'track'), 3)
    if not len(a):
        raise ValueError(f'{trip_dir}: empty track')
    df = pd.DataFrame(a, columns=['lon', 'lat', 't'])
    df.attrs['raw_audit'] = audit
    return df

def load_nodes(trip_dir: str | Path) -> np.ndarray:
    return read_records(raw_path(trip_dir, 'nodes'), 2)[0]

def load_arcs(trip_dir: str | Path, n_nodes: int | None = None) -> np.ndarray:
    a = read_records(raw_path(trip_dir, 'arcs'), 2, True)[0]
    n_nodes = len(load_nodes(trip_dir)) if n_nodes is None else n_nodes
    bad = np.flatnonzero(np.any((a < 0) | (a >= n_nodes), axis=1))
    if len(bad):
        _range_error(trip_dir, 'arcs', int(bad[0]), 'node endpoint out of range')
    return a

def _range_error(trip_dir, kind, record, message):
    path = raw_path(trip_dir, kind)
    with path.open() as stream:
        index = -1
        for line_no, line in enumerate(stream, 1):
            if line.split():
                index += 1
                if index == record:
                    raise ValueError(f'{path.parent.name}/{path.name}: physical line {line_no}: {message}')

def load_route(trip_dir: str | Path, n_arcs: int | None = None) -> np.ndarray:
    a = read_records(raw_path(trip_dir, 'route'), 1, True)[0].ravel()
    n_arcs = len(load_arcs(trip_dir)) if n_arcs is None else n_arcs
    bad = np.flatnonzero((a < 0) | (a >= n_arcs))
    if len(bad):
        _range_error(trip_dir, 'route', int(bad[0]), 'arc ID out of range')
    return a
