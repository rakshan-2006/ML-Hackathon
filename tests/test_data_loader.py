from pathlib import Path
import numpy as np
import pytest
from src.data.dataset_loader import load_track, load_nodes, load_arcs, load_route, list_trips

FIXTURE = Path(__file__).parent / 'fixtures/tiny_trip'

def test_fixture():
    assert load_track(FIXTURE).iloc[0].tolist() == [0.0002, 0, 1]
    assert load_nodes(FIXTURE).shape == (6, 2)
    assert load_arcs(FIXTURE).tolist()[0] == [0, 1]
    assert load_route(FIXTURE).tolist() == [0, 2]

def test_whitespace_invalid_and_blanks(tmp_path):
    p = tmp_path / f'{tmp_path.name}.track'
    p.write_bytes(b' 1\t2 3\r\n\r\nnan 4 5\n')
    df = load_track(tmp_path)
    assert len(df) == 2 and np.isnan(df.lon.iloc[1])
    assert df.attrs['raw_audit']['blank_lines'] == 1
    p.write_text('1 2\n')
    with pytest.raises(ValueError, match='physical line 1'):
        load_track(tmp_path)
    p.write_text('')
    with pytest.raises(ValueError, match='empty track'):
        load_track(tmp_path)

def test_ids_ranges_and_missing(tmp_path):
    for value, match in [('0.5 1', 'integer'), ('-1 1', 'out of range'), ('0 9', 'out of range')]:
        (tmp_path / f'{tmp_path.name}.arcs').write_text(value)
        with pytest.raises(ValueError, match=match):
            load_arcs(tmp_path, 2)
    (tmp_path / f'{tmp_path.name}.route').write_text('2')
    with pytest.raises(ValueError, match='out of range'):
        load_route(tmp_path, 2)
    with pytest.raises(FileNotFoundError):
        load_nodes(tmp_path)

def test_discovery(tmp_path):
    (tmp_path / 'train/00000001').mkdir(parents=True)
    assert list_trips(tmp_path, ['train'])[0][1] == '00000001'
    with pytest.raises(ValueError, match='Forbidden'):
        list_trips(tmp_path, ['evaluator_test'])
