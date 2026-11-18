import json
import shutil
from pathlib import Path
import pytest
from src.data.build_dataset import build
from src.data.validate_dataset import validate
from src.data.model_data import load_model_data
from src.common.schema import FEATURES
from tests.test_data_loader import FIXTURE

def test_atomic_build_and_repeat(tmp_path):
    trip = tmp_path/'raw/train/00000001'; trip.mkdir(parents=True)
    for kind in ['track','nodes','arcs','route']:
        shutil.copyfile(FIXTURE/f'tiny_trip.{kind}',trip/f'00000001.{kind}')
    config = Path(__file__).parents[1]/'experiments/data_config.yaml'
    build(tmp_path/'raw',tmp_path/'first',config,['train'])
    build(tmp_path/'raw',tmp_path/'second',config,['train'])
    assert validate(tmp_path/'first',tmp_path/'second')['ordered_semantic_content_identical']
    X,y,groups,keys = load_model_data(tmp_path/'first')
    assert list(X.columns)==FEATURES and set(y)=={0,1}
    assert set(groups)=={'00000001'} and len(keys)==len(X)==len(y)
    assert X.heading_diff_deg.isna().any()
    with pytest.raises(FileExistsError):
        build(tmp_path/'raw',tmp_path/'first',config,['train'])
    (trip/'00000001.route').write_text('99\n')
    with pytest.raises(ValueError,match='out of range'):
        build(tmp_path/'raw',tmp_path/'first',config,['train'],overwrite=True)
    assert not (tmp_path/'first/manifest.json').exists()
