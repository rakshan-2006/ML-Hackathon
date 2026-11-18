import hashlib
import json
import pytest
from src.data.download_dataset import download

def test_verified_download_and_corruption(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    payload = b'synthetic processed table for downloader test'
    (source/'train_candidates.parquet').write_bytes(payload)
    (source/'manifest.json').write_text('{}')
    metadata = tmp_path/'release.json'
    assets = [dict(name=p.name,url=p.as_uri(),size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source.iterdir()]
    metadata.write_text(json.dumps({'assets':assets}))
    output = tmp_path/'download'
    download(metadata,output)
    assert (output/'train_candidates.parquet').read_bytes()==payload
    download(metadata,output)
    (output/'train_candidates.parquet').write_text('corrupt')
    with pytest.raises(FileExistsError):
        download(metadata,output)
    download(metadata,output,overwrite=True)
    assets[0]['sha256']='0'*64
    metadata.write_text(json.dumps({'assets':assets}))
    with pytest.raises(ValueError,match='integrity mismatch'):
        download(metadata,output,overwrite=True)
    assert not (output/'manifest.json').exists()

def test_unsafe_asset_name(tmp_path):
    metadata = tmp_path/'release.json'
    metadata.write_text(json.dumps({'assets':[{'name':'../escape'}]}))
    with pytest.raises(ValueError,match='Unsafe'):
        download(metadata,tmp_path/'download')
