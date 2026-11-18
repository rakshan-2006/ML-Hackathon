from pathlib import Path
import stat
import zipfile
import pytest
from src.api.import_evaluation import import_archive
from src.common.paths import PROJECT_ROOT


def synthetic_zip(path,extra=None):
    with zipfile.ZipFile(path,'w') as z:
        for p in (PROJECT_ROOT/'tests/fixtures/tiny_trip').iterdir():
            z.write(p,f'wrapper/tiny_trip/{p.name}')
        if extra:
            z.writestr(extra,b'x')


def test_gate_precedes_any_archive_access(tmp_path,monkeypatch):
    monkeypatch.setattr(zipfile,'ZipFile',lambda *a,**kw:pytest.fail('Archive opened before gate'))
    with pytest.raises(PermissionError):
        import_archive(tmp_path/'does-not-exist.zip',tmp_path/'out')


def test_synthetic_wrapped_import_is_atomic_and_route_optional(tmp_path):
    archive=tmp_path/'synthetic.zip'; synthetic_zip(archive)
    result=import_archive(archive,tmp_path/'out',evaluation_authorized=True)
    assert result['trip_count']==1 and result['count_matches_expectation'] is False
    assert (tmp_path/'out/tiny_trip/tiny_trip.track').is_file()
    assert not list(tmp_path.glob('.pathsnap-import-*'))
    with pytest.raises(FileExistsError):
        import_archive(archive,tmp_path/'out',evaluation_authorized=True)


@pytest.mark.parametrize('name',['../escape.track','C:/escape.track','/escape.track',
                                 'wrapper/tiny_trip/TINY_TRIP.track','wrapper/NUL.track'])
def test_unsafe_paths_are_rejected_without_partial_install(tmp_path,name):
    archive=tmp_path/'synthetic.zip'; synthetic_zip(archive,name)
    with pytest.raises(ValueError):
        import_archive(archive,tmp_path/'out',evaluation_authorized=True)
    assert not (tmp_path/'out').exists()
    assert not list(tmp_path.glob('.pathsnap-import-*'))


def test_symlink_and_size_limits(tmp_path):
    archive=tmp_path/'synthetic.zip'
    with zipfile.ZipFile(archive,'w') as z:
        info=zipfile.ZipInfo('link.track'); info.create_system=3
        info.external_attr=(stat.S_IFLNK|0o777)<<16
        z.writestr(info,'target')
    with pytest.raises(ValueError,match='Links'):
        import_archive(archive,tmp_path/'out',evaluation_authorized=True)
    synthetic_zip(archive)
    with pytest.raises(ValueError,match='size limit'):
        import_archive(archive,tmp_path/'out',evaluation_authorized=True,max_file_bytes=1)
