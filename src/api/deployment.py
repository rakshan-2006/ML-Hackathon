"""Prepare a checksummed final runtime bundle only after A and decoder gates pass."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
from src.api.service import ArtifactLoader
from src.common.paths import PROJECT_ROOT
from src.common.schema import SCHEMA_VERSION
from src.hmm.tune import sha256


def build_bundle(source,decoder_path,confirmation_path,destination):
    source=Path(source); decoder_path=Path(decoder_path)
    card=json.loads((source/'model_card.json').read_text(encoding='utf8'))
    decoder=json.loads(decoder_path.read_text(encoding='utf8'))
    confirmation=json.loads(Path(confirmation_path).read_text(encoding='utf8'))
    if card.get('status')!='final' or decoder.get('status')!='final_tuned':
        raise ValueError('Final selected model and measured tuned decoder are required')
    if decoder.get('model_version')!=card.get('model_version') or not decoder.get('tuning_provenance'):
        raise ValueError('Model/decoder binding or tuning provenance is missing')
    if decoder.get('data_config_sha256')!=sha256(PROJECT_ROOT/'configs/data.yaml'):
        raise ValueError('Decoder/data configuration mismatch')
    if (confirmation.get('status')!='final_student_test_confirmed' or
        confirmation.get('model_version')!=card['model_version'] or
        confirmation.get('decoder_sha256')!=sha256(decoder_path) or
        confirmation.get('trip_count')!=10 or not confirmation.get('all_http_offline_geometry_equal')):
        raise ValueError('Repeat the ten-trip HTTP/offline confirmation with final artifacts first')
    target=Path(destination)
    if target.exists(): raise FileExistsError('Bundle destination must be new')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=PROJECT_ROOT,text=True).strip():
        raise ValueError('Commit reviewed source and evidence before freezing the final bundle')
    target.mkdir(parents=True)
    for name in ['best_model.joblib','preprocessing.joblib','model_card.json']:
        shutil.copyfile(source/name,target/name)
    shutil.copyfile(decoder_path,target/'decoder_config.json')
    shutil.copyfile(PROJECT_ROOT/'configs/data.yaml',target/'data.yaml')
    manifest=dict(status='final',schema_version=SCHEMA_VERSION,model_version=card['model_version'],
        decoder_version=decoder['version'],data_config_sha256=sha256(target/'data.yaml'),
        sha256={name:sha256(target/name) for name in ArtifactLoader.REQUIRED_FILES},
        dependencies={name:version(name) for name in ['scikit-learn','numpy','pandas','joblib','pyproj','shapely']},
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip(),
        tuning_provenance=decoder['tuning_provenance'],confirmation_sha256=sha256(confirmation_path))
    (target/'deployment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    ArtifactLoader(target).load()
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','decoder','confirmation','destination']: p.add_argument('--'+name,required=True)
    a=p.parse_args()
    print(json.dumps(build_bundle(a.source,a.decoder,a.confirmation,a.destination),indent=2))
