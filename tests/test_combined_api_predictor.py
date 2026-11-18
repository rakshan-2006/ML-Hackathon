"""Exercise Person A's scorer through Person B's real API, without real training."""
import json
import shutil
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.common.paths import PROJECT_ROOT,Settings
from src.inference.predictor import file_sha256
from tests.runtime_helpers import write_bundle


def test_real_api_uses_checksumming_person_a_predictor(tmp_path):
    trip=tmp_path/'trips/tiny_trip'
    shutil.copytree(PROJECT_ROOT/'tests/fixtures/tiny_trip',trip)
    models=tmp_path/'models';write_bundle(models)
    card_path=models/'model_card.json'
    card=json.loads(card_path.read_text())
    card.update(status='final',artifact_sha256={name:file_sha256(models/name)
        for name in ['best_model.joblib','preprocessing.joblib']})
    card_path.write_text(json.dumps(card))
    settings=Settings('real',tmp_path/'trips',models,tmp_path/'logs','synthetic','synthetic')
    with TestClient(create_app(settings)) as client:
        assert client.get('/health').json()['status']=='ok'
        response=client.post('/predict',json={'trip_id':'tiny_trip'})
        assert response.status_code==200,response.text
        assert response.json()['n_obs']==4 and not response.json()['is_stub']
        scorer=client.app.state.prediction_service.scorer
        assert scorer.__self__.__class__.__module__=='src.inference.predictor'
