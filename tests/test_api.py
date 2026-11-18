"""Focused tests for the Member 4 API contract and walking skeleton."""

from pathlib import Path
import json

from fastapi.testclient import TestClient
import joblib
import pytest

from src.api.main import create_app
from src.common.paths import Settings
from src.common.schema import SCHEMA_VERSION
from src.api.service import PipelineFailureError


def make_settings(tmp_path: Path, mode: str = "stub") -> Settings:
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    (evaluation_dir / "trip-alpha").mkdir(exist_ok=True)
    return Settings(
        mode=mode,
        evaluation_dir=evaluation_dir,
        artifacts_dir=tmp_path / "models",
        log_dir=tmp_path / "logs",
        model_name="test-model",
        model_version="test-version",
    )


def test_health_has_contract_fields(tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_name": "test-model (stub)",
        "model_version": "test-version",
        "schema_version": SCHEMA_VERSION,
        "decoder_config_version": "not-loaded",
    }


def test_trips_are_discovered_from_directory(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    (settings.evaluation_dir / "trip-beta").mkdir()

    with TestClient(create_app(settings)) as client:
        response = client.get("/trips")

    assert response.status_code == 200
    assert response.json() == ["trip-alpha", "trip-beta"]


def test_predict_returns_explicit_stub_without_fake_outputs(tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as client:
        response = client.post("/predict", json={"trip_id": "trip-alpha"})

    assert response.status_code == 200
    body = response.json()
    assert {key: value for key, value in body.items() if key != "timings"} == {
        "trip_id": "trip-alpha",
        "n_obs": 0,
        "gps_coordinates": [],
        "route_hmm": [],
        "route_classifier": None,
        "metrics": {},
        "diagnostics": {
            "status": "stub",
            "message": (
                "No GPS observations or route predictions are loaded. "
                "Member 1-3 inference integration is pending."
            ),
        },
        "mode": "stub",
        "is_stub": True,
    }
    assert set(body["timings"]) == {"total"}
    assert isinstance(body["timings"]["total"], (int, float))


def test_unknown_trip_returns_404(tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as client:
        response = client.post("/predict", json={"trip_id": "missing-trip"})

    assert response.status_code == 404


def test_invalid_prediction_request_returns_422(tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as client:
        missing_field = client.post("/predict", json={})
        extra_field = client.post(
            "/predict", json={"trip_id": "trip-alpha", "route": "must-not-send"}
        )

    assert missing_field.status_code == 422
    assert extra_field.status_code == 422


def test_missing_evaluation_directory_returns_empty_trip_list(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings = Settings(
        mode=settings.mode,
        evaluation_dir=tmp_path / "does-not-exist",
        artifacts_dir=settings.artifacts_dir,
        log_dir=settings.log_dir,
        model_name=settings.model_name,
        model_version=settings.model_version,
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/trips")

    assert response.status_code == 200
    assert response.json() == []


def test_real_mode_fails_startup_when_artifacts_are_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Missing required artifacts"):
        with TestClient(create_app(make_settings(tmp_path, mode="real"))):
            pass


def test_missing_artifact_after_startup_returns_503(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, mode="real")
    settings.artifacts_dir.mkdir()
    joblib.dump({"test": True}, settings.artifacts_dir / "best_model.joblib")
    joblib.dump({"test": True}, settings.artifacts_dir / "preprocessing.joblib")
    (settings.artifacts_dir / "model_card.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION}), encoding="utf-8"
    )
    (settings.artifacts_dir / "decoder_config.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION}), encoding="utf-8"
    )

    with TestClient(create_app(settings)) as client:
        (settings.artifacts_dir / "best_model.joblib").unlink()
        response = client.post("/predict", json={"trip_id": "trip-alpha"})

    assert response.status_code == 503
    assert "best_model.joblib" in response.json()["detail"]


def test_pipeline_failure_returns_500(tmp_path: Path, monkeypatch) -> None:
    app = create_app(make_settings(tmp_path))

    def fail_prediction(_: str):
        raise PipelineFailureError("synthetic pipeline failure")

    monkeypatch.setattr(app.state.prediction_service, "predict", fail_prediction)
    with TestClient(app) as client:
        response = client.post("/predict", json={"trip_id": "trip-alpha"})

    assert response.status_code == 500
