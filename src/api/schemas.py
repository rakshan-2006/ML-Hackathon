"""Pydantic request and response models for the public API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.common.paths import SCHEMA_VERSION


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    schema_version: str
    decoder_config_version: str


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    trip_id: str = Field(min_length=1, max_length=128)


class GeoPoint(BaseModel):
    latitude: float
    longitude: float


class RoutePolyline(BaseModel):
    points: list[GeoPoint]


class PredictionResponse(BaseModel):
    trip_id: str
    n_obs: int = Field(ge=0)
    gps_coordinates: list[GeoPoint]
    route_hmm: list[RoutePolyline]
    route_classifier: list[RoutePolyline] | None = None
    metrics: dict[str, float]
    diagnostics: dict[str, str]
    timings: dict[str, float]
    mode: Literal["stub", "real"]
    is_stub: bool


def stub_health(model_name: str, model_version: str) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=f"{model_name} (stub)",
        model_version=model_version,
        schema_version=SCHEMA_VERSION,
        decoder_config_version="not-loaded",
    )
