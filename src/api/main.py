"""FastAPI application factory and endpoint definitions."""

from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request

from src.api.logging_config import configure_logging, log_event
from src.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictionResponse,
    stub_health,
)
from src.api.service import (
    ArtifactLoader,
    ArtifactUnavailableError,
    InferenceUnavailableError,
    PipelineFailureError,
    PredictionService,
    TripNotFoundError,
)
from src.common.paths import SCHEMA_VERSION, Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()
    logger = configure_logging(runtime_settings.log_dir)
    service = PredictionService(runtime_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if runtime_settings.mode == "real":
            try:
                ArtifactLoader(runtime_settings.artifacts_dir).load()
            except ArtifactUnavailableError as exc:
                logger.error("Real-mode startup failed: %s", exc)
                raise RuntimeError(f"Real-mode startup failed: {exc}") from exc
        yield

    app = FastAPI(
        title="PathSnap API",
        description="API foundation for PathSnap route reconstruction.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = runtime_settings
    app.state.prediction_service = service
    app.state.logger = logger

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        started = perf_counter()
        client_ip = request.client.host if request.client else None
        error = None
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            if status_code >= 400:
                error = f"HTTP {status_code}"
            return response
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
            log_event(
                logger,
                client_ip=client_ip,
                trip_id=None,
                stage=f"request:{request.method}:{request.url.path}:{status_code}",
                timings={"total": round(elapsed_ms, 3)},
                error=error,
            )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        if runtime_settings.mode == "stub":
            return stub_health(
                runtime_settings.model_name, runtime_settings.model_version
            )
        try:
            bundle = service.artifact_loader.load()
        except ArtifactUnavailableError:
            return HealthResponse(
                status="degraded",
                model_name=runtime_settings.model_name,
                model_version=runtime_settings.model_version,
                schema_version=SCHEMA_VERSION,
                decoder_config_version="unknown",
            )
        return HealthResponse(
            status="degraded",
            model_name=str(
                bundle.model_card.get("model_name", runtime_settings.model_name)
            ),
            model_version=str(
                bundle.model_card.get("model_version", runtime_settings.model_version)
            ),
            schema_version=str(bundle.model_card["schema_version"]),
            decoder_config_version=str(
                bundle.decoder_config.get("version", "unknown")
            ),
        )

    @app.get("/trips", response_model=list[str])
    def trips() -> list[str]:
        return service.registry.list_trip_ids()

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictionResponse:
        client_ip = request.client.host if request.client else None
        started = perf_counter()
        try:
            result = service.predict(payload.trip_id)
        except TripNotFoundError as exc:
            log_event(
                logger,
                client_ip=client_ip,
                trip_id=payload.trip_id,
                stage="trip_lookup",
                timings={"total": round((perf_counter() - started) * 1000, 3)},
                error=str(exc),
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ArtifactUnavailableError as exc:
            log_event(
                logger,
                client_ip=client_ip,
                trip_id=payload.trip_id,
                stage="artifact_loading",
                timings={"total": round((perf_counter() - started) * 1000, 3)},
                error=str(exc),
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except InferenceUnavailableError as exc:
            log_event(
                logger,
                client_ip=client_ip,
                trip_id=payload.trip_id,
                stage="inference",
                timings={"total": round((perf_counter() - started) * 1000, 3)},
                error=str(exc),
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except PipelineFailureError as exc:
            log_event(
                logger,
                client_ip=client_ip,
                trip_id=payload.trip_id,
                stage="inference",
                timings={"total": round((perf_counter() - started) * 1000, 3)},
                error=str(exc),
            )
            raise HTTPException(
                status_code=500, detail="Inference pipeline failed."
            ) from exc

        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        result.timings["total"] = elapsed_ms
        log_event(
            logger,
            client_ip=client_ip,
            trip_id=payload.trip_id,
            stage="inference",
            timings={"total": elapsed_ms},
        )
        return result

    return app


app = create_app()
