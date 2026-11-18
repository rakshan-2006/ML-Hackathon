"""Trip discovery, artifact validation, and inference service boundary."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib

from src.api.schemas import GeoPoint, PredictionResponse, RoutePolyline
from src.common.paths import PROJECT_ROOT, SCHEMA_VERSION, Settings

DATA_CONFIG_PATH = PROJECT_ROOT / "experiments" / "data_config.yaml"


class TripNotFoundError(LookupError):
    """Raised when a requested trip directory is not available."""


class ArtifactUnavailableError(RuntimeError):
    """Raised when required model/decoder artifacts cannot be loaded."""


class InferenceUnavailableError(RuntimeError):
    """Raised when a required inference component (e.g. the M2 scorer) is missing."""


class PipelineFailureError(RuntimeError):
    """Raised when an installed inference pipeline fails."""


@dataclass(frozen=True)
class ArtifactBundle:
    model: Any
    preprocessing: Any
    model_card: dict[str, Any]
    decoder_config: dict[str, Any]


class TripRegistry:
    """Discover trip folders without assuming or hardcoding any trip IDs."""

    def __init__(self, evaluation_dir: Path) -> None:
        self.evaluation_dir = evaluation_dir

    def list_trip_ids(self) -> list[str]:
        if not self.evaluation_dir.is_dir():
            return []
        return sorted(
            child.name
            for child in self.evaluation_dir.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        )

    def get_trip_dir(self, trip_id: str) -> Path:
        trip_dir = self.evaluation_dir / trip_id
        if not trip_dir.is_dir() or trip_id not in self.list_trip_ids():
            raise TripNotFoundError(f"Unknown trip ID: {trip_id}")
        return trip_dir


class ArtifactLoader:
    """Load and validate the agreed artifact filenames and schema version."""

    REQUIRED_FILES = (
        "best_model.joblib",
        "preprocessing.joblib",
        "model_card.json",
        "decoder_config.json",
    )

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir

    def load(self) -> ArtifactBundle:
        missing = [
            name for name in self.REQUIRED_FILES
            if not (self.artifacts_dir / name).is_file()
        ]
        if missing:
            names = ", ".join(missing)
            raise ArtifactUnavailableError(
                f"Missing required artifacts in {self.artifacts_dir}: {names}"
            )

        model_card = self._load_json("model_card.json")
        decoder_config = self._load_json("decoder_config.json")
        self._validate_schema(model_card, "model_card.json")
        self._validate_schema(decoder_config, "decoder_config.json")
        try:
            model = joblib.load(self.artifacts_dir / "best_model.joblib")
            preprocessing = joblib.load(
                self.artifacts_dir / "preprocessing.joblib"
            )
        except Exception as exc:
            raise ArtifactUnavailableError(
                f"Could not load joblib artifacts in {self.artifacts_dir}: {exc}"
            ) from exc
        return ArtifactBundle(
            model=model,
            preprocessing=preprocessing,
            model_card=model_card,
            decoder_config=decoder_config,
        )

    def _load_json(self, filename: str) -> dict[str, Any]:
        try:
            with (self.artifacts_dir / filename).open(
                "r", encoding="utf-8"
            ) as artifact_file:
                document = json.load(artifact_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactUnavailableError(
                f"Could not read {filename} from {self.artifacts_dir}: {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise ArtifactUnavailableError(f"{filename} must contain a JSON object.")
        return document

    @staticmethod
    def _validate_schema(document: dict[str, Any], filename: str) -> None:
        actual = document.get("schema_version")
        if actual != SCHEMA_VERSION:
            raise ArtifactUnavailableError(
                f"{filename} schema_version {actual!r} is incompatible; "
                f"expected {SCHEMA_VERSION!r}."
            )


class PredictionService:
    """Provides contract-valid stub responses until real components are ready."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = TripRegistry(settings.evaluation_dir)
        self.artifact_loader = ArtifactLoader(settings.artifacts_dir)

    def predict(self, trip_id: str) -> PredictionResponse:
        trip_dir = self.registry.get_trip_dir(trip_id)
        if self.settings.mode == "real":
            bundle = self.artifact_loader.load()
            return self._predict_real(trip_id, trip_dir, bundle)

        return PredictionResponse(
            trip_id=trip_id,
            n_obs=0,
            gps_coordinates=[],
            route_hmm=[],
            route_classifier=None,
            metrics={},
            diagnostics={
                "status": "stub",
                "message": (
                    "No GPS observations or route predictions are loaded. "
                    "Member 1-3 inference integration is pending."
                ),
            },
            timings={},
            mode="stub",
            is_stub=True,
        )

    def _predict_real(
        self, trip_id: str, trip_dir: Path, bundle: ArtifactBundle
    ) -> PredictionResponse:
        # Imported lazily so stub mode runs without the M1/M3 dependencies.
        from src.data.config import load_config
        from src.hmm.adapter import decoder_config_from_dict
        from src.hmm.route_metrics import route_continuity
        from src.inference.pipeline import run_trip

        score_candidates = load_scorer(self.settings.artifacts_dir)
        try:
            data_config = load_config(DATA_CONFIG_PATH)
            decoder_config = decoder_config_from_dict(bundle.decoder_config)
            out = run_trip(trip_dir, score_candidates, data_config, decoder_config)
        except Exception as exc:
            raise PipelineFailureError(f"{type(exc).__name__}: {exc}") from exc

        hmm, classifier, graph = out["hmm"], out["classifier"], out["graph"]
        obs = out["observations"]
        gps = obs.loc[obs.coordinate_valid, ["lat", "lon"]].to_numpy()
        metrics = {
            "route_continuity_hmm": route_continuity(hmm["route"], graph)["continuity"],
            "route_continuity_classifier": route_continuity(
                classifier["route"], graph
            )["continuity"],
        }
        return PredictionResponse(
            trip_id=trip_id,
            n_obs=len(obs),
            gps_coordinates=[GeoPoint(latitude=lat, longitude=lon) for lat, lon in gps],
            route_hmm=_polylines(out["route_hmm"]),
            route_classifier=_polylines(out["route_classifier"]),
            # Reference-free metrics only; NaN (nothing to measure) is not JSON-safe.
            metrics={k: float(v) for k, v in metrics.items() if v == v},
            diagnostics={
                "status": hmm["status"],
                "warnings": "; ".join(hmm["warnings"]),
                "n_segments": str(hmm["n_segments"]),
                "n_breaks_gap": str(hmm["n_breaks_gap"]),
                "n_breaks_no_transition": str(hmm["n_breaks_no_transition"]),
                "n_no_candidates": str(hmm["n_no_candidates"]),
                "n_candidates": str(len(out["candidates"])),
            },
            timings=out["timings"],
            mode="real",
            is_stub=False,
        )


def _polylines(lines: list[list[tuple[float, float]]]) -> list[RoutePolyline]:
    return [
        RoutePolyline(points=[GeoPoint(latitude=lat, longitude=lon) for lat, lon in line])
        for line in lines
    ]


def load_scorer(artifacts_dir: Path):
    """Member 2's candidate scorer: Predictor.load(dir).score_candidates(features)."""
    try:
        from src.inference.predictor import Predictor
    except ImportError as exc:
        raise InferenceUnavailableError(
            "Classifier probabilities are unavailable: Member 2's "
            "src.inference.predictor.Predictor (load(models_dir) -> "
            "score_candidates(features)) is not integrated yet."
        ) from exc
    try:
        return Predictor.load(artifacts_dir).score_candidates
    except Exception as exc:
        raise PipelineFailureError(f"Could not load Member 2's Predictor: {exc}") from exc
