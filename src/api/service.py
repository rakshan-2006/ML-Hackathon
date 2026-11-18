"""Trip discovery, artifact validation, and inference service boundary."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib

from src.api.schemas import PredictionResponse
from src.common.paths import SCHEMA_VERSION, Settings


class TripNotFoundError(LookupError):
    """Raised when a requested trip directory is not available."""


class ArtifactUnavailableError(RuntimeError):
    """Raised when required model/decoder artifacts cannot be loaded."""


class InferenceUnavailableError(RuntimeError):
    """Raised until the Member 1-3 inference adapter is integrated."""


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
        self.registry.get_trip_dir(trip_id)
        if self.settings.mode == "real":
            self.artifact_loader.load()
            raise InferenceUnavailableError(
                "Real inference is unavailable until the Member 1-3 pipeline "
                "adapter is integrated."
            )

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
