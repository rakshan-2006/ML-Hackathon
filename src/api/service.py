"""Trip discovery, artifact validation, and inference service boundary."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import hashlib
from collections import OrderedDict
from time import perf_counter
import numpy as np

from src.api.schemas import PredictionResponse, GeoPoint, RoutePolyline
from src.common.paths import SCHEMA_VERSION, Settings, PROJECT_ROOT
from src.common.schema import FEATURES
from src.data.config import load_config
from src.data.road_graph import load_road_graph
from src.hmm.adapter import decoder_config_from_dict
from src.inference.pipeline import run_trip


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


def load_scorer(artifact_dir, bundle):
    """Person A's final scorer, or an explicitly labelled real development model."""
    if bundle.model_card.get('status') == 'development_baseline':
        from src.inference.development_baseline import DevelopmentScorer
        return DevelopmentScorer(bundle.model,bundle.preprocessing,bundle.model_card).score_candidates
    try:
        from src.inference.predictor import Predictor
    except ModuleNotFoundError as exc:
        if exc.name != 'src.inference.predictor':
            raise
        raise ArtifactUnavailableError('Person A final Predictor is not available; use an explicit development baseline bundle') from exc
    return Predictor.load(artifact_dir).score_candidates


class TripRegistry:
    """Discover trip folders without assuming or hardcoding any trip IDs."""

    def __init__(self, evaluation_dir: Path, require_inputs=False) -> None:
        self.evaluation_dir = evaluation_dir
        self.require_inputs = require_inputs

    def list_trip_ids(self) -> list[str]:
        if not self.evaluation_dir.is_dir():
            return []
        return sorted(
            child.name
            for child in self.evaluation_dir.iterdir()
            if child.is_dir() and not child.name.startswith('.')
            and child.resolve().parent == self.evaluation_dir.resolve()
            and (not self.require_inputs or all((child/f'{child.name}.{kind}').is_file()
                                                for kind in ['track','nodes','arcs']))
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
        if model_card.get('ordered_features',model_card.get('features')) != FEATURES:
            raise ArtifactUnavailableError('Model feature order is incompatible with the shared schema')
        if model_card.get('artifact_layout') != 'split_estimator_preprocessor':
            raise ArtifactUnavailableError('Expected split_estimator_preprocessor artifact layout')
        if decoder_config.get('model_version') not in (None, model_card.get('model_version')):
            raise ArtifactUnavailableError('Decoder was configured for a different model version')
        try:
            decoder_config_from_dict(decoder_config)
        except (ValueError,TypeError) as exc:
            raise ArtifactUnavailableError(f'Invalid decoder configuration: {exc}') from exc
        manifest_path=self.artifacts_dir/'deployment_manifest.json'
        if manifest_path.is_file():
            manifest=self._load_json('deployment_manifest.json')
            for name,digest in manifest.get('sha256',{}).items():
                if Path(name).name!=name or name not in self.REQUIRED_FILES:
                    raise ArtifactUnavailableError('Unknown deployment manifest entry')
                if hashlib.sha256((self.artifacts_dir/name).read_bytes()).hexdigest()!=digest:
                    raise ArtifactUnavailableError(f'Artifact checksum mismatch: {name}')
        try:
            model = joblib.load(self.artifacts_dir / "best_model.joblib")
            preprocessing = joblib.load(
                self.artifacts_dir / "preprocessing.joblib"
            )
            if not hasattr(model,'predict_proba') or not hasattr(preprocessing,'transform'):
                raise ValueError('Artifact objects do not implement the scorer contract')
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
        self.registry = TripRegistry(settings.evaluation_dir,require_inputs=settings.mode=='real')
        self.artifact_loader = ArtifactLoader(settings.artifacts_dir)
        self.bundle = None
        self.graph_cache = OrderedDict()

    def startup(self):
        if self.settings.mode!='real' or self.bundle is not None:
            return
        bundle=self.artifact_loader.load()
        if self.settings.require_final:
            if bundle.model_card.get('status')!='final' or bundle.decoder_config.get('status')!='final_tuned':
                raise ArtifactUnavailableError('Final model and tuned decoder are still pending')
            if not (self.settings.artifacts_dir/'deployment_manifest.json').is_file():
                raise ArtifactUnavailableError('Final deployment manifest is missing')
        self.data_config=load_config(self.settings.data_config_path or PROJECT_ROOT/'configs/data.yaml')
        expected=bundle.decoder_config.get('data_config_sha256')
        if expected and hashlib.sha256((self.settings.data_config_path or PROJECT_ROOT/'configs/data.yaml').read_bytes()).hexdigest()!=expected:
            raise ArtifactUnavailableError('Frozen data configuration fingerprint mismatch')
        self.decoder_config=decoder_config_from_dict(bundle.decoder_config)
        self.scorer=load_scorer(self.settings.artifacts_dir,bundle)
        self.bundle=bundle
        self.artifact_state=self._artifact_state()

    def _artifact_state(self):
        try:
            return tuple((name,(self.settings.artifacts_dir/name).stat().st_mtime_ns,
                          (self.settings.artifacts_dir/name).stat().st_size)
                         for name in self.artifact_loader.REQUIRED_FILES)
        except OSError as exc:
            raise ArtifactUnavailableError(f'Artifact removed after startup: {exc.filename}') from exc

    def predict(self, trip_id: str) -> PredictionResponse:
        trip_dir=self.registry.get_trip_dir(trip_id)
        if self.settings.mode == "real":
            self.startup()
            if self._artifact_state()!=self.artifact_state:
                raise ArtifactUnavailableError('Artifacts changed after startup; restart to load a verified bundle')
            try:
                started=perf_counter()
                key=(str(trip_dir.resolve()),tuple((trip_dir/f'{trip_id}.{k}').stat().st_mtime_ns for k in ['nodes','arcs']))
                cached=self.graph_cache.get(key)
                if cached is None:
                    cached=load_road_graph(trip_dir)
                    if self.settings.graph_cache_size:
                        self.graph_cache[key]=cached
                        while len(self.graph_cache)>self.settings.graph_cache_size:
                            self.graph_cache.popitem(last=False)
                else:
                    self.graph_cache.move_to_end(key)
                graph_ms=(perf_counter()-started)*1000
                output=run_trip(trip_dir,self.scorer,self.data_config,self.decoder_config,graph=cached)
                output['timings']['load_graph']=graph_ms
                # Explicitly after route-free prediction, never inside run_trip.
                from src.api.evaluation import evaluate_prediction
                evaluation_start=perf_counter()
                metrics,evaluation=evaluate_prediction(trip_dir,output,self.data_config)
                output['timings']['evaluate']=(perf_counter()-evaluation_start)*1000
                obs=output['observations']
                geometry=lambda lines:[RoutePolyline(points=[GeoPoint(latitude=lat,longitude=lon)
                                                             for lat,lon in line]) for line in lines]
                return PredictionResponse(trip_id=trip_id,n_obs=len(obs),
                    gps_coordinates=[GeoPoint(latitude=float(lat),longitude=float(lon))
                                     for lat,lon in zip(obs.lat,obs.lon) if np.isfinite(lat) and np.isfinite(lon)
                                     and -90<=lat<=90 and -180<=lon<=180],
                    route_hmm=geometry(output['route_hmm']),route_classifier=geometry(output['route_classifier']),
                    metrics=metrics,diagnostics={'status':output['hmm']['status'],
                        'deployment_status':self.bundle.model_card.get('status','unknown'),
                        'model_version':self.bundle.model_card.get('model_version','unknown'),
                        'decoder_version':self.bundle.decoder_config.get('version','unknown'),
                        'n_candidates':len(output['candidates']),
                        **{k:output['hmm'][k] for k in ['n_segments','n_breaks_gap','n_breaks_no_transition','n_no_candidates','n_inferred_arcs']},
                        **evaluation},timings=output['timings'],mode='real',is_stub=False)
            except ArtifactUnavailableError:
                raise
            except Exception as exc:
                raise PipelineFailureError('Real pipeline failed; inspect server logs') from exc

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
