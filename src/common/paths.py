"""Centralized, environment-configurable project paths."""

from dataclasses import dataclass
import os
from pathlib import Path

from src.common.schema import SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configured_path(variable: str, default: str) -> Path:
    value = Path(os.environ.get(variable, default)).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from environment variables."""

    mode: str
    evaluation_dir: Path
    artifacts_dir: Path
    log_dir: Path
    model_name: str
    model_version: str

    @classmethod
    def from_environment(cls) -> "Settings":
        mode = os.environ.get("PATHSNAP_MODE", "stub").strip().lower()
        if mode not in {"stub", "real"}:
            raise ValueError("PATHSNAP_MODE must be either 'stub' or 'real'.")
        return cls(
            mode=mode,
            evaluation_dir=_configured_path(
                "PATHSNAP_EVALUATION_DIR", "data/evaluation"
            ),
            artifacts_dir=_configured_path("PATHSNAP_ARTIFACTS_DIR", "models"),
            log_dir=_configured_path("PATHSNAP_LOG_DIR", "logs"),
            model_name=os.environ.get(
                "PATHSNAP_MODEL_NAME",
                "walking-skeleton" if mode == "stub" else "not-configured",
            ),
            model_version=os.environ.get("PATHSNAP_MODEL_VERSION", "not-loaded"),
        )


def get_settings() -> Settings:
    """Return a fresh settings snapshot, suitable for environment-based launch."""
    return Settings.from_environment()
