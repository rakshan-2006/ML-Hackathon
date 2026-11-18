"""Request and inference logging configuration."""

from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Mapping


LOGGER_NAME = "pathsnap"


def configure_logging(log_dir: Path) -> logging.Logger:
    """Log to the console and a size-limited file under the configured directory."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        log_dir / "pathsnap.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def log_event(
    logger: logging.Logger,
    *,
    client_ip: str | None,
    trip_id: str | None,
    stage: str,
    timings: Mapping[str, float] | None = None,
    error: str | None = None,
) -> None:
    """Emit the requested request/inference fields in a structured log message."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_ip": client_ip,
        "trip_id": trip_id,
        "processing_stage": stage,
        "stage_timings_ms": dict(timings or {}),
        "error": error,
    }
    logger.info("%s", json.dumps(payload, sort_keys=True))
