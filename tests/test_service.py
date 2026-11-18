"""Tests for required artifact reporting and schema compatibility."""

import json
from pathlib import Path

import pytest

from src.api.service import ArtifactLoader, ArtifactUnavailableError
from src.common.schema import SCHEMA_VERSION


def test_missing_artifacts_are_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ArtifactUnavailableError, match="best_model.joblib"):
        ArtifactLoader(tmp_path).load()


def test_incompatible_schema_version_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "best_model.joblib").touch()
    (tmp_path / "preprocessing.joblib").touch()
    (tmp_path / "model_card.json").write_text(
        json.dumps({"schema_version": "incompatible"}), encoding="utf-8"
    )
    (tmp_path / "decoder_config.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION}), encoding="utf-8"
    )

    with pytest.raises(ArtifactUnavailableError, match="incompatible"):
        ArtifactLoader(tmp_path).load()
