"""M1 -> scores -> M3 decoder -> M4 API integration on Member 1's tiny_trip fixture.

SYNTHETIC_SCORER below is NOT a classifier: it is a stand-in so the pipeline can run
before Member 2's Predictor exists. No result here is a model result.
"""

import json
import yaml
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.common.paths import PROJECT_ROOT, SCHEMA_VERSION, Settings
from src.data.config import load_config
from src.hmm.adapter import decoder_config_from_dict, decoder_inputs
from src.hmm.trip_evaluation import evaluate_trip
from src.inference.pipeline import run_trip

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_trip"
DATA_CONFIG = load_config(PROJECT_ROOT / "configs" / "data.yaml")
DECODER_CONFIG_FILE = PROJECT_ROOT / "configs" / "decoder.yaml"


def SYNTHETIC_SCORER(features):
    """Placeholder probabilities from candidate rank only (not a trained model)."""
    return 1.0 / (1.0 + features.cand_rank.to_numpy(dtype=float))


def route_free_copy(tmp_path):
    """Copy of the fixture without its .route file, as seen by inference."""
    trip = tmp_path / "evaluation" / "tiny_trip"
    trip.mkdir(parents=True)
    for kind in ("track", "nodes", "arcs"):
        shutil.copy(FIXTURE / f"tiny_trip.{kind}", trip / f"tiny_trip.{kind}")
    return trip


def test_run_trip_decodes_without_reference_route(tmp_path):
    trip = route_free_copy(tmp_path)
    out = run_trip(trip, SYNTHETIC_SCORER, DATA_CONFIG)
    hmm, graph = out["hmm"], out["graph"]

    assert hmm["status"] == "ok"
    assert len(hmm["pred_arc_ids"]) == len(out["observations"])
    assert all(0 <= a < len(graph.arc_from) for a in hmm["pred_arc_ids"])
    # Every predicted arc is one of that observation's M1 candidates.
    cands = out["candidates"]
    for t, arc in enumerate(hmm["pred_arc_ids"]):
        assert arc in set(cands.loc[cands.obs_idx == t, "arc_id"])
    # Polylines are (lat, lon) taken directly from M1's node lon/lat.
    first_arc = hmm["route"][0]
    lon, lat = graph.node_lonlat[graph.arc_from[first_arc]]
    assert out["route_hmm"][0][0] == (lat, lon)


def test_decoder_inputs_keep_candidate_score_correspondence(tmp_path):
    out = run_trip(route_free_copy(tmp_path), SYNTHETIC_SCORER, DATA_CONFIG)
    obs, cand, scores = out["observations"], out["candidates"], out["scores"]
    expected = decoder_inputs(obs, cand, scores)

    perm = np.random.default_rng(0).permutation(len(cand))
    shuffled = decoder_inputs(obs, cand.iloc[perm], scores[perm])
    assert shuffled[:3] == expected[:3]

    # Removing one observation's candidates yields an empty list there.
    kept = cand.obs_idx != 1
    arcs, probs, *_ = decoder_inputs(obs, cand[kept], scores[kept.to_numpy()])
    assert arcs[1] == [] and probs[1] == []

    with pytest.raises(ValueError):
        decoder_inputs(obs, cand, scores[:-1])


def test_committed_decoder_config_is_valid():
    document = yaml.safe_load(DECODER_CONFIG_FILE.read_text(encoding="utf-8"))
    assert document["schema_version"] == SCHEMA_VERSION
    decoder_config_from_dict(document)
    with pytest.raises(ValueError):
        decoder_config_from_dict({"beta": 1.0})


def test_evaluate_trip_on_fixture_reference(tmp_path):
    out = run_trip(route_free_copy(tmp_path), SYNTHETIC_SCORER, DATA_CONFIG)
    metrics = evaluate_trip(FIXTURE, out["observations"], out["hmm"], out["graph"],
                            DATA_CONFIG)
    assert {"top1_accuracy", "n_evaluated", "route_continuity", "arc_f1", "lcs_similarity"} <= set(metrics)
    assert metrics["route_continuity"] == 1.0
