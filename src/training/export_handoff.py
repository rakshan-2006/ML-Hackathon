"""Package the selected model for Person B: deployment bundle, keyed scores, OOF manifest.

    python -m src.training.export_handoff --name gradient_boosting_tuned --output artifacts/handoff

Writes <output>/<model_version>/:
  best_model.joblib, preprocessing.joblib, model_card.json   (src.training.export_bundle)
  scores_train_oof.parquet      C4 keys + p_model, model_name, model_version, fold
  scores_student_test.parquet   C4 keys + p_model, model_name, model_version
  oof_manifest.json             provenance consumed by Person B's src.hmm.tune
  SHA256SUMS.txt
The OOF scores are the shared runner's out-of-fold predictions for this exact
parameter set: each trip scored by the fold pipeline (fold-fitted
preprocessing) that never saw it. Every key and fold is re-validated here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT
from src.common.schema import SCHEMA_VERSION
from src.evaluation.audit_scores import DEFAULT_EXPORTS, EXPORTS
from src.inference.predictor import file_sha256
from src.training.cross_validation import FOLDS_PATH, fold_of_rows, load_folds
from src.training.export_bundle import export
from src.training.train_models import ARTIFACTS, DATA, KEY

DATA_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"


def keyed(scores, cand, name):
    if not scores[KEY].reset_index(drop=True).equals(cand[KEY].reset_index(drop=True)):
        raise AssertionError(f"{name}: keys/order differ from C2")
    if scores[KEY].duplicated().any():
        raise AssertionError(f"{name}: duplicate keys")
    p = scores.p_model.to_numpy()
    if p.dtype != np.float64 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise AssertionError(f"{name}: invalid probabilities")
    if scores.model_version.nunique() != 1:
        raise AssertionError(f"{name}: more than one model_version")
    out = scores.copy()
    out["trip_id"] = out.trip_id.astype("string")
    return out


def handoff(name, output_root):
    card = json.loads((PROJECT_ROOT / "results" / "models" / name / "model_card.json")
                      .read_text(encoding="utf-8"))
    version = card["model_version"]
    out = Path(output_root) / version
    if out.exists():
        raise FileExistsError(f"{out} exists; handoffs are immutable")
    bundle_card = export(name, "final", out)

    folds = load_folds()
    train_cand = pd.read_parquet(DATA / "train_candidates.parquet", columns=KEY)
    test_cand = pd.read_parquet(DATA / "student_test_candidates.parquet", columns=KEY)
    oof_file, test_file, _ = EXPORTS.get(name, DEFAULT_EXPORTS)
    oof = keyed(pd.read_parquet(ARTIFACTS / name / oof_file), train_cand, "OOF")
    test = keyed(pd.read_parquet(ARTIFACTS / name / test_file), test_cand, "student_test")
    for frame in (oof, test):
        if set(frame.model_version) != {version}:
            raise AssertionError("score export version differs from the model card")
    canonical = fold_of_rows(oof.trip_id, folds).astype("int64")
    if "fold" in oof.columns and not np.array_equal(oof.fold.to_numpy(), canonical):
        raise AssertionError("exported OOF fold IDs differ from the canonical folds")
    oof["fold"] = canonical
    oof.to_parquet(out / "scores_train_oof.parquet", index=False)
    test.to_parquet(out / "scores_student_test.parquet", index=False)

    consumed = {"candidates": DATA / "train_candidates.parquet",
                "observations": DATA / "train_observations.parquet",
                "alignment": DATA / "train_alignment.parquet",
                "scores": out / "scores_train_oof.parquet",
                "model_card": out / "model_card.json",
                "folds": FOLDS_PATH, "data_config": DATA_CONFIG}
    manifest = {
        "status": "final_parameter_oof", "schema_version": SCHEMA_VERSION,
        "model_version": version, "model_name": bundle_card["model_name"],
        "partition": "train", "score_kind": "out_of_fold",
        "fold_holdout_trip_ids": {str(k): sorted(g.trip_id.astype(str)) for k, g in folds.groupby("fold")},
        "sha256": {k: file_sha256(v) for k, v in consumed.items()},
        "rows": {"train_oof": int(len(oof)), "student_test": int(len(test))},
        "generation": ("src.training.train_models with the final selected parameters: for each "
                       "canonical fold the full factory pipeline (preprocessing fitted on the four "
                       "training folds' labelled rows only) scored every candidate of the held-out "
                       "trips; no held-out trip was used for fitting. The final refit on all 80 "
                       "train trips produced scores_student_test only."),
        "parameters": bundle_card["params"],
    }
    (out / "oof_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sums = [f"{file_sha256(p)}  {p.name}" for p in sorted(out.iterdir()) if p.name != "SHA256SUMS.txt"]
    (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"handoff written to {out}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", default=str(ARTIFACTS / "handoff"))
    args = parser.parse_args(argv)
    handoff(args.name, args.output)


if __name__ == "__main__":
    main()
