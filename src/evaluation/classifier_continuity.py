"""Classifier-only route continuity for each model's saved score exports.

    python src/evaluation/classifier_continuity.py --hmm-root <checkout with src/hmm> \
        --raw-root "<Student folder>" knn random_forest ...

Uses Person B's implementation unchanged (PR #12: src.hmm.adapter.decode_tables
for the per-observation argmax and the shared GPS-gap segmentation, and
src.hmm.trip_evaluation.evaluate_trip -> src.hmm.route_metrics.route_continuity),
because route metrics are owned by Person B. --hmm-root is prepended to
sys.path, so every `src` import in this run comes from that checkout; the
commit is recorded. Only the classifier-only result is kept; the untuned HMM
output that decode_tables also produces is discarded (decoder results are B's).
Continuity is aggregated as sum(connected) / sum(transitions) over trips.
Writes results/models/comparison/classifier_only_continuity{,_per_trip}.csv.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPORTS = {"gradient_boosting": ("scores_train_oof.parquet", "scores_student_test.parquet")}
DEFAULT = ("oof_probabilities.parquet", "student_test_probabilities.parquet")
KEY = ["trip_id", "obs_idx", "arc_id"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hmm-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(args.hmm_root).resolve()))
    import pandas as pd
    from src.data.config import load_config
    from src.data.road_graph import load_road_graph
    from src.hmm.adapter import decode_tables
    from src.hmm.route_decoder import DecoderConfig
    from src.hmm.trip_evaluation import evaluate_trip

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.hmm_root, capture_output=True,
                            text=True).stdout.strip()
    data_config = load_config(REPO / "configs" / "data.yaml")
    gap_split_s = DecoderConfig().gap_split_s
    per_trip = []
    for partition, export_index in (("train_oof", 0), ("student_test", 1)):
        split = "train" if partition == "train_oof" else "student_test"
        processed = REPO / "data" / "processed"
        cand = pd.read_parquet(processed / f"{split}_candidates.parquet")
        obs = pd.read_parquet(processed / f"{split}_observations.parquet")
        align = pd.read_parquet(processed / f"{split}_alignment.parquet")
        graphs = {t: load_road_graph(Path(args.raw_root) / split / t) for t in sorted(obs.trip_id.unique())}
        rows_of = cand.groupby("trip_id", sort=False).indices
        for name in args.models:
            scores = pd.read_parquet(REPO / "artifacts" / name / EXPORTS.get(name, DEFAULT)[export_index])
            if not scores[KEY].reset_index(drop=True).equals(cand[KEY].reset_index(drop=True)):
                raise AssertionError(f"{name} {partition}: score keys/order differ from C2")
            p = scores.p_model.to_numpy()
            for trip, g in graphs.items():
                idx = rows_of.get(trip, [])
                o = obs[obs.trip_id == trip]
                _, base = decode_tables(o, cand.iloc[idx], p[idx], g)
                m = evaluate_trip(Path(args.raw_root) / split / trip, o, base, g, data_config,
                                  alignment=align[align.trip_id == trip])
                per_trip.append({"model": name, "partition": partition, "trip_id": trip,
                                 "n_continuity_transitions": int(m["n_continuity_transitions"]),
                                 "n_connected_transitions": int(m["n_connected_transitions"]),
                                 "n_route_breaks": int(m["n_route_breaks"]),
                                 "route_continuity": m["route_continuity"]})
            print(f"{name} {partition}: {len(graphs)} trips", flush=True)
    detail = pd.DataFrame(per_trip)
    totals = detail.groupby(["model", "partition"], as_index=False)[
        ["n_continuity_transitions", "n_connected_transitions", "n_route_breaks"]].sum()
    totals["continuity"] = totals.n_connected_transitions / totals.n_continuity_transitions
    totals["trips"] = detail.groupby(["model", "partition"]).size().to_numpy()
    totals["definition"] = ("classifier-only argmax per observation; sum connected / sum "
                            f"transitions between distinct consecutive arcs within GPS-gap segments "
                            f"(gap_split_s={gap_split_s})")
    totals["person_b_commit"] = commit
    out = REPO / "results" / "models" / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    # Replace only the models in this run; earlier models' rows are kept.
    for frame, filename in ((detail, "classifier_only_continuity_per_trip.csv"),
                            (totals, "classifier_only_continuity.csv")):
        path = out / filename
        if path.is_file():
            old = pd.read_csv(path, dtype={"trip_id": str})
            frame = pd.concat([old[~old.model.isin(args.models)], frame], ignore_index=True)
        frame.sort_values(["model", "partition"], kind="mergesort").to_csv(path, index=False)
    print(totals.to_string(index=False))


if __name__ == "__main__":
    main()
