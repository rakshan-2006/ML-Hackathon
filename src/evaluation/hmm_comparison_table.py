"""Classifier-only vs classifier+HMM summary table from Person B's per-trip results.

    python -m src.evaluation.hmm_comparison_table
    python -m src.evaluation.hmm_comparison_table --source results/rehearsal/student_test_api.csv

Reads the per-trip CSV written by src.api.rehearse (one row per student_test trip,
`<metric>_classifier` and `<metric>_hmm` columns) and writes
results/hmm/development/classifier_vs_hmm_summary.{csv,md}. Ratios are pooled over
trips (sum of numerators / sum of denominators); arc F1 and LCS similarity are
unweighted means of the per-trip values; break counts are summed. No metric is
redefined: every per-trip value comes from src.hmm.route_metrics via the rehearsal.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common.paths import PROJECT_ROOT

SOURCE = PROJECT_ROOT / "results" / "rehearsal" / "student_test_api.csv"
OUT = PROJECT_ROOT / "results" / "hmm" / "development"
METHODS = {"classifier": "Classifier only", "hmm": "Classifier + HMM"}


def summarise(trips):
    rows = []
    pooled = [("Top-1 accuracy (trusted observations)", "n_correct", "n_evaluated"),
              ("Route continuity (connected / transitions)", "n_connected_transitions",
               "n_continuity_transitions")]
    for label, num, den in pooled:
        rows.append({"metric": label, "aggregation": f"sum {num} / sum {den}",
                     **{m: trips[f"{num}_{m}"].sum() / trips[f"{den}_{m}"].sum() for m in METHODS}})
    rows.append({"metric": "Route breaks", "aggregation": "sum n_route_breaks",
                 **{m: int(trips[f"n_route_breaks_{m}"].sum()) for m in METHODS}})
    for label, col in (("Arc-overlap F1 vs reference route", "arc_f1"),
                       ("Order-aware LCS similarity vs reference route", "lcs_similarity")):
        rows.append({"metric": label, "aggregation": f"mean of per-trip {col}",
                     **{m: trips[f"{col}_{m}"].mean() for m in METHODS}})
    table = pd.DataFrame(rows)
    better = trips.top1_accuracy_hmm > trips.top1_accuracy_classifier
    worse = trips.top1_accuracy_hmm < trips.top1_accuracy_classifier
    return table, {"trips": int(len(trips)), "hmm_top1_better": int(better.sum()),
                   "hmm_top1_worse": int(worse.sum()),
                   "worse_trip_ids": trips.loc[worse, "trip_id"].tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(SOURCE))
    args = parser.parse_args(argv)
    trips = pd.read_csv(args.source, dtype={"trip_id": "string"})
    try:
        source = Path(args.source).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source = Path(args.source).name
    versions = {c: sorted(trips[c].unique()) for c in ("model_version", "decoder_version")}
    table, counts = summarise(trips)
    OUT.mkdir(parents=True, exist_ok=True)
    table.rename(columns={"classifier": "classifier_only", "hmm": "classifier_plus_hmm"}) \
        .to_csv(OUT / "classifier_vs_hmm_summary.csv", index=False)

    def fmt(row, m):
        v = getattr(row, m)
        return f"{int(v):,}" if row.metric == "Route breaks" else (
            f"{100 * v:.2f}%" if "accuracy" in row.metric or "continuity" in row.metric else f"{v:.3f}")
    lines = [f"Classifier-only vs classifier+HMM on {counts['trips']} student_test trips "
             f"(model {', '.join(versions['model_version'])}, decoder {', '.join(versions['decoder_version'])}; "
             f"development measurements, not evaluator results). Source: `{source}`.", "",
             "| Metric | Classifier only | Classifier + HMM | Aggregation |", "|---|---:|---:|---|"]
    lines += [f"| {r.metric} | {fmt(r, 'classifier')} | {fmt(r, 'hmm')} | {r.aggregation} |"
              for r in table.itertuples()]
    lines += ["", f"HMM Top-1 better on {counts['hmm_top1_better']} trips, worse on "
              f"{counts['hmm_top1_worse']} ({', '.join(counts['worse_trip_ids']) or 'none'})."]
    (OUT / "classifier_vs_hmm_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(versions))


if __name__ == "__main__":
    main()
