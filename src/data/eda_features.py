"""Feature-level EDA on the permitted partitions (train and student_test only).

    python -m src.data.eda_features --raw-root "<Student folder>"

Complements `src.data.eda audit` (inventory, graph, projection and alignment
evidence) with what the report's EDA section also needs: per-feature missing
values, label availability, train feature distributions by label, sampling
intervals, and one GPS-trajectory/road-network/reference plot per partition.
Every table and figure names its partition. The representative trip of each
partition is the one with the median observation count (deterministic).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.paths import PROJECT_ROOT  # noqa: E402
from src.common.schema import FEATURES  # noqa: E402
from src.data.dataset_loader import list_trips, load_route  # noqa: E402
from src.data.road_graph import load_road_graph  # noqa: E402

PARTITIONS = ("train", "student_test")
COLOURS = {"train": "#2a78d6", "student_test": "#eb6834"}
DATA = PROJECT_ROOT / "data" / "processed"
EVIDENCE = PROJECT_ROOT / "results" / "eda"
FIGURES = PROJECT_ROOT / "results" / "figures" / "eda"
INK, MUTED = "#0b0b0b", "#52514e"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def missing_values(cands):
    rows = []
    for partition, c in cands.items():
        status = np.where(c.label.isna(), "unlabelled", np.where(c.label == 1, "positive", "negative"))
        for feature in FEATURES:
            miss = c[feature].isna()
            row = {"partition": partition, "feature": feature, "rows": len(c),
                   "missing": int(miss.sum()), "missing_fraction": float(miss.mean())}
            for s in ("positive", "negative", "unlabelled"):
                mask = status == s
                row[f"missing_fraction_{s}"] = float(miss[mask].mean()) if mask.any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def label_summary(cands, obs):
    rows = []
    for partition, c in cands.items():
        o = obs[partition]
        rows.append({"partition": partition, "trips": int(c.trip_id.nunique()),
                     "observations": len(o), "candidates": len(c),
                     "labelled_candidates": int(c.label.notna().sum()),
                     "unlabelled_candidates": int(c.label.isna().sum()),
                     "positive_candidates": int((c.label == 1).sum()),
                     "positive_rate_labelled": float((c.label == 1).sum() / c.label.notna().sum()),
                     "candidates_per_observation_mean": float(o.n_candidates.mean()),
                     "zero_candidate_observations": int((o.n_candidates == 0).sum()),
                     "dt_prev_p50_s": float(o.dt_prev.median()),
                     "dt_prev_p99_s": float(o.dt_prev.quantile(0.99)),
                     "dt_prev_max_s": float(o.dt_prev.max()),
                     "gap_flag_observations": int(o.gap_flag.sum()),
                     "speed_mps_p50": float(o.speed_mps.median()),
                     "speed_mps_p99": float(o.speed_mps.quantile(0.99))})
    return pd.DataFrame(rows)


def feature_by_label(train):
    lab = train[train.label.notna()]
    rows = []
    for feature in FEATURES:
        for label, group in lab.groupby("label"):
            v = group[feature].astype(float)
            rows.append({"partition": "train", "feature": feature, "label": int(label),
                         "rows": len(v), "missing": int(v.isna().sum()),
                         "p10": v.quantile(0.1), "median": v.median(), "p90": v.quantile(0.9),
                         "mean": v.mean()})
    return pd.DataFrame(rows)


def plot_missing(table, out):
    features = [f for f in FEATURES if table[table.feature == f].missing.sum() > 0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(features))
    width = 0.38
    for i, partition in enumerate(PARTITIONS):
        sub = table[table.partition == partition].set_index("feature").loc[features]
        bars = ax.bar(x + (i - 0.5) * width, 100 * sub.missing_fraction, width - 0.02,
                      color=COLOURS[partition], label=partition)
        for bar, value in zip(bars, 100 * sub.missing_fraction):
            ax.annotate(f"{value:.2f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8, color=INK,
                        xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, features)
    ax.set_ylabel("Candidate rows with NaN (%)")
    ax.set_title("Missing model features by partition (all C2 candidate rows)\n"
                 "Features not shown have no missing values", fontsize=10)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "missing_values.png", dpi=150)
    plt.close(fig)


def plot_sampling(obs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.arange(0.5, 12.5, 1)
    for partition in PARTITIONS:
        o = obs[partition]
        dt = o.dt_prev.dropna()
        axes[0].hist(dt, bins=bins, color=COLOURS[partition], alpha=0.6, label=partition,
                     weights=np.full(len(dt), 100 / len(dt)))
        speed = o.speed_mps.dropna().clip(upper=60)
        axes[1].hist(speed, bins=60, color=COLOURS[partition], alpha=0.6, label=partition,
                     weights=np.full(len(speed), 100 / len(speed)), histtype="stepfilled")
    axes[0].set(xlabel="Seconds since previous GPS point (dt_prev)", ylabel="Observations (%)",
                yscale="log", title="Sampling interval")
    axes[1].set(xlabel="GPS speed (m/s, clipped at 60)", ylabel="Observations (%)",
                title="Point speed")
    for ax in axes:
        ax.legend(frameon=False)
        style(ax)
    fig.suptitle("train vs student_test observations (first point of each trip has no dt/speed)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "sampling_and_speed.png", dpi=150)
    plt.close(fig)


def plot_features_by_label(train, out):
    lab = train[train.label.notna()]
    shown = ["dist_m", "dist_minus_min", "heading_diff_deg", "cand_rank", "prev_dist_m", "proj_frac"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, feature in zip(axes.flat, shown):
        values = lab[feature].astype(float)
        hi = values.quantile(0.99)
        bins = np.linspace(values.min(), hi, 40) if feature != "cand_rank" else np.arange(-0.5, 16.5)
        for label, colour, name in ((0, "#52514e", "negative (not reference)"),
                                    (1, "#1baf7a", "positive (reference arc)")):
            v = values[lab.label == label].dropna().clip(upper=hi)
            ax.hist(v, bins=bins, color=colour, alpha=0.55, label=name,
                    weights=np.full(len(v), 100 / len(v)))
        ax.set_title(feature, fontsize=10)
        ax.set_ylabel("Rows within class (%)")
        style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("train only: labelled candidate feature distributions by class "
                 "(clipped at the 99th percentile)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "features_by_label.png", dpi=150)
    plt.close(fig)


def representative_trip(obs):
    counts = obs.groupby("trip_id").size().sort_values(kind="mergesort")
    return counts.index[len(counts) // 2]


def plot_trajectories(raw_root, obs, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    chosen = {}
    trips = {(p, t): path for p, t, path in list_trips(raw_root, list(PARTITIONS))}
    for ax, partition in zip(axes, PARTITIONS):
        trip = representative_trip(obs[partition])
        path = trips[(partition, trip)]
        g = load_road_graph(path)
        route = load_route(path, len(g.arc_from))
        o = obs[partition][obs[partition].trip_id == trip]
        pad = 0.003
        box = (o.lon.min() - pad, o.lon.max() + pad, o.lat.min() - pad, o.lat.max() + pad)
        ll = g.node_lonlat
        a, b = ll[g.arc_from], ll[g.arc_to]
        inside = ((np.minimum(a[:, 0], b[:, 0]) <= box[1]) & (np.maximum(a[:, 0], b[:, 0]) >= box[0])
                  & (np.minimum(a[:, 1], b[:, 1]) <= box[3]) & (np.maximum(a[:, 1], b[:, 1]) >= box[2]))
        for i in np.flatnonzero(inside):
            ax.plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], color="#c9c8c2", linewidth=0.6, zorder=1)
        ref = np.asarray(route, dtype=int)
        for i in ref:
            ax.plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], color="#1baf7a", linewidth=4, zorder=2,
                    solid_capstyle="round")
        ax.scatter(o.lon, o.lat, s=1.5, color=COLOURS[partition], alpha=0.7, zorder=3, linewidths=0)
        ax.plot([], [], color="#c9c8c2", label="road network (directed arcs)")
        ax.plot([], [], color="#1baf7a", linewidth=4, label="reference route")
        ax.scatter([], [], s=12, color=COLOURS[partition], label="GPS observations")
        ax.set(xlim=box[:2], ylim=box[2:], xlabel="Longitude", ylabel="Latitude")
        ax.set_aspect(1 / np.cos(np.deg2rad(o.lat.mean())))
        ax.set_title(f"{partition} trip {trip}: {len(o):,} points, {int(inside.sum()):,} arcs in view",
                     fontsize=10)
        ax.legend(fontsize=8, loc="best", framealpha=0.95)
        chosen[partition] = {"trip_id": trip, "observations": int(len(o)),
                             "arcs_in_view": int(inside.sum()), "route_arcs": int(len(ref))}
    fig.suptitle("Representative trips (median observation count per partition)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "trajectory_network.png", dpi=150)
    plt.close(fig)
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-root", required=True)
    args = parser.parse_args(argv)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    cands = {p: pd.read_parquet(DATA / f"{p}_candidates.parquet",
                                columns=["trip_id", "label"] + FEATURES) for p in PARTITIONS}
    for c in cands.values():
        c["label"] = c.label.astype("float64")   # nullable label: NA (untrusted) becomes NaN
    obs = {p: pd.read_parquet(DATA / f"{p}_observations.parquet") for p in PARTITIONS}

    missing = missing_values(cands)
    missing.to_csv(EVIDENCE / "missing_values.csv", index=False)
    labels = label_summary(cands, obs)
    labels.to_csv(EVIDENCE / "label_and_sampling_summary.csv", index=False)
    feature_by_label(cands["train"]).to_csv(EVIDENCE / "train_features_by_label.csv", index=False)
    plot_missing(missing, FIGURES)
    plot_sampling(obs, FIGURES)
    plot_features_by_label(cands["train"], FIGURES)
    chosen = plot_trajectories(args.raw_root, obs, FIGURES)
    (EVIDENCE / "representative_trips.json").write_text(json.dumps(
        {"selection": "median observation count within each partition; train and student_test only",
         "trips": chosen}, indent=2), encoding="utf-8")
    print(labels.to_string(index=False))
    print(missing[missing.missing > 0].to_string(index=False))


if __name__ == "__main__":
    main()
