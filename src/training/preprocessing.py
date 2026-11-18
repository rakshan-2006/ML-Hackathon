
from pathlib import Path

import pandas as pd

FEATURES = [
    "dist_m",
    "cand_rank",
    "dist_minus_min",
    "heading_diff_deg",
    "speed_mps",
    "arc_len_m",
    "proj_frac",
    "n_candidates",
    "is_oneway",
    "source_degree",
    "target_degree",
    "prev_dist_m",
    "next_dist_m",
]

TARGET = "label"
GROUP = "trip_id"

DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "pathsnap-m1"
    / "data"
    / "processed"
)


def load_training_data():
    """Load labelled training candidates and preserve trip groups."""
    path = DATA_PATH / "train_candidates.parquet"
    df = pd.read_parquet(path)

    required_columns = FEATURES + [TARGET, GROUP]
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Null labels represent ambiguous or unaligned candidates.
    df = df.dropna(subset=[TARGET]).copy()

    X = df[FEATURES].copy()
    y = df[TARGET].astype("int8")
    groups = df[GROUP].copy()

    # Convert the boolean feature into a numeric feature.
    X["is_oneway"] = X["is_oneway"].astype("int8")

    return X, y, groups


if __name__ == "__main__":
    X, y, groups = load_training_data()

    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)
    print("Unique trips:", groups.nunique())
    print("\nTarget distribution:")
    print(y.value_counts())
    print("\nMissing feature values:")
    print(X.isna().sum())
