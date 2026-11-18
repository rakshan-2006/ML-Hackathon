"""Shared five-fold trip-grouped CV folds for the 80 training trips.

Folds are assigned once with sklearn GroupKFold (deterministic, balances labelled
rows per fold) and saved to FOLDS_PATH; every model must load that file so all
comparisons use identical folds.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from src.common.paths import PROJECT_ROOT

N_SPLITS = 5
FOLDS_PATH = PROJECT_ROOT / "configs" / "cv_folds.csv"


def assign_folds(groups, n_splits=N_SPLITS):
    """trip_id -> fold table from the labelled training rows' trip groups."""
    groups = pd.Series(groups).astype(str).reset_index(drop=True)
    fold_of = {}
    for fold, (_, val) in enumerate(GroupKFold(n_splits=n_splits).split(groups, groups=groups)):
        for trip in groups.iloc[val].unique():
            fold_of[trip] = fold
    return (pd.DataFrame({"trip_id": list(fold_of), "fold": list(fold_of.values())})
            .sort_values("trip_id").reset_index(drop=True))


def load_folds(path=FOLDS_PATH):
    path = Path(path)
    if path.suffix == '.json':
        # Explicitly supplied original GB Release fold maps remain readable.
        mapping = json.loads(path.read_text(encoding='utf8'))['trip_to_fold']
        folds = pd.DataFrame({'trip_id': list(mapping), 'fold': list(mapping.values())})
        folds = folds.astype({'trip_id': 'string', 'fold': 'int64'}).sort_values('trip_id')
    else:
        folds = pd.read_csv(path, dtype={"trip_id": "string", "fold": "int64"})
    if folds.trip_id.duplicated().any():
        raise ValueError("a trip appears in more than one fold")
    if set(folds.fold) != set(range(N_SPLITS)):
        raise ValueError("fold IDs must cover exactly 0 through 4")
    return folds


def folds_sha256(path=FOLDS_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fold_of_rows(trip_ids, folds):
    """Fold index for each row; every trip must be in the saved fold table."""
    mapping = dict(zip(folds.trip_id.astype(str), folds.fold))
    trips = pd.Series(trip_ids).astype(str)
    unknown = set(trips) - set(mapping)
    if unknown:
        raise ValueError(f"trips missing from fold table: {sorted(unknown)[:5]}")
    return trips.map(mapping).to_numpy()


def main():
    from src.data.model_data import load_model_data

    _, _, groups, _ = load_model_data("data/processed", "train")
    folds = assign_folds(groups)
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLDS_PATH, index=False)
    print(folds.fold.value_counts().sort_index().to_string(), f"\nwrote {FOLDS_PATH}")


if __name__ == "__main__":
    main()
