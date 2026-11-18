
from sklearn.model_selection import StratifiedGroupKFold


def get_cv_splits(X, y, groups, n_splits=5, random_state=42):
    """Create stratified folds while keeping each trip in one fold."""

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits = list(splitter.split(X, y, groups))

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])

        if train_groups.intersection(val_groups):
            raise ValueError(f"Trip leakage detected in fold {fold}")

        print(f"\nFold {fold}")
        print("Training trips:", len(train_groups))
        print("Validation trips:", len(val_groups))
        print("Training samples:", len(train_idx))
        print("Validation samples:", len(val_idx))

    return splits
