
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_knn_model(
    n_neighbors=5,
    weights="distance",
    metric="minkowski",
    p=2,
):
    """Build a KNN pipeline with fold-fitted imputation and scaling."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric,
                    p=p,
                    n_jobs=-1,
                ),
            ),
        ]
    )
