
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.data.model_data import load_model_data
from src.models.knn import build_knn_model


DATA_PATH = r"..\pathsnap-m1\data\processed"

X, y, groups, keys = load_model_data(DATA_PATH, "train")

# Select at most 1,000 candidates from each of 10 trips.
selected_trips = groups.drop_duplicates().iloc[:10].tolist()

parts = []
for trip in selected_trips:
    indices = groups[groups == trip].index[:1000]
    parts.extend(indices)

X = X.loc[parts].reset_index(drop=True)
y = y.loc[parts].reset_index(drop=True)
groups = groups.loc[parts].reset_index(drop=True)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_idx, val_idx = next(splitter.split(X, y, groups))

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

model = build_knn_model()
model.fit(X_train, y_train)

positive_index = list(
    model.named_steps["classifier"].classes_
).index(1)

probabilities = model.predict_proba(X_val)[:, positive_index]
predictions = model.predict(X_val)

tn, fp, fn, tp = confusion_matrix(
    y_val, predictions, labels=[0, 1]
).ravel()

specificity = tn / (tn + fp) if (tn + fp) else 0

print("\n========== KNN EVALUATION ==========")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Training trips:", groups.iloc[train_idx].nunique())
print("Validation trips:", groups.iloc[val_idx].nunique())

print("\nAccuracy:", accuracy_score(y_val, predictions))
print("Precision:", precision_score(y_val, predictions, zero_division=0))
print("Recall:", recall_score(y_val, predictions, zero_division=0))
print("F1-score:", f1_score(y_val, predictions, zero_division=0))
print("Specificity:", specificity)
print("Balanced accuracy:", balanced_accuracy_score(y_val, predictions))
print("ROC-AUC:", roc_auc_score(y_val, probabilities))
print("PR-AUC:", average_precision_score(y_val, probabilities))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, predictions, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(y_val, predictions, zero_division=0))
