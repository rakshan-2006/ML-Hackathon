
import joblib

from src.data.model_data import load_model_data
from src.models.knn import build_knn_model

DATA_PATH = r"..\pathsnap-m1\data\processed"
MODEL_PATH = "knn_model.joblib"

X, y, groups, keys = load_model_data(DATA_PATH, "train")

model = build_knn_model()
model.fit(X, y)

joblib.dump(model, MODEL_PATH)

print(f"KNN model saved to {MODEL_PATH}")
