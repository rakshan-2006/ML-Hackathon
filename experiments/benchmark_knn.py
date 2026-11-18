
import time

from src.data.model_data import load_model_data
from src.models.knn import build_knn_model


DATA_PATH = r"..\pathsnap-m1\data\processed"

X, y, groups, keys = load_model_data(DATA_PATH, "train")

# Benchmark using a clearly identified 10,000-row subset.
X_sample = X.iloc[:10000]
y_sample = y.iloc[:10000]

model = build_knn_model(n_neighbors=5)

start = time.perf_counter()
model.fit(X_sample, y_sample)
fit_time = time.perf_counter() - start

start = time.perf_counter()
probabilities = model.predict_proba(X_sample.iloc[:1000])
predict_time = time.perf_counter() - start

print("Benchmark rows:", len(X_sample))
print("Prediction rows:", 1000)
print("Fit time (seconds):", round(fit_time, 2))
print("Prediction time (seconds):", round(predict_time, 2))
print("Probability shape:", probabilities.shape)
