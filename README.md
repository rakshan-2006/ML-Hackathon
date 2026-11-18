# PathSnap

GPS candidate scoring and route reconstruction for the PES University ML hackathon.
Preprocessing and four model baselines (Logistic Regression, Decision Tree,
Gradient Boosting and Random Forest) are available. The API/UI currently runs as a walking skeleton;
the production decoder and model-to-API adapter are still required.

KNN is also implemented through the shared factory/trainer. Its full five-fold
evaluation and comparable accuracy results are pending; see [training](docs/training.md).

## Start here

Use Python 3.11 or later, from the repository root:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m src.data.download_dataset
.venv/Scripts/python -m src.data.validate_dataset
.venv/Scripts/python -m pytest -q
```

`requirements.txt` installs the complete team environment. For a smaller training
environment use `requirements/training.txt`; see [setup and deployment](docs/setup.md).
Cloning does not download dataset/model assets.

## Repository map

```text
configs/        Frozen preprocessing config and the one shared CV fold assignment
data/           Local raw/processed/evaluation data; Release indexes in releases/
docs/           Setup, data, training, architecture, and member reports
requirements/   Shared dependency pins grouped by workflow
src/            Python code: data, models, training, evaluation, API, common
tests/          Synthetic fixtures and automated checks
ui/             Streamlit dashboard
results/        Committed measured metrics, evidence and report figures
artifacts/      Downloaded/fitted per-model pipelines and scores (local, ignored)
models/         Selected runtime model/decoder bundle for the API (local, ignored)
```

Keep code under `src/`, configuration under `configs/`, and measured outputs under
`results/`. Never commit raw/evaluator data, environments, or large fitted artifacts.

## Train and compare

```powershell
.venv/Scripts/python -m src.training.train_models --model logistic_regression
.venv/Scripts/python -m src.training.train_models --model decision_tree
.venv/Scripts/python -m src.training.train_gradient_boosting --threads 4
.venv/Scripts/python -m src.training.train_models --model random_forest
.venv/Scripts/python -m src.evaluation.compare_models logistic_regression decision_tree gradient_boosting random_forest
```

All models use [configs/cv_folds.csv](configs/cv_folds.csv), the same 80 training
trips, 13 ordered physical features, nullable-label policy, and held-out 10-trip
student-test partition. [Training and artifact loading](docs/training.md) explains
commands, model contracts, metrics, and existing Release downloads.

## Run the demo foundation

```powershell
.venv/Scripts/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
# In another terminal:
.venv/Scripts/python -m streamlit run ui/app.py
```

The default mode is explicitly `stub`; it returns no fabricated predictions.
[Setup/deployment](docs/setup.md) covers environment variables and two laptops.

## Documentation and results

- [Dataset download, regeneration, schema and leakage rules](docs/data.md)
- [Model training, comparison and saved pipelines](docs/training.md)
- [Architecture and layout migration](docs/architecture.md)
- [Refactor validation](docs/validation.md)
- [Member handoffs and reports](docs/reports/)
- [Comparison of the three implemented models](docs/reports/THREE_MODEL_COMPARISON.md)
- [Gradient Boosting graph gallery](results/models/gradient_boosting/figures/)
- [Random Forest training and review notes](docs/reports/m2_random_forest.md)
- [Measured model results](results/models/)

Historical manifests/cards retain the paths and hashes from their original runs.
The layout migration moves configuration and documentation without changing
dataset values, fold membership, trained weights, or recorded metrics.
