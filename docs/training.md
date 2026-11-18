# Model training and artifacts

Install `requirements/training.txt` in Python 3.11, then download and validate the
data with `python -m src.data.download_dataset` and
`python -m src.data.validate_dataset`. For all repository tests/API checks, install
the full root `requirements.txt` instead.

## Shared model contract

`src/common/schema.py` defines the ordered 13 features and schema version.
`configs/cv_folds.csv` is the only checked-in fold assignment; all 80 trip-to-fold
mappings are identical to the baseline runs. Saved historical cards keep the
original fold-file paths/hashes as provenance; moving the unchanged CSV changes
its location, not its bytes or membership.

`src.models.model_factory.get_model(name, params=None, seed=2026)` returns a full
pipeline for each implemented model. LR fits median imputation/missing indicators
and scaling; DT fits median imputation/missing indicators; histogram GB handles
NaNs natively without a fitted imputer or scaler. Learned processing belongs
inside each CV training fold. No student-test fitting is permitted.

## Commands

```powershell
python -m src.training.train_models --model logistic_regression
python -m src.training.train_models --model decision_tree
python -m src.training.train_gradient_boosting --threads 4
python -m src.evaluation.compare_models logistic_regression decision_tree gradient_boosting
```

The shared runner also accepts `--model gradient_boosting` and `--smoke-only`.
The dedicated GB runner remains available to reproduce its baseline workflow and
probability export layout. Both read the same canonical fold assignment. RF and
KNN modules have not yet been delivered to main; they must implement the shared
module metadata and `build_estimator(params, seed)` interface.

Small measured metrics/cards/figures go to `results/models/<name>/`. New fitted
pipelines and probability exports go to ignored `artifacts/<name>/`. The `models/`
directory is reserved for the chosen API deployment bundle, not every experiment.
Do not overwrite an existing baseline just to reorganize or compare its results.

## Download the trained baselines

```powershell
python -m src.data.download_dataset --metadata results/models/logistic_regression/artifacts.json --output-root artifacts/logistic_regression
python -m src.data.download_dataset --metadata results/models/decision_tree/artifacts.json --output-root artifacts/decision_tree
python -m src.data.download_dataset --metadata data/releases/gradient_boosting.json --output-root artifacts/gradient_boosting
```

These downloads verify sizes/SHA-256. LR/DT use `pipeline.joblib`; the GB baseline
uses `gradient_boosting_pipeline.joblib`. The entire processing pipeline is inside
the saved file. Use the trusted project artifacts with matching library versions;
do not apply an extra fitted preprocessor.

Candidate metrics exclude null labels and use threshold 0.5. Top-1 ranks all
candidates by class-1 probability, breaking ties by `cand_rank`. Trusted valid
observations with reference misses or no candidates count as failures; uncertain
reference labels are excluded from that denominator. Derived reference agreement
is not independently verified route accuracy.

## Compare and plot

The comparison reader normalizes the two historical CSV naming conventions in
memory. Original evidence files are not rewritten. Comparisons accept LR, DT and
GB results and preserve the distinction between fold means and pooled metrics.

```powershell
python -m src.evaluation.plot_gradient_boosting
```

This recreates the GB PNG/SVG graphs after downloading the data and model assets;
it does not train the model. See [the GB report](reports/gradient_boosting.md) and
[graph gallery](../results/models/gradient_boosting/figures/).
