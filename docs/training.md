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
and scaling; DT and RF fit median imputation/missing indicators; histogram GB handles
NaNs natively without a fitted imputer or scaler. Learned processing belongs
inside each CV training fold. No student-test fitting is permitted.

## Commands

```powershell
python -m src.training.train_models --model logistic_regression
python -m src.training.train_models --model decision_tree
python -m src.training.train_gradient_boosting --threads 4
python -m src.training.train_models --model random_forest
python -m src.evaluation.compare_models logistic_regression decision_tree gradient_boosting random_forest
```

The shared runner also accepts `--model gradient_boosting` and `--smoke-only`.
The dedicated GB runner remains available to reproduce its baseline workflow and
probability export layout. Both read the same canonical fold assignment.
Random Forest uses the shared runner and factory; its old
`src.models.train_random_forest` command delegates to that runner with five folds.
See [the RF report](reports/m2_random_forest.md) for original-run provenance and
the correction to its original conditional Top-1 denominator. KNN implements the
same factory contract with median imputation, missing indicators, scaling and
distance-weighted five-neighbour classification. Its full evaluation is pending;
do not compare the bounded integration check with measured baseline accuracies.

```powershell
python -m src.training.train_models --model knn --smoke-only
python -m src.training.train_models --model knn
```

The first command uses 20,000 training rows and 5,000 rows from held-out trips to
check wiring and latency. Only the second performs canonical five-fold evaluation,
final fitting, model export and student-test evaluation. It may take substantially
longer because KNN queries the stored training examples. KD-tree search and four
workers avoid the default brute-force query-by-training distance matrix. Legacy
`experiments/evaluate_knn.py` and `save_knn.py` delegate to the full shared runner;
`benchmark_knn.py` delegates to the smoke check. These scripts are import-safe and
support direct or module execution. Outputs use the canonical artifacts/results
directories; no sibling checkout or root-level model file is required.

Review checks fitted all 1,070,140 labelled rows from the 80 training trips, queried
128 held-out candidate rows, and verified compressed serialization/reload. This
is integration evidence, not an accuracy estimate or full five-fold result. See
[KNN integration evidence](../results/validation/knn_integration.json).

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
memory. Original evidence files are not rewritten. Comparisons accept LR, DT, GB
and RF results and preserve the distinction between fold means and pooled metrics.

```powershell
python -m src.evaluation.plot_gradient_boosting
```

This recreates the GB PNG/SVG graphs after downloading the data and model assets;
it does not train the model. See [the GB report](reports/gradient_boosting.md) and
[graph gallery](../results/models/gradient_boosting/figures/).

### Random Forest

The fourth baseline's fitted pipeline, embedded-feature model card and probability
exports are published separately from the historical three-fold metadata:

```powershell
python -m src.data.download_dataset --metadata results/models/random_forest/artifacts.json --output-root artifacts/random_forest
```

This verifies sizes and SHA-256 hashes. Load with
`RandomForestPredictor('artifacts/random_forest')` from `src.models.random_forest`,
or load `pipeline.joblib` through joblib. Fitted preprocessing is already inside.
See [the RF report](reports/m2_random_forest.md) for the reviewed measurements.

## Shared loading and exploratory splits

`src.training.preprocessing.load_training_data(processed_root=None)` delegates to
the schema-validated M1 loader and returns all trusted training rows with their
trip groups. Features still contain NaNs; learned preprocessing stays inside CV.
The default root is this repository's data/processed, not a sibling checkout.

`src.training.cross_validation.get_cv_splits(X, y, groups)` loads the canonical
five-fold membership. `get_stratified_cv_splits` is an explicitly separate,
deterministic experimental helper; it never overwrites configs/cv_folds.csv and
must not be substituted into comparisons with the published baselines.
