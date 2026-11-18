# Architecture and layout migration

## Ownership by purpose

| Directory | Put here |
| --- | --- |
| `configs/` | Frozen preprocessing settings and shared trip folds |
| `requirements/` | One composable set of dependency pins per workflow |
| `src/data/` | Parsers, geometry, candidates, derived labels, feature tables |
| `src/models/` | Classifier definitions and factory |
| `src/training/` | Fold-aware model fitting and export runners |
| `src/evaluation/` | Metrics, comparison, figures |
| `src/api/`, `ui/` | Backend service and dashboard |
| `src/common/` | Shared schema and path settings |
| `data/releases/` | Published dataset/model download indexes |
| `data/{raw,processed,evaluation}/` | Local-only datasets |
| `results/` | Small committed measured evidence and figures |
| `artifacts/<model>/` | Local large pipelines/probability exports |
| `models/` | Selected deployment artifacts for API configuration |
| `docs/reports/` | Member handoffs and report contributions |
| `tests/` | Synthetic fixtures and focused regression checks |

The data flow is raw trip -> projected observations and directed graph -> indexed
candidates -> physical features -> classifier probability. Offline reference
alignment produces supervision separately. The production HMM/decoder and API
inference adapter remain integration work; the current API is an explicit stub.

## Migration from the initial team layout

| Previous path | Current path |
| --- | --- |
| `experiments/data_config.yaml` | `configs/data.yaml` |
| `results/models/shared/train_folds_5.csv` | `configs/cv_folds.csv` (same bytes) |
| `experiments/cv_folds.json` | Duplicate removed; GB reads the same canonical CSV |
| `data/dataset_release.json` | `data/releases/preprocessing.json` |
| `data/gradient_boosting_release.json` | `data/releases/gradient_boosting.json` |
| `data/requirements-*.txt` | `requirements/` workflow files |
| `report/*.md` | `docs/reports/*.md` |
| Long data README | `docs/data.md`, with `data/README.md` as a directory guide |
| New LR/DT pipelines/scores inside results | `artifacts/<model>/` |

Existing Python module names and CLI module entry points stay stable, including
`src.training.preprocessing.to_float64`, which downloaded joblib pipelines import.
No trained weights, recorded metrics, raw data, or published Release assets are
changed by this migration. Explicit old configuration/dependency paths in shell
scripts must be updated using the table; current CLI defaults are already updated.
The GB runner still accepts an explicitly supplied downloaded legacy JSON fold map.

Historical JSON manifests/model cards/figure manifests are audit evidence and keep
their original source paths and hashes. This is intentional: rewriting them would
misrepresent the original run. Source documentation and current commands use the
new paths. The original M1 dependency snapshot remains archived separately.
