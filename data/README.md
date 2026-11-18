# Data directory

See [the dataset guide](../docs/data.md) for download, regeneration, schemas,
quality/label policies, and exact commands.

| Path | Purpose | Versioned? |
| --- | --- | --- |
| `releases/preprocessing.json` | Dataset asset URLs, sizes and SHA-256 checksums | Yes |
| `releases/gradient_boosting.json` | Baseline pipeline/probability Release index | Yes |
| `processed/` | Downloaded/generated Parquet and successful-build manifest | No |
| `raw/` | Optional local supplied source data | No |
| `evaluation/` | Evaluation-only trip folders, configured on the server | No |

Download from the repository root with `python -m src.data.download_dataset`.
Dependencies are in `requirements/`; frozen data settings are in `configs/`.
