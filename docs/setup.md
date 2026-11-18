# Setup and deployment

Use Python 3.11+ from the repository root. Python 3.11.9 was used for baseline
training. The default full environment includes data, models, API/UI and tests:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
```

Activation is optional; using the interpreter directly avoids PowerShell policy
issues. Smaller install sets are `requirements/data.txt`, `training.txt`,
`api.txt`, and `ui.txt`. `requirements/dev.txt` combines them with test tools.
Common pins are shared rather than repeated with conflicting values. The
historical M1 release snapshot lives at `requirements/reproducibility/m1.txt`.
Released joblib models use scikit-learn 1.7.2 and joblib 1.5.1.

## API and dashboard

```powershell
$env:PATHSNAP_MODE = 'stub'
.venv/Scripts/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. In another terminal:

```powershell
$env:PATHSNAP_API_BASE_URL = 'http://127.0.0.1:8000'
.venv/Scripts/python -m streamlit run ui/app.py
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `PATHSNAP_MODE` | `stub` | Explicit skeleton or real-artifact validation |
| `PATHSNAP_EVALUATION_DIR` | `data/evaluation` | Server-only evaluation trip folders |
| `PATHSNAP_ARTIFACTS_DIR` | `models` | Selected deployment bundle |
| `PATHSNAP_LOG_DIR` | `logs` | Request/stage logs |
| `PATHSNAP_API_BASE_URL` | `http://127.0.0.1:8000` | Dashboard API origin |

Relative server paths resolve from the repository root. Real mode requires
`best_model.joblib`, `preprocessing.joblib`, `model_card.json`, and
`decoder_config.json`, with the shared schema version. The selected bundle and
model-to-decoder inference adapter are still required; downloading an individual
baseline does not make real prediction available automatically. Stub mode reads
no trip contents and produces no fake predictions or metrics.

## Two laptops

1. Connect both to the same trusted Wi-Fi or hotspot.
2. Find the API laptop's private IPv4 address (`ipconfig`).
3. Bind Uvicorn with `--host 0.0.0.0 --port 8000`.
4. Allow inbound port 8000 on that trusted local network.
5. On the dashboard laptop, set `PATHSNAP_API_BASE_URL` to the API laptop's IP.
6. Verify `/health`, `/trips`, and the explicit stub response before integration.

Keep evaluator data and reference routes server-side. Never open evaluator data
during model development or return its reference geometry to the dashboard.
