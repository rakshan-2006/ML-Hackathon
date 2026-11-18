# PathSnap

PathSnap is a PES University hackathon project for distributed map-matching and
route reconstruction. This repository currently contains the **Member 4 API,
UI, integration, and deployment foundation**. Inference is in walking-skeleton
mode until the data and model components are supplied by the other members.

## Project structure

```text
data/
  raw/                 # Local-only source data
  processed/           # Processed inputs and trip folders
  evaluation/          # Local-only evaluator trips; never commit this data
src/
  common/              # Shared paths and configuration
  inference/           # Reserved integration point; no inference logic yet
  api/                 # FastAPI contract, service, artifact and logging interfaces
ui/                    # Streamlit/Folium evaluator dashboard
models/                # Local model artifacts
results/               # Local generated outputs
experiments/            # Experiment outputs
tests/
  fixtures/             # Safe, synthetic test fixtures only
report/                 # Member 4 implementation notes
```

## Python environment and dependencies

Python 3.10 or later is required. The initial dependency pins were selected for
compatibility with the available Python 3.13.15 environment.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run commands using
`.\.venv\Scripts\python.exe -m pip ...` directly.

## Configure and run the API

The API defaults to `PATHSNAP_MODE=stub`. Stub mode does not load model artifacts,
read trip contents, or produce fabricated predictions or metrics.

```powershell
$env:PATHSNAP_MODE = "stub"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API contract. Configuration
uses environment variables; relative paths resolve from the repository root:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PATHSNAP_MODE` | `stub` | `stub` or `real`; real mode validates artifacts at startup |
| `PATHSNAP_EVALUATION_DIR` | `data/evaluation` | Directory containing one folder per trip |
| `PATHSNAP_ARTIFACTS_DIR` | `models` | Directory containing required model/decoder artifacts |
| `PATHSNAP_LOG_DIR` | `logs` | Rotating application log output |

`GET /trips` returns names of immediate child directories in the configured
evaluation directory. An absent directory produces an empty list. A prediction
for an ID not discovered there returns 404.

Real mode requires `best_model.joblib`, `preprocessing.joblib`,
`model_card.json`, and `decoder_config.json`; both JSON files must declare a
matching `schema_version`. The artifact-loading interface is ready, but the
Member 1/2/3 inference adapter is not implemented. Consequently, the API reports
an explicit 503 for predictions in real mode until that integration is supplied.

## Run the Streamlit evaluator dashboard

Start the API, then in another terminal:

```powershell
$env:PATHSNAP_API_BASE_URL = "http://127.0.0.1:8000"
python -m streamlit run ui/app.py
```

Set `PATHSNAP_API_BASE_URL` to the reachable API origin, without an endpoint
suffix. The UI gets health and trip IDs from the API and sends prediction
requests to `/predict`; it does not implement inference itself.

### Connect two laptops over Wi-Fi or hotspot

1. Connect both laptops to the same trusted Wi-Fi/hotspot and find the API
   laptop's private IPv4 address (for example, with `ipconfig` on Windows).
2. Start the API bound to that interface:
   `python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000`.
3. Allow inbound TCP port 8000 in the API laptop's firewall, limited to the
   trusted local network.
4. On the UI laptop, set
   `$env:PATHSNAP_API_BASE_URL = "http://<API-LAPTOP-IP>:8000"` and start
   Streamlit. Keep both devices on the trusted network; do not expose the
   development API publicly.

## Test

```powershell
python -m pytest
```

Tests use temporary directories and synthetic trip IDs. They do not open or use
evaluator data.

## Integration still awaited

- **Member 1:** input data-processing/preprocessing outputs and the documented
  trip input format.
- **Member 2:** trained `best_model.joblib`, `preprocessing.joblib`, and a
  `model_card.json` declaring `schema_version`.
- **Member 3:** decoder implementation/configuration and `decoder_config.json`
  declaring the same `schema_version`, plus the inference integration contract.

No Member 1, 2, or 3 logic is implemented here. Inference code must not read
reference `.route` files or return reference routes to the evaluator UI. When
evaluation is implemented, only the server-side evaluation module may read
reference routes.
