# Member 4 API, UI, and deployment foundation

## Architecture

The Streamlit dashboard (`ui/app.py`) is a client of the FastAPI service; it
does not load data, run a model, reconstruct routes, or calculate metrics.
FastAPI exposes `/health`, `/trips`, and `/predict`. The prediction service
validates trip membership and separates the HTTP contract from the future
inference adapter. Shared environment-variable path handling lives in
`src/common/paths.py`.

The default walking-skeleton mode returns explicit empty route/GPS/metric values
and a diagnostic identifying the stub. It exists to support UI and integration
work only; it is not a prediction. `PATHSNAP_MODE=real` checks for the agreed
model, preprocessing, model-card, and decoder-config artifacts and validates
their shared schema version at startup. Actual inference remains unavailable
until Members 1-3 supply their components and the adapter is implemented.

## API contract

- `GET /health`: API status, model name/version, schema version, and decoder
  config version.
- `GET /trips`: sorted trip-folder names discovered under
  `PATHSNAP_EVALUATION_DIR`.
- `POST /predict`: accepts a `trip_id`; returns observation count and GPS,
  HMM/classifier route, metrics, diagnostics, timings, and explicit mode.
- Unknown trips return 404, invalid payloads return 422, unavailable artifacts
  or inference integration return 503, and pipeline failures return 500.

## Data-leakage boundary

The current inference/service/UI code never opens route references. The
`src/api/evaluation.py` module is only an unimplemented server-side extension
point; reference-route evaluation must remain server-side. No route reference
is included in an API response. Development tests use only temporary synthetic
folder names and do not read evaluator data.

## Logging and UI

Requests and inference stages are written to console and to a rotating
`logs/pathsnap.log` file. Events include timestamp, client IP, trip ID, stage,
timings, and error. The dashboard reads `PATHSNAP_API_BASE_URL`, displays API
health and dynamic trip IDs, submits predictions to the API, and renders GPS
points and route polylines with Folium. Empty skeleton results are visibly
identified as such.

## Running and connecting devices

Install `requirements.txt`, launch the API with Uvicorn, and launch the
dashboard using Streamlit in another terminal. Defaults and Wi-Fi/hotspot
instructions are maintained in the repository README. For another laptop,
bind Uvicorn to `0.0.0.0`, allow only trusted local-network access to port 8000,
and set `PATHSNAP_API_BASE_URL` on the UI laptop to the API laptop's private
address.

## Pending integrations

The API adapter and data contract still need agreement with the other members:
Member 1's processed GPS input format; Member 2's serialized model,
preprocessing artifact, and model card; Member 3's decoder/configuration and
route output format. Predictions and server-side evaluation should be enabled
only after these contracts and artifact schema versions are agreed.
