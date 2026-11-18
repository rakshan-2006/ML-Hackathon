# Person B: real application and remaining acceptance gates

## Current state

Branch `completion/decoder-demo-and-submission` integrates main through `55e90cb`
(RF, shared training and KNN source). The original checkout and teammate branches
are preserved. The reviewed route-reconstruction branch supplied the HMM kernel;
the adapter, correctness fixes, evaluation isolation and real runtime are here.

The working application uses the verified existing histogram GB release
`hgb-baseline-ac522dfa` and **fixed untuned** decoder defaults. This is preparation,
not final-model selection or official evaluation. No evaluator archive contents,
names, geography or labels have been inspected. Do not unlock that archive.

Measured evidence: [HTTP results](../results/rehearsal/student_test_api.csv),
[summary](../results/rehearsal/summary.json), [server records](../results/rehearsal/server_inference.jsonl),
[extreme-trip profile](../results/rehearsal/profile/profile.json),
[inventory](../results/rehearsal/profile/permitted_inventory.csv),
[UI](evidence/local_ui.jpg), [map](evidence/development_map.jpg).
All ten HTTP outputs matched offline geometry **and metrics**. Weighted trusted
derived-alignment Top-1: classifier 83.5687%, HMM 91.4515%. Trip 00000007 worsens
from 91.8161% to 88.1528%; smoothing is not uniformly beneficial. Scores use
derived reference alignment, not independently verified physical ground truth.

## Clean-clone preparation on Laptop 1

Use Python 3.11.9 for the released joblib assets. In the repository root:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m src.inference.development_baseline
```

The last command downloads and checksum-verifies this team's public GB release,
then converts its pipeline to a distinctly versioned estimator/identity-transform
bundle in ignored `models/development`. It does not train or change weights.
Do not load untrusted joblib assets. Raw permitted data must be available locally
as `Student folder/train` and `Student folder/student_test`; see [data setup](data.md).

```powershell
$env:OMP_NUM_THREADS = '4'
$env:PATHSNAP_MODE = 'real'
$env:PATHSNAP_ARTIFACTS_DIR = 'models/development'
$env:PATHSNAP_EVALUATION_DIR = 'C:\Acad\Sem 5\ML Hackathon\Student folder\student_test'
$env:PATHSNAP_GRAPH_CACHE_SIZE = '1'
.venv/Scripts/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8013
```

Run a second terminal from the same repository:

```powershell
$env:PATHSNAP_API_BASE_URL = 'http://127.0.0.1:8013'
$env:PATHSNAP_REQUEST_TIMEOUT_SECONDS = '300'
.venv/Scripts/python -m streamlit run ui/app.py --server.port 8513
```

Open the UI at `http://localhost:8513` or Swagger at
`http://127.0.0.1:8013/docs`. `/trips` remains a JSON list. `/predict` takes
`{"trip_id":"00000052"}`. Trip IDs are discovered from coherent folders, not
hardcoded. The UI receives GPS, predicted geometry, counts and aggregate reference
metrics; reference geometry/labels are never returned. Missing references leave
metrics unavailable without blocking prediction. Real errors never fall back to
stub results. The explicit stub mode remains for compatibility.

Startup loads artifacts once and verifies schema/features/layout/version/hashes.
After changing artifacts, restart the service. Requests are serialized to bound
peak memory; the graph cache holds one graph by default. This is a small-demo
service, without authentication or multi-user capacity guarantees. Keep it on a
trusted private LAN during the actual demonstration. UI requests run from the
Streamlit server to FastAPI; browser CORS is not needed.

## Reproduce the permitted rehearsal, maps and profile

Keep the real development API running, then:

```powershell
.venv/Scripts/python -m src.api.rehearse --api-url http://127.0.0.1:8013 --raw-root 'C:\Acad\Sem 5\ML Hackathon\Student folder\student_test' --models-dir models/development --create-maps
.venv/Scripts/python -m src.api.profile_development --raw-root 'C:\Acad\Sem 5\ML Hackathon\Student folder' --models-dir models/development
.venv/Scripts/python -m src.api.profile_development --raw-root 'C:\Acad\Sem 5\ML Hackathon\Student folder' --models-dir models/development --sequence-only
```

Rehearsal compares every server response against an independently called offline
pipeline and evaluation. Maps select shortest/median/longest tracks **before**
looking at accuracy. HTMLs are under `maps/development`; Folium/Leaflet and the OSM
basemap need Internet/CDN access. Samples affect display only. Repeated generation
can change Folium element IDs without changing predictions. Profile discovers
extremes only within train/student_test, starts a fresh process for each, measures
cold/warm graph behavior and the OS peak working set. It does not flush the OS
file cache or claim absolute cold-machine latency. Optional alignment/evaluation
and HTTP are excluded from this separate profile; HTTP CSV includes evaluation.

Observed slowest student-test server inference was 12.10 seconds under concurrent
local checks; the 300-second UI timeout is a conservative configurable ceiling,
not a guaranteed SLA. Longest train trip was 19,313 points (8.89s cold/6.11s warm,
about 303 MiB peak); largest network 307,392 arcs (about 333 MiB peak). Final model
latency and actual LAN behavior must be measured again.

## Person A handoff, then decoder selection

The [shared contract](contracts/person_b.md) lists required scorer semantics.
Wait for A's five measured baselines, actual Bayesian trials, selected split
model/preprocessor/card, final `src/inference/predictor.py`, final-parameter
train-OOF scores and student-test exports. KNN source on main is not proof of a
completed KNN experiment. No final decoder selection was run in this task.

`src.hmm.tune` prepares a deterministic 12-configuration selection: beta 5/10/20/50
metres crossed with emission weight 0.5/1/2; other declared defaults fixed.
Objective is observation-weighted trusted-reference Top-1 on all canonical train
OOF trips; exact ties select earliest fixed trial index. Student-test only confirms
the selection. This is decoder grid selection; A separately owns **Bayesian model
optimisation**. A's OOF manifest contract and checksums must be satisfied first.

```powershell
.venv/Scripts/python -m src.hmm.tune --raw-root '<allowed-raw>/train' --processed data/processed --scores '<A-final-oof.parquet>' --manifest '<A-oof-manifest.json>' --model-card '<A-final-models>/model_card.json' --output results/hmm/final-oof-run
```

Keep trials immutable. If a run fails, retain the partial evidence, fix the cause
and start a new output directory. Do not relabel provisional defaults as tuned.
Copy the selected decoder config alongside A's split artifacts in a local staging
directory, restart the real API against that directory, and repeat all ten
student-test HTTP/offline checks with a new output root. The resulting summary
must say `final_student_test_confirmed` and bind final model/decoder versions.

```powershell
.venv/Scripts/python -m src.api.deployment --source '<A-final-models>' --decoder results/hmm/final-oof-run/decoder_config.json --confirmation '<final-rehearsal>/summary.json' --destination models/final-bundle
```

This prepares hashes for model/preprocessor/card/decoder, effective data config,
dependency versions, source revision and tuning/confirmation provenance. It
refuses development or incomplete bundles. Before official use set
`PATHSNAP_REQUIRE_FINAL=1`, `PATHSNAP_ARTIFACTS_DIR=models/final-bundle`, and
`PATHSNAP_DATA_CONFIG=models/final-bundle/data.yaml`. Publish final artifacts as a
checksummed GitHub Release after review, keeping large weights out of Git history.
Release publishing/download validation remains pending until these files exist.
Commit reviewed source/results before freezing the bundle; the builder rejects
tracked local changes so its source revision has a concrete meaning.
Copy the generated hashes unchanged into a release index, download into a new
directory, compare each SHA-256 and perform a fresh final-required startup.

## Actual two-laptop rehearsal: pending human hardware step

1. Connect two physical laptops to the same trusted private Wi-Fi/hotspot. Keep
   the ten allowed student-test trips and all references on Laptop 1 only.
2. On Laptop 1 find its private IPv4 (`ipconfig`). Run the frozen API with
   `--host 0.0.0.0 --port 8000`. If blocked, allow only that API port on the trusted
   private network using the machine's normal firewall approval. Do not disable
   the firewall or expose the endpoint publicly.
3. On Laptop 2 clone this branch/reviewed merge. Install only `requirements/ui.txt`;
   it needs no raw data, model weights or local inference modules. Set
   `PATHSNAP_API_BASE_URL=http://<LAPTOP_1_PRIVATE_IP>:8000` and timeout 300, then
   run `python -m streamlit run ui/app.py`.
4. On Laptop 2 check server `/health`, all ten discovered IDs and every prediction.
   Confirm final versions, routes, metrics and error/timeout handling. Match UI
   request IDs to Laptop 1's receiving-console inference logs.
5. Capture actual screenshots of Laptop 2's UI and Laptop 1's console responses.
   Record real hostnames/addresses privately, versions, timings and both operators.
   Repeat connectivity over the fallback hotspot. Do not call loopback evidence a
   successful two-laptop rehearsal. The checked-in screenshots are local only.

## Future official evaluation: still locked

Only after explicit official-stage authorization, freeze code/artifact/config
hashes and use `python -m src.api.import_evaluation --archive <authorized-zip>
--destination data/evaluation/<new-session> --evaluation-authorized` on Laptop 1.
That flag acknowledges an actual authorization; it is not a development bypass.
The command has been tested exclusively on synthetic ZIPs and rejects unsafe paths,
links, duplicate IDs, oversized content, inconsistent/orphan files and overwrites.
It validates before atomic readiness and records count/checksum locally. Compare
the discovered count to ten; do not invent missing inputs. Point the frozen API
to the imported root, run all reserved trips and capture receiving-console/UI
evidence. Never tune or edit predictions after viewing official results. No real
archive import or inference occurred during this preparation task.

## Report and integrity

The working evidence PDF is a draft contribution, not the final submission. One
consolidated reviewed PDF must integrate A's actual EDA/model/Bayesian sections,
B's final decoder results/maps and physical deployment evidence. No model result,
human approval or hardware success may be invented. Codex assisted implementation,
test execution, evidence generation and draft prose in this task; humans must
review/understand the code and measured findings and use any organizer-required
disclosure format. See [acceptance matrix](reports/person_b_acceptance.md).

To regenerate the draft charts and PDF, install `requirements/reporting.txt` and
run `python -m src.visualization.build_development_report`. The editable builder
reads committed measurements and actual map/UI captures. Its output is
`output/pdf/pathsnap-development-evidence.pdf`; render the pages and visually
check them before sharing. It deliberately cannot masquerade as the final report.
