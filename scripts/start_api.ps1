param([string]$RawRoot = '', [int]$Port = 8020)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not $RawRoot) { $RawRoot = Join-Path $projectRoot '../Student folder/student_test' }
$pythonExe = 'python'
foreach ($candidate in @('.venv-m1/Scripts/python.exe','.venv/Scripts/python.exe')) { if (Test-Path $candidate) { $pythonExe = (Resolve-Path $candidate).Path; break } }
if (-not (Test-Path 'models/demo/best_model.joblib')) {
 New-Item -ItemType Directory -Force 'models/demo' | Out-Null
 Invoke-WebRequest 'https://github.com/rakshan-2006/ML-Hackathon/releases/download/demo-gb-hmm-2026-11-18/pathsnap-demo-model.zip' -OutFile 'models/demo-bundle.zip'
 Expand-Archive 'models/demo-bundle.zip' -DestinationPath 'models/demo' -Force
}
if (-not (Test-Path $RawRoot)) { throw "Trip folder not found: $RawRoot. Supply -RawRoot with your student_test folder." }
$env:OMP_NUM_THREADS='2'
$env:PATHSNAP_MODE='real'
$env:PATHSNAP_REQUIRE_FINAL='0'
$env:PATHSNAP_ARTIFACTS_DIR=(Join-Path $projectRoot 'models/demo')
$env:PATHSNAP_EVALUATION_DIR=(Resolve-Path $RawRoot).Path
$env:PATHSNAP_DATA_CONFIG=(Join-Path $projectRoot 'configs/data.yaml')
$env:PATHSNAP_GRAPH_CACHE_SIZE='1'
Write-Host "API on port $Port. Connect Laptop 2 to the same Wi-Fi/hotspot. Keep this terminal open."
& $pythonExe -m uvicorn src.api.main:app --host 0.0.0.0 --port $Port
