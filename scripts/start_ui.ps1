param([Parameter(Mandatory=$true)][string]$ApiHost, [int]$Port=8020)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$pythonExe='python'
foreach ($candidate in @('.venv/Scripts/python.exe','.venv-m1/Scripts/python.exe')) { if (Test-Path $candidate) { $pythonExe=(Resolve-Path $candidate).Path; break } }
& $pythonExe -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('streamlit') and importlib.util.find_spec('streamlit_folium') else 1)"
if ($LASTEXITCODE -ne 0) {
 & $pythonExe -m pip install -r requirements/ui.txt
 if ($LASTEXITCODE -ne 0) { throw 'UI dependency installation failed' }
}
$env:PATHSNAP_API_BASE_URL="http://${ApiHost}:$Port"
$env:PATHSNAP_REQUEST_TIMEOUT_SECONDS='300'
Invoke-RestMethod "$env:PATHSNAP_API_BASE_URL/health" | Format-List
& $pythonExe -m streamlit run ui/app.py --server.port 8520
