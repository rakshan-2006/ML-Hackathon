# Quick two-laptop demo

This runs the selected untuned GB model with real HMM decoding and fixed untuned
decoder defaults. It is a working development demo, not a declaration that final
decoder tuning or physical LAN acceptance has been completed.

1. Connect both laptops to the same hotspot. This laptop's hotspot adapter is
   192.168.137.1; its campus Wi-Fi address at setup was 10.30.202.28. Addresses may
   change. Prefer the hotspot if campus Wi-Fi isolates clients.
2. Laptop 1, repository root:
   `powershell -ExecutionPolicy Bypass -File scripts/start_api.ps1`
   Supply `-RawRoot 'C:\path\to\Student folder\student_test'` if needed.
3. Laptop 2: `git pull origin main`, then
   `powershell -ExecutionPolicy Bypass -File scripts/start_ui.ps1 -ApiHost 192.168.137.1`
4. Open http://localhost:8520 on Laptop 2. Select a trip and predict.

Laptop 2 needs Python and `pip install -r requirements/ui.txt` (prefer its .venv).
Laptop 1 needs the full requirements. No model or trip data is needed on Laptop 2.
To test connectivity first, open http://192.168.137.1:8020/health on Laptop 2.
If blocked, allow Python through Windows Firewall, or in an administrator
PowerShell on Laptop 1:

```powershell
New-NetFirewallRule -DisplayName 'PathSnap LAN API' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8020 -Profile Any -RemoteAddress LocalSubnet
```

Keep the API terminal open on Laptop 1 to show incoming requests and timings.
The model ZIP downloads from the team's demo Release; no weights enter Git history.
No evaluator archive is opened by these commands. Final decoder tuning, full
final-bundle gating and a physical two-laptop acceptance test remain separate.
