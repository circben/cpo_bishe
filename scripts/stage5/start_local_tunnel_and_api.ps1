param(
    [string]$RemoteHost = "root@connect.westb.seetacloud.com",
    [string]$RemotePort = "50734",
    [string]$TunnelPort = "8088",
    [string]$ApiBase = "http://127.0.0.1:8088",
    [string]$ApiScript = "scripts/stage5/web/run_stage5_api_server.py"
)

$env:REMOTE_API_BASE = $ApiBase

Write-Host "[INFO] REMOTE_API_BASE=$env:REMOTE_API_BASE"
Write-Host "[INFO] Starting SSH tunnel: ${RemoteHost}:${RemotePort} -> 127.0.0.1:${TunnelPort}"

Start-Process -FilePath "ssh" -ArgumentList @(
    "-p", $RemotePort,
    "-L", "$TunnelPort`:127.0.0.1:$TunnelPort",
    $RemoteHost
) -WindowStyle Normal

Write-Host "[INFO] Starting local Stage5 API: $ApiScript"

& python $ApiScript
