param(
    [string]$RemoteHost = "root@connect.westb.seetacloud.com",
    [string]$RemotePort = "50734",
    [string]$RemoteDir = "/root/autodl-tmp/cpo_project/outputs/remote_inference/jobs",
    [string]$LocalDir = "D:/001_softwares/VS Code Demo/cpo_project/outputs/remote_inference/jobs"
)

$null = New-Item -ItemType Directory -Path $LocalDir -Force

$distros = wsl.exe --list --quiet 2>$null
if (-not $distros) {
    Write-Error "WSL distro not found. Run: wsl --install -d Ubuntu, then install rsync inside WSL."
    exit 1
}

$localDirWsl = [regex]::Replace($LocalDir, "^([A-Za-z]):", { "/mnt/" + $args[0].Groups[1].Value.ToLowerInvariant() })
$localDirWsl = $localDirWsl -replace "\\", "/"

$cmd = "rsync -av --partial --info=stats1 --exclude '*.tmp' -e 'ssh -p $RemotePort' '${RemoteHost}:${RemoteDir}/' '${localDirWsl}/'"

wsl.exe bash -lc "$cmd"
