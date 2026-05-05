param(
    [string]$RemoteHost = "root@connect.westb.seetacloud.com",
    [string]$RemotePort = "50734",
    [string]$RemoteDir = "~/cpo_project/outputs/remote_inference/jobs",
    [string]$LocalDir = "D:/001_softwares/VS Code Demo/cpo_project/outputs/remote_inference/jobs",
    [string]$RsyncPath = "rsync"
)

# Requires rsync (via WSL, cwRsync, or Git Bash)
& $RsyncPath -av --partial --info=stats1 --exclude "*.tmp" -e "ssh -p $RemotePort" "$RemoteHost:$RemoteDir/" "$LocalDir/"
