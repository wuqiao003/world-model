# Ship the code tree to the cluster pod.
#
# Two hops: scp to the jump host, then kubectl cp into the pod. Remote commands
# are single-line (PowerShell here-strings emit CRLF, which bash mis-parses as
# part of the last argument). Data files are NOT shipped this way -- kubectl cp
# streams over the exec channel and drops transfers of this size; subsets are
# regenerated on the cluster from the full dataset instead.
param(
    [string]$Pod = "raytrainv2-jxdong-ray-worker-worker-25mhn",
    [string]$Jump = "root@124.223.202.234",
    [string]$Remote = "/mnt/group/jxdong/wm_exp",
    [int]$Retries = 3
)

$ErrorActionPreference = "Continue"
Push-Location (Split-Path -Parent $PSScriptRoot)

$tar = Join-Path $env:TEMP "wm_code.tar.gz"
Remove-Item $tar -ErrorAction SilentlyContinue
tar --exclude="__pycache__" --exclude="*.pyc" -czf $tar code eval
if (-not (Test-Path $tar)) { Write-Output "TAR FAILED"; Pop-Location; exit 1 }
$kb = [math]::Round((Get-Item $tar).Length / 1KB, 1)
Write-Output "code tarball: $kb KB"

$ok = $false
for ($i = 1; $i -le $Retries; $i++) {
    scp -o ConnectTimeout=30 -o ServerAliveInterval=15 $tar "${Jump}:/tmp/wm_code.tar.gz"
    if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    Write-Output "  scp attempt $i failed, retrying"
    Start-Sleep -Seconds 5
}
if (-not $ok) { Write-Output "SCP FAILED"; Pop-Location; exit 1 }

$cmd = "kubectl cp /tmp/wm_code.tar.gz default/${Pod}:/tmp/wm_code.tar.gz; " +
       "kubectl exec -n default $Pod -- bash -c 'cd $Remote && tar xzf /tmp/wm_code.tar.gz && ls code/ | tail -3 && echo CODE_OK'"
for ($i = 1; $i -le $Retries; $i++) {
    $out = ssh -o ConnectTimeout=30 $Jump $cmd 2>&1
    $out | Where-Object { $_ -notmatch "Defaulted container" }
    if ($out -match "CODE_OK") { break }
    Write-Output "  kubectl cp attempt $i failed, retrying"
    Start-Sleep -Seconds 5
}

Pop-Location
