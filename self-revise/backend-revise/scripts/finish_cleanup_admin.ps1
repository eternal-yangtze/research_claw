$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script in an Administrator PowerShell window. No changes made.'
}
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$outer = Split-Path $project -Parent
$manifest = Join-Path $project 'self-revise\backend-revise\environment-evidence\environment-cleanup-manifest.json'
$entries = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
$targets = $entries | Where-Object action -eq 'blocked' | Select-Object -ExpandProperty path -Unique
foreach ($target in $targets) {
    $full = [IO.Path]::GetFullPath($target)
    $parent = Split-Path $full -Parent
    $leaf = Split-Path $full -Leaf
    $allowed = ($parent -eq $outer -and $leaf -match '^\.rt[a-zA-Z0-9_]+$') -or
        ($parent -eq $project -and $leaf -match '^\.(w27|web_check|pytest_cache)')
    if (-not $allowed) { throw "Unexpected permission repair target: $full" }
    if (-not (Test-Path -LiteralPath $full)) { continue }
    if ((Get-Item -LiteralPath $full -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refuse junction: $full" }
    & takeown.exe /F $full /R /D Y
    if ($LASTEXITCODE) { throw "Ownership repair failed: $full" }
    & icacls.exe $full /grant ('*'+$identity.User.Value+':(OI)(CI)F') /T /C /Q
    if ($LASTEXITCODE) { throw "Permission repair failed: $full" }
}
& (Join-Path $PSScriptRoot 'teardown_environment.ps1') -Apply
