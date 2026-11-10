param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$outer = Split-Path $root -Parent
$records = [Collections.Generic.List[object]]::new()
function Remove-Checked([string]$path) {
    $full = [IO.Path]::GetFullPath($path)
    if (-not $full.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)) { throw "Outside project: $full" }
    if (-not (Test-Path -LiteralPath $full)) { return }
    $item = Get-Item -LiteralPath $full -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refuse recursive junction deletion: $full" }
    if ($Apply) { try { Remove-Item -LiteralPath $full -Force -Recurse } catch { $records.Add([pscustomobject]@{action='blocked';path=$full;error=$_.Exception.Message}); return } }
    $records.Add([pscustomobject]@{action='delete-environment';path=$full})
}
# Preserve historical experimental evidence in one visible location, rather than
# guessing whether individual traces/SQLite files will be needed for an audit.
$archive = Join-Path $root 'self-revise\backend-revise\internal_experiments\runs\legacy_test_evidence'
$candidates = @()
$candidates += Get-ChildItem -LiteralPath $outer -Force -Directory | Where-Object Name -match '^\.rt[a-zA-Z0-9_]+$'
$candidates += Get-ChildItem -LiteralPath $root -Force -Directory | Where-Object Name -match '^\.(w27|live27|browser27|reuse27|web_check|download_demo_check)'
foreach ($item in $candidates) {
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Unexpected junction: $($item.FullName)" }
    $destination = Join-Path $archive ($item.Name.TrimStart('.'))
    if ($Apply) { try {
        New-Item -ItemType Directory -Force -Path $archive | Out-Null
        robocopy $item.FullName $destination /E /XJ /R:0 /W:0 /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "Evidence copy failed: $LASTEXITCODE" }
        Remove-Item -LiteralPath $item.FullName -Recurse -Force
    } catch { $records.Add([pscustomobject]@{action='blocked';path=$item.FullName;error=$_.Exception.Message}); continue } }
    $records.Add([pscustomobject]@{action='preserve-test-evidence';path=$item.FullName;destination=$destination})
}
foreach ($name in @('.venv','.venv-v03','.venv-ragas','.pytest_cache','__pycache__','ResearchClaw.egg-info','frontend\node_modules','frontend\dist')) { Remove-Checked (Join-Path $root $name) }
# Never traverse junctions, backups, source history, or dependency trees.
$pending = [Collections.Generic.Stack[string]]::new(); $pending.Push($root)
while ($pending.Count) {
    $current = $pending.Pop()
    try { $entries = @(Get-ChildItem -LiteralPath $current -Force -Directory) } catch { $records.Add([pscustomobject]@{action='blocked-scan';path=$current;error=$_.Exception.Message}); continue }
    foreach ($item in $entries) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
        if ($item.Name -eq '.git' -or $item.Name -eq 'docker_backup_20261030') { continue }
        if ($item.Name -in @('__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','node_modules') -or $item.Name -match '^\.venv($|-)') { Remove-Checked $item.FullName }
        else { $pending.Push($item.FullName) }
    }
}
if ($Apply) { $records | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $root 'self-revise\backend-revise\environment-evidence\environment-cleanup-manifest.json') -Encoding UTF8 }
$records | Format-Table -AutoSize
