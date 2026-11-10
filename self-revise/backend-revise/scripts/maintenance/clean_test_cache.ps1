param(
    [switch]$Apply,
    [string]$ReportPath = '.maintenance/reorganization_cleanup.json'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path.TrimEnd('\')
$reportTarget = [IO.Path]::GetFullPath((Join-Path $projectRoot $ReportPath))
if (-not $reportTarget.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Report must stay inside project' }
New-Item -ItemType Directory -Path (Split-Path -Parent $reportTarget) -Force | Out-Null
$targets = @(Get-ChildItem -LiteralPath $projectRoot -Directory -Force | Where-Object { $_.Name -match '^\.pytest(?:_|$)' })
$legacyTemp = Join-Path $projectRoot '.test_tmp'
if (Test-Path -LiteralPath $legacyTemp) {
    $targets += Get-ChildItem -LiteralPath $legacyTemp -Directory -Force
}
foreach ($area in @('src', 'tests', 'scripts')) {
    $targets += Get-ChildItem -LiteralPath (Join-Path $projectRoot $area) -Directory -Filter '__pycache__' -Recurse
}
$topCache = Join-Path $projectRoot '__pycache__'
if (Test-Path -LiteralPath $topCache) { $targets += Get-Item -LiteralPath $topCache }
$results = @()
foreach ($item in $targets) {
    $path = (Resolve-Path -LiteralPath $item.FullName).Path
    $relative = $path.Substring($projectRoot.Length + 1)
    $valid = $path.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase) -and (
        ($item.Parent.FullName -eq $projectRoot -and $item.Name -match '^\.pytest(?:_|$)') -or
        ($item.Name -eq '__pycache__' -and ($relative -eq '__pycache__' -or $relative -match '^(src|tests|scripts)\\')) -or
        ($item.Parent.FullName -eq $legacyTemp -and $item.Name -match '^(phase[1-4]|batch_phase|final_neural_dag|neural_dag(?:_2)?|sha_final)$'))
    if (-not $valid -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Unsafe cache target: $path" }
    try {
        $contents = @(Get-ChildItem -LiteralPath $path -Force -Recurse)
    } catch {
        $results += [pscustomobject]@{Path=$path;Bytes=0;Removed=$false;Status='skipped_access_denied'}
        continue
    }
    if (@($contents | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        throw "Cache has links; refusing recursive deletion: $path"
    }
    $bytes = ($contents | Where-Object { -not $_.PSIsContainer } | Measure-Object Length -Sum).Sum
    $removed = $false
    $status = 'verified_cache'
    if ($Apply) {
        try {
            Remove-Item -LiteralPath $path -Recurse -Force
            $removed = $true
        } catch {
            $status = 'removal_incomplete'
        }
    }
    $results += [pscustomobject]@{Path=$path;Bytes=[long]$bytes;Removed=$removed;Status=$status}
}
if ($Apply) {
    $results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $reportTarget -Encoding UTF8
}
[pscustomobject]@{Targets=$results.Count;Removed=@($results | Where-Object Removed).Count;RemovedBytes=($results | Where-Object Removed | Measure-Object Bytes -Sum).Sum;Applied=[bool]$Apply;Skipped=@($results | Where-Object {$_.Status -ne 'verified_cache'}).Count}
