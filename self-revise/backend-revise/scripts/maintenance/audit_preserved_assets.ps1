param(
    [switch]$Verify,
    [string]$BaselinePath = '.maintenance/asset_inventory_20261019.json'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path.TrimEnd('\')
$baselineTarget = [IO.Path]::GetFullPath((Join-Path $projectRoot $BaselinePath))
if (-not $baselineTarget.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Baseline must stay inside project' }

# Read every workspace's file identity, size and modification time without changing it.
# This is a metadata fingerprint, not a full content checksum of every dataset image.
$workspaces = @(
    foreach ($directory in (Get-ChildItem -LiteralPath $projectRoot -Directory -Force -Filter 'workspace*' | Sort-Object Name)) {
        $builder = New-Object Text.StringBuilder
        [long]$bytes = 0
        $files = @([IO.Directory]::EnumerateFiles($directory.FullName, '*', [IO.SearchOption]::AllDirectories) | Sort-Object)
        foreach ($file in $files) {
            $info = [IO.FileInfo]$file
            $bytes += $info.Length
            [void]$builder.AppendLine(('{0}|{1}|{2}' -f $file.Substring($projectRoot.Length + 1), $info.Length, $info.LastWriteTimeUtc.Ticks))
        }
        $algorithm = [Security.Cryptography.SHA256]::Create()
        try { $digest = [BitConverter]::ToString($algorithm.ComputeHash([Text.Encoding]::UTF8.GetBytes($builder.ToString()))).Replace('-', '') }
        finally { $algorithm.Dispose() }
        [pscustomobject]@{Name=$directory.Name;Files=$files.Count;Bytes=$bytes;MetadataSHA256=$digest}
    }
)

$historical = Get-Content -LiteralPath (Join-Path $projectRoot '.maintenance/reorganization_protected_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$paths = @($historical | ForEach-Object { $_.Path })
$paths += Get-ChildItem -LiteralPath (Join-Path $projectRoot 'workspace_batch_vit_v2/artifacts/coding') -File -Filter '*.json' | ForEach-Object FullName
foreach ($runName in @('coding_b47a5e29638086d91812_scratch_c_replanned_20261019', 'coding_c9236140044de88f0b73_scratch_final_replanned_20261019')) {
    $runRoot = Join-Path $projectRoot ('workspace_batch_vit_v2/coding/' + $runName)
    $paths += Get-ChildItem -LiteralPath (Join-Path $runRoot 'project') -File -Recurse | Where-Object { $_.FullName -notmatch '\\(__pycache__|\.pytest_cache)\\' } | ForEach-Object FullName
    $paths += Join-Path $runRoot 'report.json'
    $paths += Join-Path $runRoot 'checkpoint.json'
}
$protected = @(
    foreach ($path in ($paths | Sort-Object -Unique)) {
        $item = Get-Item -LiteralPath $path
        [pscustomobject]@{Path=$path;Length=$item.Length;SHA256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash}
    }
)
$snapshot = [pscustomobject]@{CreatedAt=[DateTime]::UtcNow.ToString('o');ProjectRoot=$projectRoot;Workspaces=$workspaces;ProtectedFiles=$protected}
if (-not $Verify) {
    if (Test-Path -LiteralPath $baselineTarget) { throw 'Baseline already exists; choose a new filename to preserve evidence' }
    $snapshot | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $baselineTarget -Encoding UTF8
    [pscustomobject]@{Mode='snapshot';Workspaces=$workspaces.Count;Files=($workspaces | Measure-Object Files -Sum).Sum;ProtectedFiles=$protected.Count;Baseline=$baselineTarget}
    exit
}
$before = Get-Content -LiteralPath $baselineTarget -Raw -Encoding UTF8 | ConvertFrom-Json
$differences = @()
foreach ($entry in $before.Workspaces) {
    $current = @($workspaces | Where-Object Name -eq $entry.Name)
    if ($current.Count -ne 1 -or $current[0].MetadataSHA256 -ne $entry.MetadataSHA256) { $differences += $entry.Name }
}
foreach ($entry in $before.ProtectedFiles) {
    $current = @($protected | Where-Object Path -eq $entry.Path)
    if ($current.Count -ne 1 -or $current[0].SHA256 -ne $entry.SHA256) { $differences += $entry.Path }
}
if ($workspaces.Count -ne $before.Workspaces.Count) { $differences += 'workspace_count_changed' }
$result = [pscustomobject]@{Mode='verify';Passed=($differences.Count -eq 0);Workspaces=$workspaces.Count;Files=($workspaces | Measure-Object Files -Sum).Sum;ProtectedFiles=$protected.Count;Differences=$differences}
$result | ConvertTo-Json -Depth 4
if ($differences.Count) { exit 1 }
