param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..')).TrimEnd('\')
$baseline = Join-Path $projectRoot '.maintenance/protected_before.json'
if (-not (Test-Path -LiteralPath $baseline)) { throw 'Run inventory_cleanup.py inventory first' }
$rows = Get-Content -LiteralPath $baseline -Raw -Encoding UTF8 | ConvertFrom-Json
$moves = [Collections.Generic.List[object]]::new()
function Assert-ProjectPath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside project: $full"
    }
    return $full
}
function Relocate-Directory([string]$Source, [string]$Target) {
    $sourcePath = Assert-ProjectPath $Source
    $targetPath = Assert-ProjectPath $Target
    if (-not (Test-Path -LiteralPath $sourcePath)) { return }
    $item = Get-Item -LiteralPath $sourcePath -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { return }
    if (Test-Path -LiteralPath $targetPath) { throw "Target exists: $targetPath" }
    $moves.Add([pscustomobject]@{Source=$sourcePath;Target=$targetPath})
    if ($Apply) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force | Out-Null
        Move-Item -LiteralPath $sourcePath -Destination $targetPath
        try { New-Item -ItemType Junction -Path $sourcePath -Target $targetPath | Out-Null }
        catch {
            Move-Item -LiteralPath $targetPath -Destination $sourcePath
            throw
        }
    }
}
$workspaces = @(Get-ChildItem -LiteralPath $projectRoot -Directory -Force | Where-Object {
    $_.Name -like 'workspace*' -and -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint)
})
foreach ($workspace in $workspaces) {
    # Moving the entire directory keeps SQLite databases and their WAL/SHM together.
    Relocate-Directory $workspace.FullName (Join-Path $projectRoot ('knowledge_base/' + $workspace.Name))
}
# Use the pre-migration manifest: traversal never follows the compatibility junctions.
$paperDirs = @($rows | Where-Object { $_.path -match '^workspace[^/]*/' -and $_.path -match '\.pdf$' } |
    ForEach-Object { ($_.path -replace '/[^/]+$', '') } | Sort-Object -Unique)
$sourceDirs = @($rows | Where-Object { $_.path -match '^workspace[^/]*/(?:artifacts/)?sources/' } |
    ForEach-Object { $_.path -replace '/sources/.*$', '/sources' } | Sort-Object -Unique)
$assetDirs = @(@($paperDirs) + @($sourceDirs) | Sort-Object Length, {$_})
$selected = [Collections.Generic.List[string]]::new()
foreach ($relative in $assetDirs) {
    if (($relative -split '/').Count -lt 2) { throw "Paper at workspace root needs manual handling: $relative" }
    $covered = $false
    foreach ($parent in $selected) { if ($relative.StartsWith($parent + '/')) { $covered = $true; break } }
    if ($covered) { continue }
    $selected.Add($relative)
    $parts = $relative -split '/', 2
    $source = Join-Path $projectRoot ('knowledge_base/' + $relative)
    # Dry run reads the old physical directory; actual run uses the moved one.
    if (-not $Apply) { $source = Join-Path $projectRoot $relative }
    Relocate-Directory $source (Join-Path $projectRoot ('papers/' + $relative))
}
foreach ($workspace in $workspaces) {
    $relative = $workspace.Name + '/datasets'
    $source = Join-Path $projectRoot ('knowledge_base/' + $relative)
    if (-not $Apply) { $source = Join-Path $projectRoot $relative }
    Relocate-Directory $source (Join-Path $projectRoot ('datasets/' + $workspace.Name))
}
if ($Apply -and $moves.Count) {
    foreach ($workspace in $workspaces) {
        $alias = Join-Path $projectRoot $workspace.Name
        # .NET on a junction may change the target. cmd attrib /L addresses the link itself.
        & attrib.exe +H /L $alias
        if ($LASTEXITCODE -ne 0) { throw "Could not hide compatibility entry: $alias" }
    }
    $moves | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $projectRoot '.maintenance/asset_moves.json') -Encoding UTF8
}
[pscustomobject]@{Applied=[bool]$Apply;Workspaces=$workspaces.Count;Directories=$moves.Count;PaperGroups=$selected.Count}
