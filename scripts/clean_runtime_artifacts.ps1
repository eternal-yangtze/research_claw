param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$workspaceRoot = [IO.Directory]::GetParent($projectRoot).FullName
$requiredFixture = Join-Path $projectRoot 'self-revise\backend-revise\internal_experiments\runs\v03_p8\20261023_a'
$protectedPrefixes = @(
    (Join-Path $requiredFixture 'study'),
    (Join-Path $requiredFixture 'implementation_continued\coding\study_coding_a99bc09cbc6909af5978_continued\inputs'),
    (Join-Path $requiredFixture 'implementation_continued\coding\study_coding_a99bc09cbc6909af5978_continued\project')
)
$candidateRoots = @()
Get-ChildItem -LiteralPath $workspaceRoot -Force -Directory | ForEach-Object {
    if ($_.Name -match '^\.rt[a-zA-Z0-9_]+$' -and -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        $candidateRoots += $_.FullName
    }
}
Get-ChildItem -LiteralPath $projectRoot -Force -Directory | ForEach-Object {
    if ($_.Name -match '^\.(w27|live27|browser27|reuse27|pytest_cache|web_check|web_live)' -and -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        $candidateRoots += $_.FullName
    }
}
foreach ($name in @('runs','evaluation','installation','verification','maintenance','historical_maintenance','planning','documentation')) {
    $path = Join-Path $projectRoot "self-revise\backend-revise\internal_experiments\$name"
    if (Test-Path -LiteralPath $path) { $candidateRoots += $path }
}
# Keep all source, scripts, tests, configs, environments, datasets, original PDFs,
# full knowledge-base directories and current user conversation data.
$cacheParents = @('src','backend','tests','scripts','frontend\dist')
foreach ($name in $cacheParents) {
    $path = Join-Path $projectRoot $name
    if (-not (Test-Path -LiteralPath $path)) { continue }
    if ($name -eq 'frontend\dist') { $candidateRoots += $path; continue }
    $pending = [Collections.Generic.Stack[string]]::new(); $pending.Push($path)
    while ($pending.Count) {
        foreach ($entry in Get-ChildItem -LiteralPath $pending.Pop() -Force -Directory) {
            if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
            if ($entry.Name -eq '__pycache__') { $candidateRoots += $entry.FullName }
            else { $pending.Push($entry.FullName) }
        }
    }
}
$removedFiles = 0; [long]$removedBytes = 0; $keptDocs = 0
$inaccessible = [Collections.Generic.List[string]]::new()
foreach ($candidate in ($candidateRoots | Sort-Object -Unique)) {
    $absolute = [IO.Path]::GetFullPath($candidate)
    $parentTestRoot = ([IO.Directory]::GetParent($absolute).FullName -eq $workspaceRoot -and [IO.Path]::GetFileName($absolute) -match '^\.rt[a-zA-Z0-9_]+$')
    if (-not $absolute.StartsWith($projectRoot + '\',[StringComparison]::OrdinalIgnoreCase) -and -not $parentTestRoot) { throw "Unsafe target: $absolute" }
    if ((Get-Item -LiteralPath $absolute -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refuse junction root: $absolute" }
    $pending = [Collections.Generic.Stack[string]]::new(); $pending.Push($absolute)
    $directories = [Collections.Generic.List[string]]::new(); [long]$bytes = 0; $count = 0
    while ($pending.Count) {
        $current = $pending.Pop(); $directories.Add($current)
        try { $entries = @(Get-ChildItem -LiteralPath $current -Force -ErrorAction Stop) }
        catch { $inaccessible.Add($current); continue }
        foreach ($entry in $entries) {
            if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
            $protected = $false
            foreach ($prefix in $protectedPrefixes) {
                if ($entry.FullName -eq $prefix -or $entry.FullName.StartsWith($prefix+'\',[StringComparison]::OrdinalIgnoreCase)) { $protected = $true; break }
            }
            if ($protected) { continue }
            if ($entry.PSIsContainer) { $pending.Push($entry.FullName); continue }
            # Retain authored explanatory material, not bundled dependency READMEs.
            if ($entry.Extension -in @('.md','.rst','.tex') -and $entry.FullName -notmatch '\\node_modules\\|\\__pycache__\\|\\dist\\') { $keptDocs++; continue }
            if ($Apply) {
                try { Remove-Item -LiteralPath $entry.FullName -Force -ErrorAction Stop }
                catch { $inaccessible.Add($entry.FullName); continue }
            }
            $count++; $bytes += $entry.Length
        }
    }
    if ($Apply) {
        foreach ($directory in ($directories | Sort-Object Length -Descending)) {
            try {
                if (-not (Get-ChildItem -LiteralPath $directory -Force -ErrorAction Stop | Select-Object -First 1)) { Remove-Item -LiteralPath $directory -Force -ErrorAction Stop }
            } catch { if (-not $inaccessible.Contains($directory)) { $inaccessible.Add($directory) } }
        }
    }
    $removedFiles += $count; $removedBytes += $bytes
    $display = if ($parentTestRoot) { '..\'+[IO.Path]::GetFileName($absolute) } else { $absolute.Substring($projectRoot.Length+1) }
    [pscustomobject]@{Target=$display;Files=$count;MiB=[Math]::Round($bytes/1MB,2);Applied=[bool]$Apply}
}
# Top-level archival manifests are migration intermediates, not runtime config.
foreach ($name in @('archive_audit.json','migration_manifest.json','public_docs_check.json','requirements_resolution.json')) {
    $path=Join-Path $projectRoot "self-revise\backend-revise\internal_experiments\$name"
    if (Test-Path -LiteralPath $path) {
        $item=Get-Item -LiteralPath $path; $removedFiles++; $removedBytes+=$item.Length
        if ($Apply) { Remove-Item -LiteralPath $path -Force }
    }
}
"TOTAL files=$removedFiles bytes=$removedBytes authored_documents_retained=$keptDocs applied=$Apply"
"INACCESSIBLE count=$($inaccessible.Count)"
$inaccessible | Select-Object -First 20
