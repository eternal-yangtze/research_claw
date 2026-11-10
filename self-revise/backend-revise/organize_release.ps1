$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location $root
$backend = Join-Path $root 'self-revise/backend-revise'
$front = Join-Path $root 'self-revise/frontend-revise'
New-Item -ItemType Directory -Force $front | Out-Null
$moves = [System.Collections.Generic.List[object]]::new()
function Relocate($source, $target) {
    if (!(Test-Path -LiteralPath $source)) { return }
    New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
    Move-Item -LiteralPath $source -Destination $target
    $moves.Add(@{old=$source;new=$target})
}
Get-ChildItem self-revise -Force | Where-Object Name -NotIn @('backend-revise','frontend-revise') | ForEach-Object {
    if ($_.Name -eq 'frontend-evidence') { Relocate $_.FullName "$front/evidence" }
    else { Relocate $_.FullName (Join-Path $backend $_.Name) }
}
Relocate "$root/README_frontend.md" "$front/README_frontend.md"
Relocate "$root/tests" "$backend/tests"
Relocate "$root/internal_experiments" "$backend/internal_experiments"
$keep = @('serve_backend.py','research_rag.py','search_research.ps1','backup_rag_release.py','restore_qdrant_backup.py','inspect_qdrant_backup.py','warm_qdrant_knowledge.py','build_rag_lexical.py','teardown_environment.ps1','clean_runtime_artifacts.ps1')
$scriptMap = @{}
Get-ChildItem scripts -File | Where-Object Name -NotIn $keep | ForEach-Object {
    $dest = if ($_.Name -match '(chat_ui|real_browser|readme_cases|research_chat|web_browser|web_compact|web_workbench)') { 'self-revise/frontend-revise/scripts' } else { 'self-revise/backend-revise/scripts' }
    $scriptMap['scripts/'+$_.Name] = "$dest/$($_.Name)"
    Relocate $_.FullName "$root/$dest/$($_.Name)"
}
Get-ChildItem scripts -Directory | ForEach-Object { Relocate $_.FullName "$backend/scripts/$($_.Name)" }
# These archives are moved as opaque directories; no recursive traversal of junctions.
foreach ($name in @('.maintenance','.pytest_cache','__pycache__','researchclaw.egg-info')) {
    try { Relocate "$root/$name" "$backend/local-runtime/$name" } catch { $moves.Add(@{old=$name;error=$_.Exception.Message}) }
}
$utf8 = [System.Text.UTF8Encoding]::new($false)
$files = @(Get-ChildItem "$backend/tests" -Recurse -File -Filter '*.py') + @(Get-ChildItem "$backend/scripts","$front/scripts", "$root/scripts" -File | Where-Object Extension -In @('.py','.cjs','.ps1'))
foreach ($file in $files) {
    $text = [IO.File]::ReadAllText($file.FullName)
    $isMovedScript = $file.DirectoryName -in @("$backend\scripts", "$front\scripts")
    if ($isMovedScript -and $file.Extension -eq '.py') {
        $text = $text -replace 'Path\(__file__\)\.resolve\(\)\.parents\[1\]', 'next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())'
    }
    if ($isMovedScript -and $file.Extension -eq '.cjs') {
        $text = $text.Replace("require('../frontend/", "require('../../../frontend/")
        $text = $text.Replace("path.resolve(__dirname, '..')", "path.resolve(__dirname, '../../..')")
    }
    if ($file.FullName -like "$backend\tests\*") {
        $text = $text.Replace('Path(__file__).resolve().parents[2]', 'next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())')
        $text = $text.Replace('/ "scripts" / "benchmark_scratch_coding.py"', '/ "self-revise" / "backend-revise" / "scripts" / "benchmark_scratch_coding.py"')
    }
    $text = $text.Replace('self-revise/frontend-evidence/', 'self-revise/frontend-revise/evidence/')
    $text = $text.Replace("ROOT / 'README_frontend.md'", "ROOT / 'self-revise/frontend-revise/README_frontend.md'")
    $text = $text.Replace("ROOT / '.maintenance'", "ROOT / 'self-revise/frontend-revise/local-runtime'")
    $text = $text.Replace('.maintenance/', 'self-revise/frontend-revise/local-runtime/')
    foreach ($key in $scriptMap.Keys) { $text = $text.Replace($key, $scriptMap[$key]) }
    # Existing backend report paths retain their directory layouts after relocation.
    foreach ($name in @('agent-evidence','rag-evidence','memory-evidence','environment-evidence','stage1-evidence','stage23-evidence','question-analysis-v20261110','history','cleanup-v20261110')) {
        $text = $text.Replace("self-revise/$name/", "self-revise/backend-revise/$name/")
    }
    [IO.File]::WriteAllText($file.FullName, $text, $utf8)
}
# Repair navigation documents only; raw evidence JSON, source snapshots and hashes stay immutable.
Get-ChildItem "$backend","$front" -File -Filter '*.md' | ForEach-Object {
    $text = [IO.File]::ReadAllText($_.FullName)
    foreach ($key in $scriptMap.Keys) { $text = $text.Replace($key, $scriptMap[$key]) }
    $text = $text.Replace('self-revise/frontend-evidence/', 'self-revise/frontend-revise/evidence/')
    [IO.File]::WriteAllText($_.FullName, $text, $utf8)
}
[IO.File]::WriteAllText("$backend/directory-migration-v20261110.json", ($moves | ConvertTo-Json -Depth 5), $utf8)
Write-Output "Relocated $($moves.Count) entries. Raw experiment records preserved."
