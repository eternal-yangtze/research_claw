[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Query,
    [ValidateRange(1, 20)]
    [int]$Limit = 5,
    [switch]$EnglishOnly
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$libraryRoot = Join-Path $projectRoot 'knowledge_base\rag_corpora\time_series_domain_release_v3_1'
$manifestPath = Join-Path $libraryRoot 'rag-search-index.json'
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw 'Build the library BM25 index first: .\.venv\Scripts\python.exe -B scripts/build_rag_lexical.py'
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$embeddingParts = $manifest.embedding_key -split '@', 2
if ($embeddingParts.Count -ne 2) {
    throw 'Invalid embedding identity in the library search manifest'
}
$queryMode = 'auto'
if ($EnglishOnly) { $queryMode = 'off' }
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cliPath = Join-Path $projectRoot 'researchclaw.py'
& $pythonPath -X utf8 -B $cliPath --workspace $libraryRoot rag-search $Query `
    --corpus-version $manifest.corpus_version `
    --embedding-model $embeddingParts[0] --embedding-revision $embeddingParts[1] `
    --limit $Limit --candidate-limit 40 --query-language-mode $queryMode
exit $LASTEXITCODE
