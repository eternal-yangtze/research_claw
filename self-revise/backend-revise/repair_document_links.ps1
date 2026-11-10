$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$back = Join-Path $root 'self-revise/backend-revise'
$front = Join-Path $root 'self-revise/frontend-revise'
$utf8 = [Text.UTF8Encoding]::new($false)
$files = @(Get-ChildItem $back,$front -File -Filter '*.md')
foreach ($folder in @('rag-evidence','agent-evidence','memory-evidence','stage1-evidence','stage23-evidence','environment-evidence')) {
    $files += @(Get-ChildItem (Join-Path $back $folder) -Recurse -File -Filter '*.md' -ErrorAction SilentlyContinue | Where-Object FullName -NotMatch '(source|snapshot|vendor|node_modules|templates)')
}
$files += @(Get-ChildItem "$front/evidence" -Recurse -File -Filter '*.md' -ErrorAction SilentlyContinue)
foreach($f in $files) {
    if($f.Name -in @('RELEASE_LAYOUT.md','INDEX.md')) { continue }
    $oldFile = $f.FullName
    if($oldFile.StartsWith("$front\evidence\")) { $oldFile=$oldFile.Replace("$front\evidence\","$root\self-revise\frontend-evidence\") }
    elseif($oldFile -eq "$front\README_frontend.md") { $oldFile="$root\README_frontend.md" }
    elseif($oldFile.StartsWith("$back\")) { $oldFile=$oldFile.Replace("$back\","$root\self-revise\") }
    $content = [IO.File]::ReadAllText($f.FullName)
    $result = [regex]::Replace($content, '(?<!!)\[([^\]]+)\]\(([^\)]+)\)', [Text.RegularExpressions.MatchEvaluator]{param($match)
        $target=$match.Groups[2].Value
        if($target -match '^(https?:|mailto:|#)' -or $target -match '[<>]') { return $match.Value }
        try {
            $parts=$target.Split('#',2)
            $oldTarget=[IO.Path]::GetFullPath((Join-Path (Split-Path $oldFile) $parts[0]))
            $newTarget=$oldTarget
            if($newTarget -eq "$root\README_frontend.md") { $newTarget="$front\README_frontend.md" }
            elseif($newTarget.StartsWith("$root\self-revise\frontend-evidence")) { $newTarget=$newTarget.Replace("$root\self-revise\frontend-evidence","$front\evidence") }
            elseif($newTarget.StartsWith("$root\self-revise\") -and !$newTarget.StartsWith("$back\") -and !$newTarget.StartsWith("$front\")) { $newTarget=$newTarget.Replace("$root\self-revise\","$back\") }
            elseif($newTarget.StartsWith("$root\internal_experiments\")) { $newTarget=$newTarget.Replace("$root\internal_experiments\","$back\internal_experiments\") }
            if(!(Test-Path -LiteralPath $newTarget)) { return $match.Value }
            $base=[Uri]((Split-Path $f.FullName)+'\')
            $relative=[Uri]::UnescapeDataString($base.MakeRelativeUri([Uri]$newTarget).ToString())
            if($parts.Length -gt 1) { $relative+='#'+$parts[1] }
            return '['+$match.Groups[1].Value+']('+$relative+')'
        } catch { return $match.Value }
    })
    if($result -ne $content) { [IO.File]::WriteAllText($f.FullName,$result,$utf8) }
}
