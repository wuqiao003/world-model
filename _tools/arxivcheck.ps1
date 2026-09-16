param([Parameter(Mandatory=$true)][string]$Ids)
$list = $Ids -replace '\s',''
$url = "https://export.arxiv.org/api/query?id_list=$list&max_results=100"
try { $r = curl.exe -sSL $url } catch { Write-Output "FETCH FAIL"; exit 1 }
[xml]$x = $r -join "`n"
foreach ($e in $x.feed.entry) {
  $id = ($e.id -replace '.*abs/','')
  $t  = ($e.title -replace '\s+',' ').Trim()
  $p  = $e.published
  $c  = if ($e.comment) { ($e.comment -replace '\s+',' ').Trim() } else { '' }
  $j  = if ($e.'journal_ref') { ($e.'journal_ref' -replace '\s+',' ').Trim() } else { '' }
  Write-Output "$id | $p | $t"
  if ($c) { Write-Output "    COMMENT: $c" }
  if ($j) { Write-Output "    JOURNAL: $j" }
}
