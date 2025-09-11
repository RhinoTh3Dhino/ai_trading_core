param(
  [string]$Path = ".github/workflows"
)
# Gem som UTF-8 UDEN BOM
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# Find alle YAML-filer
$files = Get-ChildItem $Path -Recurse -Include *.yml, *.yaml
foreach ($f in $files) {
  $s = Get-Content $f.FullName -Raw

  # 1) Normalisér linjeskift til LF
  $s = $s -replace "`r`n", "`n"
  $s = $s -replace "`r", "`n"

  # 2) Fjern alle kontroltegn (C0/C1) undtagen tab (\x09) og newline (\x0A)
  $s = [Regex]::Replace($s, "[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x80-\x9F]", "")

  # 3) (Valgfrit) Normalisér smarte anførselstegn & lange streger til ASCII
  $s = $s -replace "[\u2018\u2019\u201A\u201B]", "'"
  $s = $s -replace "[\u201C\u201D\u201E\u201F]", '"'
  $s = $s -replace "[\u2013\u2014\u2212]", "-"

  [System.IO.File]::WriteAllText($f.FullName, $s, $utf8NoBom)
  Write-Host "✔ Renset: $($f.FullName)"
}
