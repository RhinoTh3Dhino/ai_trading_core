# tools\lint.ps1 (PowerShell 5.1 kompatibel)
$ErrorActionPreference = "Stop"

# --- 1) Hent bærbar actionlint, hvis den mangler ---
$toolsDir = Join-Path $PSScriptRoot "actionlint"
$alExe    = Join-Path $toolsDir "actionlint.exe"

if (-not (Test-Path $alExe)) {
  New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
  Write-Host "Henter actionlint (Windows amd64)..." -ForegroundColor Cyan
  try {
    $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/rhysd/actionlint/releases/latest" -Headers @{ "User-Agent" = "actionlint-installer" }
    $asset = $rel.assets | Where-Object { $_.name -match "windows_amd64\.zip$" } | Select-Object -First 1
    $url = $asset.browser_download_url
  } catch {
    Write-Warning "Kunne ikke læse GitHub API – bruger fallback v1.7.1"
    $version = "v1.7.1"
    $plain   = $version.TrimStart('v')
    $url     = "https://github.com/rhysd/actionlint/releases/download/$version/actionlint_${plain}_windows_amd64.zip"
  }
  $zip = Join-Path $toolsDir "actionlint.zip"
  Invoke-WebRequest -Uri $url -OutFile $zip
  Expand-Archive -Path $zip -DestinationPath $toolsDir -Force
  Remove-Item $zip -Force
}

# Tilføj til PATH for denne session og vis version
$env:PATH = "$toolsDir;$env:PATH"
& $alExe -version

# --- 2) Sørg for yamllint (via pip i din venv) ---
# (brug tempfil i stedet for heredoc)
$tempPy = [System.IO.Path]::GetTempFileName() + ".py"
@'
import sys, subprocess
try:
    import yamllint  # noqa: F401
    print("yamllint already installed")
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yamllint"])
'@ | Set-Content -Path $tempPy -NoNewline -Encoding UTF8
python $tempPy
Remove-Item $tempPy -Force

# --- 3) Normalisér EOL -> LF i workflows (undgår "expected \n") ---
$wfRoot = Join-Path (Get-Location) ".github/workflows"
if (Test-Path $wfRoot) {
  Get-ChildItem $wfRoot -Filter *.yml | ForEach-Object {
    $raw = Get-Content $_.FullName -Raw
    $fixed = $raw -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($_.FullName, $fixed, (New-Object System.Text.UTF8Encoding($false)))
  }
}

# --- 4) Læg en mild .yamllint-konfig hvis mangler ---
$ylCfg = Join-Path (Get-Location) ".yamllint"
if (-not (Test-Path $ylCfg)) {
@"
extends: default
rules:
  truthy: disable
  line-length:
    max: 140
    allow-non-breakable-words: true
  document-start: disable
  new-lines:
    type: unix
"@ | Set-Content $ylCfg -NoNewline -Encoding UTF8
}

# --- 5) Kør linters ---
Write-Host "`n== actionlint ==" -ForegroundColor Yellow
actionlint -color

Write-Host "`n== yamllint ==" -ForegroundColor Yellow
yamllint .github/workflows
