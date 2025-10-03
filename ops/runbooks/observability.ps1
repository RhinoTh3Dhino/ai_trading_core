# ops/runbooks/observability.ps1
# Observability runbook til lokal udvikling og validering (ASCII only).

[CmdletBinding()]
param(
  [switch]$Ui,
  [switch]$Alerting,
  [switch]$PromtoolTests,
  [switch]$Recreate,
  [switch]$Pull,
  [switch]$Down,
  [switch]$Open
)

$ErrorActionPreference = 'Stop'

function Write-Info([string]$msg) { Write-Host "INFO  $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "OK    $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "WARN  $msg" -ForegroundColor Yellow }
function Write-Err([string]$msg)  { Write-Host "ERR   $msg" -ForegroundColor Red }

# Stier (fra dette scripts placering)
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Resolve-Path (Join-Path $ScriptRoot "..\..")
$ComposeDir = Resolve-Path (Join-Path $ScriptRoot "..\compose")
$PromDir    = Resolve-Path (Join-Path $RepoRoot "ops\prometheus")

# Endpoints
$AppUrl      = "http://localhost:8000"
$AppHealth   = "$AppUrl/healthz"
$PromUrl     = "http://localhost:9090"
$PromReady   = "$PromUrl/-/ready"
$PromTargets = "$PromUrl/api/v1/targets"
$GrafanaUrl  = "http://localhost:3000"
$AlertmUrl   = "http://localhost:9093"

function Test-HttpOk {
  param([string]$Url, [int]$TimeoutSec = 120)
  for ($i=0; $i -lt $TimeoutSec; $i++) {
    try {
      $r = Invoke-WebRequest -UseBasicParsing -Uri $Url -Method GET -TimeoutSec 5
      if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400) { return $true }
    } catch {
      Start-Sleep -Seconds 1
    }
  }
  return $false
}

function Invoke-PromtoolCheckAndTests {
  param([string]$PromVersion = "v3.6.0")
  Write-Info "promtool check config/rules og test rules"
  $alerts  = Join-Path $PromDir "alerts.yml"
  $records = Join-Path $PromDir "recording_rules.yml"
  $cfg     = Join-Path $PromDir "prometheus.yml"
  $tests   = Join-Path $PromDir "tests\alerts_test.yml"

  if (-not (Test-Path $cfg))     { throw "Mangler: $cfg" }
  if (-not (Test-Path $alerts))  { Write-Warn "Mangler: $alerts (skipping alerts check)" }
  if (-not (Test-Path $records)) { Write-Warn "Mangler: $records (skipping recording rules check)" }

  docker run --rm `
    -v "${cfg}:/etc/prometheus/prometheus.yml:ro" `
    -v "${alerts}:/etc/prometheus/alerts.yml:ro" `
    -v "${records}:/etc/prometheus/recording_rules.yml:ro" `
    --entrypoint=promtool "prom/prometheus:$PromVersion" `
    check config /etc/prometheus/prometheus.yml | Write-Host

  if (Test-Path $alerts) {
    docker run --rm `
      -v "${alerts}:/etc/prom/alerts.yml:ro" `
      --entrypoint=promtool "prom/prometheus:$PromVersion" `
      check rules /etc/prom/alerts.yml | Write-Host
  }

  if (Test-Path $records) {
    docker run --rm `
      -v "${records}:/etc/prom/recording_rules.yml:ro" `
      --entrypoint=promtool "prom/prometheus:$PromVersion" `
      check rules /etc/prom/recording_rules.yml | Write-Host
  }

  if (Test-Path $tests) {
    docker run --rm `
      -v "${PromDir}:/etc/prom" `
      --entrypoint=promtool "prom/prometheus:$PromVersion" `
      test rules /etc/prom/tests/alerts_test.yml | Write-Host
  } else {
    Write-Warn "Ingen tests/alerts_test.yml - springer promtool test rules over."
  }
  Write-Ok "promtool checks/tests OK"
}

function Get-PromTargetsSummary {
  $tmp = Join-Path $env:TEMP "targets.json"
  Invoke-WebRequest -UseBasicParsing -Uri $PromTargets -OutFile $tmp | Out-Null
  $json = Get-Content $tmp -Raw | ConvertFrom-Json
  $list = $json.data.activeTargets
  if (-not $list) { Write-Warn "Ingen aktive targets returneret"; return }
  $list |
    Select-Object @{n='job';e={$_.labels.job}}, health, scrapeUrl, lastError |
    Sort-Object job |
    Format-Table -AutoSize
}

function Start-Stack {
  Push-Location $ComposeDir
  try {
    if ($Pull) {
      Write-Info "docker compose pull"
      docker compose -f docker-compose.yml pull
    }

    $opts = @('up','-d','--wait','live_connector','prometheus')
    if ($Recreate) { $opts = @('up','-d','--force-recreate','--wait','live_connector','prometheus') }

    Write-Info ("docker compose {0}" -f ($opts -join " "))
    docker compose -f docker-compose.yml @opts
    docker compose -f docker-compose.yml ps | Write-Host

    Write-Info "Venter paa app /healthz"
    if (-not (Test-HttpOk -Url $AppHealth -TimeoutSec 120)) {
      docker compose -f docker-compose.yml logs --no-color --tail=200 live_connector | Write-Host
      throw "App /healthz blev ikke klar i tide"
    }
    Write-Ok "App er klar"

    Write-Info "Venter paa Prometheus /-/ready"
    if (-not (Test-HttpOk -Url $PromReady -TimeoutSec 240)) {
      docker compose -f docker-compose.yml logs --no-color --tail=200 prometheus | Write-Host
      throw "Prometheus /-/ready blev ikke klar i tide"
    }
    Write-Ok "Prometheus er klar"

    Write-Info "Targets (UP) resume"
    Get-PromTargetsSummary
  } finally {
    Pop-Location
  }
}

function Start-Alerting {
  Push-Location $ComposeDir
  try {
    Write-Info "Starter Alertmanager profil"
    docker compose -f docker-compose.yml --profile alerting up -d am_init alertmanager
    if (-not (Test-HttpOk -Url "$AlertmUrl/-/ready" -TimeoutSec 120)) {
      docker compose -f docker-compose.yml logs --no-color --tail=200 alertmanager | Write-Host
      throw "Alertmanager blev ikke klar i tide"
    }
    Write-Ok "Alertmanager klar: $AlertmUrl"
  } finally {
    Pop-Location
  }
}

function Start-UI {
  Push-Location $ComposeDir
  try {
    Write-Info "Starter Grafana (UI profil)"
    docker compose -f docker-compose.yml --profile ui up -d grafana
    if (-not (Test-HttpOk -Url "$GrafanaUrl/api/health" -TimeoutSec 180)) {
      docker compose -f docker-compose.yml logs --no-color --tail=200 grafana | Write-Host
      throw "Grafana blev ikke klar i tide"
    }
    Write-Ok "Grafana klar: $GrafanaUrl"
  } finally {
    Pop-Location
  }
}

function Stop-Stack {
  Push-Location $ComposeDir
  try {
    Write-Info "docker compose down -v"
    docker compose -f docker-compose.yml down -v
    Write-Ok "Stack stoppet og volumes fjernet"
  } finally {
    Pop-Location
  }
}

# MAIN
try {
  if ($Down) {
    Stop-Stack
    return
  }

  # Pre-flight sanity
  if (-not (Test-Path (Join-Path $RepoRoot "bot\live_connector"))) {
    Write-Warn "Kunne ikke finde bot\live_connector - koer fra repo-roden hvis muligt"
  }
  if (-not (Test-Path (Join-Path $PromDir "prometheus.yml"))) {
    throw "Mangler ops\prometheus\prometheus.yml - kan ikke fortsaette"
  }

  if ($PromtoolTests) { Invoke-PromtoolCheckAndTests }

  Start-Stack

  if ($Alerting) { Start-Alerting }
  if ($Ui)       { Start-UI }

  if ($Open) {
    Write-Info "Aabner web-UI"
    try { Start-Process $AppUrl    } catch {}
    try { Start-Process $PromUrl   } catch {}
    if ($Ui)       { try { Start-Process $GrafanaUrl } catch {} }
    if ($Alerting) { try { Start-Process $AlertmUrl  } catch {} }
  }

  Write-Ok "Observability runbook faerdig"
  Write-Host ""
  Write-Host "Tip: Stop alt igen med:" -ForegroundColor DarkGray
  Write-Host "  powershell -File ops\runbooks\observability.ps1 -Down" -ForegroundColor DarkGray

} catch {
  Write-Err $_.Exception.Message
  exit 1
}
