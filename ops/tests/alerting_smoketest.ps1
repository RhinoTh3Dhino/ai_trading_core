# ops/tests/alerting_smoketest.ps1

$nowUtc = (Get-Date).ToUniversalTime()
$startsAt = $nowUtc.AddMinutes(-1).ToString("s") + "Z"
$endsAt   = $nowUtc.AddMinutes(+4).ToString("s") + "Z"

$testAlert = @"
[
  {
    "labels": {
      "alertname": "TEST_Alert",
      "severity": "warning",
      "job": "manual",
      "instance": "local"
    },
    "annotations": {
      "summary": "Manuel test via API",
      "description": "Dette er en engangstest fra PowerShell"
    },
    "startsAt": "$startsAt"
  }
]
"@

$null = Invoke-RestMethod -Uri "http://localhost:9093/api/v2/alerts" -Method Post -ContentType "application/json" -Body $testAlert
Write-Host "POST /alerts -> OK"

Start-Sleep -Seconds 5

$resolved = @"
[
  {
    "labels": { "alertname": "TEST_Alert", "severity": "warning", "job": "manual", "instance": "local" },
    "annotations": { "summary": "Manuel test via API" },
    "endsAt": "$endsAt"
  }
]
"@

$null = Invoke-RestMethod -Uri "http://localhost:9093/api/v2/alerts" -Method Post -ContentType "application/json" -Body $resolved
Write-Host "POST /alerts (resolved) -> OK"
