Param(
  [string]$RepoRoot = "$PSScriptRoot\..",
  [string]$Venv = "$PSScriptRoot\..\ .venv\Scripts\python.exe",
  [int]$KeepCsv = 50000,
  [int]$KeepText = 200000
)
$py = (Resolve-Path $Venv).Path
Push-Location $RepoRoot
& $py -m utils.logs_utils --glob "$RepoRoot\logs\*.csv" --keep $KeepCsv
& $py -m utils.log_utils  --rotate "$RepoRoot\logs\bot.log" --keep $KeepText
Pop-Location
