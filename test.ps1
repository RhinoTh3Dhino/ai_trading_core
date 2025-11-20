# test.ps1 - Kør testpipeline med pytest og coverage

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "Aktiverer virtuel environment (.venv)..."
    .\.venv\Scripts\Activate.ps1
}

Write-Host "Rydder Python-cache..."
if (Test-Path ".pytest_cache") { Remove-Item ".pytest_cache" -Recurse -Force }
Get-ChildItem -Recurse -Include *.pyc | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Include "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Installerer nøglepakker..."
pip install --upgrade --force-reinstall "numpy<2.0" pandas_ta matplotlib kiwisolver pytest pytest-cov

Write-Host "Kører pytest med coverage..."
pytest --maxfail=1 --disable-warnings -q --cov=. --cov-report=term-missing

Write-Host "Testkørsel færdig!"
