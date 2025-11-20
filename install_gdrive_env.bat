@echo off
title Google Drive Miljø Installer
color 0A
setlocal enabledelayedexpansion

REM === Logfil ===
set LOGFILE=install_gdrive_env.log
echo [INFO] Starter installation af Google Drive miljø > %LOGFILE%
echo.

echo ============================================
echo  Opretter virtuelt miljø: gdrive_env
echo ============================================
python -m venv gdrive_env >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [FEJL] Kunne ikke oprette venv. Se %LOGFILE% for detaljer.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Aktiverer miljø
echo ============================================
call gdrive_env\Scripts\activate >> %LOGFILE% 2>&1

echo.
echo ============================================
echo  Opgraderer pip, setuptools og wheel
echo ============================================
pip install --upgrade pip setuptools wheel >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [FEJL] Kunne ikke opgradere pip/setuptools/wheel. Se %LOGFILE% for detaljer.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Installerer GDrive-afhængigheder fra requirements-gdrive.txt
echo ============================================
pip install -r requirements-gdrive.txt >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [FEJL] Installation af requirements fejlede. Se %LOGFILE% for detaljer.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  [✔] Installation færdig!
echo ============================================
echo Aktiver med: call gdrive_env\Scripts\activate
echo Logfil gemt i: %LOGFILE%
echo.

pause
