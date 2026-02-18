@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [HERMES] Ambiente virtuale non trovato.
    echo Esegui prima SETUP_LAB.bat.
    pause
    exit /b 1
)

"%VENV_PY%" hermes_unified.py %*
set "APP_EXIT=%errorlevel%"

if not "%APP_EXIT%"=="0" (
    echo.
    echo [HERMES] Applicazione chiusa con codice %APP_EXIT%.
    pause
)

exit /b %APP_EXIT%
