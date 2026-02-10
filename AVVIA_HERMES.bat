@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [HERMES] Ambiente virtuale non trovato.
    echo Esegui prima SETUP_LAB.bat.
    pause
    exit /b 1
)

call "venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo ERRORE: Impossibile attivare l'ambiente virtuale.
    pause
    exit /b 1
)

python hermes_unified.py
set "APP_EXIT=%errorlevel%"

if not "%APP_EXIT%"=="0" (
    echo.
    echo [HERMES] Applicazione chiusa con codice %APP_EXIT%.
    pause
)

exit /b %APP_EXIT%
