@echo off
setlocal
cd /d "%~dp0"

echo [HERMES] Inizio configurazione automatica...

set "UV_CMD="

where uv >nul 2>&1
if %errorlevel% equ 0 (
    set "UV_CMD=uv"
)

if not defined UV_CMD if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV_CMD=%USERPROFILE%\.local\bin\uv.exe"
)

if not defined UV_CMD (
    echo [HERMES] uv non trovato. Installazione automatica...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo ERRORE: Installazione di uv fallita.
        pause
        exit /b 1
    )

    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV_CMD=%USERPROFILE%\.local\bin\uv.exe"
    ) else (
        where uv >nul 2>&1
        if %errorlevel% equ 0 set "UV_CMD=uv"
    )
)

if not defined UV_CMD (
    echo ERRORE: uv non disponibile dopo l'installazione.
    echo Installa uv manualmente da https://docs.astral.sh/uv/
    pause
    exit /b 1
)

echo [HERMES] Uso uv: %UV_CMD%
echo [HERMES] Installazione di Python 3.12...
"%UV_CMD%" python install 3.12
if %errorlevel% neq 0 (
    echo ERRORE: Impossibile installare Python 3.12 con uv.
    pause
    exit /b 1
)

if exist "venv" (
    echo [HERMES] Rimozione ambiente virtuale precedente...
    rmdir /s /q venv
)

echo [HERMES] Creazione ambiente virtuale con Python 3.12...
"%UV_CMD%" venv --python 3.12 --seed venv
if %errorlevel% neq 0 (
    echo ERRORE: Impossibile creare l'ambiente virtuale con Python 3.12.
    pause
    exit /b 1
)

set "VENV_PY=venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo ERRORE: Python del venv non trovato in %VENV_PY%.
    pause
    exit /b 1
)

echo [HERMES] Installazione di pip nel venv tramite uv...
"%UV_CMD%" pip install --python "%VENV_PY%" pip
if %errorlevel% neq 0 (
    echo AVVISO: Installazione esplicita di pip non riuscita. Procedo con uv pip.
)

echo [HERMES] Verifica Python nell'ambiente...
for /f "tokens=2 delims= " %%V in ('"%VENV_PY%" --version 2^>^&1') do set "PY_VER=%%V"
echo [HERMES] Python nell'ambiente: %PY_VER%
echo %PY_VER% | findstr /b "3.12." >nul
if %errorlevel% neq 0 (
    echo ERRORE: L'ambiente virtuale non sta usando Python 3.12.
    pause
    exit /b 1
)

echo [HERMES] Installazione librerie (potrebbe richiedere tempo)...
"%UV_CMD%" pip install --python "%VENV_PY%" gdown
if %errorlevel% neq 0 (
    echo AVVISO: gdown non installato. Usero fallback urllib nel downloader.
)

"%UV_CMD%" pip install --python "%VENV_PY%" -r requirements.txt
if %errorlevel% neq 0 (
    echo ERRORE: Installazione dipendenze da requirements.txt fallita.
    pause
    exit /b 1
)

echo [HERMES] Controllo e download modelli mancanti...
"%VENV_PY%" tools\download_models.py
if %errorlevel% neq 0 (
    echo AVVISO: Download modello non completato. Verifica URL o connessione.
)

echo.
echo [HERMES] Installazione completata.
pause
endlocal
