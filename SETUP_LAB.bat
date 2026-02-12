@echo off
setlocal
cd /d "%~dp0"
set "SETUP_VERSION=2026-02-11-uv5"
set "PYTHON_VERSION=3.12"

echo [HERMES] Inizio configurazione automatica...
echo [HERMES] Script: SETUP_LAB.bat ^| Versione: %SETUP_VERSION%

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
echo [HERMES] Installazione di Python %PYTHON_VERSION%...
"%UV_CMD%" python install %PYTHON_VERSION%
if %errorlevel% neq 0 (
    echo ERRORE: Impossibile installare Python %PYTHON_VERSION% con uv.
    pause
    exit /b 1
)

if exist "venv" (
    echo [HERMES] Rimozione ambiente virtuale precedente...
    rmdir /s /q venv
)

echo [HERMES] Creazione ambiente virtuale con Python %PYTHON_VERSION%...
"%UV_CMD%" venv --python %PYTHON_VERSION% --seed venv
if %errorlevel% neq 0 (
    echo ERRORE: Impossibile creare l'ambiente virtuale con Python %PYTHON_VERSION%.
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
echo %PY_VER% | findstr /b "%PYTHON_VERSION%." >nul
if %errorlevel% neq 0 (
    echo ERRORE: L'ambiente virtuale non sta usando Python %PYTHON_VERSION%.
    pause
    exit /b 1
)

echo [HERMES] Installazione librerie (potrebbe richiedere tempo)...
"%UV_CMD%" pip install --python "%VENV_PY%" gdown
if %errorlevel% neq 0 (
    echo AVVISO: gdown non installato. Usero fallback urllib nel downloader.
)

set "REQ_FILE=requirements.txt"
set "REQ_FILE_SETUP=requirements.setup.txt"

if not exist "%REQ_FILE%" (
    echo ERRORE: File %REQ_FILE% non trovato.
    pause
    exit /b 1
)

echo [HERMES] Preparazione requirements compatibili con uv...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$c = Get-Content -LiteralPath 'requirements.txt'; $c = $c -replace '^\s*torch==(.+)\\+cu\\d+$','torch==$1' -replace '^\s*torchaudio==(.+)\\+cu\\d+$','torchaudio==$1' -replace '^\s*torchvision==(.+)\\+cu\\d+$','torchvision==$1'; Set-Content -LiteralPath 'requirements.setup.txt' -Value $c -Encoding ASCII"
if %errorlevel% neq 0 (
    echo AVVISO: Impossibile generare requirements.setup.txt. Uso requirements.txt originale.
) else (
    set "REQ_FILE=%REQ_FILE_SETUP%"
)

"%UV_CMD%" pip install --python "%VENV_PY%" -r "%REQ_FILE%"
if %errorlevel% neq 0 (
    echo.
    echo [HERMES] ERRORE: Installazione dipendenze fallita.
    echo.
    echo Se il problema riguarda CUDA o versioni non trovate:
    echo 1. Modifica manualmente "requirements.txt" rimuovendo suffissi come "+cu130".
    echo 2. Salva il file.
    echo.
    echo Premi un tasto per riprovare l'installazione usando "requirements.txt" manuale...
    pause
    
    "%UV_CMD%" pip install --python "%VENV_PY%" -r "requirements.txt"
    if %errorlevel% neq 0 (
        echo.
        echo [HERMES] ATTENZIONE: Anche il secondo tentativo e' fallito.
        echo Continuo con il resto dello script, ma l'applicazione potrebbe non funzionare correttamente. Installa i requisiti manualmente (pip install -r requirements.txt).
        pause
    )
)

if exist "%REQ_FILE_SETUP%" del /q "%REQ_FILE_SETUP%" >nul 2>&1

echo [HERMES] Controllo e download modelli mancanti...
"%VENV_PY%" tools\download_models.py
if %errorlevel% neq 0 (
    echo AVVISO: Download modello non completato. Verifica URL o connessione.
)

echo.
echo [HERMES] Installazione completata.
pause
endlocal
