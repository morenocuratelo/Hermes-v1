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
"%UV_CMD%" venv --python 3.12 venv
if %errorlevel% neq 0 (
    echo ERRORE: Impossibile creare l'ambiente virtuale con Python 3.12.
    pause
    exit /b 1
)

echo [HERMES] Installazione di pip nel venv tramite uv...
"%UV_CMD%" pip install --python "venv\Scripts\python.exe" pip
if %errorlevel% neq 0 (
    echo ERRORE: Impossibile installare pip nel venv con uv.
    pause
    exit /b 1
)

call "venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo ERRORE: Impossibile attivare l'ambiente virtuale.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set "PY_VER=%%V"
echo [HERMES] Python nell'ambiente: %PY_VER%
echo %PY_VER% | findstr /b "3.12." >nul
if %errorlevel% neq 0 (
    echo ERRORE: L'ambiente virtuale non sta usando Python 3.12.
    pause
    exit /b 1
)

echo [HERMES] Installazione librerie (potrebbe richiedere tempo)...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERRORE: Aggiornamento pip fallito.
    pause
    exit /b 1
)

python -m pip install gdown
if %errorlevel% neq 0 (
    echo AVVISO: gdown non installato. Usero fallback urllib nel downloader.
)

python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERRORE: Installazione dipendenze da requirements.txt fallita.
    pause
    exit /b 1
)

echo [HERMES] Controllo e download modelli mancanti...
python tools\download_models.py
if %errorlevel% neq 0 (
    echo AVVISO: Download modello non completato. Verifica URL o connessione.
)

echo.
echo [HERMES] Installazione completata.
pause
endlocal
