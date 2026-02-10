@echo off
setlocal
cd /d "%~dp0"

echo [HERMES] Inizio configurazione automatica...

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRORE: Python non trovato. Installa Python 3.10+ e riprova.
    pause
    exit /b 1
)

python --version

if not exist "venv\Scripts\python.exe" (
    echo [HERMES] Creazione ambiente virtuale...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERRORE: Impossibile creare l'ambiente virtuale.
        pause
        exit /b 1
    )
) else (
    echo [HERMES] Ambiente virtuale gia presente.
)

call "venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo ERRORE: Impossibile attivare l'ambiente virtuale.
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

pip install gdown
if %errorlevel% neq 0 (
    echo AVVISO: gdown non installato. Usero fallback urllib nel downloader.
)

pip install -r requirements.txt
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
