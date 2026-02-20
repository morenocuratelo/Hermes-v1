# HERMES - Master TOI Module Developer Guide

Questo documento descrive la logica interna, i flussi di dati e le trasformazioni implementate nel modulo **Master TOI** (`hermes_master_toi.py`). Il modulo è responsabile della sincronizzazione temporale tra diverse sorgenti dati (Eye-tracker, Log comportamentali, Video) e della definizione degli intervalli temporali di interesse (TOI) per l'analisi.

## 1. Logica di Gestione: `MasterToiLogic`

La classe `MasterToiLogic` gestisce l'importazione dei log, il calcolo degli offset di sincronizzazione e la strutturazione della tabella dei TOI.

### 1.1. Importazione e Parsing Log

*   **Input:**
    *   Log Eye-Tracker (es. Tobii TSV/Excel) contenente timestamp e eventi/trigger.
    *   Log Esterno (es. E-Prime, MATLAB, CSV manuale) contenente la sequenza degli eventi sperimentali.
*   **Logica:**
    *   Normalizza i nomi delle colonne (es. cerca colonne "Event", "Timestamp").
    *   Identifica i marker di sincronizzazione comuni (es. "VideoStart", "TrialStart").

### 1.2. Sincronizzazione Temporale

Allinea la timeline dell'eye-tracker (tempo assoluto o relativo alla macchina) con la timeline del video o dell'esperimento.

*   **Metodo:** Calcolo dell'Offset Lineare.
    *   `Offset = Timestamp_Evento_EyeTracker - Timestamp_Evento_LogEsterno`
    *   Applica questo offset a tutti i tempi per portarli in un sistema di riferimento comune (solitamente relativo all'inizio del video).

### 1.3. Definizione TOI (Time of Interest)

Costruisce la tabella principale che guida l'analisi statistica.

*   **Struttura Dati:** DataFrame con colonne:
    *   `Phase`: Nome della fase (es. "Fixation", "Stimulus").
    *   `Condition`: Condizione sperimentale.
    *   `Start`: Tempo inizio (secondi, sincronizzato).
    *   `End`: Tempo fine (secondi, sincronizzato).
    *   `Trial`: Numero progressivo (opzionale).
*   **Output:** File `_TOI.tsv` usato da `hermes_stats.py`.

## 2. Data Pruning e Export: `DataCropper`

Questa classe è specializzata nel "ritagliare" (crop) i dataset voluminosi per mantenere solo i dati pertinenti ai TOI definiti, riducendo il rumore e la dimensione dei file per le analisi successive.

### 2.1. Logica di Cropping

*   **Input:**
    *   Tabella TOI definita.
    *   Dataset completi (YOLO Raw, Gaze Mapped).
*   **Processo:**
    1.  Itera su ogni TOI.
    2.  Estrae le righe dei dataset originali dove `Start_TOI <= Timestamp <= End_TOI`.
    3.  Aggiunge metadati del TOI (Phase, Condition) alle righe estratte.
    4.  Concatena i risultati in un nuovo dataset "Cropped".

### 2.2. Generazione Output Appiattito

Il modulo esporta i dati ritagliati in formato CSV "Long" (appiattito) per facilitare l'analisi in R/Pandas, replicando la struttura dell'export di `hermes_human.py`.

*   **Trasformazione YOLO:**
    *   Input: JSON/CSV con struttura annidata o colonne multiple per keypoints.
    *   Output: `_video_yolo_CROPPED.csv`.
    *   Struttura: `Frame`, `Timestamp`, `TrackID`, `Box`, `Keypoints` (appiattiti), `Phase`, `Condition`.

## 3. Interfaccia Utente: `MasterToiView`

Gestisce l'interazione per la definizione manuale o assistita dei TOI.

### 3.1. Editor Tabellare

*   Permette all'utente di visualizzare la tabella dei TOI importata o generata.
*   Consente modifiche manuali a `Start`, `End` e etichette (`Phase`, `Condition`).

### 3.2. Visualizzazione Sincronizzazione

*   Mostra i timestamp degli eventi rilevati nei due log per permettere una verifica visiva dell'allineamento.
*   Permette di aggiustare manualmente l'offset se la sincronizzazione automatica fallisce.

---

### Sommario Flusso Dati

| Input | Processo | Output |
| :--- | :--- | :--- |
| **Log Tobii + Log Exp** | **Sincronizzazione**<br>Allineamento timestamp su eventi comuni | **Offset Temporale** |
| **Offset + Log Exp** | **Definizione Fasi**<br>Mappatura eventi su intervalli temporali | **Tabella TOI (.tsv)** |
| **Tabella TOI + Dati Raw** | **Data Cropper**<br>Filtraggio temporale e arricchimento | **Dataset Cropped (.csv)**<br>(YOLO, Gaze ridotti) |