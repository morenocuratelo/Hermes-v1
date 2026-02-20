# HERMES - Eye Mapping Module Developer Guide

Questo documento descrive la logica interna, i flussi di dati e le trasformazioni implementate nel modulo **Eye Mapping** (`hermes_eye.py`). Il modulo è responsabile dell'incrocio geometrico tra i dati di sguardo (Gaze) e le Aree di Interesse (AOI) definite dinamicamente frame per frame.

## 1. Logica di Calcolo: `GazeLogic`

La classe `GazeLogic` gestisce la matematica del mapping spaziale e la sincronizzazione temporale.

### 1.1. Caricamento e Indicizzazione AOI (`load_aoi_data`)

*   **Input:** File CSV generato dal modulo Region (contenente bounding box per ogni frame).
*   **Trasformazione:**
    1.  Carica il CSV in un DataFrame pandas.
    2.  Identifica dinamicamente la colonna ID (`ID` o `TrackID`).
    3.  **Raggruppamento:** Raggruppa le righe per `Frame`.
    4.  **Indicizzazione:** Crea un dizionario (Hash Map) `{ frame_index : [lista_di_aoi_dict] }`.
*   **Scopo:** Permette l'accesso istantaneo (O(1)) a tutte le AOI attive in un dato frame durante lo streaming dei dati di sguardo.

### 1.2. Hit-Testing Geometrico (`calculate_hit`)

Questa è la funzione core che determina "cosa sta guardando il soggetto".

*   **Input:** Coordinate sguardo in pixel (x, y), Lista AOI nel frame corrente.
*   **Logica:**
    1.  Itera su tutte le AOI presenti nel frame.
    2.  Per ogni AOI, verifica l'intersezione geometrica in base alla forma (`_shape_hit_and_area`):
        *   **Box:** Semplice controllo dei limiti `x1 <= x <= x2` e `y1 <= y <= y2`.
        *   **Cerchio:** Distanza euclidea dal centro <= raggio.
        *   **Poligono:** Algoritmo Ray Casting (`_point_in_polygon`).
    3.  **Risoluzione Sovrapposizioni:** Se lo sguardo cade su più AOI contemporaneamente (es. "Faccia" dentro "Corpo"), viene selezionata l'AOI con l'**area minore**. Questo garantisce la massima specificità (es. hit su "Occhio" vince su "Faccia").
*   **Output:** L'oggetto AOI vincitore o `None` (sguardo sul background).

### 1.3. Sincronizzazione Temporale (`timestamp_to_frame`)

Converte il tempo assoluto dell'eye-tracker nel tempo relativo del video.

*   **Formula:** `Frame = int((Timestamp_Gaze - Sync_Offset) * FPS)`
*   **Parametri:**
    *   `Timestamp_Gaze`: Tempo in secondi dal file Tobii.
    *   `Sync_Offset`: Delta temporale per allineare l'inizio del video con l'inizio della registrazione eye-tracking.
    *   `FPS`: Frame rate del video.

### 1.4. Pipeline di Mapping (`run_mapping`)

Orchestra l'intero processo in modalità streaming per gestire file di grandi dimensioni senza saturare la RAM.

*   **Input:** File AOI (CSV), File Gaze (JSON.GZ), Risoluzione Video, FPS, Offset.
*   **Flusso Dati:**
    1.  **Caricamento AOI:** Esegue `load_aoi_data` per avere la mappa spaziale in memoria.
    2.  **Streaming Gaze:** Apre il file `.gz` e legge riga per riga.
    3.  **Parsing:**
        *   Ignora pacchetti non validi o senza coordinate `gaze2d`.
        *   Estrae `timestamp` e coordinate normalizzate `(gx, gy)` [0.0 - 1.0].
    4.  **Conversione Spaziale:**
        *   `Pixel_X = gx * Video_Width`
        *   `Pixel_Y = gy * Video_Height`
    5.  **Conversione Temporale:** Calcola il `frame_idx` usando la formula di sincronizzazione.
    6.  **Hit-Test:** Recupera le AOI per `frame_idx` ed esegue `calculate_hit`.
    7.  **Accumulo:** Salva il risultato (Hit/Miss, Role, AOI, TrackID) in una lista buffer.
    8.  **Export:** Scrive il file finale `_MAPPED.csv`.

## 2. Output Dati (`_MAPPED.csv`)

Il file generato contiene una riga per ogni campionamento dell'eye-tracker mappato sul video.

| Colonna | Descrizione |
| :--- | :--- |
| **Timestamp** | Tempo originale dell'eye-tracker. |
| **Frame_Est** | Frame video stimato corrispondente. |
| **Gaze_X, Gaze_Y** | Coordinate dello sguardo in pixel sul video. |
| **Hit_Role** | Ruolo colpito (es. "Target", "Confederate"). "None" se miss. |
| **Hit_AOI** | Nome AOI colpita (es. "Face", "Hands"). "None" se miss. |
| **Hit_TrackID** | ID numerico del soggetto colpito. -1 se miss. |
| **Hit_Shape** | Forma geometrica colpita (box, circle, polygon). |

## 3. Interfaccia Utente

### 3.1. `GazeView`

Pannello di configurazione per lanciare il processo.

*   **Input:** Selettori file per AOI e Gaze Data.
*   **Parametri:** Risoluzione video (default 1920x1080), FPS, Offset di sincronizzazione.
*   **Thread:** Esegue `run_mapping` in un thread separato (`_thread_worker`) per mantenere la UI responsiva e mostra una progress bar indeterminata (poiché la lettura stream non conosce la lunghezza totale a priori).

### 3.2. `GazeResultPlayer`

Player video dedicato alla verifica qualitativa del mapping.

*   **Funzione:** Carica il video e il CSV mappato appena generato.
*   **Visualizzazione:**
    *   Disegna il punto di sguardo (cerchio giallo/rosso).
    *   Se c'è un HIT, mostra il nome dell'AOI colpita in sovraimpressione.
    *   Permette di navigare frame per frame per verificare la precisione della sincronizzazione e del tracking geometrico.
    *   **Ottimizzazione:** Indicizza il CSV in memoria (`data_map`) per accesso rapido durante il playback video.