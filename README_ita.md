# H.E.R.M.E.S. - Human-centric Eye-tracking & Robust Motion Estimation Suite

**HERMES** è un framework di ricerca modulare sviluppato per sincronizzare, analizzare e visualizzare dati di eye-tracking insieme all'estrazione cinematica basata su computer vision. La suite affronta la sfida metodologica di definire Aree di Interesse (AOI) dinamiche su target umani in movimento senza annotazione manuale.

Integrando la stima della posa basata su **YOLO** con rigorosi protocolli di sincronizzazione temporale, HERMES permette ai ricercatori di mappare i dati dello sguardo su regioni semantiche del corpo (es. Volto, Mani, Spazio Peripersonale) in contesti sperimentali complessi.

---

## 📥 Installazione e Setup

### Prerequisiti
*   **OS:** Windows 10/11 (64-bit).
*   **Hardware:** GPU NVIDIA raccomandata (supporto CUDA) per il modulo Human; l'inferenza CPU è supportata ma significativamente più lenta.
*   **Software:** Python 3.10+ (Lo script di setup gestisce l'installazione automatica delle dipendenze).

### Procedura di Installazione

1.  **Clona la repository:**
    ```bash
    git clone https://github.com/morenocuratelo/Hermes-v1
    ```
2.  **Esegui il Setup:**
    *   Naviga nella cartella del progetto.
    *   Esegui il file `SETUP_LAB.bat`.
    *   *Nota:* Questo script installerà `uv` (package manager), creerà un ambiente virtuale isolato, installerà le dipendenze e scaricherà i pesi dei modelli AI necessari.

3.  **Avvio:**
    *   Una volta completato il setup, avvia l'applicazione con `AVVIA_HERMES.bat`.

---

## 🚀 Flusso di Lavoro (Workflow)

Il software impone un flusso sequenziale per garantire l'integrità dei dati. Segui i moduli numerati nell'interfaccia:

1.  **Human (Pose Estimation):** Estrae lo scheletro e traccia le persone nel video.
2.  **Entity (Identity):** Assegna ruoli (es. "Target") ai tracciati anonimi e corregge errori.
3.  **Region (AOI Definition):** Definisce le regole geometriche per le AOI (es. Faccia = Naso + Occhi).
4.  **Master TOI (Sync):** Sincronizza i log dell'eye-tracker con il video e definisce le fasi temporali.
5.  **Eye Mapping:** Incrocia geometricamente lo sguardo con le AOI dinamiche.
6.  **Stats:** Genera report statistici e file Excel completi.

---

## ⚙️ Guida al Tuning e Configurazione Moduli

Questa sezione descrive come configurare e ottimizzare ogni modulo per le tue esigenze sperimentali.

### 1. Human (Kinematic Extraction)
Utilizza YOLO per estrarre 17 keypoints scheletrici.

*   **Parametri Chiave:**
    *   **Confidence Threshold (`CONF_THRESHOLD`):** Default `0.6`. Soglia conservativa per bilanciare precisione e richiamo. Abbassare se il soggetto non viene rilevato in condizioni di scarsa luce.
    *   **Tracker:** Supporta `BoT-SORT` (default) e `ByteTrack`.
    *   **Re-Identification (ReID):** Abilitare per ridurre gli scambi di ID (ID switch) quando i soggetti si incrociano. Richiede il download automatico di modelli extra (es. `resnet50`).
*   **Output:** Genera un file `.json.gz` (raw data) e un `.csv` appiattito con coordinate e confidenza per ogni keypoint.

### 2. Entity (Identity Assignment)
Interfaccia di post-processing per correggere errori di tracking.

*   **Funzionalità:**
    *   **Merge:** Unisce due tracciati frammentati (es. ID 5 diventa parte di ID 2).
    *   **Split:** Divide un tracciato in due se l'ID è saltato da una persona all'altra.
    *   **Auto-Stitch:** Tenta di unire automaticamente frammenti basandosi su prossimità spaziale e temporale.
*   **Tuning Auto-Stitch:**
    *   `Lookahead`: Quanti frame futuri cercare per un match.
    *   `Time Gap`: Massimo intervallo di tempo (sec) consentito per unire due tracce.
    *   `Stitch Dist`: Massima distanza in pixel tra la fine della traccia A e l'inizio della B.

### 3. Region (Dynamic AOIs)
Definisce le Aree di Interesse basandosi sui keypoints.

*   **Profili JSON:** Le regole sono salvate in `assets/profiles_aoi`. Esempio regola per "Volto":
    *   `kps`: [0, 1, 2, 3, 4] (Naso, Occhi, Orecchie).
    *   `margin_px`: Padding aggiunto al bounding box dei keypoints.
    *   `shape`: `box`, `circle`, `oval`, o `polygon`.
*   **Ghost Tracks:** Il sistema rileva automaticamente frame dove il tracking è perso ma presente nei frame adiacenti, permettendo di interpolare o copiare la posizione dell'AOI ("Force Add").

### 4. Master TOI (Synchronization)
Allinea flussi dati asincroni (Tobii vs Video).

*   **Logica di Sync:** Calcola un offset lineare basandosi su un evento comune (es. "VideoStart" nel log Tobii e nel log sperimentale).
*   **Data Cropping:** Una volta definiti i TOI (Time of Interest), il modulo può generare file `_CROPPED.csv` contenenti solo i dati pertinenti alle fasi di interesse, riducendo drasticamente la dimensione dei file per l'analisi statistica.

### 5. Eye Mapping
Esegue l'hit-testing geometrico.

*   **Logica:**
    1.  Carica le AOI generate dal modulo Region.
    2.  Converte il timestamp dello sguardo in frame video: `Frame = (Timestamp - Offset) * FPS`.
    3.  Verifica se il punto di sguardo cade dentro una o più AOI.
    4.  **Sovrapposizioni:** Se lo sguardo colpisce più AOI (es. Faccia dentro Corpo), vince l'AOI con l'**area minore** (più specifica).

### 6. Statistics
Genera il report finale.

*   **Metriche Calcolate:**
    *   **Duration:** Tempo totale trascorso nell'AOI.
    *   **Percentage:** % del tempo della fase trascorso nell'AOI.
    *   **Latency:** Tempo al primo ingresso nell'AOI dall'inizio della fase.
    *   **Glances:** Numero di volte che lo sguardo entra nell'AOI.
*   **Master Report:** Può generare un file Excel multi-foglio contenente Stats, Raw Data, Mapping, e configurazioni, pronto per l'archiviazione.

---

## 🛠 Architettura del Sistema

Il software è costruito su **Python 3.12** e utilizza **Tkinter** per l'interfaccia grafica, garantendo compatibilità nativa su Windows senza framework pesanti. Utilizza un'architettura "Hub & Spoke" gestita da un `AppContext` centrale che assicura la persistenza dello stato tra i moduli.

### Stack Tecnico
*   **GUI:** Tkinter / Tcl
*   **Computer Vision:** OpenCV, Ultralytics (YOLOv8/v11)
*   **Data Manipulation:** Pandas, NumPy, SciPy
*   **Packaging:** uv (environment), PyInstaller (distribuzione)

---

## 📄 Citazione e Disclaimer

Se utilizzi HERMES nella tua ricerca, fai riferimento alla documentazione interna del laboratorio per il formato di citazione appropriato.

**Disclaimer:** Questo software è fornito "così com'è" per scopi di ricerca. Assicurati di rispettare il GDPR e le linee guida etiche quando elabori dati video contenenti soggetti umani identificabili.


# HERMES - Entity Module Developer Guide

Questo documento descrive la logica interna, i flussi di dati e le trasformazioni implementate nel modulo **Entity** (`hermes_entity.py`). Il modulo è responsabile dell'assegnazione delle identità (Role) ai tracciati generati da YOLO, permettendo la correzione manuale e semi-automatica degli errori di tracking (frammentazione, ID switch).

## 1. Gestione della Memoria: `HistoryManager`

Per supportare operazioni distruttive come Merge e Split con funzionalità di Undo/Redo, il modulo implementa un gestore di stati ibrido RAM/Disco.

*   **Logica:**
    *   Mantiene uno stack di stati (`undo_stack`).
    *   Ogni stato è una copia profonda (pickle) dei dati dei tracciati.
    *   **RAM Buffer:** I primi N stati (default 5) sono mantenuti in RAM per accesso rapido.
    *   **Disk Spilling:** Gli stati più vecchi vengono serializzati su file temporanei su disco per evitare di saturare la memoria, specialmente con video lunghi e molti tracciati.
    *   **Cleanup:** Alla chiusura, i file temporanei vengono eliminati.

## 2. Logica di Manipolazione Tracciati: `IdentityLogic`

Questa classe gestisce la struttura dati principale e le operazioni algoritmiche.

### 2.1. Struttura Dati (`self.tracks`)

I dati non sono mantenuti come lista di frame (come in YOLO), ma aggregati per ID ("Track-Oriented").

*   **Struttura:** Dizionario `{ TrackID : TrackData }`
*   **TrackData:**
    *   `frames`: Lista ordinata dei frame in cui l'ID appare.
    *   `boxes`: Lista corrispondente delle bounding box `[x1, y1, x2, y2]`.
    *   `role`: Ruolo assegnato (es. "Target", "Ignore"). Default: "Ignore".
    *   `merged_from`: Lista di ID originali che sono stati fusi in questo tracciato.
*   **Lineage (`self.id_lineage`):** Mappa `{ Original_ID : Current_Master_ID }`. Fondamentale per l'export finale: permette di sapere che l'ID 5 di YOLO ora fa parte dell'ID 2.

### 2.2. Caricamento Dati (`load_from_json_gz`)

*   **Input:** File `.json.gz` generato dal modulo Human (YOLO).
*   **Trasformazione:**
    1.  Legge riga per riga (streaming).
    2.  **Gestione ID -1 (Untracked):** Se YOLO non ha assegnato un ID (detection isolata), viene generato un **ID Sintetico** univoco: `9000000 + (frame_idx * 1000) + detection_idx`. Questo rende ogni detection "untracked" un tracciato a sé stante manipolabile.
    3.  Aggrega le detection nel dizionario `self.tracks`.

### 2.3. Operazioni di Merge (`merge_logic`)

Unisce due tracciati ("Master" e "Slave") in uno solo.

*   **Logica:**
    1.  Trasferisce tutti i frame e box dallo Slave al Master.
    2.  Aggiorna `merged_from` del Master.
    3.  Aggiorna `id_lineage`: tutti gli ID che puntavano allo Slave ora puntano al Master.
    4.  Elimina la chiave dello Slave da `self.tracks`.
    5.  **Riordino:** Ordina le liste `frames` e `boxes` del Master per indice temporale.

### 2.4. Operazioni di Split (`split_track`)

Divide un tracciato in due parti in un punto specifico.

*   **Input:** `track_id`, `split_frame`, `keep_head` (booleano).
*   **Logica:**
    1.  Trova l'indice di taglio nelle liste `frames`.
    2.  Genera un `new_id` (Max ID esistente + 1).
    3.  Divide le liste `frames` e `boxes` in `head` (prima del taglio) e `tail` (dopo).
    4.  Se `keep_head` è True:
        *   L'ID originale mantiene la parte `head`.
        *   Il `new_id` riceve la parte `tail`.
    5.  Se `keep_head` è False (default per correzione ID switch):
        *   L'ID originale mantiene la parte `tail`.
        *   Il `new_id` riceve la parte `head`.

### 2.5. Algoritmi di Correzione Automatica

*   **Auto-Stitch (`auto_stitch`):**
    *   Tenta di unire frammenti consecutivi non assegnati.
    *   **Criteri:**
        1.  Gap temporale < `time_gap` (es. 2 secondi).
        2.  Distanza spaziale (Euclidea tra centro box finale A e iniziale B) < `stitch_dist`.
*   **Absorb Noise (`absorb_noise`):**
    *   Tenta di unire frammenti "Ignore" (rumore) ai tracciati principali ("Target", ecc.).
    *   Utile per recuperare arti persi o detection momentanee che appartengono al soggetto principale.
    *   Usa criteri di prossimità spaziale molto stretti.

## 3. Interfaccia Utente: `IdentityView`

Gestisce l'interazione visuale e la sincronizzazione tra Video, Timeline e Lista Tracciati.

### 3.1. Timeline Visiva (`_draw_timeline`)

Disegna una rappresentazione temporale dei tracciati.

*   **Rendering:**
    *   I tracciati "Ignore" sono disegnati come sfondo grigio.
    *   I tracciati assegnati a un Ruolo sono disegnati con il colore del ruolo.
    *   Ogni Ruolo ha una "corsia" (riga) dedicata per evitare sovrapposizioni visive.

### 3.2. Selezione Sincronizzata (`_on_video_click`)

Permette di selezionare un tracciato cliccando direttamente sul video.

*   **Logica:**
    1.  Riceve le coordinate click (x, y) sul widget video.
    2.  Le converte in coordinate video originali (gestendo il ridimensionamento/letterboxing dell'immagine).
    3.  Interroga `logic.get_track_at_point` per trovare quale ID possiede una bounding box che contiene quel punto nel frame corrente.
    4.  Seleziona l'ID corrispondente nella Treeview laterale.

## 4. Output Dati (`save_mapping`)

Genera il file finale di mappatura identità.

*   **File:** `_identity.json`
*   **Contenuto:** Un dizionario piatto `{ Original_YOLO_ID : "RoleName" }`.
*   **Trasformazione:**
    1.  Itera su `id_lineage`.
    2.  Per ogni ID originale, controlla chi è il suo "Master" attuale.
    3.  Se il Master ha un ruolo diverso da "Ignore", scrive la mappatura.
*   **Scopo:** Questo file viene usato dai moduli successivi (Region, Eye Mapping) per sapere che, ad esempio, l'ID 45, l'ID 46 e l'ID 98 sono tutti "Target".

---

### Nota sui File Autosave
Il modulo salva periodicamente lo stato in `hermes_autosave_identity.json` nella cartella del progetto per prevenire perdita di dati in caso di crash. Al riavvio, chiede se ripristinare.

# HERMES - Region Module Developer Guide

This document details the internal logic, parameters, and workflows of the **Region** module (Spatial AOI Definition). It is intended for developers or researchers wishing to modify the underlying scripts for dynamic Area of Interest generation.

## 1. Global Constants & Keypoints

The module relies on the standard COCO Keypoint format used by YOLO-Pose. These indices are mapped to anatomical names in `KEYPOINTS_MAP`.

```python
KEYPOINTS_MAP = {
    0: "Nose", 1: "L_Eye", 2: "R_Eye", 3: "L_Ear", 4: "R_Ear",
    5: "L_Shoulder", 6: "R_Shoulder", 7: "L_Elbow", 8: "R_Elbow",
    9: "L_Wrist", 10: "R_Wrist", 11: "L_Hip", 12: "R_Hip",
    13: "L_Knee", 14: "R_Knee", 15: "L_Ankle", 16: "R_Ankle"
}
```

## 2. Profile Management (`AOIProfileManager`)

Profiles define how raw keypoints are converted into semantic Areas of Interest (e.g., "Face", "Hands"). They are stored as JSON files in `assets/profiles_aoi`.

### 2.1. Profile Structure

A profile consists of **Roles** (mapped from the Entity module) and **Rules** for each role.

```json
{
    "name": "Invasion Profile",
    "roles": {
        "Target": [
            {
                "name": "Face",
                "shape": "box",
                "kps": [0, 1, 2, 3, 4],
                "margin_px": 30,
                "scale_w": 1.0,
                "scale_h": 1.0
            }
        ],
        "DEFAULT": [...]
    }
}
```

### 2.2. Rule Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **name** | String | Name of the AOI (e.g., "Face"). |
| **shape** | String | Geometry type: `box`, `circle`, `oval`, `polygon`. |
| **kps** | List[int] | Indices of keypoints used to calculate the base bounding box. |
| **margin_px** | Int | Padding added to the raw bounding box of keypoints. |
| **scale_w/h** | Float | Multiplier to expand/shrink width or height from the center. |
| **offset_y_bottom** | Int | Extra pixels added to the bottom edge (useful for torsos/legs). |

---

## 3. Logic Layer: `RegionLogic`

The `RegionLogic` class handles the geometric computations and state management, decoupled from the UI.

### 3.1. Geometry Engine (`calculate_shape`)

This is the core function that transforms keypoints into shapes.

1.  **Base Box Calculation:**
    *   Extracts valid keypoints (confidence > threshold).
    *   Computes min/max X and Y.
    *   Applies `margin_px`.
    *   Applies `scale_w` and `scale_h` relative to the center.
    *   Applies `offset_y_bottom`.

2.  **Shape Morphing:**
    *   **Box:** Returns the calculated rectangle.
    *   **Circle:** Calculates center and radius based on the box dimensions.
    *   **Oval:** Calculates center and semi-axes (rx, ry).
    *   **Polygon:** Maps the relative positions of keypoints from the source box to the expanded box, ordering points to form a convex hull-like shape.

### 3.2. Data Hierarchy & Overrides

The system uses a strict hierarchy to determine what is shown on screen for a given frame/track/AOI:

1.  **Level 0 (Base):** Automatic calculation from YOLO keypoints using Profile Rules.
2.  **Level 1 (Manual Override):** Explicit shapes stored in `self.manual_overrides`. These take precedence over Level 0.

**Storage Key:** `(frame_idx, track_id, role, aoi_name)`

### 3.3. Ghost Tracks (`find_ghost_tracks`)

*   **Function:** Identifies tracks that are missing in the current frame but present in neighbors.
*   **Logic:**
    *   Scans a window of +/- `ghost_window_var` frames.
    *   If a track exists in a neighbor but not in the current frame (and hasn't been manually overridden), it is returned as a "Ghost".
    *   **Purpose:** Allows the user to "Force Add" an AOI by copying the position from a nearby frame, useful for occlusions or detection failures.

### 3.4. Export (`export_csv`)

Generates the final dataset.

*   **Columns:** Frame, Timestamp, TrackID, Role, AOI, ShapeType, Coordinates (x1, y1, x2, y2), Geometric details (Radius, Angle, etc.), Corrected (Bool).
*   **Logic:** Iterates through all frames, applying the Profile Rules and Manual Overrides to generate the final geometry state.

---

## 4. Presentation Layer: `RegionView`

The UI orchestrates the workflow and visualization.

### 4.1. Manual Correction Mode

A stateful mode that enables editing tools.

*   **State Snapshot:** When entering, the current session state is saved to allow cancellation.
*   **Force Add:**
    *   **Auto/Ghost:** Uses `find_ghost_tracks` to seed the new box position.
    *   **Center:** Seeds the box at the screen center if no ghost is found.
    *   **Draw:** Allows drawing the box/circle directly on the video canvas.
*   **Interpolation:**
    *   **Linear:** Linearly interpolates box coordinates between two existing anchor frames (manual or automatic) over a selected scope.

### 4.2. Scopes

Operations (like applying a manual edit or interpolation) act on a specific scope:
*   **Frame:** Only the current frame.
*   **Current TOI:** All frames within the active Time Interval of Interest.
*   **Whole Video:** Every frame in the dataset.

### 4.3. Session Management

*   **Autosave:** State is saved to `_aoi_edit_session.json` on every commit action.
*   **Undo/Redo:** Implements a command pattern stack storing before/after states of overrides.

### 4.4. Profile Wizard

A GUI tool to generate JSON profiles without editing text files.
*   **Visual Feedback:** Lists available keypoints.
*   **Strategies:** Allows defining different logic for Targets vs. Non-Targets.

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

    # HERMES - Stats Module Developer Guide

Questo documento descrive la logica interna, i flussi di dati e le trasformazioni implementate nel modulo **Stats** (`hermes_stats.py`). Il modulo è responsabile dell'aggregazione dei dati di sguardo (Gaze) mappati sugli AOI e del calcolo delle metriche statistiche basate sulle finestre temporali (TOI).

## 1. Logica di Calcolo: `StatsLogic`

La classe `StatsLogic` incapsula il motore matematico. Non dipende dall'interfaccia grafica.

### 1.1. Calcolo Frequenza di Campionamento (`calculate_actual_sampling_rate`)

*   **Input:** DataFrame dei dati di sguardo (`df_gaze`) contenente la colonna `Timestamp`.
*   **Logica:**
    1.  Calcola le differenze temporali ($\Delta t$) tra righe consecutive: `df['Timestamp'].diff()`.
    2.  **Filtro Gap:** Ignora differenze superiori a 0.1s (100ms) per evitare che buchi nei dati (blink, perdita di tracking) falsino la media.
    3.  Calcola la media dei $\Delta t$ validi.
    4.  Frequenza ($Hz$) = $1.0 / \text{avg\_dt}$.
*   **Fallback:** Se i dati sono insufficienti, ritorna 50.0 Hz di default.
*   **Scopo:** Fondamentale per convertire il *numero di campioni* in *tempo (secondi)*.

### 1.2. Generazione Dataset Raw (`generate_raw_dataset`)

Questa funzione arricchisce il file "Mapped" (livello campione) con le informazioni contestuali dei TOI (livello fase).

*   **Input:**
    *   `mapped_path`: CSV generato dal modulo Eye Mapping.
    *   `toi_path`: TSV generato dal modulo Master TOI.
*   **Trasformazione:**
    1.  **Caricamento:** Legge i file in DataFrame pandas.
    2.  **Ordinamento:** Ordina i dati di sguardo per `Timestamp` (critico per l'efficienza).
    3.  **Inizializzazione:** Aggiunge colonne vuote `Phase`, `Condition`, `Trial` al DataFrame Gaze.
    4.  **Iterazione TOI:** Per ogni riga nel file TOI (che rappresenta una fase temporale):
        *   Estrae `Start` e `End`.
        *   **Binary Search:** Usa `np.searchsorted` sui timestamp dello sguardo per trovare istantaneamente gli indici di inizio (`idx_start`) e fine (`idx_end`) nel DataFrame Gaze che corrispondono alla finestra temporale.
        *   **Assegnazione Vettoriale:** Assegna i valori di `Phase`, `Condition` e `Trial` a tutte le righe nel range `[idx_start:idx_end]` in un colpo solo.
*   **Output:** Un DataFrame dove ogni singolo campionamento di sguardo sa a quale fase sperimentale appartiene.

### 1.3. Motore di Analisi Statistica (`run_analysis`)

È il cuore del modulo. Incrocia i dati spaziali (AOI hit) con quelli temporali (TOI).

*   **Input:** File Mapped, File TOI, Frequenza (opzionale), Flag formato (Wide/Long).
*   **Flusso Dati:**
    1.  **Validazione:** Verifica la presenza delle colonne essenziali (`Hit_Role`, `Hit_AOI`, `Timestamp` nel Gaze; `Start`, `End` nel TOI).
    2.  **Setup Frequenza:** Se l'utente non forza una frequenza, la calcola usando `calculate_actual_sampling_rate`.
        *   `sample_dur` = $1.0 / \text{freq}$.
    3.  **Discovery Combinazioni:** Scansiona l'intero file Gaze per trovare tutte le coppie uniche `(Hit_Role, Hit_AOI)` esistenti (es. "Target_Face", "Confederate_Hand"). Questo assicura che nel report finale ci siano colonne per tutte le AOI, anche se in una specifica fase non vengono mai guardate (valore 0).
    4.  **Loop Fasi (TOI):** Itera su ogni intervallo temporale definito nel TOI.
        *   **Slicing:** Estrae il sottoinsieme di campioni Gaze che cadono nel tempo della fase (`t_start` -> `t_end`).
        *   **Metriche Generali Fase:**
            *   `Gaze_Samples_Total`: Numero campioni nel subset.
            *   `Gaze_Valid_Time`: `Samples * sample_dur`.
            *   `Tracking_Ratio`: `Valid_Time / (t_end - t_start)`.
        *   **Calcolo Metriche per AOI:** Raggruppa il subset per `Hit_Role` e `Hit_AOI`.
            *   **Duration:** `Count * sample_dur`.
            *   **Percentage:** `Duration / Phase_Duration`.
            *   **Latency:** `Timestamp_Primo_Hit - t_start`. Se non ci sono hit, è vuoto.
            *   **Glances (Sguardi):** Conta quante volte lo sguardo *entra* nell'AOI.
                *   Logica: `(Current == AOI) AND (Previous != AOI)`.
    5.  **Formattazione Output:**
        *   **Wide Format (Classico):** Una riga per Fase. Le metriche delle AOI sono colonne aggiunte orizzontalmente (es. `Target_Face_Dur`, `Target_Face_Perc`, `Target_Face_Lat`).
        *   **Long Format (Tidy):** Una riga per ogni combinazione Fase-AOI. Colonne fisse: `Phase`, `Condition`, `Hit_Role`, `Hit_AOI`, `Duration`, `Percentage`, ecc.

## 2. Generazione Master Report (`export_master_report`)

Questa funzione crea un file Excel complesso contenente tutti i dati dell'esperimento.

*   **Input:** Un dizionario di DataFrame (`data_frames_dict`) contenente Stats, Raw Data, Mapping, AOI, Identity, YOLO, TOI.
*   **Logica:**
    1.  Usa `xlsxwriter` come engine.
    2.  Per ogni DataFrame nel dizionario:
        *   Crea un foglio **Legenda** (`L - NomeFoglio`) usando un dizionario statico di descrizioni (`legends_dict`).
        *   Crea il foglio **Dati** (`NomeFoglio`) scrivendo il DataFrame.
        *   Applica auto-fit alla larghezza delle colonne per leggibilità.
*   **Integrazione Dati:**
    *   Il controller (`GazeStatsView`) tenta di caricare automaticamente i file correlati basandosi sulla convenzione dei nomi (es. se il file mapped è `P01_MAPPED.csv`, cerca `P01_AOI.csv`, `P01_video_yolo.csv`, etc.) per popolare il dizionario.

## 3. Interfaccia Utente: `GazeStatsView`

Gestisce l'orchestrazione dei thread per non bloccare la UI durante i calcoli pesanti.

*   **Thread Worker:**
    1.  Esegue `run_analysis` per ottenere le statistiche.
    2.  Se richiesto, esegue `generate_raw_dataset`.
    3.  Se richiesto "Master Report", raccoglie tutti i file CSV/JSON ausiliari dalla cartella del progetto e chiama `export_master_report`.
    4.  Altrimenti, salva i singoli CSV (`_STATS.csv` e opzionalmente `_RAW.csv`).

---

### Sommario Trasformazioni

| Input | Processo | Output |
| :--- | :--- | :--- |
| **Gaze Mapped (.csv)**<br>(Timestamp, X, Y, Hit_AOI) | **Slicing Temporale**<br>Taglio basato su Start/End del TOI | **Subset Gaze**<br>(Campioni specifici per la fase) |
| **Subset Gaze** | **Aggregazione**<br>Count, Sum(Duration), Min(Timestamp) | **Metriche AOI**<br>(Durata, Latenza, Glances) |
| **Metriche AOI** | **Pivoting (Wide)**<br>Flattening delle chiavi AOI in colonne | **Stats Row**<br>(Phase, Cond, Face_Dur, Hand_Dur...) |
| **Tutti i DataFrame** | **Excel Writer**<br>Merge in fogli multipli + Legende | **Master Report (.xlsx)** |