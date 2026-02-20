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