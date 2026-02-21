# HERMES - Stats Module Developer Guide

Questo documento descrive la logica interna del modulo **Stats** (`hermes_stats.py`), responsabile dell'aggregazione dei dati e della generazione del report scientifico finale.

## 1. Logica di Calcolo: `StatsLogic`

### 1.1. Sampling Rate
Il modulo calcola automaticamente la frequenza di campionamento reale analizzando i timestamp del file di input (`1 / avg_delta_time`). Questo rende il calcolo delle durate robusto anche in presenza di frame drop o frequenze variabili.

### 1.2. Generazione Dataset Raw
La funzione `generate_raw_dataset` arricchisce il file "Mapped" (livello campione) con le informazioni contestuali dei TOI (livello fase).
*   Usa `np.searchsorted` per un mapping ultra-veloce dei timestamp alle finestre temporali.
*   Aggiunge le colonne `Phase`, `Condition`, `Trial` a ogni riga di sguardo.

### 1.3. Motore di Analisi (`run_analysis`)
Incrocia i dati spaziali (AOI hit) con quelli temporali (TOI) e, opzionalmente, con i dati di fissazione (I-VT).

#### Metriche Standard (Sample-based)
*   **Duration:** Tempo totale trascorso nell'AOI (`Count * Sample_Duration`).
*   **Percentage:** `Duration / Phase_Duration`.
*   **Latency:** Tempo dal `TOI_Start` al primo campione valido nell'AOI.
*   **Glances:** Numero di ingressi distinti nell'AOI (transizioni In -> Out).
*   **Avg Glance Duration:** `Duration / Glances`.

#### Metriche Avanzate (Fixation-based)
Se il file di input contiene la colonna `EventType` (generata dal modulo Filters):
*   **Fixation Duration:** Somma delle durate dei campioni classificati come 'Fixation' che cadono nell'AOI. (Più preciso della Duration standard perché esclude le saccadi di passaggio).
*   **Fixation Count:** Numero di eventi di fissazione unici (`grp` ID) sull'AOI.
*   **TTFF (Time To First Fixation):** Latenza della prima fissazione stabile sull'AOI.

## 2. Formati di Output

### 2.1. Wide Format (Classico)
Una riga per ogni Fase (TOI). Le metriche delle AOI sono appiattite in colonne.
*   Esempio: `Phase_1`, `Condition_A`, `Target_Face_Dur`, `Target_Face_FixCount`, `Target_Hand_Dur`...

### 2.2. Long Format (Tidy Data)
Una riga per ogni combinazione Fase-AOI. Ideale per analisi in R (ggplot2) o Python (seaborn).
*   Colonne: `Phase`, `Condition`, `Hit_Role`, `Hit_AOI`, `Duration`, `Fixation_Count`, etc.

## 3. Master Report (`export_master_report`)

Genera un file Excel (`.xlsx`) complesso che funge da archivio unico per il soggetto.

**Struttura Fogli:**
1.  **Stats_Summary:** Il report statistico calcolato.
2.  **Stats_Raw:** Il dataset raw arricchito (opzionale).
3.  **Mapping:** I dati di eye-tracking mappati originali.
4.  **TOI:** La definizione delle fasi temporali.
5.  **AOI:** Le definizioni geometriche delle aree di interesse.
6.  **Identity:** La mappa delle identità (TrackID -> Ruolo).
7.  **YOLO / Enriched:** I dati cinematici grezzi.
8.  **Legende (L - ...):** Fogli descrittivi per ogni tabella dati.

## 4. Integrazione Automatica
Il modulo tenta di caricare automaticamente tutti i file ausiliari (`_AOI.csv`, `_identity.json`, `_FIXATIONS.csv`) basandosi sulla convenzione di nomenclatura dei file nella cartella di output, permettendo la generazione del Master Report con un solo click.