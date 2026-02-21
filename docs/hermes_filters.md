# HERMES - Gaze Filters Module Developer Guide

Questo documento descrive la logica interna del modulo **Gaze Filters** (`hermes_filters.py`), che implementa l'algoritmo standard I-VT (Velocity-Threshold Identification) per la classificazione degli eventi oculomotori (Fissazioni vs Saccadi).

## 1. Logica di Filtro: `FilterLogic`

La classe implementa una pipeline sequenziale ispirata a Tobii Pro Lab.

### 1.1. Pre-processing
*   **Gap Fill:** Interpolazione lineare dei campioni mancanti (NaN) fino a un massimo di `max_gap_ms` (default 75ms).
*   **Noise Reduction:** Applicazione di un filtro (Moving Median o Moving Average) su una finestra mobile (default size 3) per ridurre il rumore ad alta frequenza prima del calcolo della velocità.

### 1.2. Calcolo Velocità Angolare
Il modulo supporta due modalità di calcolo a seconda dei dati disponibili:

1.  **Vettoriale 3D (Accurato):** Se i dati `Gaze_3D` sono presenti (es. Tobii Glasses 3), calcola l'angolo tra vettori di sguardo consecutivi usando il prodotto scalare.
    *   $\theta = \arccos(\vec{v}_i \cdot \vec{v}_{i-1})$
2.  **Geometrico 2D (Fallback):**
    *   **Wearable:** Stima l'angolo basandosi sul FOV della scene camera (default 95°x63°).
    *   **Screen-Based:** Stima l'angolo basandosi sulle dimensioni fisiche dello schermo e la distanza dell'utente (configurabili).

La velocità istantanea è calcolata come: $v = \theta / \Delta t$ (°/s).

### 1.3. Classificazione I-VT
*   Ogni campione con $v < \text{velocity\_threshold}$ (default 100°/s per Glasses 3) è classificato come **Fixation**.
*   Altrimenti è **Saccade**.
*   I campioni mancanti rimangono **Unknown**.

### 1.4. Post-processing (Merge & Discard)
Per evitare la frammentazione delle fissazioni dovuta a rumore o micro-saccadi:

1.  **Merge Adjacent:** Unisce due fissazioni consecutive se:
    *   Intervallo temporale < `merge_max_time` (75ms).
    *   Distanza angolare < `merge_max_angle` (0.5°).
2.  **Discard Short:** Elimina le fissazioni che durano meno di `discard_min_dur` (60ms), riclassificandole come Unknown/Saccade.

## 2. Output Dati

### 2.1. `_FIXATIONS.csv`
Tabella degli eventi discreti.
*   `start`, `end`: Timestamp inizio/fine.
*   `duration`: Durata in ms.
*   `x`, `y`: Coordinate del centroide della fissazione (media dei campioni).
*   `count`: Numero di campioni raw inclusi.

### 2.2. `_FILTERED.csv`
Dataset raw arricchito.
*   Mantiene le colonne originali.
*   Aggiunge: `Velocity` (°/s), `EventType` (Fixation/Saccade/Unknown), `grp` (ID gruppo evento).

Questo file viene utilizzato dal modulo **Stats** per calcolare metriche avanzate come il *Time To First Fixation* (TTFF).