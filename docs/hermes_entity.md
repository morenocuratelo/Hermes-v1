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

La funzione di salvataggio genera ora due output distinti per garantire sia la compatibilità con i moduli successivi che la portabilità dei dati.

### 4.1. Mappa Identità (`_identity.json`)
*   **Contenuto:** Un dizionario piatto `{ Original_YOLO_ID : "RoleName" }`.
*   **Trasformazione:**
    1.  Itera su `id_lineage`.
    2.  Per ogni ID originale, controlla chi è il suo "Master" attuale.
    3.  Se il Master ha un ruolo diverso da "Ignore", scrive la mappatura.
*   **Scopo:** Questo file viene usato dai moduli successivi (Region, Eye Mapping) per sapere che, ad esempio, l'ID 45, l'ID 46 e l'ID 98 sono tutti "Target".

### 4.2. Dati Arricchiti (`_enriched.json.gz`)
*   **Contenuto:** Una copia esatta del file di tracking originale (YOLO), arricchita con metadati semantici iniettati direttamente in ogni detection.
*   **Campi Aggiunti:**
    *   `role`: Il ruolo assegnato all'identità (es. "Target", "Confederate").
    *   `master_id`: L'ID consolidato finale (utile per tracciare i merge effettuati).
*   **Processo:**
    1.  Apre il file originale in lettura (streaming) e il nuovo file in scrittura.
    2.  Per ogni detection, risolve l'ID (gestendo anche gli ID sintetici per detection non tracciate `9000000+`).
    3.  Consulta la `id_lineage` per recuperare il `master_id` e il `role` corrente.
    4.  Scrive il nuovo JSON compresso.
*   **Vantaggio:** Rende il dataset autoconsistente, contenendo sia i dati cinematici (keypoints) che quelli semantici (ruoli) in un unico file, facilitando l'analisi esterna senza dipendenze dalla mappa di identità.

---

### Nota sui File Autosave
Il modulo salva periodicamente lo stato in `hermes_autosave_identity.json` nella cartella del progetto per prevenire perdita di dati in caso di crash. Al riavvio, chiede se ripristinare.