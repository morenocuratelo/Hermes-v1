import gzip
import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
# MODEL — Stats Logic (Calculation Engine)
# ═══════════════════════════════════════════════════════════════════


class StatsLogic:
    """
    Engine per il calcolo delle statistiche di Eye-Tracking.
    """

    def __init__(self):
        self._cancel_flag = False

    def cancel(self):
        self._cancel_flag = True

    def calculate_actual_sampling_rate(self, df_gaze):
        """
        Calcola la frequenza di campionamento reale dai timestamp.
        Utile se l'utente sbaglia a inserire gli Hz o se ci sono drop.
        """
        if len(df_gaze) < 2:
            return 50.0
        # Calcola la differenza media tra timestamp consecutivi
        diffs = df_gaze["Timestamp"].diff().dropna()
        # Filtra gap troppo grandi (buchi dati) per non falsare la media (>100ms)
        valid_diffs = diffs[diffs < 0.1]
        if valid_diffs.empty:
            return 50.0

        avg_dt = valid_diffs.mean()
        if avg_dt == 0:
            return 50.0
        return 1.0 / avg_dt

    def generate_raw_dataset(self, mapped_path, toi_path, progress_callback=None):
        """
        Genera un dataset 'Raw' (Sample-level) arricchito con le info del TOI.
        Ogni riga è un campionamento, con colonne Phase, Condition, Trial aggiunte.
        """
        if progress_callback:
            progress_callback("Generating Raw Dataset...")

        try:
            df_gaze = pd.read_csv(mapped_path)
            df_toi = pd.read_csv(toi_path, sep="\t")

            # Sort e reset index fondamentali per slicing posizionale
            df_gaze = df_gaze.sort_values("Timestamp").reset_index(drop=True)

            # Init colonne
            df_gaze["Phase"] = "None"
            df_gaze["Phase"] = df_gaze["Phase"].astype(object)
            df_gaze["Condition"] = "None"
            df_gaze["Condition"] = df_gaze["Condition"].astype(object)

            if "Trial" in df_toi.columns:
                df_gaze["Trial"] = np.nan
                df_gaze["Trial"] = df_gaze["Trial"].astype(object)

            # Numpy array per ricerca veloce
            gaze_ts = df_gaze["Timestamp"].to_numpy(dtype=np.float64)

            total = len(df_toi)
            for i, (_, row) in enumerate(df_toi.iterrows()):
                if self._cancel_flag:
                    raise InterruptedError("Stopped by user")
                if progress_callback and i % 50 == 0:
                    progress_callback(f"Raw Data: Mapping TOI {i+1}/{total}...")

                t_start = float(row["Start"])
                t_end = float(row["End"])

                idx_start = np.searchsorted(gaze_ts, t_start)
                idx_end = np.searchsorted(gaze_ts, t_end)

                if idx_end > idx_start:
                    # Assegna valori alle righe nel range (usando iloc per slicing posizionale)
                    # Cast a int per soddisfare il type checker (get_loc può ritornare slice/array)
                    col_phase = int(df_gaze.columns.get_loc("Phase"))  # type: ignore
                    col_cond = int(df_gaze.columns.get_loc("Condition"))  # type: ignore

                    # Cast to string to ensure compatibility if Phase/Condition are numeric in TOI
                    df_gaze.iloc[idx_start:idx_end, col_phase] = str(row["Phase"])
                    df_gaze.iloc[idx_start:idx_end, col_cond] = str(row["Condition"])
                    if "Trial" in df_toi.columns:
                        col_trial = int(df_gaze.columns.get_loc("Trial"))  # type: ignore
                        df_gaze.iloc[idx_start:idx_end, col_trial] = row["Trial"]

            return df_gaze
        except Exception as e:
            raise ValueError(f"Error generating Raw Data: {e}") from e

    def run_analysis(self, mapped_path, toi_path, user_freq=None, progress_callback=None, long_format=False):
        """
        Esegue l'analisi incrociando Gaze Data (Mapped) e TOI (Time Windows).
        Restituisce un DataFrame con una riga per ogni fase (TOI).
        """
        self._cancel_flag = False

        # 1. Caricamento Dati
        if progress_callback:
            progress_callback("Loading files...")

        # Carica Gaze (MAPPED)
        try:
            df_gaze = pd.read_csv(mapped_path)
            # FIX: Verifica immediata delle colonne necessarie
            required_cols = ["Hit_Role", "Hit_AOI", "Timestamp"]
            missing = [c for c in required_cols if c not in df_gaze.columns]
            if missing:
                raise ValueError(
                    f"Il file CSV manca delle colonne: {missing}. Verifica lo step di Mapping o il separatore CSV."
                )

            df_gaze = df_gaze.sort_values("Timestamp").reset_index(drop=True)

            # --- FIX PYLANCE 2: Conversione esplicita a numpy float64 ---
            gaze_ts = df_gaze["Timestamp"].to_numpy(dtype=np.float64)

        except Exception as e:
            raise ValueError(f"Error loading Mapped CSV: {e}") from e

        # Carica TOI (TSV)
        try:
            df_toi = pd.read_csv(toi_path, sep="\t")
            required_toi = ["Start", "End", "Condition", "Phase"]
            # Controllo case-insensitive delle colonne per robustezza
            cols_upper = [c.upper() for c in df_toi.columns]
            req_upper = [c.upper() for c in required_toi]

            if not all(r in cols_upper for r in req_upper):
                raise ValueError(f"TOI file missing columns. Required: {required_toi}")
        except Exception as e:
            raise ValueError(f"Error loading TOI TSV: {e}") from e

        # 2. Calcolo Frequenza
        # Usiamo quella reale calcolata dai dati, o quella utente se forzata/fallback
        real_freq = self.calculate_actual_sampling_rate(df_gaze)
        freq = user_freq if user_freq and user_freq > 0 else real_freq
        sample_dur = 1.0 / freq

        if progress_callback:
            progress_callback(f"Frequency detected: {real_freq:.1f}Hz (Using {freq:.1f}Hz)")

        # Check for Fixation data columns (Output from I-VT Filter)
        has_fixations = "EventType" in df_gaze.columns
        has_groups = "grp" in df_gaze.columns
        if has_fixations and progress_callback:
            progress_callback("Fixation data detected (EventType). Calculating fixation metrics...")

        # 3. Analisi Per-Fase
        results = []
        total_phases = len(df_toi)

        # Pre-calcoliamo tutte le AOI uniche presenti nel file per creare colonne coerenti
        # Es: Target_Face, Target_Hands, Confederate_Face...
        unique_combinations = df_gaze[["Hit_Role", "Hit_AOI"]].drop_duplicates().dropna()

        # --- FIX PYLANCE 1: Usa enumerate per avere un indice intero sicuro ---
        for i, (_orig_idx, row) in enumerate(df_toi.iterrows()):
            if self._cancel_flag:
                raise InterruptedError("Stopped by user")

            if progress_callback and i % 10 == 0:
                progress_callback(f"Analyzing phase {i+1}/{total_phases}...")

            t_start = float(row["Start"])
            t_end = float(row["End"])
            phase_dur = t_end - t_start

            # --- FAST SLICING (Binary Search) ---
            # numpy.searchsorted ora riceve array float64 espliciti, risolvendo l'errore di tipo
            idx_start = np.searchsorted(gaze_ts, t_start)
            idx_end = np.searchsorted(gaze_ts, t_end)

            # Slice del dataframe
            subset = df_gaze.iloc[idx_start:idx_end]

            # Dati Base (copia tutto ciò che c'era nel TOI)
            res_row = row.to_dict()
            res_row["Gaze_Samples_Total"] = len(subset)
            res_row["Gaze_Valid_Time"] = len(subset) * sample_dur
            res_row["Tracking_Ratio"] = (len(subset) * sample_dur) / phase_dur if phase_dur > 0 else 0

            # --- METRICHE DINAMICHE ---
            if not subset.empty:
                # Conta occorrenze per (Ruolo, AOI)
                counts = subset.groupby(["Hit_Role", "Hit_AOI"]).size()

                # Latency Logic: Trova il primo timestamp per ogni Role/AOI
                first_hits = subset.groupby(["Hit_Role", "Hit_AOI"])["Timestamp"].min()
            else:
                counts = pd.Series(dtype=float)
                first_hits = pd.Series(dtype=float)

            # Itera su tutte le combinazioni possibili trovate nel file intero (per avere colonne fisse anche se 0)
            for _, comb in unique_combinations.iterrows():
                r, a = comb["Hit_Role"], comb["Hit_AOI"]
                if r == "None" or r == "Ignore":
                    continue  # Skip rumore

                key_base = f"{r}_{a}"  # Es: Target_Face

                # --- Calcolo Metriche Comuni ---
                # Usa .get con un default sicuro se la combinazione non esiste in questo subset
                c_val = counts.get((r, a), 0)
                duration = c_val * sample_dur
                perc = duration / phase_dur if phase_dur > 0 else 0

                # Latency
                latency = None
                if (r, a) in first_hits.index:
                    first_ts = first_hits.get((r, a))
                    if first_ts is not None:
                        latency = first_ts - t_start

                # Glances
                glances = 0
                if c_val > 0:
                    mask = (subset["Hit_Role"] == r) & (subset["Hit_AOI"] == a)
                    glances = (mask & ~mask.shift(1).fillna(False)).sum()

                # --- NEW METRICS ---
                # 5. Avg Glance Duration
                avg_glance_dur = duration / glances if glances > 0 else 0.0

                # 6. Fixation Metrics (if available from I-VT Filter)
                fix_dur = 0.0
                fix_count = 0
                ttff = None

                if has_fixations:
                    # Filter for Fixation samples on this AOI
                    mask_fix = (
                        (subset["Hit_Role"] == r) & (subset["Hit_AOI"] == a) & (subset["EventType"] == "Fixation")
                    )
                    fix_subset = subset[mask_fix]

                    if not fix_subset.empty:
                        fix_dur = len(fix_subset) * sample_dur

                        if has_groups:
                            fix_count = fix_subset["grp"].nunique()
                        else:
                            fix_count = (mask_fix & ~mask_fix.shift(1).fillna(False)).sum()

                        first_fix_ts = fix_subset["Timestamp"].min()
                        ttff = first_fix_ts - t_start

                # --- Output Formatting ---
                if long_format:
                    # LONG FORMAT (Tidy Data): Una riga per ogni combinazione (Fase, Ruolo, AOI)
                    # Copia i dati base della fase
                    row_long = res_row.copy()
                    # Rimuovi metriche aggregate se presenti (opzionale, qui non ci sono ancora)

                    row_long["Hit_Role"] = r
                    row_long["Hit_AOI"] = a
                    row_long["Duration"] = duration
                    row_long["Percentage"] = perc
                    row_long["Latency"] = latency
                    row_long["Glances"] = glances
                    row_long["Avg_Glance_Dur"] = avg_glance_dur
                    if has_fixations:
                        row_long["Fixation_Dur"] = fix_dur
                        row_long["Fixation_Count"] = fix_count
                        row_long["TTFF"] = ttff

                    results.append(row_long)

                else:
                    # WIDE FORMAT (Classic): Una riga per Fase, colonne estese

                    # 1. Total Duration
                    res_row[f"{key_base}_Dur"] = duration

                    # 2. Percentage of Phase
                    res_row[f"{key_base}_Perc"] = perc

                    # 3. Latency (Time to First Fixation)
                    res_row[f"{key_base}_Latency"] = latency

                    # 4. Glance Count
                    res_row[f"{key_base}_Glances"] = glances

                    # 5. Avg Glance Duration
                    res_row[f"{key_base}_Avg_Glance_Dur"] = avg_glance_dur

                    if has_fixations:
                        res_row[f"{key_base}_Fix_Dur"] = fix_dur
                        res_row[f"{key_base}_Fix_Count"] = fix_count
                        res_row[f"{key_base}_TTFF"] = ttff

            if not long_format:
                results.append(res_row)

        if not results:
            return None

        # Crea DataFrame Finale
        df_results = pd.DataFrame(results)

        # Riordina colonne per pulizia (TOI info first, then Metrics)
        cols = list(df_results.columns)
        # Identifica le colonne base (dal TOI o generali)
        base_cols = [
            c for c in cols if c in df_toi.columns or c in ["Gaze_Samples_Total", "Gaze_Valid_Time", "Tracking_Ratio"]
        ]

        if long_format:
            # Per Long format aggiungiamo le chiavi di raggruppamento alle base cols
            base_cols.extend(["Hit_Role", "Hit_AOI"])
            # Rimuovi duplicati mantenendo ordine
            base_cols = list(dict.fromkeys(base_cols))

        # Le altre sono metriche
        metric_cols = sorted([c for c in cols if c not in base_cols])

        return df_results[base_cols + metric_cols]

    def export_master_report(self, output_path, data_frames_dict):
        """
        Genera il Master Report Excel alternando fogli dati e fogli legenda.
        Richiede 'xlsxwriter' installato (pip install xlsxwriter).
        """
        # Check if we have at least one valid dataframe
        if not any(df is not None for df in data_frames_dict.values()):
            return

        # Definizione statica delle legende per garantire la coerenza formale con il template
        _kp_names = [
            "Nose",
            "L_Eye",
            "R_Eye",
            "L_Ear",
            "R_Ear",
            "L_Shoulder",
            "R_Shoulder",
            "L_Elbow",
            "R_Elbow",
            "L_Wrist",
            "R_Wrist",
            "L_Hip",
            "R_Hip",
            "L_Knee",
            "R_Knee",
            "L_Ankle",
            "R_Ankle",
        ]
        _kp_cols = [f"{kp}_{axis}" for kp in _kp_names for axis in ["X", "Y", "C"]]
        _kp_descs = [
            f"{kp}: {desc}"
            for kp in _kp_names
            for desc in [
                "Coordinata X in pixel del keypoint.",
                "Coordinata Y in pixel del keypoint.",
                "Confidenza (0.0-1.0) del keypoint rilevato.",
            ]
        ]

        _yolo_base_cols = [
            "Frame",
            "Timestamp",
            "TrackID",
            "Conf",
            "Box_X1",
            "Box_Y1",
            "Box_X2",
            "Box_Y2",
        ]
        _yolo_base_descs = [
            "Indice del frame video di riferimento.",
            "Timestamp in secondi dall'inizio del video.",
            "Identificativo numerico del soggetto tracciato da YOLO.",
            "Confidenza del detector YOLO per la detection (0.0-1.0).",
            "Coordinata X del vertice alto-sinistra della bounding box.",
            "Coordinata Y del vertice alto-sinistra della bounding box.",
            "Coordinata X del vertice basso-destra della bounding box.",
            "Coordinata Y del vertice basso-destra della bounding box.",
        ]

        _yolo_legend = pd.DataFrame(
            {
                "Column": _yolo_base_cols + _kp_cols,
                "Description": _yolo_base_descs + _kp_descs,
            }
        )

        legends_dict = {
            # 1. HUMAN (Kinematic Extraction)
            "YOLO": _yolo_legend,
            # 2. MASTER TOI (Sync & Cut)
            "TOI": pd.DataFrame(
                {
                    "Column": [
                        "Name",
                        "Start",
                        "End",
                        "Duration",
                        "Trial",
                        "Condition",
                        "Phase",
                    ],
                    "Description": [
                        "Nome univoco del TOI/fase sperimentale.",
                        "Tempo di inizio del TOI in secondi.",
                        "Tempo di fine del TOI in secondi.",
                        "Durata del TOI in secondi (End - Start).",
                        "Indice o etichetta del trial sperimentale.",
                        "Condizione sperimentale associata al campione/fase.",
                        "Nome della fase sperimentale.",
                    ],
                }
            ),
            "YOLO_Cropped": _yolo_legend,
            # 3. ENTITY (Identity Assignment)
            "Identity": pd.DataFrame(
                {
                    "Column": ["TrackID", "Role"],
                    "Description": [
                        "Identificativo numerico del soggetto tracciato da YOLO.",
                        "Ruolo semantico del soggetto (es. Target, Confederate).",
                    ],
                }
            ),
            "Enriched": pd.DataFrame(
                {
                    "Column": [
                        "Frame",
                        "Timestamp",
                        "TrackID",
                        "MasterID",
                        "Role",
                        "Conf",
                        "Box_X1",
                        "Box_Y1",
                        "Box_X2",
                        "Box_Y2",
                    ]
                    + _kp_cols,
                    "Description": [
                        "Indice del frame video di riferimento.",
                        "Timestamp in secondi dall'inizio del video.",
                        "Identificativo numerico originale (YOLO).",
                        "Identificativo numerico consolidato (dopo merge/split).",
                        "Ruolo semantico assegnato (es. Target).",
                        "Confidenza del detector YOLO.",
                        "Coordinata X del vertice alto-sinistra della bounding box.",
                        "Coordinata Y del vertice alto-sinistra della bounding box.",
                        "Coordinata X del vertice basso-destra della bounding box.",
                        "Coordinata Y del vertice basso-destra della bounding box.",
                    ]
                    + _kp_descs,
                }
            ),
            # 4. REGION (AOI Definition)
            "AOI": pd.DataFrame(
                {
                    "Column": [
                        "Frame",
                        "Timestamp",
                        "TrackID",
                        "Role",
                        "AOI",
                        "ShapeType",
                        "ShapePoints",
                        "CenterX",
                        "CenterY",
                        "Radius",
                        "RadiusX",
                        "RadiusY",
                        "Angle",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "Corrected",
                    ],
                    "Description": [
                        "Indice del frame video di riferimento.",
                        "Timestamp in secondi dall'inizio del video.",
                        "Identificativo numerico del soggetto tracciato da YOLO.",
                        "Ruolo semantico del soggetto (es. Target, Confederate).",
                        "Area of Interest anatomica (es. Face, Hands, Torso).",
                        "Tipo geometrico AOI (box, polygon, ellipse, circle).",
                        "Coordinate serializzate dei punti della forma AOI.",
                        "Coordinata X del centro (forme circolari/ellittiche).",
                        "Coordinata Y del centro (forme circolari/ellittiche).",
                        "Raggio (forme circolari).",
                        "Semiasse/raggio X (forme ellittiche).",
                        "Semiasse/raggio Y (forme ellittiche).",
                        "Angolo di rotazione della forma (gradi).",
                        "Coordinata X del vertice alto-sinistra AOI box.",
                        "Coordinata Y del vertice alto-sinistra AOI box.",
                        "Coordinata X del vertice basso-destra AOI box.",
                        "Coordinata Y del vertice basso-destra AOI box.",
                        "1 se AOI corretta manualmente, 0 se solo automatica.",
                    ],
                }
            ),
            # 5. EYE MAPPING
            "Mapping": pd.DataFrame(
                {
                    "Column": [
                        "Timestamp",
                        "Frame_Est",
                        "Gaze_X",
                        "Gaze_Y",
                        "Hit_Role",
                        "Hit_AOI",
                        "Hit_TrackID",
                        "Hit_Shape",
                        "Hit_x1",
                        "Hit_y1",
                        "Hit_x2",
                        "Hit_y2",
                        "Raw_Gaze2D_X",
                        "Raw_Gaze2D_Y",
                    ],
                    "Description": [
                        "Timestamp in secondi dall'inizio del video.",
                        "Frame stimato a partire dal timestamp del campione gaze.",
                        "Coordinata X gaze sul frame video (pixel).",
                        "Coordinata Y gaze sul frame video (pixel).",
                        "Ruolo del soggetto colpito dal gaze; vuoto se background.",
                        "AOI specifica colpita dal gaze (es. Face).",
                        "TrackID del soggetto colpito dal gaze.",
                        "Tipo geometrico della AOI colpita.",
                        "Limite sinistro AOI colpita.",
                        "Limite alto AOI colpita.",
                        "Limite destro AOI colpita.",
                        "Limite basso AOI colpita.",
                        "Coordinata gaze normalizzata X (0-1) dal tracker.",
                        "Coordinata gaze normalizzata Y (0-1) dal tracker.",
                    ],
                }
            ),
            # 6. GAZE FILTERS (I-VT)
            "Filtered_Gaze": pd.DataFrame(
                {
                    "Column": [
                        "Timestamp",
                        "Gaze_X",
                        "Gaze_Y",
                        "Velocity",
                        "EventType",
                    ],
                    "Description": [
                        "Timestamp del campione (secondi).",
                        "Coordinata X dello sguardo (pixel), filtrata/interpolata.",
                        "Coordinata Y dello sguardo (pixel), filtrata/interpolata.",
                        "Velocità angolare istantanea (°/s).",
                        "Classificazione I-VT (Fixation, Saccade, Unknown).",
                    ],
                }
            ),
            "Fixations": pd.DataFrame(
                {
                    "Column": [
                        "start",
                        "end",
                        "duration",
                        "x",
                        "y",
                        "count",
                    ],
                    "Description": [
                        "Timestamp di inizio della fissazione (secondi).",
                        "Timestamp di fine della fissazione (secondi).",
                        "Durata della fissazione in millisecondi.",
                        "Coordinata X media del centroide della fissazione (pixel).",
                        "Coordinata Y media del centroide della fissazione (pixel).",
                        "Numero di campioni di sguardo inclusi nella fissazione.",
                    ],
                }
            ),
            # 7. ANALYTICS
            "Stats_Raw": pd.DataFrame(
                {
                    "Column": [
                        "Timestamp",
                        "Frame_Est",
                        "Gaze_X",
                        "Gaze_Y",
                        "Hit_Role",
                        "Hit_AOI",
                        "Hit_TrackID",
                        "Hit_Shape",
                        "Hit_x1",
                        "Hit_y1",
                        "Hit_x2",
                        "Hit_y2",
                        "Raw_Gaze2D_X",
                        "Raw_Gaze2D_Y",
                        "Phase",
                        "Condition",
                        "Trial",
                    ],
                    "Description": [
                        "Timestamp in secondi dall'inizio del video.",
                        "Frame stimato a partire dal timestamp del campione gaze.",
                        "Coordinata X gaze sul frame video (pixel).",
                        "Coordinata Y gaze sul frame video (pixel).",
                        "Ruolo del soggetto colpito dal gaze; vuoto se background.",
                        "AOI specifica colpita dal gaze (es. Face).",
                        "TrackID del soggetto colpito dal gaze.",
                        "Tipo geometrico della AOI colpita.",
                        "Limite sinistro AOI colpita.",
                        "Limite alto AOI colpita.",
                        "Limite destro AOI colpita.",
                        "Limite basso AOI colpita.",
                        "Coordinata gaze normalizzata X (0-1) dal tracker.",
                        "Coordinata gaze normalizzata Y (0-1) dal tracker.",
                        "Nome della fase sperimentale.",
                        "Condizione sperimentale associata al campione/fase.",
                        "Indice o etichetta del trial sperimentale.",
                    ],
                }
            ),
            "Stats_Summary": pd.DataFrame(
                {
                    "Column": [
                        "Name",
                        "Start",
                        "End",
                        "Duration",
                        "Trial",
                        "Condition",
                        "Phase",
                        "Gaze_Samples_Total",
                        "Gaze_Valid_Time",
                        "Tracking_Ratio",
                        "Hit_Role",
                        "Hit_AOI",
                        "Glances",
                        "Latency",
                        "Percentage",
                        "Avg_Glance_Dur",
                        "Fixation_Dur",
                        "Fixation_Count",
                        "TTFF",
                    ],
                    "Description": [
                        "Nome univoco del TOI/fase sperimentale.",
                        "Tempo di inizio del TOI in secondi.",
                        "Tempo di fine del TOI in secondi.",
                        "Durata del TOI in secondi (End - Start).",
                        "Indice o etichetta del trial sperimentale.",
                        "Condizione sperimentale associata al campione/fase.",
                        "Nome della fase sperimentale.",
                        "Numero totale campioni gaze nella fase.",
                        "Tempo (s) con gaze valido nella fase.",
                        "Qualità tracking = Gaze_Valid_Time / Duration.",
                        "Ruolo del soggetto colpito dal gaze; vuoto se background.",
                        "AOI specifica colpita dal gaze (es. Face).",
                        "Numero di ingressi dello sguardo nella AOI.",
                        "Secondi dall'inizio fase alla prima fissazione AOI.",
                        "Percentuale del tempo fase passato sulla AOI.",
                        "Durata media di uno sguardo (Duration / Glances).",
                        "Tempo totale di fissazione (somma campioni Fixation).",
                        "Numero di fissazioni distinte.",
                        "Time To First Fixation (latenza prima fissazione).",
                    ],
                }
            ),
        }

        try:
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                for sheet_name, df in data_frames_dict.items():
                    if df is not None:
                        # 1. Scrivi il foglio della Legenda (L - NomeFoglio)
                        legend_sheet_name = f"L - {sheet_name}"
                        if sheet_name in legends_dict:
                            legends_dict[sheet_name].to_excel(writer, sheet_name=legend_sheet_name, index=False)

                        # 2. Scrivi il foglio dei Dati
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

                        # (Opzionale) Auto-fit delle colonne per leggibilità
                        worksheet = writer.sheets[sheet_name]
                        for i, col in enumerate(df.columns):
                            try:
                                # Handle empty dataframe and non-string column headers (e.g. float)
                                max_val_len = 0
                                if not df.empty:
                                    series = df[col]
                                    # Check if series is actually a Series (not DataFrame due to duplicate cols)
                                    if isinstance(series, pd.Series):
                                        max_val_len = series.astype(str).map(len).max()
                                        if pd.isna(max_val_len):
                                            max_val_len = 0

                                col_header_len = len(str(col))
                                column_len = max(int(max_val_len), int(col_header_len)) + 2
                                worksheet.set_column(i, i, min(column_len, 80))
                            except Exception:
                                worksheet.set_column(i, i, 15)
                    else:
                        # Gestione file mancante
                        pd.DataFrame({"Status": ["File mancante!"]}).to_excel(
                            writer, sheet_name=sheet_name, index=False
                        )
                        worksheet = writer.sheets[sheet_name]
                        worksheet.set_column(0, 0, 30)
        except Exception as e:
            raise ValueError(f"Error generating Master Excel: {e}") from e


# ═══════════════════════════════════════════════════════════════════
# VIEW — UI Logic
# ═══════════════════════════════════════════════════════════════════


class GazeStatsView:
    def __init__(self, parent, context):
        self.parent = parent
        self.context = context
        self.logic = StatsLogic()

        # Variabili File
        self.mapped_csv_path = tk.StringVar()
        self.toi_tsv_path = tk.StringVar()

        # Variabili UI
        self.gaze_freq = tk.DoubleVar(value=0.0)  # 0 = Auto-detect
        self.var_raw = tk.BooleanVar(value=False)
        self.var_long = tk.BooleanVar(value=False)
        self.var_master = tk.BooleanVar(value=False)  # NUOVA VARIABILE per esportazione Master Report

        self.status_var = tk.StringVar(value="Ready.")

        self._build_ui()
        self._auto_load()

    def _auto_load(self):
        # 1. Cerca il file MAPPED (Output del Tab 5)
        if hasattr(self.context, "mapped_csv_path") and self.context.mapped_csv_path:
            if os.path.exists(self.context.mapped_csv_path):
                self.mapped_csv_path.set(self.context.mapped_csv_path)

        # 2. Cerca il file TOI (Output del Tab 4)
        if hasattr(self.context, "toi_path") and self.context.toi_path:
            if os.path.exists(self.context.toi_path):
                self.toi_tsv_path.set(self.context.toi_path)

    def _build_ui(self):
        tk.Label(self.parent, text="6. Statistics & Reporting", font=("Segoe UI", 18, "bold"), bg="white").pack(
            pady=(0, 10), anchor="w"
        )

        main = tk.Frame(self.parent, padx=20, pady=20, bg="white")
        main.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            main, text="Gaze Analyzer: Generate Scientific Report", font=("Segoe UI", 16, "bold"), bg="white"
        ).pack(pady=(0, 20))

        # 1. Input Files
        lf_in = tk.LabelFrame(main, text="1. Input Files", padx=10, pady=10, bg="white")
        lf_in.pack(fill=tk.X, pady=5)

        self._add_picker(lf_in, "MAPPED CSV (from Step 5):", self.mapped_csv_path, "*.csv")
        self._add_picker(lf_in, "TOI TSV (from Step 4):", self.toi_tsv_path, "*.tsv")

        # 2. Settings
        lf_set = tk.LabelFrame(main, text="2. Analysis Settings", padx=10, pady=10, bg="white")
        lf_set.pack(fill=tk.X, pady=5)

        f_freq = tk.Frame(lf_set, bg="white")
        f_freq.pack(fill=tk.X)
        tk.Label(f_freq, text="Forced Frequency (Hz):", bg="white").pack(side=tk.LEFT)
        tk.Entry(f_freq, textvariable=self.gaze_freq, width=8).pack(side=tk.LEFT, padx=5)
        tk.Label(f_freq, text="(Leave 0 for Auto-Detect from timestamps)", fg="gray", bg="white").pack(side=tk.LEFT)

        # Checkboxes
        f_chk = tk.Frame(lf_set, bg="white")
        f_chk.pack(fill=tk.X, pady=5)
        tk.Checkbutton(
            f_chk, text="Export Raw Data (Sample-level with Phase info)", variable=self.var_raw, bg="white"
        ).pack(anchor="w")
        tk.Checkbutton(
            f_chk, text="Use Long Format (Tidy Data) for Stats Report", variable=self.var_long, bg="white"
        ).pack(anchor="w")
        tk.Checkbutton(
            f_chk,
            text="Generate Full Excel MASTER REPORT (Requires all source files in same folder)",
            variable=self.var_master,
            fg="darkred",
            bg="white",
        ).pack(anchor="w", pady=(5, 0))

        # 3. Action
        self.btn_run = tk.Button(
            main,
            text="GENERATE FULL REPORT",
            bg="#4CAF50",
            fg="white",
            font=("Bold", 12),
            height=2,
            command=self.run_analysis_thread,
        )
        self.btn_run.pack(fill=tk.X, pady=20)

        # 4. Status & Progress
        self.progress = ttk.Progressbar(main, mode="indeterminate")
        self.progress.pack(fill=tk.X)

        tk.Label(main, textvariable=self.status_var, bg="white", fg="blue").pack(pady=5)

    def _add_picker(self, p, lbl, var, ft):
        f = tk.Frame(p, bg="white")
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=lbl, width=25, anchor="w", bg="white").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(f, text="...", width=3, command=lambda: self.browse(var, ft)).pack(side=tk.LEFT)

    def browse(self, var, ft):
        f = filedialog.askopenfilename(filetypes=[("File", ft)])
        if f:
            var.set(f)

    def _ask_user_for_file(self, label, expected_path, filetypes):
        """
        Thread-safe method to ask user for a missing file.
        Returns the path string or None.
        """
        result: list[Any] = [None]
        event = threading.Event()

        def _ask():
            if messagebox.askyesno(
                "File mancante",
                f"Impossibile trovare il file '{label}'.\n\nPercorso atteso:\n"
                f"{os.path.basename(expected_path)}\n\nVuoi selezionarlo manualmente?",
                parent=self.parent,
            ):
                path = filedialog.askopenfilename(title=f"Seleziona {label}", filetypes=filetypes, parent=self.parent)
                if path:
                    result[0] = path
            event.set()

        self.parent.after(0, _ask)
        event.wait()
        return result[0]

    def _load_and_flatten_enriched_json(self, path):
        if not os.path.exists(path):
            return None

        rows = []
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    f_idx = frame.get("f_idx")
                    ts = frame.get("ts")

                    for det in frame.get("det", []):
                        row = {
                            "Frame": f_idx,
                            "Timestamp": ts,
                            "TrackID": det.get("track_id"),
                            "MasterID": det.get("master_id"),
                            "Role": det.get("role"),
                            "Conf": det.get("conf"),
                        }
                        box = det.get("box", {})
                        row.update(
                            {
                                "Box_X1": box.get("x1"),
                                "Box_Y1": box.get("y1"),
                                "Box_X2": box.get("x2"),
                                "Box_Y2": box.get("y2"),
                            }
                        )
                        kps = det.get("keypoints", [])
                        kp_names = [
                            "Nose",
                            "L_Eye",
                            "R_Eye",
                            "L_Ear",
                            "R_Ear",
                            "L_Shoulder",
                            "R_Shoulder",
                            "L_Elbow",
                            "R_Elbow",
                            "L_Wrist",
                            "R_Wrist",
                            "L_Hip",
                            "R_Hip",
                            "L_Knee",
                            "R_Knee",
                            "L_Ankle",
                            "R_Ankle",
                        ]
                        for i, kp in enumerate(kp_names):
                            if i < len(kps):
                                p = kps[i]
                                row[f"{kp}_X"] = p[0]
                                row[f"{kp}_Y"] = p[1]
                                row[f"{kp}_C"] = p[2] if len(p) > 2 else 0
                            else:
                                row[f"{kp}_X"] = 0
                                row[f"{kp}_Y"] = 0
                                row[f"{kp}_C"] = 0
                        rows.append(row)
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"Error loading enriched JSON: {e}")
            return None

    # ── Threading ──────────────────────────────────────────────────

    def run_analysis_thread(self):
        mapped = self.mapped_csv_path.get()
        toi = self.toi_tsv_path.get()

        if not mapped or not toi:
            messagebox.showwarning("Error", "Please select both MAPPED CSV and TOI TSV files.")
            return

        if not os.path.exists(mapped) or not os.path.exists(toi):
            messagebox.showerror("Error", "One or more files do not exist.")
            return

        self.btn_run.config(state="disabled")
        self.progress.start(10)
        self.status_var.set("Initializing analysis...")

        freq: float | None = self.gaze_freq.get()
        # Ensure 0.0 or negative is treated as None for auto-detect
        if freq is not None and freq <= 0.0:
            freq = None

        def worker():
            try:
                # 1. Stats Analysis
                df_stats = self.logic.run_analysis(
                    mapped, toi, user_freq=freq, progress_callback=self._update_status, long_format=self.var_long.get()
                )

                # 2. Raw Data (Optional)
                df_raw = None
                if self.var_raw.get():
                    df_raw = self.logic.generate_raw_dataset(mapped, toi, progress_callback=self._update_status)

                if df_stats is not None:
                    if self.var_master.get():
                        # --- LOGICA MASTER REPORT ---
                        self.parent.after(0, lambda: self.status_var.set("Compiling Master Report..."))
                        base_dir = os.path.dirname(mapped)
                        # Assumiamo che il file mapped si chiami es: BWWW_TD_Inv_11_GC_MAPPED.csv
                        # Ricaviamo il prefisso pulito (per AOI/YOLO) e quello gaze (per Fixations)
                        base_name = os.path.basename(mapped)
                        if "_gaze_MAPPED.csv" in base_name:
                            clean_prefix = base_name.replace("_gaze_MAPPED.csv", "")
                            gaze_prefix = clean_prefix + "_gaze"
                        else:
                            clean_prefix = base_name.replace("_MAPPED.csv", "")
                            gaze_prefix = clean_prefix

                        # Caricamento dinamico dei file extra (con fallback a None se non trovati)
                        dfs: dict[str, Any] = {}
                        dfs["Stats_Summary"] = df_stats
                        dfs["Stats_Raw"] = (
                            df_raw
                            if df_raw is not None
                            else (self.logic.generate_raw_dataset(mapped, toi) if not self.var_raw.get() else None)
                        )
                        dfs["Mapping"] = pd.read_csv(mapped) if os.path.exists(mapped) else None
                        dfs["TOI"] = pd.read_csv(toi, sep="\t") if os.path.exists(toi) else None

                        # Helper per caricamento interattivo
                        def smart_load(label, path, loader, ftypes) -> Any:
                            if os.path.exists(path):
                                try:
                                    return loader(path)
                                except Exception:
                                    return None

                            # Se non esiste, chiedi all'utente
                            user_path = self._ask_user_for_file(label, path, ftypes)
                            if user_path and os.path.exists(user_path):
                                try:
                                    return loader(user_path)
                                except Exception:
                                    return None
                            return None

                        # Auto-discovery degli altri file basati sul prefisso
                        aoi_path = os.path.join(base_dir, f"{clean_prefix}_AOI.csv")
                        identity_path = os.path.join(base_dir, f"{clean_prefix}_video_yolo_CROPPED_identity.json")
                        yolo_cropped_path = os.path.join(base_dir, f"{clean_prefix}_video_yolo_CROPPED.csv")

                        # Tentativo di caricare enriched (preferenza per CROPPED se esiste)
                        enriched_path = os.path.join(base_dir, f"{clean_prefix}_video_yolo_CROPPED_enriched.json.gz")
                        if not os.path.exists(enriched_path):
                            fallback = os.path.join(base_dir, f"{clean_prefix}_video_yolo_enriched.json.gz")
                            if os.path.exists(fallback):
                                enriched_path = fallback

                        dfs["AOI"] = smart_load("AOI Data", aoi_path, pd.read_csv, [("CSV", "*.csv")])
                        dfs["Identity"] = smart_load("Identity Map", identity_path, pd.read_json, [("JSON", "*.json")])
                        dfs["Enriched"] = smart_load(
                            "Enriched Data",
                            enriched_path,
                            self._load_and_flatten_enriched_json,
                            [("JSON GZ", "*.json.gz")],
                        )

                        # Caricamento del file YOLO Cropped (ora in formato CSV appiattito)
                        dfs["YOLO_Cropped"] = smart_load(
                            "YOLO Cropped", yolo_cropped_path, pd.read_csv, [("CSV", "*.csv")]
                        )

                        # Il file YOLO raw base (generato automaticamente come CSV)
                        yolo_raw_path = os.path.join(base_dir, f"{clean_prefix}_video_yolo.csv")
                        dfs["YOLO"] = smart_load("YOLO Raw", yolo_raw_path, pd.read_csv, [("CSV", "*.csv")])

                        # Caricamento Output Filtri I-VT (se presenti)
                        fix_path = os.path.join(base_dir, f"{gaze_prefix}_FIXATIONS.csv")
                        filt_path = os.path.join(base_dir, f"{gaze_prefix}_FILTERED.csv")
                        dfs["Fixations"] = smart_load("Fixations", fix_path, pd.read_csv, [("CSV", "*.csv")])
                        dfs["Filtered_Gaze"] = smart_load("Filtered Gaze", filt_path, pd.read_csv, [("CSV", "*.csv")])

                        default_excel_name = os.path.join(base_dir, f"{gaze_prefix}_MASTER_REPORT.xlsx")
                        self.parent.after(0, lambda: self._save_master_results(dfs, default_excel_name))

                    else:
                        # --- LOGICA STANDARD CSV ---
                        default_name = mapped.replace("_MAPPED.csv", "_FINAL_STATS.csv")
                        if default_name == mapped:
                            default_name += "_stats.csv"
                        self.parent.after(0, lambda: self._save_results(df_stats, df_raw, default_name))

            except Exception as e:
                import traceback

                traceback.print_exc()
                # FIX: Salviamo il messaggio in una variabile locale prima di passarlo alla lambda
                error_message = str(e)
                self.parent.after(0, lambda: self._show_error(error_message))

        threading.Thread(target=worker, daemon=True).start()

    def _update_status(self, msg):
        self.parent.after(0, lambda: self.status_var.set(msg))

    def _save_results(self, df_stats, df_raw, default_path):
        self.progress.stop()
        out = filedialog.asksaveasfilename(
            initialfile=os.path.basename(default_path), defaultextension=".csv", filetypes=[("CSV Report", "*.csv")]
        )

        if out:
            try:
                # Save Stats
                df_stats.to_csv(out, index=False)
                msg = f"Stats Report saved: {len(df_stats)} rows.\nPath: {out}"

                # Save Raw if present
                if df_raw is not None:
                    out_raw = out.replace(".csv", "_RAW.csv")
                    df_raw.to_csv(out_raw, index=False)
                    msg += f"\n\nRaw Data saved: {len(df_raw)} rows.\nPath: {out_raw}"

                messagebox.showinfo("Success", msg)
            except Exception as e:
                messagebox.showerror("Save Error", str(e))

        self._reset_ui()

    def _save_master_results(self, dfs_dict, default_path):
        self.progress.stop()
        out = filedialog.asksaveasfilename(
            initialfile=os.path.basename(default_path),
            defaultextension=".xlsx",
            filetypes=[("Excel Master Report", "*.xlsx")],
        )
        if out:
            try:
                self.logic.export_master_report(out, dfs_dict)

                # Also export CSVs automatically
                base = os.path.splitext(out)[0]
                if base.endswith("_MASTER_REPORT"):
                    base = base[:-14]

                msg_extra = ""
                if "Stats_Summary" in dfs_dict and dfs_dict["Stats_Summary"] is not None:
                    p = f"{base}_FINAL_STATS.csv"
                    dfs_dict["Stats_Summary"].to_csv(p, index=False)
                    msg_extra += f"\n- {os.path.basename(p)}"
                if "Stats_Raw" in dfs_dict and dfs_dict["Stats_Raw"] is not None:
                    p = f"{base}_FINAL_STATS_RAW.csv"
                    dfs_dict["Stats_Raw"].to_csv(p, index=False)
                    msg_extra += f"\n- {os.path.basename(p)}"

                messagebox.showinfo(
                    "Success", f"Master Report saved successfully.\nPath: {out}\n\nAlso exported CSVs:{msg_extra}"
                )
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save Master Report: {str(e)}")

        self._reset_ui()

    def _show_error(self, msg):
        self.progress.stop()
        self.status_var.set("Error occurred.")
        messagebox.showerror("Analysis Failed", msg)
        self._reset_ui()

    def _reset_ui(self):
        self.btn_run.config(state="normal")
        self.progress.stop()
        self.status_var.set("Ready.")
