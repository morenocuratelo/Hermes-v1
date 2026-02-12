import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import os
import threading

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
        if len(df_gaze) < 2: return 50.0
        # Calcola la differenza media tra timestamp consecutivi
        diffs = df_gaze['Timestamp'].diff().dropna()
        # Filtra gap troppo grandi (buchi dati) per non falsare la media (>100ms)
        valid_diffs = diffs[diffs < 0.1] 
        if valid_diffs.empty: return 50.0
        
        avg_dt = valid_diffs.mean()
        if avg_dt == 0: return 50.0
        return 1.0 / avg_dt

    def generate_raw_dataset(self, mapped_path, toi_path, progress_callback=None):
        """
        Genera un dataset 'Raw' (Sample-level) arricchito con le info del TOI.
        Ogni riga è un campionamento, con colonne Phase, Condition, Trial aggiunte.
        """
        if progress_callback: progress_callback("Generating Raw Dataset...")
        
        try:
            df_gaze = pd.read_csv(mapped_path)
            df_toi = pd.read_csv(toi_path, sep='\t')
            
            # Sort e reset index fondamentali per slicing posizionale
            df_gaze = df_gaze.sort_values('Timestamp').reset_index(drop=True)
            
            # Init colonne
            df_gaze['Phase'] = 'None'
            df_gaze['Phase'] = df_gaze['Phase'].astype(object)
            df_gaze['Condition'] = 'None'
            df_gaze['Condition'] = df_gaze['Condition'].astype(object)
            
            if 'Trial' in df_toi.columns:
                df_gaze['Trial'] = np.nan
                df_gaze['Trial'] = df_gaze['Trial'].astype(object)
            
            # Numpy array per ricerca veloce
            gaze_ts = df_gaze['Timestamp'].to_numpy(dtype=np.float64)
            
            total = len(df_toi)
            for i, (_, row) in enumerate(df_toi.iterrows()):
                if self._cancel_flag: raise InterruptedError("Stopped by user")
                if progress_callback and i % 50 == 0:
                    progress_callback(f"Raw Data: Mapping TOI {i+1}/{total}...")
                
                t_start = float(row['Start'])
                t_end = float(row['End'])
                
                idx_start = np.searchsorted(gaze_ts, t_start)
                idx_end = np.searchsorted(gaze_ts, t_end)
                
                if idx_end > idx_start:
                    # Assegna valori alle righe nel range (usando iloc per slicing posizionale)
                    # Cast a int per soddisfare il type checker (get_loc può ritornare slice/array)
                    col_phase = int(df_gaze.columns.get_loc('Phase')) # type: ignore
                    col_cond = int(df_gaze.columns.get_loc('Condition')) # type: ignore
                    
                    # Cast to string to ensure compatibility if Phase/Condition are numeric in TOI
                    df_gaze.iloc[idx_start:idx_end, col_phase] = str(row['Phase'])
                    df_gaze.iloc[idx_start:idx_end, col_cond] = str(row['Condition'])
                    if 'Trial' in df_toi.columns:
                        col_trial = int(df_gaze.columns.get_loc('Trial')) # type: ignore
                        df_gaze.iloc[idx_start:idx_end, col_trial] = row['Trial']
                        
            return df_gaze
        except Exception as e:
            raise ValueError(f"Error generating Raw Data: {e}")

    def run_analysis(self, mapped_path, toi_path, user_freq=None, progress_callback=None, long_format=False):
        """
        Esegue l'analisi incrociando Gaze Data (Mapped) e TOI (Time Windows).
        Restituisce un DataFrame con una riga per ogni fase (TOI).
        """
        self._cancel_flag = False
        
        # 1. Caricamento Dati
        if progress_callback: progress_callback("Loading files...")
        
        # Carica Gaze (MAPPED)
        try:
            df_gaze = pd.read_csv(mapped_path)
            # FIX: Verifica immediata delle colonne necessarie
            required_cols = ['Hit_Role', 'Hit_AOI', 'Timestamp']
            missing = [c for c in required_cols if c not in df_gaze.columns]
            if missing:
                raise ValueError(f"Il file CSV manca delle colonne: {missing}. Verifica lo step di Mapping o il separatore CSV.")

            df_gaze = df_gaze.sort_values('Timestamp').reset_index(drop=True)
            
            # --- FIX PYLANCE 2: Conversione esplicita a numpy float64 ---
            gaze_ts = df_gaze['Timestamp'].to_numpy(dtype=np.float64)
            
        except Exception as e:
            raise ValueError(f"Error loading Mapped CSV: {e}")

        # Carica TOI (TSV)
        try:
            df_toi = pd.read_csv(toi_path, sep='\t')
            required_toi = ['Start', 'End', 'Condition', 'Phase']
            # Controllo case-insensitive delle colonne per robustezza
            cols_upper = [c.upper() for c in df_toi.columns]
            req_upper = [c.upper() for c in required_toi]
            
            if not all(r in cols_upper for r in req_upper):
                raise ValueError(f"TOI file missing columns. Required: {required_toi}")
        except Exception as e:
            raise ValueError(f"Error loading TOI TSV: {e}")

        # 2. Calcolo Frequenza
        # Usiamo quella reale calcolata dai dati, o quella utente se forzata/fallback
        real_freq = self.calculate_actual_sampling_rate(df_gaze)
        freq = user_freq if user_freq and user_freq > 0 else real_freq
        sample_dur = 1.0 / freq
        
        if progress_callback: 
            progress_callback(f"Frequency detected: {real_freq:.1f}Hz (Using {freq:.1f}Hz)")

        # 3. Analisi Per-Fase
        results = []
        total_phases = len(df_toi)
        
        # Pre-calcoliamo tutte le AOI uniche presenti nel file per creare colonne coerenti
        # Es: Target_Face, Target_Hands, Confederate_Face...
        unique_combinations = df_gaze[['Hit_Role', 'Hit_AOI']].drop_duplicates().dropna()
        
        # --- FIX PYLANCE 1: Usa enumerate per avere un indice intero sicuro ---
        for i, (orig_idx, row) in enumerate(df_toi.iterrows()):
            if self._cancel_flag: raise InterruptedError("Stopped by user")
            
            if progress_callback and i % 10 == 0:
                progress_callback(f"Analyzing phase {i+1}/{total_phases}...")

            t_start = float(row['Start'])
            t_end = float(row['End'])
            phase_dur = t_end - t_start
            
            # --- FAST SLICING (Binary Search) ---
            # numpy.searchsorted ora riceve array float64 espliciti, risolvendo l'errore di tipo
            idx_start = np.searchsorted(gaze_ts, t_start)
            idx_end = np.searchsorted(gaze_ts, t_end)
            
            # Slice del dataframe
            subset = df_gaze.iloc[idx_start:idx_end]
            
            # Dati Base (copia tutto ciò che c'era nel TOI)
            res_row = row.to_dict() 
            res_row['Gaze_Samples_Total'] = len(subset)
            res_row['Gaze_Valid_Time'] = len(subset) * sample_dur
            res_row['Tracking_Ratio'] = (len(subset) * sample_dur) / phase_dur if phase_dur > 0 else 0
            
            # --- METRICHE DINAMICHE ---
            if not subset.empty:
                # Conta occorrenze per (Ruolo, AOI)
                counts = subset.groupby(['Hit_Role', 'Hit_AOI']).size()
                
                # Latency Logic: Trova il primo timestamp per ogni Role/AOI
                first_hits = subset.groupby(['Hit_Role', 'Hit_AOI'])['Timestamp'].min()
            else:
                counts = pd.Series(dtype=float)
                first_hits = pd.Series(dtype=float)

            # Itera su tutte le combinazioni possibili trovate nel file intero (per avere colonne fisse anche se 0)
            for _, comb in unique_combinations.iterrows():
                r, a = comb['Hit_Role'], comb['Hit_AOI']
                if r == "None" or r == "Ignore": continue # Skip rumore
                
                key_base = f"{r}_{a}" # Es: Target_Face
                
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
                    mask = (subset['Hit_Role'] == r) & (subset['Hit_AOI'] == a)
                    glances = (mask & ~mask.shift(1).fillna(False)).sum()

                # --- Output Formatting ---
                if long_format:
                    # LONG FORMAT (Tidy Data): Una riga per ogni combinazione (Fase, Ruolo, AOI)
                    # Copia i dati base della fase
                    row_long = res_row.copy()
                    # Rimuovi metriche aggregate se presenti (opzionale, qui non ci sono ancora)
                    
                    row_long['Hit_Role'] = r
                    row_long['Hit_AOI'] = a
                    row_long['Duration'] = duration
                    row_long['Percentage'] = perc
                    row_long['Latency'] = latency
                    row_long['Glances'] = glances
                    
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

            if not long_format:
                results.append(res_row)

        if not results:
            return None

        # Crea DataFrame Finale
        df_results = pd.DataFrame(results)
        
        # Riordina colonne per pulizia (TOI info first, then Metrics)
        cols = list(df_results.columns)
        # Identifica le colonne base (dal TOI o generali)
        base_cols = [c for c in cols if c in df_toi.columns or c in ['Gaze_Samples_Total', 'Gaze_Valid_Time', 'Tracking_Ratio']]
        
        if long_format:
            # Per Long format aggiungiamo le chiavi di raggruppamento alle base cols
            base_cols.extend(['Hit_Role', 'Hit_AOI'])
            # Rimuovi duplicati mantenendo ordine
            base_cols = list(dict.fromkeys(base_cols))
            
        # Le altre sono metriche
        metric_cols = sorted([c for c in cols if c not in base_cols])
        
        return df_results[base_cols + metric_cols]


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
        self.gaze_freq = tk.DoubleVar(value=0.0) # 0 = Auto-detect
        self.var_raw = tk.BooleanVar(value=False)
        self.var_long = tk.BooleanVar(value=False)
        
        self.status_var = tk.StringVar(value="Ready.")
        
        self._build_ui()
        self._auto_load()

    def _auto_load(self):
        # 1. Cerca il file MAPPED (Output del Tab 5)
        if hasattr(self.context, 'mapped_csv_path') and self.context.mapped_csv_path:
            if os.path.exists(self.context.mapped_csv_path):
                self.mapped_csv_path.set(self.context.mapped_csv_path)
            
        # 2. Cerca il file TOI (Output del Tab 4)
        if hasattr(self.context, 'toi_path') and self.context.toi_path:
            if os.path.exists(self.context.toi_path):
                self.toi_tsv_path.set(self.context.toi_path)

    def _build_ui(self):
        tk.Label(self.parent, text="6. Statistics & Reporting", font=("Segoe UI", 18, "bold"), bg="white").pack(pady=(0, 10), anchor="w")

        main = tk.Frame(self.parent, padx=20, pady=20, bg="white")
        main.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main, text="Gaze Analyzer: Generate Scientific Report", font=("Segoe UI", 16, "bold"), bg="white").pack(pady=(0,20))

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
        f_chk = tk.Frame(lf_set, bg="white"); f_chk.pack(fill=tk.X, pady=5)
        tk.Checkbutton(f_chk, text="Export Raw Data (Sample-level with Phase info)", variable=self.var_raw, bg="white").pack(anchor="w")
        tk.Checkbutton(f_chk, text="Use Long Format (Tidy Data) for Stats Report", variable=self.var_long, bg="white").pack(anchor="w")

        # 3. Action
        self.btn_run = tk.Button(main, text="GENERATE FULL REPORT", bg="#4CAF50", fg="white", 
                                 font=("Bold", 12), height=2, command=self.run_analysis_thread)
        self.btn_run.pack(fill=tk.X, pady=20)
        
        # 4. Status & Progress
        self.progress = ttk.Progressbar(main, mode='indeterminate')
        self.progress.pack(fill=tk.X)
        
        tk.Label(main, textvariable=self.status_var, bg="white", fg="blue").pack(pady=5)

    def _add_picker(self, p, lbl, var, ft):
        f = tk.Frame(p, bg="white"); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=lbl, width=25, anchor="w", bg="white").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(f, text="...", width=3, command=lambda: self.browse(var, ft)).pack(side=tk.LEFT)

    def browse(self, var, ft):
        f = filedialog.askopenfilename(filetypes=[("File", ft)])
        if f: var.set(f)

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
        
        freq = self.gaze_freq.get()
        # Ensure 0.0 or negative is treated as None for auto-detect
        if freq <= 0.0: freq = None 

        def worker():
            try:
                # 1. Stats Analysis
                df_stats = self.logic.run_analysis(
                    mapped, toi, user_freq=freq,
                    progress_callback=self._update_status,
                    long_format=self.var_long.get()
                )
                
                # 2. Raw Data (Optional)
                df_raw = None
                if self.var_raw.get():
                    df_raw = self.logic.generate_raw_dataset(
                        mapped, toi, progress_callback=self._update_status
                    )
                
                if df_stats is not None:
                    # Save Logic
                    default_name = mapped.replace("_MAPPED.csv", "_FINAL_STATS.csv")
                    if default_name == mapped: default_name += "_stats.csv"
                    
                    self.parent.after(0, lambda: self._save_results(df_stats, df_raw, default_name))
                else:
                    self.parent.after(0, lambda: messagebox.showwarning("Empty", "No results generated. Check time alignment between files."))
                    self.parent.after(0, self._reset_ui)

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
            initialfile=os.path.basename(default_path),
            defaultextension=".csv",
            filetypes=[("CSV Report", "*.csv")]
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

    def _show_error(self, msg):
        self.progress.stop()
        self.status_var.set("Error occurred.")
        messagebox.showerror("Analysis Failed", msg)
        self._reset_ui()

    def _reset_ui(self):
        self.btn_run.config(state="normal")
        self.progress.stop()
        self.status_var.set("Ready.")