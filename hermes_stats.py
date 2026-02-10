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

    def run_analysis(self, mapped_path, toi_path, user_freq=None, progress_callback=None):
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
            # Assicuriamoci che i timestamp siano ordinati per usare searchsorted (molto più veloce)
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
                
                # 1. Total Duration
                # Usa .get con un default sicuro se la combinazione non esiste in questo subset
                c_val = counts.get((r, a), 0)
                duration = c_val * sample_dur
                res_row[f"{key_base}_Dur"] = duration
                
                # 2. Percentage of Phase
                res_row[f"{key_base}_Perc"] = duration / phase_dur if phase_dur > 0 else 0
                
                # 3. Latency (Time to First Fixation)
                if (r, a) in first_hits.index:
                    first_ts = first_hits.get((r, a))
                    # Check if first_ts is not None before performing subtraction
                    if first_ts is not None:
                        latency = first_ts - t_start
                        res_row[f"{key_base}_Latency"] = latency
                    else:
                        res_row[f"{key_base}_Latency"] = None
                else:
                    res_row[f"{key_base}_Latency"] = None # Mai guardato

                # 4. Glance Count (Numero di volte che lo sguardo è ENTRATO nell'AOI)
                if c_val > 0:
                    # Crea maschera booleana: 1 dove guardo questa AOI, 0 altrimenti
                    mask = (subset['Hit_Role'] == r) & (subset['Hit_AOI'] == a)
                    # Shift per trovare differenze: (Attuale=1) AND (Precedente=0/False)
                    transitions = (mask & ~mask.shift(1).fillna(False)).sum()
                    res_row[f"{key_base}_Glances"] = transitions
                else:
                    res_row[f"{key_base}_Glances"] = 0

            results.append(res_row)

        if not results:
            return None

        # Crea DataFrame Finale
        df_results = pd.DataFrame(results)
        
        # Riordina colonne per pulizia (TOI info first, then Metrics)
        cols = list(df_results.columns)
        # Identifica le colonne base (dal TOI o generali)
        base_cols = [c for c in cols if c in df_toi.columns or c in ['Gaze_Samples_Total', 'Gaze_Valid_Time', 'Tracking_Ratio']]
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
                df_result = self.logic.run_analysis(
                    mapped, toi, user_freq=freq,
                    progress_callback=self._update_status
                )
                
                if df_result is not None:
                    # Save Logic
                    default_name = mapped.replace("_MAPPED.csv", "_FINAL_STATS.csv")
                    if default_name == mapped: default_name += "_stats.csv"
                    
                    self.parent.after(0, lambda: self._save_results(df_result, default_name))
                else:
                    self.parent.after(0, lambda: messagebox.showwarning("Empty", "No results generated. Check time alignment between files."))
                    self.parent.after(0, self._reset_ui)

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.parent.after(0, lambda: self._show_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _update_status(self, msg):
        self.parent.after(0, lambda: self.status_var.set(msg))

    def _save_results(self, df, default_path):
        self.progress.stop()
        out = filedialog.asksaveasfilename(
            initialfile=os.path.basename(default_path),
            defaultextension=".csv",
            filetypes=[("CSV Report", "*.csv")]
        )
        
        if out:
            try:
                df.to_csv(out, index=False)
                messagebox.showinfo("Success", f"Report saved successfully!\n\nRows: {len(df)}\nPath: {out}")
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