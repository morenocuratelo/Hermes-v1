import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import pandas as pd
import os
from datetime import datetime
import scipy.io as sio

# --- GESTIONE PROFILI ---
class ProfileManager:
    def __init__(self, profiles_dir="profiles"):
        self.profiles_dir = profiles_dir
        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir)

    def get_available_profiles(self):
        return [f for f in os.listdir(self.profiles_dir) if f.endswith(".json")]

    def load_profile(self, filename):
        with open(os.path.join(self.profiles_dir, filename), 'r') as f:
            return json.load(f)

# --- MOTORE LOGICO ---
class TOIGenerator:
    @staticmethod
    def parse_time_string(time_str):
        """Converte stringhe 'HH:MM:SS.mmm' in secondi totali."""
        try:
            if pd.isna(time_str): return None
            # If data comes from .mat, it might already be a float/int
            if isinstance(time_str, (float, int)):
                return float(time_str)
            time_str = str(time_str).strip()
            formats = ["%H:%M:%S.%f", "%H:%M:%S"]
            dt = None
            for fmt in formats:
                try:
                    dt = datetime.strptime(time_str, fmt)
                    break
                except ValueError:
                    continue
            if dt is None: return None
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        except Exception:
            return None

    @staticmethod
    def _load_matlab_file(path):
        """
        Loads data from .csv or .mat.
        For .mat, extracts the first variable that is not an internal header.
        """
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.csv':
            return pd.read_csv(path)
        
        elif ext == '.mat':
            try:
                mat_contents = sio.loadmat(path)
                # Filter out internal keys starting with '__' (e.g., __header__, __version__)
                valid_keys = [k for k in mat_contents.keys() if not k.startswith('__')]
                
                if not valid_keys:
                    raise ValueError("The .mat file contains no variables.")
                
                # Heuristic: Take the first variable found. 
                # Ideally, the variable name should be consistent (e.g., 'results'), 
                # but we take the first valid one to be flexible.
                data_var = mat_contents[valid_keys[0]]
                
                # Convert to DataFrame
                # Case A: It's already a structured array/matrix compatible with DataFrame
                try:
                    df = pd.DataFrame(data_var)
                    return df
                except Exception:
                    # Case B: It might be a Matlab Struct (complex handling often needed)
                    # This flattens simple structs
                    return pd.DataFrame(data_var[0]) 
                    
            except Exception as e:
                raise ValueError(f"Failed to parse .mat file: {e}")
        
        else:
            raise ValueError(f"Unsupported format: {ext}")

    @staticmethod
    def process(matlab_path, json_path, profile, output_path):
        # 1. Carica Evento Tobii
        try:
            with open(json_path, 'r') as f:
                tobii_data = json.load(f)
            
            # Leggi quale label cercare dal profilo (Default: "Start")
            sync_logic = profile.get('sync_logic', {})
            target_label = sync_logic.get('tobii_event_label', 'Start')
            
            if isinstance(tobii_data, list):
                target_event = next((e for e in tobii_data if e.get('label', '').lower() == target_label.lower()), None)
                if not target_event: 
                    # Fallback sul primo evento se non trova la label specifica
                    print(f"Attenzione: Label '{target_label}' non trovata. Uso il primo evento disponibile.")
                    target_event = tobii_data[0]
            else:
                target_event = tobii_data
                
            tobii_ts = float(target_event['timestamp'])
            print(f"Tobii Sync Point ({target_label}): {tobii_ts}s")
            
        except Exception as e:
            raise ValueError(f"Errore Tobii JSON: {e}")

        # 2. Load Matlab Data (CSV or MAT) <--- MODIFIED SECTION
        try:
            df = TOIGenerator._load_matlab_file(matlab_path)
            # Ensure column names are stripped of whitespace if loaded from loose CSVs
            df.columns = df.columns.astype(str).str.strip()
        except Exception as e:
            raise ValueError(f"Matlab Data Load Error: {e}")

        # 3. Estrai configurazione Profilo
        struct = profile.get('csv_structure', {})
        cols_seq = struct.get('sequence_columns', [])
        cond_col = struct.get('condition_column')
        labels = profile.get('phases_labels', [])
        
        # --- LOGICA DI SINCRONIZZAZIONE (Nuova V4) ---
        matlab_anchor_col = sync_logic.get('matlab_anchor_column') # Es. 'baseline_end'
        offset_val = float(sync_logic.get('seconds_offset', 0.0))  # Es. -60.0
        
        if not matlab_anchor_col:
            raise ValueError("Profilo errato: Manca 'matlab_anchor_column' in sync_logic.")
            
        if matlab_anchor_col not in df.columns:
            raise ValueError(f"Errore: La colonna di sync '{matlab_anchor_col}' definita nel profilo NON esiste nel file CSV caricato.\nColonne disponibili: {list(df.columns)}")

        # Leggi orario Matlab
        matlab_sync_str = df.iloc[0][matlab_anchor_col]
        matlab_sync_sec = TOIGenerator.parse_time_string(matlab_sync_str)
        
        if matlab_sync_sec is None:
            raise ValueError(f"Impossibile leggere orario nella colonna '{matlab_anchor_col}'")

        # CALCOLO DELTA
        # Formula: Video_Time = Matlab_Time + Delta
        # Al momento del sync: Tobii_TS = (Matlab_Sync_Time + Offset) + Delta
        # Delta = Tobii_TS - (Matlab_Sync_Time + Offset)
        delta_seconds = tobii_ts - (matlab_sync_sec + offset_val)
        
        print(f"Matlab Anchor ({matlab_anchor_col}): {matlab_sync_sec}s")
        print(f"Offset applicato: {offset_val}s")
        print(f"Delta calcolato: {delta_seconds:.3f}s")

        # 4. Generazione Trial
        toi_rows = []
        fixed_phases = profile.get('append_fixed_phases', [])
        fixed_anchor_col = profile.get('fixed_phase_anchor_column', 'auto')

        for idx, row in df.iterrows():
            trial_n = row.get('TrialN', idx+1)
            condition = row.get(cond_col, 'NA')
            
            # --- FASE A: Intervalli Variabili ---
            seq_times = []
            valid_trial = True
            
            for col in cols_seq:
                ts = TOIGenerator.parse_time_string(row.get(col))
                if ts is None:
                    valid_trial = False; break
                seq_times.append(ts + delta_seconds)
            
            if not valid_trial: continue

            for i in range(len(seq_times) - 1):
                start_t = seq_times[i]
                end_t = seq_times[i+1]
                p_name = labels[i] if i < len(labels) else f"Phase_{i+1}"
                
                toi_rows.append({
                    "Name": f"T{trial_n}_{condition}_{p_name}",
                    "Start": round(start_t, 3), "End": round(end_t, 3), "Duration": round(end_t - start_t, 3),
                    "Trial": trial_n, "Condition": condition, "Phase": p_name
                })

            # --- FASE B: Fasi Fisse (ITI) ---
            anchor_time = None
            if fixed_anchor_col == "auto":
                anchor_time = seq_times[-1]
            elif fixed_anchor_col in row:
                raw_t = TOIGenerator.parse_time_string(row.get(fixed_anchor_col))
                if raw_t is not None:
                    anchor_time = raw_t + delta_seconds
            
            if anchor_time is not None:
                curr = anchor_time
                for phase in fixed_phases:
                    p_name = phase.get("name", "Extra")
                    dur = float(phase.get("duration", 0.0))
                    toi_rows.append({
                        "Name": f"T{trial_n}_{condition}_{p_name}",
                        "Start": round(curr, 3), "End": round(curr + dur, 3), "Duration": dur,
                        "Trial": trial_n, "Condition": condition, "Phase": p_name
                    })
                    curr += dur

        out_df = pd.DataFrame(toi_rows)
        out_df.to_csv(output_path, index=False, sep='\t')
        return len(toi_rows)

class TOIGeneratorView: # <--- NOME CAMBIATO
    def __init__(self, parent, context=None): # <--- ACCETTA CONTEXT
        self.parent = parent
        self.context = context
        
        self.pm = ProfileManager()
        self.tobii_file = tk.StringVar()
        self.matlab_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.selected_profile = tk.StringVar()
        
        self._build_ui()

    def _build_ui(self):
        # Header visivo (opzionale ma consigliato per coerenza)
        tk.Label(self.parent, text="4. TOI Builder (Sync & Cut)", font=("Segoe UI", 18, "bold"), bg="white").pack(pady=(0, 10), anchor="w")

        main = tk.Frame(self.parent, padx=20, pady=20, bg="white") # <--- CORRETTO: self.parent
        main.pack(fill=tk.BOTH, expand=True)
        
        # Header
        tk.Label(main, text="TOI Builder - Lab Modigliani", font=("Segoe UI", 14, "bold")).pack(pady=(0, 20))

        # 1. Profilo
        lf1 = tk.LabelFrame(main, text="1. Profilo Esperimento", padx=10, pady=10)
        lf1.pack(fill=tk.X, pady=5)
        self.cb = ttk.Combobox(lf1, textvariable=self.selected_profile, state="readonly")
        self.cb.pack(fill=tk.X)
        self.refresh_profiles()
        ttk.Button(lf1, text="Aggiorna Lista", command=self.refresh_profiles).pack(pady=5)
        # Nuovo bottone per lanciare il Wizard (importiamo dinamicamente per evitare loop)
        ttk.Button(lf1, text="➕ Crea Nuovo Profilo (Wizard)", command=self.launch_wizard).pack(pady=5)

        # 2. Input
        lf2 = tk.LabelFrame(main, text="2. Dati Input", padx=10, pady=10)
        lf2.pack(fill=tk.X, pady=10)
        self._add_picker(lf2, "Eventi Tobii (.json):", self.tobii_file, "*.json", 0)
        # VERSIONE CORRETTA
        self._add_picker(lf2, "Results Matlab (.csv, .mat):", self.matlab_file, "*.csv *.mat", 2)

        # 3. Output
        lf3 = tk.LabelFrame(main, text="3. Output", padx=10, pady=10)
        lf3.pack(fill=tk.X, pady=5)
        tk.Entry(lf3, textvariable=self.output_file).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(lf3, text="Scegli...", command=self.save_as).pack(side=tk.LEFT, padx=5)

        # Run
        tk.Button(main, text="GENERA TOI", bg="#007ACC", fg="white", font=("Arial", 11, "bold"), height=2, command=self.run).pack(fill=tk.X, pady=20)

    def _add_picker(self, p, lbl, var, ft, r):
        tk.Label(p, text=lbl).grid(row=r, column=0, sticky="w")
        tk.Entry(p, textvariable=var, width=45).grid(row=r+1, column=0, padx=5)
        tk.Button(p, text="...", width=3, command=lambda: self.browse(var, ft)).grid(row=r+1, column=1)

    def refresh_profiles(self):
        v = self.pm.get_available_profiles()
        self.cb['values'] = v
        if v: self.selected_profile.set(v[0])

    def browse(self, var, ft):
        f = filedialog.askopenfilename(filetypes=[("File", ft)])
        if f: 
            var.set(f)
            if self.matlab_file.get() and not self.output_file.get():
                self.output_file.set(os.path.splitext(self.matlab_file.get())[0] + "_TOIs.tsv")

    def save_as(self):
        f = filedialog.asksaveasfilename(defaultextension=".tsv", filetypes=[("TSV", "*.tsv")])
        if f: self.output_file.set(f)

    def run(self):
        if not self.selected_profile.get(): return
        try:
            prof = self.pm.load_profile(self.selected_profile.get())
            n = TOIGenerator.process(self.matlab_file.get(), self.tobii_file.get(), prof, self.output_file.get())
            if self.context:
                self.context.toi_path = self.output_file.get()
            messagebox.showinfo("Successo", f"Generati {n} TOI.")
        except Exception as e:
            messagebox.showerror("Errore", str(e))
    
    def launch_wizard(self):
        try:
            from hermes_master_prof import ProfileWizard
            # Creiamo una Toplevel figlia del nostro parent
            win = tk.Toplevel(self.parent)
            win.title("Wizard Profilo")
            ProfileWizard(win) # Lancia il wizard nella nuova finestra
        except ImportError:
            messagebox.showerror("Errore", "Impossibile trovare hermes_master_prof.py")