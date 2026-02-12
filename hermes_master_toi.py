import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import pandas as pd
import os
from datetime import datetime
import scipy.io as sio
import gzip

# --- GESTIONE PROFILI ---
class ProfileManager:
    def __init__(self, profiles_dir):
        self.profiles_dir = profiles_dir
        if not os.path.exists(self.profiles_dir):
            try:
                os.makedirs(self.profiles_dir)
            except OSError:
                pass # Già esiste o permessi

    def get_available_profiles(self):
        if not os.path.exists(self.profiles_dir):
            return []
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
            if pd.isna(time_str):
                return None
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
            if dt is None:
                return None
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
                    print(f"Warning: Label '{target_label}' not found. Using first available event.")
                    target_event = tobii_data[0]
            else:
                target_event = tobii_data
                
            tobii_ts = float(target_event['timestamp'])
            print(f"Tobii Sync Point ({target_label}): {tobii_ts}s")
            
        except Exception as e:
            raise ValueError(f"Tobii JSON Error: {e}")

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
            raise ValueError("Invalid Profile: Missing 'matlab_anchor_column' in sync_logic.")
            
        if matlab_anchor_col not in df.columns:
            raise ValueError(f"Error: Sync column '{matlab_anchor_col}' defined in profile does NOT exist in loaded CSV.\nAvailable columns: {list(df.columns)}")

        # Leggi orario Matlab
        matlab_sync_str = df.iloc[0][matlab_anchor_col]
        matlab_sync_sec = TOIGenerator.parse_time_string(matlab_sync_str)
        
        if matlab_sync_sec is None:
            raise ValueError(f"Cannot read time in column '{matlab_anchor_col}'")

        # CALCOLO DELTA
        # Formula: Video_Time = Matlab_Time + Delta
        # Al momento del sync: Tobii_TS = (Matlab_Sync_Time + Offset) + Delta
        # Delta = Tobii_TS - (Matlab_Sync_Time + Offset)
        delta_seconds = tobii_ts - (matlab_sync_sec + offset_val)
        
        print(f"Matlab Anchor ({matlab_anchor_col}): {matlab_sync_sec}s")
        print(f"Applied Offset: {offset_val}s")
        print(f"Calculated Delta: {delta_seconds:.3f}s")

        # 4. Generazione Trial
        toi_rows = []
        fixed_phases = profile.get('append_fixed_phases', [])
        fixed_anchor_col = profile.get('fixed_phase_anchor_column', 'auto')

        for idx, (_, row) in enumerate(df.iterrows()):
            trial_n = row.get('TrialN', idx + 1)
            condition = row.get(cond_col, 'NA')
            
            # --- FASE A: Intervalli Variabili ---
            seq_times = []
            valid_trial = True
            
            for col in cols_seq:
                ts = TOIGenerator.parse_time_string(row.get(col))
                if ts is None:
                    valid_trial = False
                    break
                seq_times.append(ts + delta_seconds)
            
            if not valid_trial:
                continue

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

class TOIGeneratorView:
    def __init__(self, parent, context=None):
        self.parent = parent
        self.context = context
        
        # --- PATH MANAGEMENT ---
        if self.context and self.context.paths["profiles_toi"]:
            p_dir = self.context.paths["profiles_toi"]
        else:
            p_dir = "profiles_toi_fallback"
            
        self.pm = ProfileManager(profiles_dir=p_dir)
        
        # Variabili UI
        self.tobii_file = tk.StringVar()
        self.matlab_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.selected_profile = tk.StringVar()
        self.yolo_raw_path = tk.StringVar()

        # Auto-load Optimization path
        if self.context and self.context.pose_data_path and self.context.pose_data_path.endswith('.json.gz'):
            self.yolo_raw_path.set(self.context.pose_data_path)
        
        self._build_ui()

    def _build_ui(self):
        # Header Principale
        header_frame = tk.Frame(self.parent, bg="white", pady=10)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="4. TOI Builder & Synchronization", font=("Segoe UI", 18, "bold"), bg="white").pack(anchor="w", padx=10)
        tk.Label(header_frame, text="Synchronizes data streams and defines Temporal Intervals of Interest (TOI).", 
                 font=("Segoe UI", 10), fg="#666", bg="white").pack(anchor="w", padx=10)

        # Container scrollabile o frame principale
        main = tk.Frame(self.parent, padx=15, pady=15)
        main.pack(fill=tk.BOTH, expand=True)

        # ==========================================
        # PARTE A: PROCESSO PRINCIPALE (Config & Run)
        # ==========================================
        
        # Frame Configurazione (Profilo + Input) affiancati o raggruppati
        lf_config = tk.LabelFrame(main, text="A. Experiment Configuration", font=("Bold", 11), padx=10, pady=10)
        lf_config.pack(fill=tk.X, pady=(0, 10))

        # Riga 1: Profilo
        f_prof = tk.Frame(lf_config)
        f_prof.pack(fill=tk.X, pady=5)
        tk.Label(f_prof, text="Selected Profile:", width=20, anchor="w").pack(side=tk.LEFT)
        self.cb = ttk.Combobox(f_prof, textvariable=self.selected_profile, state="readonly", width=40)
        self.cb.pack(side=tk.LEFT, padx=5)
        self.refresh_profiles()
        tk.Button(f_prof, text="🔄", command=self.refresh_profiles, width=3).pack(side=tk.LEFT)
        tk.Button(f_prof, text="⚙️ Manage Profiles", command=self.launch_wizard).pack(side=tk.LEFT, padx=10)

        # Separatore visivo interno
        ttk.Separator(lf_config, orient='horizontal').pack(fill='x', pady=10)

        # Riga 2: Inputs
        self._add_file_row(lf_config, "Tobii Events (.json):", self.tobii_file, "*.json")
        self._add_file_row(lf_config, "Matlab Data (.csv/.mat):", self.matlab_file, "*.csv *.mat")

        # Frame Output & Azione
        lf_action = tk.LabelFrame(main, text="B. TOI Generation", font=("Bold", 11), padx=10, pady=10)
        lf_action.pack(fill=tk.X, pady=(0, 20))

        # Riga Output
        f_out = tk.Frame(lf_action)
        f_out.pack(fill=tk.X, pady=5)
        tk.Label(f_out, text="Save TOI as:", width=20, anchor="w").pack(side=tk.LEFT)
        tk.Entry(f_out, textvariable=self.output_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(f_out, text="📂 Browse...", command=self.save_as).pack(side=tk.LEFT)

        # Big Button
        tk.Button(lf_action, text="🚀 GENERATE TOI FILE (SYNC)", bg="#007ACC", fg="white", 
                  font=("Segoe UI", 12, "bold"), height=2, cursor="hand2", 
                  command=self.run).pack(fill=tk.X, pady=10)

        # ==========================================
        # PARTE B: OTTIMIZZAZIONE (Sezione 4 Migliorata)
        # ==========================================
        
        # Separatore visivo forte tra le due fasi
        tk.Label(main, text="Advanced Tools", font=("Bold", 10), fg="#888").pack(anchor="w", pady=(10, 0))
        ttk.Separator(main, orient='horizontal').pack(fill='x', pady=5)

        # Frame dedicato con colore di sfondo leggermente diverso o bordo
        lf_opt = tk.LabelFrame(main, text="C. Data Pruning (Optional)", font=("Bold", 11), fg="#E65100", padx=10, pady=10)
        lf_opt.pack(fill=tk.X, pady=10)

        # Spiegazione User Friendly
        info_frame = tk.Frame(lf_opt)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        lbl_icon = tk.Label(info_frame, text="💡", font=("Arial", 16))
        lbl_icon.pack(side=tk.LEFT, anchor="n")
        lbl_desc = tk.Label(info_frame, justify="left", wraplength=500,
                            text="This function prunes the original YOLO file (.json.gz) removing "
                                 "data before the experiment start (Sync).\n"
                                 "Recommended to reduce file size and speed up analysis.")
        lbl_desc.pack(side=tk.LEFT, padx=10)

        # Input File Raw
        self._add_file_row(lf_opt, "Raw YOLO File (.gz):", self.yolo_raw_path, "*.json.gz")

        # Bottone Azione Ottimizzazione
        btn_opt = tk.Button(lf_opt, text="✂️ PRUNE & OPTIMIZE DATA", 
                            bg="#ff9800", fg="white", font=("Segoe UI", 10, "bold"), 
                            cursor="hand2", command=self.run_cropping)
        btn_opt.pack(fill=tk.X, pady=5)

    def _add_file_row(self, parent, label_text, var, file_types):
        """Helper per creare righe di input pulite"""
        f = tk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=label_text, width=20, anchor="w").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var, fg="#333").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(f, text="📂", width=4, command=lambda: self.browse(var, file_types)).pack(side=tk.LEFT)

    # --- LOGICA (Invariata) ---
    def refresh_profiles(self):
        v = self.pm.get_available_profiles()
        self.cb['values'] = v
        if v:
            self.selected_profile.set(v[0])

    def browse(self, var, ft):
        f = filedialog.askopenfilename(filetypes=[("File", ft)])
        if f: 
            var.set(f)
            # Auto-suggest output name if matlab input is selected
            if var == self.matlab_file and not self.output_file.get():
                base = os.path.splitext(f)[0]
                if self.context and self.context.paths["output"]:
                    name = os.path.basename(base) + "_TOIs.tsv"
                    out = os.path.join(self.context.paths["output"], name)
                else:
                    out = base + "_TOIs.tsv"
                self.output_file.set(out)

    def save_as(self):
        f = filedialog.asksaveasfilename(defaultextension=".tsv", filetypes=[("TSV", "*.tsv")])
        if f:
            self.output_file.set(f)

    def run(self):
        if not self.selected_profile.get(): 
            messagebox.showwarning("Warning", "Select a profile before proceeding.")
            return
        try:
            prof = self.pm.load_profile(self.selected_profile.get())
            n = TOIGenerator.process(self.matlab_file.get(), self.tobii_file.get(), prof, self.output_file.get())
            if self.context:
                self.context.toi_path = self.output_file.get()
            messagebox.showinfo("Success", f"Generated {n} TOI intervals.\nYou can now proceed to optimization (Step C) if needed.")
        except Exception as e:
            messagebox.showerror("Generation Error", str(e))
    
    def launch_wizard(self):
        try:
            from hermes_master_prof import ProfileWizard
            win = tk.Toplevel(self.parent)
            win.title("Profile Wizard")
            ProfileWizard(win)
        except ImportError:
            messagebox.showerror("Error", "Cannot find hermes_master_prof.py")

    def run_cropping(self):
        raw_in = self.yolo_raw_path.get().strip()
        toi_in = self.output_file.get().strip() 
        
        if not raw_in or not os.path.exists(raw_in):
            messagebox.showwarning("Missing File", "Select a valid YOLO .json.gz file in section C.")
            return

        if not toi_in or not os.path.exists(toi_in):
            messagebox.showwarning("Missing File", "TOI file not found. Run Step B (Generate) first.")
            return
            
        cropped_path = DataCropper.crop_yolo_json(raw_in, toi_in)
        
        if cropped_path:
            if self.context:
                self.context.pose_data_path = cropped_path
            messagebox.showinfo("Optimization", f"Pruned RAW file created!\n{os.path.basename(cropped_path)}")
        else:
            messagebox.showerror("Error", "Pruning failed. See console for details.")
class DataCropper:
    @staticmethod
    def crop_yolo_json(json_gz_path, toi_path, output_suffix="_CROPPED"):
        """
        Legge il JSON.GZ di YOLO e il file TOI.
        Crea un nuovo file .json.gz contenente solo i frame dal primo TOI in poi.
        """
        try:
            print(f"✂️ Starting RAW pruning on: {os.path.basename(json_gz_path)}")
            
            # 1. Trova il tempo di inizio minimo dai TOI
            try:
                df_toi = pd.read_csv(toi_path, sep='\t')
            except Exception:
                # Fallback se il separatore fosse diverso
                df_toi = pd.read_csv(toi_path)

            if df_toi.empty:
                print("⚠️ Empty TOI file, cannot prune.")
                return None
            
            start_cut_time = df_toi['Start'].min()
            print(f"⏱️ Cut-off time identified: {start_cut_time:.3f} sec")
            
            # 2. Setup percorsi
            path_parts = os.path.splitext(os.path.splitext(json_gz_path)[0]) # Rimuove .gz poi .json
            # Ricostruisce nome: base + suffix + .json.gz
            new_path = f"{path_parts[0]}{output_suffix}.json.gz"
            
            kept_frames = 0
            dropped_frames = 0
            
            # 3. Streaming Read/Write (Memoria efficiente)
            print("⏳ Processing (Streaming)...")
            with gzip.open(json_gz_path, 'rt', encoding='utf-8') as f_in, \
                 gzip.open(new_path, 'wt', encoding='utf-8') as f_out:
                
                for line in f_in:
                    try:
                        # Parsing veloce solo per leggere il timestamp
                        # Nota: se il file è enorme, json.loads su ogni riga è il collo di bottiglia,
                        # ma è comunque il metodo più sicuro.
                        frame = json.loads(line)
                        ts = frame.get('ts', 0.0)
                        
                        if ts >= start_cut_time:
                            # Scrive la riga originale intatta (veloce)
                            f_out.write(line)
                            kept_frames += 1
                        else:
                            dropped_frames += 1
                            
                    except json.JSONDecodeError:
                        continue

            print("✅ Pruning complete.")
            print(f"   Frames removed:   {dropped_frames}")
            print(f"   Frames kept: {kept_frames}")
            print(f"   Saved in:      {os.path.basename(new_path)}")
            
            return new_path

        except Exception as e:
            print(f"❌ Error during JSON pruning: {e}")
            return None
