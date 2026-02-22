import csv
import gzip
import json
import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, simpledialog, ttk

import pandas as pd
import scipy.io as sio


# --- GESTIONE PROFILI ---
class ProfileManager:
    def __init__(self, profiles_dir):
        self.profiles_dir = profiles_dir
        if not os.path.exists(self.profiles_dir):
            try:
                os.makedirs(self.profiles_dir)
            except OSError:
                pass  # Già esiste o permessi

    def get_available_profiles(self):
        if not os.path.exists(self.profiles_dir):
            return []
        return [f for f in os.listdir(self.profiles_dir) if f.endswith(".json")]

    def load_profile(self, filename):
        with open(os.path.join(self.profiles_dir, filename)) as f:
            return json.load(f)


class ProfileWizardLogic:
    """
    Gestisce la logica di business per il Profile Wizard:
    - Caricamento e parsing dei file di esempio (JSON/CSV).
    - Costruzione e salvataggio della struttura dati del profilo.
    """

    def __init__(self, profiles_dir):
        self.profiles_dir = profiles_dir
        self.loaded_json_events = []
        self.loaded_csv_columns = []

    def load_json_sample(self, path):
        with open(path) as f:
            data = json.load(f)

        labels = set()
        if isinstance(data, list):
            for e in data:
                if "label" in e:
                    labels.add(e["label"])
        elif isinstance(data, dict):
            if "label" in data:
                labels.add(data["label"])

        self.loaded_json_events = sorted(list(labels))
        return self.loaded_json_events

    def load_csv_sample(self, path):
        # sep=None con engine='python' permette a pandas di indovinare se usare , o ; o \t
        df = pd.read_csv(path, nrows=2, sep=None, engine="python")
        self.loaded_csv_columns = [str(c).strip() for c in df.columns]
        return self.loaded_csv_columns

    def save_profile(self, profile_data):
        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir)

        name = profile_data.get("name", "profile")
        safe_name = "".join([c if c.isalnum() else "_" for c in name]).lower() + ".json"
        path = os.path.join(self.profiles_dir, safe_name)

        with open(path, "w") as f:
            json.dump(profile_data, f, indent=4)
        return path


class ProfileWizard:
    def __init__(self, root, profiles_dir=None):
        self.root = root
        profiles_dir = profiles_dir or "profiles"
        self.logic = ProfileWizardLogic(profiles_dir)

        self.root.title("Universal Profile Generator - Lab Modigliani")
        self.root.geometry("900x850")

        # --- UI LAYOUT ---
        canvas = tk.Canvas(root)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # CONTENUTI
        self._build_header()
        self._build_data_loader()
        self._build_sync_logic()
        self._build_structure_logic()
        self._build_fixed_phases()
        self._build_save_section()

    def _build_header(self):
        f = tk.Frame(self.scrollable_frame, pady=10, padx=20)
        f.pack(fill=tk.X)
        tk.Label(f, text="Generatore Profili Universale", font=("Segoe UI", 16, "bold")).pack()

        tk.Label(f, text="Nome Profilo:").pack(anchor="w")
        self.entry_name = tk.Entry(f, width=50)
        self.entry_name.pack(anchor="w")
        self.entry_name.insert(0, "Nuovo Esperimento")

        tk.Label(f, text="Descrizione:").pack(anchor="w")
        self.entry_desc = tk.Entry(f, width=80)
        self.entry_desc.pack(anchor="w")

    def _build_data_loader(self):
        lf = tk.LabelFrame(self.scrollable_frame, text="1. Carica Dati Esempio (Per popolare i menu)", padx=10, pady=10)
        lf.pack(fill=tk.X, padx=20, pady=10)

        # JSON Loader
        tk.Button(lf, text="Carica User Event Tobii (.json)", command=self.load_json_sample).grid(
            row=0, column=0, padx=5, sticky="w"
        )
        self.lbl_json_status = tk.Label(lf, text="Nessun file caricato", fg="red")
        self.lbl_json_status.grid(row=0, column=1, padx=5)

        # CSV Loader
        tk.Button(lf, text="Carica Results Matlab (.csv)", command=self.load_csv_sample).grid(
            row=1, column=0, padx=5, sticky="w", pady=5
        )
        self.lbl_csv_status = tk.Label(lf, text="Nessun file caricato", fg="red")
        self.lbl_csv_status.grid(row=1, column=1, padx=5)

    def _build_sync_logic(self):
        lf = tk.LabelFrame(self.scrollable_frame, text="2. Logica di Sincronizzazione (Anchor)", padx=10, pady=10)
        lf.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(lf, text="A. Quale EVENTO Tobii è il punto di sync?").grid(row=0, column=0, sticky="w")
        self.combo_tobii_event = ttk.Combobox(lf, state="readonly", width=40)
        self.combo_tobii_event.grid(row=1, column=0, padx=5, pady=(0, 10))

        tk.Label(lf, text="B. Quale COLONNA Matlab corrisponde a quell'evento?").grid(row=0, column=1, sticky="w")
        self.combo_matlab_anchor = ttk.Combobox(lf, state="readonly", width=40)
        self.combo_matlab_anchor.grid(row=1, column=1, padx=5, pady=(0, 10))

        tk.Label(lf, text="C. Offset temporale (Secondi da aggiungere/togliere):").grid(row=2, column=0, sticky="w")
        self.entry_offset = tk.Entry(lf, width=10)
        self.entry_offset.grid(row=3, column=0, sticky="w", padx=5)
        self.entry_offset.insert(0, "0.0")
        tk.Label(lf, text="(Es: metti -60.0 se Start avviene 60s prima del marker)").grid(row=3, column=1, sticky="w")

    def _build_structure_logic(self):
        lf = tk.LabelFrame(self.scrollable_frame, text="3. Struttura Trial & Sequenza", padx=10, pady=10)
        lf.pack(fill=tk.X, padx=20, pady=10)

        # Condition Column
        tk.Label(lf, text="Colonna che contiene il nome della Condizione (es. Condition):").pack(anchor="w")
        self.combo_cond_col = ttk.Combobox(lf, state="readonly", width=40)
        self.combo_cond_col.pack(anchor="w", pady=(0, 10))

        # Sequence Builder (Dual Listbox)
        tk.Label(lf, text="Costruisci la sequenza temporale (Sposta le colonne da SX a DX nell'ordine corretto):").pack(
            anchor="w"
        )

        frame_seq = tk.Frame(lf)
        frame_seq.pack(fill=tk.X)

        # Left: Available
        tk.Label(frame_seq, text="Colonne Disponibili").grid(row=0, column=0)
        self.lb_avail = tk.Listbox(frame_seq, selectmode=tk.EXTENDED, height=10, width=30)
        self.lb_avail.grid(row=1, column=0)

        # Buttons
        btn_frame = tk.Frame(frame_seq)
        btn_frame.grid(row=1, column=1, padx=10)
        tk.Button(btn_frame, text="Aggiungi ->", command=self.add_to_sequence).pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="<- Rimuovi", command=self.remove_from_sequence).pack(fill=tk.X, pady=5)

        # Right: Selected
        tk.Label(frame_seq, text="Sequenza Selezionata (Ordine Temporale)").grid(row=0, column=2)
        self.lb_selected = tk.Listbox(frame_seq, selectmode=tk.EXTENDED, height=10, width=30)
        self.lb_selected.grid(row=1, column=2)

    def _build_fixed_phases(self):
        lf = tk.LabelFrame(self.scrollable_frame, text="4. Fasi Fisse Extra (es. ITI)", padx=10, pady=10)
        lf.pack(fill=tk.X, padx=20, pady=10)

        # Anchor selection
        tk.Label(lf, text="Punto di partenza per le fasi fisse:").grid(row=0, column=0, sticky="w")
        self.combo_fixed_anchor = ttk.Combobox(lf, state="readonly", width=30)
        self.combo_fixed_anchor.grid(row=0, column=1, sticky="w")
        # Popolato dopo con "auto" + colonne

        # List of fixed phases
        self.tree_fixed = ttk.Treeview(lf, columns=("name", "duration"), show="headings", height=4)
        self.tree_fixed.heading("name", text="Nome Fase")
        self.tree_fixed.heading("duration", text="Durata (s)")
        self.tree_fixed.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)

        btn_f = tk.Frame(lf)
        btn_f.grid(row=2, column=0, columnspan=3)
        tk.Button(btn_f, text="Aggiungi Fase Fissa...", command=self.add_fixed_phase_dialog).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_f, text="Rimuovi Selezionata", command=self.remove_fixed_phase).pack(side=tk.LEFT, padx=5)

    def _build_save_section(self):
        f = tk.Frame(self.scrollable_frame, pady=20)
        f.pack(fill=tk.X)
        tk.Button(
            f,
            text="SALVA PROFILO JSON",
            bg="#4CAF50",
            fg="white",
            font=("Arial", 14, "bold"),
            height=2,
            command=self.save_profile,
        ).pack(fill=tk.X, padx=50)

    # --- LOGICA ---

    def load_json_sample(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            events = self.logic.load_json_sample(path)

            self.combo_tobii_event["values"] = events
            if events:
                self.combo_tobii_event.current(0)

            self.lbl_json_status.config(text=f"OK! Trovati {len(events)} eventi.", fg="green")
        except Exception as e:
            messagebox.showerror("Errore JSON", str(e))

    def load_csv_sample(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Text", "*.txt"), ("All", "*.*")])
        if not path:
            return

        try:
            columns = self.logic.load_csv_sample(path)

            print(f"DEBUG - Colonne trovate: {columns}")  # Controllo console

            # 1. Popola menu Anchor Matlab (Sezione 2)
            self.combo_matlab_anchor["values"] = columns
            if columns:
                # Cerca di indovinare se c'è una colonna 'baseline'
                default_idx = next((i for i, c in enumerate(columns) if "baseline" in c.lower()), 0)
                self.combo_matlab_anchor.current(default_idx)

            # 2. Popola menu Condizione (Sezione 3 - Quello che non ti funzionava)
            self.combo_cond_col["values"] = columns
            if columns:
                # Cerca di selezionare automaticamente "Condition" se esiste
                cond_idx = next((i for i, c in enumerate(columns) if "cond" in c.lower()), 0)
                self.combo_cond_col.current(cond_idx)

            # 3. Popola lista colonne disponibili (Sezione 3)
            self.lb_avail.delete(0, tk.END)
            for c in columns:
                self.lb_avail.insert(tk.END, c)

            # 4. Popola Anchor Fasi Fisse (Sezione 4)
            fixed_opts = ["auto (Fine ultima fase)"] + columns
            self.combo_fixed_anchor["values"] = fixed_opts
            self.combo_fixed_anchor.current(0)

            self.lbl_csv_status.config(text=f"OK! Trovate {len(columns)} colonne.", fg="green")

        except Exception as e:
            print(f"Errore caricamento CSV: {e}")
            messagebox.showerror("Errore CSV", f"Impossibile leggere le colonne.\nErrore: {str(e)}")

    def add_to_sequence(self):
        indices = self.lb_avail.curselection()
        for i in indices:
            val = self.lb_avail.get(i)
            # Evita duplicati se necessario, o permetti
            all_curr = self.lb_selected.get(0, tk.END)
            if val not in all_curr:
                self.lb_selected.insert(tk.END, val)

    def remove_from_sequence(self):
        indices = self.lb_selected.curselection()
        # Delete in reverse order to keep indices valid
        for i in reversed(indices):
            self.lb_selected.delete(i)

    def add_fixed_phase_dialog(self):
        name = simpledialog.askstring("Nuova Fase", "Nome della fase (es. ITI):")
        if not name:
            return
        dur = simpledialog.askfloat("Durata", "Durata in secondi (es. 10.0):")
        if dur is None:
            return
        self.tree_fixed.insert("", tk.END, values=(name, dur))

    def remove_fixed_phase(self):
        sel = self.tree_fixed.selection()
        for item in sel:
            self.tree_fixed.delete(item)

    def save_profile(self):
        # 1. Raccolta dati UI
        p_name = self.entry_name.get()
        if not p_name:
            messagebox.showwarning("Manca nome", "Inserisci un nome per il file.")
            return

        seq_cols = list(self.lb_selected.get(0, tk.END))
        if not seq_cols:
            messagebox.showwarning("Manca Sequenza", "Seleziona almeno due colonne per la sequenza temporale.")
            return

        # Fasi fisse
        fixed_phases = []
        for child in self.tree_fixed.get_children():
            vals = self.tree_fixed.item(child)["values"]
            fixed_phases.append({"name": vals[0], "duration": float(vals[1])})

        # Anchor fissa
        raw_anchor = self.combo_fixed_anchor.get()
        fixed_anchor = "auto" if "auto" in raw_anchor else raw_anchor

        # Genera etichette automatiche per le fasi variabili se non specificate
        # Questo wizard semplice assume phase1, phase2... o usa i nomi colonne
        # Per semplicità qui usiamo nomi generici o basati sulle colonne start
        # Un sistema più complesso chiederebbe i nomi per ogni intervallo.
        auto_labels = [f"Phase_{col}" for col in seq_cols[:-1]]

        profile_data = {
            "name": p_name,
            "description": self.entry_desc.get(),
            "csv_structure": {"condition_column": self.combo_cond_col.get(), "sequence_columns": seq_cols},
            "phases_labels": auto_labels,
            "fixed_phase_anchor_column": fixed_anchor,
            "append_fixed_phases": fixed_phases,
            "sync_logic": {
                "tobii_event_label": self.combo_tobii_event.get(),
                "matlab_anchor_column": self.combo_matlab_anchor.get(),
                "seconds_offset": float(self.entry_offset.get() or 0.0),
            },
        }

        try:
            path = self.logic.save_profile(profile_data)
            messagebox.showinfo("Successo", f"Profilo salvato correttamente in:\n{path}")
        except Exception as e:
            messagebox.showerror("Errore Salvataggio", str(e))


# --- MOTORE LOGICO ---
class TOIGenerator:
    @staticmethod
    def parse_time_string(time_str):
        """Converte stringhe 'HH:MM:SS.mmm' in secondi totali."""
        try:
            if pd.isna(time_str):
                return None
            # If data comes from .mat, it might already be a float/int
            if isinstance(time_str, float | int):
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

        if ext == ".csv":
            return pd.read_csv(path)

        elif ext == ".mat":
            try:
                mat_contents = sio.loadmat(path)
                # Filter out internal keys starting with '__' (e.g., __header__, __version__)
                valid_keys = [k for k in mat_contents.keys() if not k.startswith("__")]

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
                raise ValueError(f"Failed to parse .mat file: {e}") from e

        else:
            raise ValueError(f"Unsupported format: {ext}")

    @staticmethod
    def process(matlab_path, json_path, profile, output_path):
        # 1. Carica Evento Tobii
        try:
            with open(json_path) as f:
                tobii_data = json.load(f)

            # Leggi quale label cercare dal profilo (Default: "Start")
            sync_logic = profile.get("sync_logic", {})
            target_label = sync_logic.get("tobii_event_label", "Start")

            if isinstance(tobii_data, list):
                target_event = next((e for e in tobii_data if e.get("label", "").lower() == target_label.lower()), None)
                if not target_event:
                    # Fallback sul primo evento se non trova la label specifica
                    print(f"Warning: Label '{target_label}' not found. Using first available event.")
                    target_event = tobii_data[0]
            else:
                target_event = tobii_data

            tobii_ts = float(target_event["timestamp"])
            print(f"Tobii Sync Point ({target_label}): {tobii_ts}s")

        except Exception as e:
            raise ValueError(f"Tobii JSON Error: {e}") from e

        # 2. Load Matlab Data (CSV or MAT) <--- MODIFIED SECTION
        try:
            df = TOIGenerator._load_matlab_file(matlab_path)
            # Ensure column names are stripped of whitespace if loaded from loose CSVs
            df.columns = df.columns.astype(str).str.strip()
        except Exception as e:
            raise ValueError(f"Matlab Data Load Error: {e}") from e

        # 3. Estrai configurazione Profilo
        struct = profile.get("csv_structure", {})
        cols_seq = struct.get("sequence_columns", [])
        cond_col = struct.get("condition_column")
        labels = profile.get("phases_labels", [])

        # --- LOGICA DI SINCRONIZZAZIONE (Nuova V4) ---
        matlab_anchor_col = sync_logic.get("matlab_anchor_column")  # Es. 'baseline_end'
        offset_val = float(sync_logic.get("seconds_offset", 0.0))  # Es. -60.0

        if not matlab_anchor_col:
            raise ValueError("Invalid Profile: Missing 'matlab_anchor_column' in sync_logic.")

        if matlab_anchor_col not in df.columns:
            raise ValueError(
                f"Error: Sync column '{matlab_anchor_col}' defined in profile does NOT exist in loaded CSV.\n"
                f"Available columns: {list(df.columns)}"
            )

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
        fixed_phases = profile.get("append_fixed_phases", [])
        fixed_anchor_col = profile.get("fixed_phase_anchor_column", "auto")

        for idx, (_, row) in enumerate(df.iterrows()):
            trial_n = row.get("TrialN", idx + 1)
            condition = row.get(cond_col, "NA")

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
                end_t = seq_times[i + 1]
                p_name = labels[i] if i < len(labels) else f"Phase_{i+1}"

                toi_rows.append(
                    {
                        "Name": f"T{trial_n}_{condition}_{p_name}",
                        "Start": round(start_t, 3),
                        "End": round(end_t, 3),
                        "Duration": round(end_t - start_t, 3),
                        "Trial": trial_n,
                        "Condition": condition,
                        "Phase": p_name,
                    }
                )

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
                    toi_rows.append(
                        {
                            "Name": f"T{trial_n}_{condition}_{p_name}",
                            "Start": round(curr, 3),
                            "End": round(curr + dur, 3),
                            "Duration": dur,
                            "Trial": trial_n,
                            "Condition": condition,
                            "Phase": p_name,
                        }
                    )
                    curr += dur

        out_df = pd.DataFrame(toi_rows)
        out_df.to_csv(output_path, index=False, sep="\t")
        return len(toi_rows)


class MasterToiLogic:
    """
    Logica per la vista Master TOI.
    Collega il ProfileManager, il TOIGenerator e il DataCropper.
    """

    def __init__(self, profile_manager):
        self.pm = profile_manager

    def get_available_profiles(self):
        return self.pm.get_available_profiles()

    def generate_toi(self, matlab_path, tobii_path, profile_name, output_path):
        prof = self.pm.load_profile(profile_name)
        return TOIGenerator.process(matlab_path, tobii_path, prof, output_path)

    def crop_data(self, raw_path, toi_path):
        return DataCropper.crop_yolo_json(raw_path, toi_path)


class MasterToiView:
    def __init__(self, parent, context=None):
        self.parent = parent
        self.context = context

        # --- PATH MANAGEMENT ---
        if self.context and self.context.paths["profiles_toi"]:
            p_dir = self.context.paths["profiles_toi"]
        else:
            p_dir = "profiles_toi_fallback"

        self.pm = ProfileManager(profiles_dir=p_dir)
        self.logic = MasterToiLogic(self.pm)

        # Variabili UI
        self.tobii_file = tk.StringVar()
        self.matlab_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.selected_profile = tk.StringVar()
        self.yolo_raw_path = tk.StringVar()

        # Auto-load Optimization path
        if self.context and self.context.pose_data_path and self.context.pose_data_path.endswith(".json.gz"):
            self.yolo_raw_path.set(self.context.pose_data_path)

        self._build_ui()

    def _build_ui(self):
        # Header Principale
        header_frame = tk.Frame(self.parent, bg="white", pady=10)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="4. TOI Builder & Synchronization", font=("Segoe UI", 18, "bold"), bg="white").pack(
            anchor="w", padx=10
        )
        tk.Label(
            header_frame,
            text="Synchronizes data streams and defines Temporal Intervals of Interest (TOI).",
            font=("Segoe UI", 10),
            fg="#666",
            bg="white",
        ).pack(anchor="w", padx=10)

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
        ttk.Separator(lf_config, orient="horizontal").pack(fill="x", pady=10)

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
        tk.Button(
            lf_action,
            text="🚀 GENERATE TOI FILE (SYNC)",
            bg="#007ACC",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            height=2,
            cursor="hand2",
            command=self.run,
        ).pack(fill=tk.X, pady=10)

        # ==========================================
        # PARTE B: OTTIMIZZAZIONE (Sezione 4 Migliorata)
        # ==========================================

        # Separatore visivo forte tra le due fasi
        tk.Label(main, text="Advanced Tools", font=("Bold", 10), fg="#888").pack(anchor="w", pady=(10, 0))
        ttk.Separator(main, orient="horizontal").pack(fill="x", pady=5)

        # Frame dedicato con colore di sfondo leggermente diverso o bordo
        lf_opt = tk.LabelFrame(
            main, text="C. Data Pruning (Optional)", font=("Bold", 11), fg="#E65100", padx=10, pady=10
        )
        lf_opt.pack(fill=tk.X, pady=10)

        # Spiegazione User Friendly
        info_frame = tk.Frame(lf_opt)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        lbl_icon = tk.Label(info_frame, text="💡", font=("Arial", 16))
        lbl_icon.pack(side=tk.LEFT, anchor="n")
        lbl_desc = tk.Label(
            info_frame,
            justify="left",
            wraplength=500,
            text="This function prunes the original YOLO file (.json.gz) removing "
            "data before the experiment start (Sync).\n"
            "Recommended to reduce file size and speed up analysis.",
        )
        lbl_desc.pack(side=tk.LEFT, padx=10)

        # Input File Raw
        self._add_file_row(lf_opt, "Raw YOLO File (.gz):", self.yolo_raw_path, "*.json.gz")

        # Bottone Azione Ottimizzazione
        btn_opt = tk.Button(
            lf_opt,
            text="✂️ PRUNE & OPTIMIZE DATA",
            bg="#ff9800",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            command=self.run_cropping,
        )
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
        v = self.logic.get_available_profiles()
        self.cb["values"] = v
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
            n = self.logic.generate_toi(
                self.matlab_file.get(), self.tobii_file.get(), self.selected_profile.get(), self.output_file.get()
            )
            if self.context:
                self.context.toi_path = self.output_file.get()
            messagebox.showinfo(
                "Success", f"Generated {n} TOI intervals.\nYou can now proceed to optimization (Step C) if needed."
            )
        except Exception as e:
            messagebox.showerror("Generation Error", str(e))

    def launch_wizard(self):
        win = tk.Toplevel(self.parent)
        win.title("Profile Wizard")
        ProfileWizard(win, profiles_dir=self.pm.profiles_dir)
        # Aggiorna la lista profili quando il wizard viene chiuso
        win.bind("<Destroy>", lambda e: self.refresh_profiles())

    def run_cropping(self):
        raw_in = self.yolo_raw_path.get().strip()
        toi_in = self.output_file.get().strip()

        if not raw_in or not os.path.exists(raw_in):
            messagebox.showwarning("Missing File", "Select a valid YOLO .json.gz file in section C.")
            return

        if not toi_in or not os.path.exists(toi_in):
            messagebox.showwarning("Missing File", "TOI file not found. Run Step B (Generate) first.")
            return

        cropped_path = self.logic.crop_data(raw_in, toi_in)

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
        Genera anche un CSV appiattito (long-format) corrispondente.
        """
        try:
            print(f"✂️ Starting RAW pruning on: {os.path.basename(json_gz_path)}")

            # 1. Trova il tempo di inizio minimo dai TOI
            try:
                df_toi = pd.read_csv(toi_path, sep="\t")
            except Exception:
                # Fallback se il separatore fosse diverso
                df_toi = pd.read_csv(toi_path)

            if df_toi.empty:
                print("⚠️ Empty TOI file, cannot prune.")
                return None

            start_cut_time = df_toi["Start"].min()
            print(f"⏱️ Cut-off time identified: {start_cut_time:.3f} sec")

            # 2. Setup percorsi
            path_parts = os.path.splitext(os.path.splitext(json_gz_path)[0])  # Rimuove .gz poi .json
            # Ricostruisce nome: base + suffix + .json.gz
            new_path = f"{path_parts[0]}{output_suffix}.json.gz"
            csv_path = f"{path_parts[0]}{output_suffix}.csv"

            # Header CSV
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
            header = ["Frame", "Timestamp", "TrackID", "Conf", "Box_X1", "Box_Y1", "Box_X2", "Box_Y2"]
            for kp in kp_names:
                header.extend([f"{kp}_X", f"{kp}_Y", f"{kp}_C"])

            kept_frames = 0
            dropped_frames = 0

            # 3. Streaming Read/Write (Memoria efficiente)
            print("⏳ Processing (Streaming)...")
            with (
                gzip.open(json_gz_path, "rt", encoding="utf-8") as f_in,
                gzip.open(new_path, "wt", encoding="utf-8") as f_out,
                open(csv_path, "w", newline="", encoding="utf-8") as f_csv,
            ):
                writer = csv.writer(f_csv)
                writer.writerow(header)

                for line in f_in:
                    try:
                        # Parsing veloce solo per leggere il timestamp
                        # Nota: se il file è enorme, json.loads su ogni riga è il collo di bottiglia,
                        # ma è comunque il metodo più sicuro.
                        frame = json.loads(line)
                        ts = frame.get("ts", 0.0)

                        if ts >= start_cut_time:
                            # Scrive la riga originale intatta (veloce)
                            f_out.write(line)
                            kept_frames += 1

                            # Scrive le righe nel CSV (Long Format)
                            f_idx = frame.get("f_idx", -1)
                            for det in frame.get("det", []):
                                row = [f_idx, ts, det.get("track_id", -1), det.get("conf", 0)]
                                b = det.get("box", {})
                                row.extend([b.get("x1", 0), b.get("y1", 0), b.get("x2", 0), b.get("y2", 0)])
                                kps = det.get("keypoints", [])
                                if not kps:
                                    row.extend([0] * (17 * 3))
                                else:
                                    for point in kps:
                                        row.extend(point)
                                        if len(point) < 3:
                                            row.append(0)
                                writer.writerow(row)
                        else:
                            dropped_frames += 1

                    except json.JSONDecodeError:
                        continue

            print("✅ Pruning complete.")
            print(f"   Frames removed:   {dropped_frames}")
            print(f"   Frames kept: {kept_frames}")
            print(f"   Saved in:      {os.path.basename(new_path)}")
            print(f"   Saved CSV in:  {os.path.basename(csv_path)}")

            return new_path

        except Exception as e:
            print(f"❌ Error during JSON pruning: {e}")
            return None
