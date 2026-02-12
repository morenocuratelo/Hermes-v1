import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import json
import pandas as pd
import os

class ProfileWizard:
    def __init__(self, root):
        self.root = root
        self.root.title("Universal Profile Generator - Lab Modigliani")
        self.root.geometry("900x850")
        
        # Variabili Dati Caricati
        self.loaded_json_events = []
        self.loaded_csv_columns = []
        
        # --- UI LAYOUT ---
        canvas = tk.Canvas(root)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

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
        tk.Button(lf, text="Carica User Event Tobii (.json)", command=self.load_json_sample).grid(row=0, column=0, padx=5, sticky="w")
        self.lbl_json_status = tk.Label(lf, text="Nessun file caricato", fg="red")
        self.lbl_json_status.grid(row=0, column=1, padx=5)

        # CSV Loader
        tk.Button(lf, text="Carica Results Matlab (.csv)", command=self.load_csv_sample).grid(row=1, column=0, padx=5, sticky="w", pady=5)
        self.lbl_csv_status = tk.Label(lf, text="Nessun file caricato", fg="red")
        self.lbl_csv_status.grid(row=1, column=1, padx=5)

    def _build_sync_logic(self):
        lf = tk.LabelFrame(self.scrollable_frame, text="2. Logica di Sincronizzazione (Anchor)", padx=10, pady=10)
        lf.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(lf, text="A. Quale EVENTO Tobii è il punto di sync?").grid(row=0, column=0, sticky="w")
        self.combo_tobii_event = ttk.Combobox(lf, state="readonly", width=40)
        self.combo_tobii_event.grid(row=1, column=0, padx=5, pady=(0,10))

        tk.Label(lf, text="B. Quale COLONNA Matlab corrisponde a quell'evento?").grid(row=0, column=1, sticky="w")
        self.combo_matlab_anchor = ttk.Combobox(lf, state="readonly", width=40)
        self.combo_matlab_anchor.grid(row=1, column=1, padx=5, pady=(0,10))

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
        tk.Label(lf, text="Costruisci la sequenza temporale (Sposta le colonne da SX a DX nell'ordine corretto):").pack(anchor="w")
        
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
        tk.Button(f, text="SALVA PROFILO JSON", bg="#4CAF50", fg="white", font=("Arial", 14, "bold"), height=2, command=self.save_profile).pack(fill=tk.X, padx=50)

    # --- LOGICA ---

    def load_json_sample(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Estrai label uniche
            labels = set()
            if isinstance(data, list):
                for e in data:
                    if 'label' in e:
                        labels.add(e['label'])
            elif isinstance(data, dict):
                if 'label' in data:
                    labels.add(data['label'])
            
            self.loaded_json_events = sorted(list(labels))
            self.combo_tobii_event['values'] = self.loaded_json_events
            if self.loaded_json_events:
                self.combo_tobii_event.current(0)
            
            self.lbl_json_status.config(text=f"OK! Trovati {len(labels)} eventi.", fg="green")
        except Exception as e:
            messagebox.showerror("Errore JSON", str(e))

    def load_csv_sample(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Text", "*.txt"), ("All", "*.*")])
        if not path:
            return
        
        try:
            # --- MODIFICA CRITICA: Rilevamento automatico separatore ---
            # sep=None con engine='python' permette a pandas di indovinare se usare , o ; o \t
            df = pd.read_csv(path, nrows=2, sep=None, engine='python')
            
            # Pulizia nomi colonne (rimuove spazi vuoti extra che spesso Matlab lascia)
            self.loaded_csv_columns = [str(c).strip() for c in df.columns]
            
            print(f"DEBUG - Colonne trovate: {self.loaded_csv_columns}") # Controllo console

            # 1. Popola menu Anchor Matlab (Sezione 2)
            self.combo_matlab_anchor['values'] = self.loaded_csv_columns
            if self.loaded_csv_columns:
                # Cerca di indovinare se c'è una colonna 'baseline'
                default_idx = next((i for i, c in enumerate(self.loaded_csv_columns) if 'baseline' in c.lower()), 0)
                self.combo_matlab_anchor.current(default_idx)

            # 2. Popola menu Condizione (Sezione 3 - Quello che non ti funzionava)
            self.combo_cond_col['values'] = self.loaded_csv_columns
            if self.loaded_csv_columns:
                # Cerca di selezionare automaticamente "Condition" se esiste
                cond_idx = next((i for i, c in enumerate(self.loaded_csv_columns) if 'cond' in c.lower()), 0)
                self.combo_cond_col.current(cond_idx)

            # 3. Popola lista colonne disponibili (Sezione 3)
            self.lb_avail.delete(0, tk.END)
            for c in self.loaded_csv_columns:
                self.lb_avail.insert(tk.END, c)
            
            # 4. Popola Anchor Fasi Fisse (Sezione 4)
            fixed_opts = ["auto (Fine ultima fase)"] + self.loaded_csv_columns
            self.combo_fixed_anchor['values'] = fixed_opts
            self.combo_fixed_anchor.current(0)

            self.lbl_csv_status.config(text=f"OK! Trovate {len(self.loaded_csv_columns)} colonne.", fg="green")
            
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
            vals = self.tree_fixed.item(child)['values']
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
            "csv_structure": {
                "condition_column": self.combo_cond_col.get(),
                "sequence_columns": seq_cols
            },
            "phases_labels": auto_labels,
            "fixed_phase_anchor_column": fixed_anchor,
            "append_fixed_phases": fixed_phases,
            "sync_logic": {
                "tobii_event_label": self.combo_tobii_event.get(),
                "matlab_anchor_column": self.combo_matlab_anchor.get(),
                "seconds_offset": float(self.entry_offset.get() or 0.0)
            }
        }

        # Salvataggio
        if not os.path.exists("profiles"):
            os.makedirs("profiles")
        
        # Sanitize filename
        safe_name = "".join([c if c.isalnum() else "_" for c in p_name]).lower() + ".json"
        path = os.path.join("profiles", safe_name)
        
        try:
            with open(path, 'w') as f:
                json.dump(profile_data, f, indent=4)
            messagebox.showinfo("Successo", f"Profilo salvato correttamente in:\n{path}")
        except Exception as e:
            messagebox.showerror("Errore Salvataggio", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    ProfileWizard(root)
    root.mainloop()
