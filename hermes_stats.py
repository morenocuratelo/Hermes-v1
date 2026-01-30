import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import os

class GazeAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Gaze Data Analyzer - Lab Modigliani")
        self.root.geometry("700x550")
        
        # Variabili File
        self.mapped_csv_path = tk.StringVar()
        self.toi_tsv_path = tk.StringVar()
        
        # Parametri
        self.gaze_freq = tk.DoubleVar(value=50.0) # Frequenza Tobii Glasses (solitamente 50Hz o 100Hz)
        
        self._build_ui()

    def _build_ui(self):
        main = tk.Frame(self.root, padx=20, pady=20)
        main.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main, text="Gaze Analyzer: From Raw to Stats", font=("Segoe UI", 16, "bold")).pack(pady=(0,20))

        # 1. Input
        lf_in = tk.LabelFrame(main, text="1. File di Input", padx=10, pady=10)
        lf_in.pack(fill=tk.X, pady=5)
        
        self._add_picker(lf_in, "File MAPPED (.csv):", self.mapped_csv_path, "*.csv")
        self._add_picker(lf_in, "File TOI (.tsv):", self.toi_tsv_path, "*.tsv")

        # 2. Settings
        lf_set = tk.LabelFrame(main, text="2. Impostazioni Eye-Tracker", padx=10, pady=10)
        lf_set.pack(fill=tk.X, pady=5)
        
        tk.Label(lf_set, text="Frequenza Campionamento (Hz):").grid(row=0, column=0, sticky="w")
        tk.Entry(lf_set, textvariable=self.gaze_freq, width=10).grid(row=0, column=1, padx=10)
        tk.Label(lf_set, text="(Default Tobii Glasses 2/3: 50 o 100 Hz)", fg="gray").grid(row=0, column=2, sticky="w")

        # 3. Action
        tk.Button(main, text="GENERA REPORT STATISTICO", bg="#4CAF50", fg="white", font=("Bold", 12), height=2, command=self.run_analysis).pack(fill=tk.X, pady=20)
        
        self.lbl_status = tk.Label(main, text="In attesa...")
        self.lbl_status.pack()

    def _add_picker(self, p, lbl, var, ft):
        f = tk.Frame(p); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=lbl, width=20, anchor="w").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(f, text="...", width=3, command=lambda: self.browse(var, ft)).pack(side=tk.LEFT)

    def browse(self, var, ft):
        f = filedialog.askopenfilename(filetypes=[("File", ft)])
        if f: var.set(f)

    def run_analysis(self):
        if not self.mapped_csv_path.get() or not self.toi_tsv_path.get():
            messagebox.showwarning("Errore", "Seleziona entrambi i file (Mapped CSV e TOI TSV).")
            return
            
        try:
            self.lbl_status.config(text="Caricamento dati...")
            self.root.update()
            
            # 1. Carica Dati
            # Mapped Gaze
            df_gaze = pd.read_csv(self.mapped_csv_path.get())
            # TOI (Tab separated)
            df_toi = pd.read_csv(self.toi_tsv_path.get(), sep='\t')
            
            # Ordina gaze per timestamp per sicurezza
            df_gaze = df_gaze.sort_values('Timestamp')
            
            # Campionamento (durata di un singolo sample)
            sample_dur = 1.0 / self.gaze_freq.get()
            
            results = []
            
            self.lbl_status.config(text=f"Analisi di {len(df_toi)} fasi...")
            self.root.update()
            
            # 2. Loop sulle Fasi (TOI)
            for idx, row in df_toi.iterrows():
                phase_name = row['Name'] # Es. T1_Cond105_Approach
                t_start = row['Start']
                t_end = row['End']
                trial = row['Trial']
                cond = row['Condition']
                phase_type = row.get('Phase', 'Unknown')
                
                # Filtra i dati gaze dentro questa finestra temporale
                mask = (df_gaze['Timestamp'] >= t_start) & (df_gaze['Timestamp'] <= t_end)
                subset = df_gaze[mask]
                
                # Calcola Durate (in secondi)
                # Totale tempo registrato nella fase (esclusi sample persi)
                total_track_time = len(subset) * sample_dur
                
                # Scomposizione per Ruolo
                dur_target = len(subset[subset['Hit_Role'] == 'Target']) * sample_dur
                dur_confed = len(subset[subset['Hit_Role'].str.contains('Confederate', case=False, na=False)]) * sample_dur
                dur_none = len(subset[(subset['Hit_Role'] == 'None') | (subset['Hit_Role'].isna())]) * sample_dur
                
                # Scomposizione Specifica AOI Target
                dur_target_face = len(subset[(subset['Hit_Role'] == 'Target') & (subset['Hit_AOI'].isin(['Head', 'Eyes', 'Face']))]) * sample_dur
                dur_target_body = len(subset[(subset['Hit_Role'] == 'Target') & (subset['Hit_AOI'] == 'Body')]) * sample_dur
                dur_target_peri = len(subset[(subset['Hit_Role'] == 'Target') & (subset['Hit_AOI'] == 'Peripersonal')]) * sample_dur
                
                # Creazione riga risultato
                res_row = {
                    "Trial": trial,
                    "Condition": cond,
                    "Phase": phase_type,
                    "Phase_Full_ID": phase_name,
                    "Phase_Duration_Real": (t_end - t_start),
                    "Tracked_Time": total_track_time,
                    "Valid_Ratio": total_track_time / (t_end - t_start) if (t_end-t_start)>0 else 0,
                    
                    # Metriche Chiave
                    "Dur_Target_Total": dur_target,
                    "Dur_Confederate_Total": dur_confed,
                    "Dur_Background": dur_none,
                    
                    # Dettagli Target
                    "Dur_Target_Face": dur_target_face,
                    "Dur_Target_Body": dur_target_body,
                    "Dur_Target_Peripersonal": dur_target_peri,
                    
                    # Indici Derivati (Opzionali)
                    "Prop_Target_Face": dur_target_face / total_track_time if total_track_time > 0 else 0
                }
                
                results.append(res_row)
            
            # 3. Export
            df_res = pd.DataFrame(results)
            
            # Salva
            out_path = self.mapped_csv_path.get().replace("_MAPPED.csv", "_RESULTS.csv")
            if out_path == self.mapped_csv_path.get(): out_path += "_results.csv"
            
            df_res.to_csv(out_path, index=False)
            
            self.lbl_status.config(text="Fatto.")
            messagebox.showinfo("Analisi Completata", f"Report generato con successo:\n{out_path}\n\nOra puoi aprirlo in Excel/SPSS.")
            
        except Exception as e:
            messagebox.showerror("Errore Analisi", str(e))
            self.lbl_status.config(text="Errore.")

if __name__ == "__main__":
    root = tk.Tk()
    GazeAnalyzer(root)
    root.mainloop()