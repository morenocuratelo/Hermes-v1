import os
import json
import pandas as pd
import scipy.io as sio
from datetime import datetime

class HermesDiagnostics:
    """
    Strumento di debug per analizzare file di input e configurazioni
    senza avviare l'intera interfaccia grafica.
    """
    
    def __init__(self):
        self.logs = []

    def log(self, msg, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{level}] {msg}"
        print(entry)
        self.logs.append(entry)

    def analyze_data_file(self, path):
        self.log(f"Analisi file dati: {path}")
        
        if not os.path.exists(path):
            self.log("File non trovato!", "ERROR")
            return

        ext = os.path.splitext(path)[1].lower()
        
        try:
            if ext == '.csv' or ext == '.txt':
                # Tenta di indovinare il separatore
                df = pd.read_csv(path, nrows=5, sep=None, engine='python')
                self.log(f"CSV caricato. Colonne rilevate ({len(df.columns)}):")
                self.log(f"   {list(df.columns)}")
                self.log(f"Esempio prima riga: {df.iloc[0].to_dict()}")
                
                # Check per colonne vuote o con spazi
                dirty_cols = [c for c in df.columns if str(c).startswith(' ') or str(c).endswith(' ')]
                if dirty_cols:
                    self.log(f"ATTENZIONE: Rilevati spazi nei nomi colonne: {dirty_cols}", "WARN")

            elif ext == '.mat':
                mat = sio.loadmat(path)
                keys = [k for k in mat.keys() if not k.startswith('__')]
                self.log(f"MAT file caricato. Variabili trovate: {keys}")
                if keys:
                    data = mat[keys[0]]
                    self.log(f"Struttura prima variabile ({keys[0]}): {type(data)}")
                    if hasattr(data, 'shape'):
                        self.log(f"Shape: {data.shape}")
            elif ext == '.json':
                 with open(path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.log(f"JSON Lista eventi: {len(data)} elementi.")
                        if len(data) > 0:
                            self.log(f"Chiavi primo evento: {list(data[0].keys())}")
                    elif isinstance(data, dict):
                        self.log(f"JSON Dizionario. Chiavi: {list(data.keys())}")
            else:
                self.log(f"Estensione {ext} non supportata ufficialmente.", "WARN")
                
        except Exception as e:
            self.log(f"Errore durante la lettura del file: {e}", "ERROR")

    def validate_profile(self, profile_path):
        self.log(f"Validazione profilo: {profile_path}")
        
        if not os.path.exists(profile_path):
            self.log("File profilo non trovato!", "ERROR")
            return

        try:
            with open(profile_path, 'r') as f:
                data = json.load(f)
            
            # Controlli strutturali
            required = ['name', 'sync_logic', 'csv_structure']
            missing = [k for k in required if k not in data]
            
            if missing:
                self.log(f"Profilo incompleto. Mancano: {missing}", "ERROR")
            else:
                self.log("Struttura JSON base valida.")
                
            # Verifica logica sync
            sync = data.get('sync_logic', {})
            anchor = sync.get('matlab_anchor_column')
            if not anchor:
                self.log("Manca 'matlab_anchor_column' in sync_logic", "WARN")
            else:
                self.log(f"Anchor Column attesa: '{anchor}'")
            
            # Verifica struttura CSV
            struct = data.get('csv_structure', {})
            seq = struct.get('sequence_columns', [])
            if not seq:
                self.log("Nessuna colonna sequenza definita in csv_structure.", "WARN")
            else:
                self.log(f"Sequenza temporale definita su {len(seq)} colonne.")

        except json.JSONDecodeError:
            self.log("Il file non è un JSON valido.", "ERROR")
        except Exception as e:
            self.log(f"Errore generico: {e}", "ERROR")

if __name__ == "__main__":
    diag = HermesDiagnostics()
    print("--- HERMES DIAGNOSTICS TOOL ---")
    print("Trascina qui un file (.csv, .mat, .json) per analizzarlo e premi Invio.")
    
    while True:
                 
        try:
            user_input = input("\nFile > ").strip().strip('"')
            if not user_input:
                break


            if "profile" in user_input.lower() and user_input.endswith('.json'):
                diag.validate_profile(user_input)
            else:
                diag.analyze_data_file(user_input)
        except KeyboardInterrupt:
            break
