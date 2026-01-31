import torch
import os
import shutil
import json

class AppContext:
    def __init__(self):
        # 1. Hardware Initialization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "Nessuna GPU"
        print(f"SISTEMA: Context inizializzato su {self.device} ({self.gpu_name})")

        # 2. Project Management
        self.project_path = None
        self.paths = {
            "root": "",
            "input": "",
            "output": "",
            "profiles_toi": "",
            "profiles_aoi": "",
            "models": ""
        }
        
        # 3. Shared Data References
        self.video_path = None
        self.pose_data_path = None      # .json.gz
        self.identity_map_path = None   # .json
        self.toi_path = None            # .tsv
        self.mapped_csv_path = None     # .csv
        self.aoi_csv_path = None        # .csv
        self.export_path = None         # Ultimo export CSV AOI
        
        # Mapping Ruoli/Colori condiviso
        self.cast = {} 

    def update_video(self, path):
        """Aggiorna il percorso video nel contesto."""
        if path and os.path.exists(path):
            self.video_path = path
            print(f"CONTESTO: Video aggiornato -> {path}")

    def initialize_project(self, folder_path):
        """Crea la struttura delle cartelle e i file di default se non esistono."""
        self.project_path = folder_path
        
        # Definisci sottocartelle
        self.paths["root"] = folder_path
        self.paths["input"] = os.path.join(folder_path, "Input")
        self.paths["output"] = os.path.join(folder_path, "Output")
        self.paths["profiles_toi"] = os.path.join(folder_path, "Profiles", "TOI")
        self.paths["profiles_aoi"] = os.path.join(folder_path, "Profiles", "AOI")
        self.paths["models"] = os.path.join(folder_path, "Models")

        # Crea cartelle fisiche
        for key, path in self.paths.items():
            if key != "root" and not os.path.exists(path):
                os.makedirs(path)
                print(f"Creata cartella: {path}")

        # Crea Profili Default (Se vuoti)
        self._create_default_toi_profile()
        # self._create_default_aoi_profile() # Opzionale, gestito da RegionView se manca
        
        print(f"PROGETTO INIZIALIZZATO: {folder_path}")

    def import_file(self, source_path, category="input"):
        """Copia un file nella cartella del progetto."""
        if not source_path or not os.path.exists(source_path): return None
        
        filename = os.path.basename(source_path)
        dest_folder = self.paths["input"] # Default a Input
        
        dest_path = os.path.join(dest_folder, filename)
        
        # Evita di copiare se è già lì
        if os.path.abspath(source_path) == os.path.abspath(dest_path):
            return dest_path
            
        try:
            shutil.copy2(source_path, dest_path)
            print(f"Importato: {filename}")
            return dest_path
        except Exception as e:
            print(f"Errore importazione {filename}: {e}")
            return source_path

    def _create_default_toi_profile(self):
        # Crea un JSON di esempio se la cartella TOI è vuota
        if not os.path.exists(self.paths["profiles_toi"]): return
        if not os.listdir(self.paths["profiles_toi"]):
            default_prof = {
                "sync_logic": {"tobii_event_label": "DigIn", "matlab_anchor_column": "StudioEventData", "seconds_offset": 0.0},
                "csv_structure": {"sequence_columns": [], "condition_column": "Condition"},
                "phases_labels": ["Phase1", "Phase2"]
            }
            with open(os.path.join(self.paths["profiles_toi"], "Default_TOI.json"), 'w') as f:
                json.dump(default_prof, f, indent=4)