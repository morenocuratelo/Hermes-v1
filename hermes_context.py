import torch
import os
import shutil
import json

class AppContext:
    def __init__(self):
        # 1. Hardware Initialization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "No GPU"
        print(f"SYSTEM: Context initialized on {self.device} ({self.gpu_name})")

        # Config Persistence
        self.config_file = "hermes_config.json"
        self.last_project = None
        self.recent_files = {"video": [], "data": []}

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
        self.gaze_data_path = None      # .gz (Tobii)
        self.pose_data_path = None      # .json.gz
        self.identity_map_path = None   # .json
        self.toi_path = None            # .tsv
        self.mapped_csv_path = None     # .csv
        self.aoi_csv_path = None        # .csv
        self.export_path = None         # Ultimo export CSV AOI
        self.yolo_model_path = None     # .pt
        
        # Mapping Ruoli/Colori condiviso
        self.cast = {} 

        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.last_project = data.get("last_project")
                    self.recent_files = data.get("recent_files", {"video": [], "data": []})
            except Exception as e:
                print(f"Config Load Error: {e}")

    def save_config(self):
        data = {
            "last_project": self.project_path,
            "recent_files": self.recent_files
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Config Save Error: {e}")

    def add_recent_file(self, category, path):
        if not path: return
        if category not in self.recent_files: self.recent_files[category] = []
        if path in self.recent_files[category]:
            self.recent_files[category].remove(path)
        self.recent_files[category].insert(0, path)
        self.recent_files[category] = self.recent_files[category][:10]
        self.save_config()

    def update_video(self, path):
        """Aggiorna il percorso video nel contesto."""
        if path and os.path.exists(path):
            self.video_path = path
            self.add_recent_file("video", path)
            print(f"CONTEXT: Video updated -> {path}")

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
                print(f"Created folder: {path}")

        # Crea Profili Default (Se vuoti)
        self._create_default_toi_profile()
        # self._create_default_aoi_profile() # Opzionale, gestito da RegionView se manca
        
        # Scansiona file esistenti
        self._scan_existing_files()
        
        self.save_config()
        print(f"PROJECT INITIALIZED: {folder_path}")

    def import_file(self, source_path, category="input"):
        """Copia un file nella cartella del progetto."""
        if not source_path or not os.path.exists(source_path): return None
        
        # Track recent data files
        if source_path.lower().endswith(('.json', '.csv', '.mat', '.tsv', '.gz')):
            self.add_recent_file("data", source_path)

        filename = os.path.basename(source_path)
        dest_folder = self.paths["input"] # Default a Input
        
        dest_path = os.path.join(dest_folder, filename)
        
        # Evita di copiare se è già lì
        if os.path.abspath(source_path) == os.path.abspath(dest_path):
            return dest_path
            
        try:
            shutil.copy2(source_path, dest_path)
            print(f"Imported: {filename}")
            return dest_path
        except Exception as e:
            print(f"Import error {filename}: {e}")
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

    def _scan_existing_files(self):
        """Scansiona ricorsivamente tutte le sottocartelle per popolare il contesto con file esistenti."""
        if not self.project_path or not os.path.exists(self.project_path):
            return

        print(f"CONTEXT: Recursive scan in {self.project_path}...")

        for root, _, files in os.walk(self.project_path):
            for f in files:
                f_path = os.path.join(root, f)
                lower = f.lower()
                
                # 1. Video
                if lower.endswith(('.mp4', '.avi', '.mov')):
                    self.video_path = f_path
                    print(f"CONTEXT: Video detected -> {f}")
                
                # 2. Pose Data (YOLO)
                elif lower.endswith('_yolo.json.gz'):
                    self.pose_data_path = f_path
                    print(f"CONTEXT: Pose Data detected -> {f}")

                # 3. Gaze Data (Tobii) - .gz ma non yolo
                elif lower.endswith('.gz') and not lower.endswith('_yolo.json.gz'):
                    self.gaze_data_path = f_path
                    print(f"CONTEXT: Gaze Data detected -> {f}")

                # 4. Identity Map
                elif lower.endswith('_identity.json'):
                    self.identity_map_path = f_path
                    print(f"CONTEXT: Identity Map detected -> {f}")

                # 5. TOI
                elif lower.endswith('.tsv'):
                    self.toi_path = f_path
                    print(f"CONTEXT: TOI detected -> {f}")

                # 6. Mapped CSV
                elif 'mapped' in lower and lower.endswith('.csv'):
                    self.mapped_csv_path = f_path
                    print(f"CONTEXT: Mapped CSV detected -> {f}")

                # 7. AOI CSV
                elif lower.endswith('.csv') and 'results' not in lower:
                    self.aoi_csv_path = f_path
                    print(f"CONTEXT: AOI CSV detected -> {f}")

                # 8. YOLO Model
                elif lower.endswith('.pt'):
                    self.yolo_model_path = f_path
                    print(f"CONTEXT: YOLO Model detected -> {f}")