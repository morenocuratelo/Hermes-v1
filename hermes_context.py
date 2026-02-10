import torch
import os
import shutil
import json
import pandas as pd # Usato per timestamp, assicurati di averlo o usa datetime

class AppContext:
    def __init__(self):
        # 1. Hardware Initialization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "No GPU"
        print(f"SYSTEM: Context initialized on {self.device} ({self.gpu_name})")

        # Config Persistence (Globale dell'App, es. ultimi progetti aperti)
        self.config_file = "hermes_config.json"
        self.last_project = None
        self.recent_files = {"video": [], "data": []}

        # 2. Project State
        self.project_root = None
        self.project_config = {}
        
        # 3. Participant Management
        self.participants = []      # Lista ID stringhe: ["P001", "P002"]
        self.current_participant = None # ID attivo: "P001"
        
        # 4. Global Settings (Cast, Models)
        self.cast = {} 
        self.yolo_model_path = None # Modello globale per il progetto

        self.load_global_config()

    # ════════════════════════════════════════════════════════════════
    # GLOBAL APP CONFIG (Persistenza tra riavvi)
    # ════════════════════════════════════════════════════════════════

    def load_global_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.last_project = data.get("last_project")
                    self.recent_files = data.get("recent_files", {"video": [], "data": []})
            except Exception as e:
                print(f"Config Load Error: {e}")

    def save_global_config(self):
        data = {
            "last_project": self.project_root, # Salva il progetto corrente come ultimo
            "recent_files": self.recent_files
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Config Save Error: {e}")

    # ════════════════════════════════════════════════════════════════
    # PROJECT MANAGEMENT (Hub & Spoke)
    # ════════════════════════════════════════════════════════════════

    def create_project(self, parent_folder, name):
        """Crea una nuova cartella progetto strutturata."""
        self.project_root = os.path.join(parent_folder, name)
        
        # Struttura Cartelle
        os.makedirs(os.path.join(self.project_root, "assets", "models"), exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "assets", "profiles_aoi"), exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "assets", "profiles_toi"), exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "participants"), exist_ok=True)
        
        # Configurazione Iniziale
        self.project_config = {
            "name": name,
            "created_at": str(pd.Timestamp.now()),
            "hermes_version": "1.0",
            "description": "Multi-participant eye-tracking project"
        }
        self.participants = []
        self.current_participant = None
        self.save_project()
        self.save_global_config()
        print(f"PROJECT: Created at {self.project_root}")

    def load_project(self, project_dir):
        """Carica un progetto esistente e scansiona i partecipanti."""
        config_path = os.path.join(project_dir, "hermes_project.json")
        if not os.path.exists(config_path):
            raise ValueError("Not a valid HERMES project folder (missing hermes_project.json).")
        
        self.project_root = project_dir
        with open(config_path, 'r') as f:
            self.project_config = json.load(f)
            
        # Scansiona cartelle partecipanti
        p_dir = os.path.join(self.project_root, "participants")
        if os.path.exists(p_dir):
            self.participants = [d for d in os.listdir(p_dir) if os.path.isdir(os.path.join(p_dir, d))]
            self.participants.sort()
            
        # Imposta il primo come attivo se esiste
        if self.participants:
            self.set_active_participant(self.participants[0])
            
        self.save_global_config()
        print(f"PROJECT: Loaded from {project_dir}. Found {len(self.participants)} participants.")

    def save_project(self):
        """Salva lo stato del progetto (configurazione)."""
        if not self.project_root: return
        with open(os.path.join(self.project_root, "hermes_project.json"), 'w') as f:
            json.dump(self.project_config, f, indent=4)

    # ════════════════════════════════════════════════════════════════
    # PARTICIPANT MANAGEMENT
    # ════════════════════════════════════════════════════════════════

    def add_participant(self, pid):
        """Aggiunge un nuovo partecipante e crea le sue cartelle."""
        if not self.project_root: return
        if pid in self.participants:
            print(f"PARTICIPANT: {pid} already exists.")
            return

        p_path = os.path.join(self.project_root, "participants", pid)
        os.makedirs(os.path.join(p_path, "input"), exist_ok=True)
        os.makedirs(os.path.join(p_path, "output"), exist_ok=True)
        
        self.participants.append(pid)
        self.participants.sort()
        self.set_active_participant(pid)
        print(f"PARTICIPANT: Created {pid}")

    def set_active_participant(self, pid):
        """Cambia il focus del contesto su un altro partecipante."""
        if pid in self.participants:
            self.current_participant = pid
            print(f"PARTICIPANT: Active changed to -> {pid}")
            # Qui potremmo ricaricare dati specifici se necessario (es. cast specifico)

    # ════════════════════════════════════════════════════════════════
    # DYNAMIC PATH RESOLUTION (The Magic)
    # Espone proprietà compatibili con i vecchi script, ma che puntano
    # dinamicamente alle cartelle del partecipante attivo.
    # ════════════════════════════════════════════════════════════════

    def _get_active_input_dir(self):
        if not self.project_root or not self.current_participant: return None
        return os.path.join(self.project_root, "participants", self.current_participant, "input")

    def _get_active_output_dir(self):
        if not self.project_root or not self.current_participant: return None
        return os.path.join(self.project_root, "participants", self.current_participant, "output")

    def _find_file(self, folder, extensions):
        """Helper per trovare il primo file che matcha le estensioni."""
        if not folder or not os.path.exists(folder): return None
        for f in os.listdir(folder):
            if f.lower().endswith(extensions):
                return os.path.join(folder, f)
        return None

    # --- VIDEO (Read/Write) ---
    @property
    def video_path(self):
        # Cerca video nella cartella input del partecipante attivo
        return self._find_file(self._get_active_input_dir(), ('.mp4', '.avi', '.mov', '.mkv'))
    
    @video_path.setter
    def video_path(self, path):
        # Se viene settato un path esterno, lo copiamo/importiamo nel progetto?
        # Per ora stampiamo solo un warning, l'import deve essere esplicito.
        print(f"WARNING: Setting video_path directly is deprecated. Use import_file() instead.")

    # --- GAZE DATA (Tobii .gz) ---
    @property
    def gaze_data_path(self):
        # Attenzione: anche i file pose sono .gz, quindi escludiamo quelli che contengono "_yolo"
        folder = self._get_active_input_dir()
        if not folder or not os.path.exists(folder): return None
        for f in os.listdir(folder):
            if f.lower().endswith('.gz') and '_yolo' not in f.lower():
                return os.path.join(folder, f)
        return None
        
    @gaze_data_path.setter
    def gaze_data_path(self, val): pass

    # --- POSE DATA (YOLO Output) ---
    @property
    def pose_data_path(self):
        # Cerca nell'output (se generato) o nell'input (se importato manualmente)
        out_f = self._find_file(self._get_active_output_dir(), ('_yolo.json.gz',))
        if out_f: return out_f
        return self._find_file(self._get_active_input_dir(), ('_yolo.json.gz',))

    @pose_data_path.setter
    def pose_data_path(self, val): pass

    # --- IDENTITY MAP ---
    @property
    def identity_map_path(self):
        return self._find_file(self._get_active_output_dir(), ('_identity.json',))

    @identity_map_path.setter
    def identity_map_path(self, val): pass

    # --- TOI FILE ---
    @property
    def toi_path(self):
        # Cerca il TSV generato dal modulo MasterTOI nell'output
        out_f = self._find_file(self._get_active_output_dir(), ('_tois.tsv',))
        if out_f: return out_f
        # Fallback all'input (se fornito manualmente)
        return self._find_file(self._get_active_input_dir(), ('.tsv', '.txt'))

    @toi_path.setter
    def toi_path(self, val): pass

    # --- AOI / MAPPED FILES ---
    @property
    def aoi_csv_path(self):
        # Cerca file che finiscono con .csv ma NON contengono "mapped" o "results"
        folder = self._get_active_output_dir()
        if not folder: return None
        for f in os.listdir(folder):
            if f.endswith('.csv') and 'mapped' not in f.lower() and 'results' not in f.lower():
                return os.path.join(folder, f)
        return None
        
    @aoi_csv_path.setter
    def aoi_csv_path(self, val): pass

    @property
    def mapped_csv_path(self):
        return self._find_file(self._get_active_output_dir(), ('_mapped.csv',))

    @mapped_csv_path.setter
    def mapped_csv_path(self, val): pass

    # --- DIZIONARIO PATHS (Compatibilità) ---
    @property
    def paths(self):
        """Restituisce un dizionario compatibile con i vecchi script."""
        if not self.project_root:
            return {"profiles_aoi": "", "profiles_toi": "", "output": ""}
        
        return {
            "profiles_aoi": os.path.join(self.project_root, "assets", "profiles_aoi"),
            "profiles_toi": os.path.join(self.project_root, "assets", "profiles_toi"),
            "models": os.path.join(self.project_root, "assets", "models"),
            "output": self._get_active_output_dir() or ""
        }

    # ════════════════════════════════════════════════════════════════
    # FILE IMPORT UTILITIES
    # ════════════════════════════════════════════════════════════════

    def import_file_for_participant(self, pid, source_path, file_type):
        """
        Importa un file (copia) nella cartella input del partecipante specificato.
        file_type: 'video', 'gaze', 'toi'
        """
        if not self.project_root: return None
        dest_dir = os.path.join(self.project_root, "participants", pid, "input")
        if not os.path.exists(dest_dir): return None
        
        filename = os.path.basename(source_path)
        dest_path = os.path.join(dest_dir, filename)
        
        try:
            shutil.copy2(source_path, dest_path)
            print(f"IMPORT: {filename} -> {pid}/input")
            return dest_path
        except Exception as e:
            print(f"IMPORT ERROR: {e}")
            return None