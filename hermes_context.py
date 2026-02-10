import torch
import os
import shutil
import json
import datetime

class AppContext:
    def __init__(self):
        # 1. Hardware Initialization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "No GPU"
        print(f"SYSTEM: Context initialized on {self.device} ({self.gpu_name})")

        # Global Persistence (File di configurazione generale dell'app)
        self.config_file = "hermes_global_config.json"
        self.last_project = None
        self.recent_files = {"video": [], "data": []}

        # 2. Project State (In Memory)
        self.project_root = None
        self.project_config = {}
        
        # 3. Participant Management
        self.participants = []          # List of IDs: ["P001", "P002"]
        self.current_participant = None # Active ID: "P001"
        
        # 4. Global Settings (Shared across project)
        self.cast = {} 
        self.yolo_model_path = None 

        # Load global app config on startup
        self.load_global_config()

    # ════════════════════════════════════════════════════════════════
    # GLOBAL APP CONFIG (Last opened project, etc.)
    # ════════════════════════════════════════════════════════════════

    def load_global_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.last_project = data.get("last_project")
                    self.recent_files = data.get("recent_files", {"video": [], "data": []})
            except Exception as e:
                print(f"Global Config Load Error: {e}")

    def save_global_config(self):
        data = {
            "last_project": self.project_root,
            "recent_files": self.recent_files
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Global Config Save Error: {e}")

    # ════════════════════════════════════════════════════════════════
    # PROJECT MANAGEMENT (Hub & Spoke Structure)
    # ════════════════════════════════════════════════════════════════

    def create_project(self, parent_folder, name):
        """Creates the folder structure for a new multi-participant project."""
        self.project_root = os.path.join(parent_folder, name)
        
        if os.path.exists(self.project_root):
            raise FileExistsError(f"Folder '{name}' already exists in that location.")

        # Create Hierarchy
        os.makedirs(os.path.join(self.project_root, "assets", "models"), exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "assets", "profiles_aoi"), exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "assets", "profiles_toi"), exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "participants"), exist_ok=True)
        
        # Initial Config
        self.project_config = {
            "name": name,
            "created_at": str(datetime.datetime.now()),
            "hermes_version": "1.0",
            "description": "Multi-participant eye-tracking project"
        }
        self.participants = []
        self.current_participant = None
        
        self.save_project()
        self.save_global_config()
        print(f"PROJECT: Created at {self.project_root}")

    def load_project(self, project_dir):
        """Loads an existing project and scans for participants."""
        config_path = os.path.join(project_dir, "hermes_project.json")
        if not os.path.exists(config_path):
            raise ValueError("Not a valid HERMES project folder (missing hermes_project.json).")
        
        self.project_root = project_dir
        with open(config_path, 'r') as f:
            self.project_config = json.load(f)
            
        # Scan participant folders
        p_dir = os.path.join(self.project_root, "participants")
        if os.path.exists(p_dir):
            self.participants = [d for d in os.listdir(p_dir) if os.path.isdir(os.path.join(p_dir, d))]
            self.participants.sort()
            
        # Auto-select first participant if available
        if self.participants:
            self.set_active_participant(self.participants[0])
        else:
            self.current_participant = None
            
        self.save_global_config()
        print(f"PROJECT: Loaded from {project_dir}. Found {len(self.participants)} participants.")

    def save_project(self):
        """Persist project metadata to JSON."""
        if not self.project_root: return
        with open(os.path.join(self.project_root, "hermes_project.json"), 'w') as f:
            json.dump(self.project_config, f, indent=4)

    # ════════════════════════════════════════════════════════════════
    # PARTICIPANT MANAGEMENT
    # ════════════════════════════════════════════════════════════════

    def add_participant(self, pid):
        """Adds a new participant folder structure."""
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
        """Sets the 'focus' of the context to a specific participant."""
        if pid in self.participants:
            self.current_participant = pid
            print(f"PARTICIPANT: Active changed to -> {pid}")
            # Optional: Clear cached paths if any
        else:
            print(f"PARTICIPANT: Error, {pid} not found.")

    # ════════════════════════════════════════════════════════════════
    # DYNAMIC PATH RESOLUTION (The Magic Trick)
    # These properties allow modules (Human, Entity, etc.) to ask for
    # "video_path" and get the file for the ACTIVE participant automatically.
    # ════════════════════════════════════════════════════════════════

    def _get_active_input_dir(self):
        if not self.project_root or not self.current_participant: return None
        return os.path.join(self.project_root, "participants", self.current_participant, "input")

    def _get_active_output_dir(self):
        if not self.project_root or not self.current_participant: return None
        return os.path.join(self.project_root, "participants", self.current_participant, "output")

    def _find_file(self, folder, extensions_or_suffix, exclude=None):
        """Helper to find files in the active folders."""
        if not folder or not os.path.exists(folder): return None
        
        candidates = []
        for f in os.listdir(folder):
            f_lower = f.lower()
            
            # Check exclusions
            if exclude and exclude in f_lower:
                continue
                
            # Check match (extension tuple OR suffix string)
            is_match = False
            if isinstance(extensions_or_suffix, tuple):
                if f_lower.endswith(extensions_or_suffix): is_match = True
            elif isinstance(extensions_or_suffix, str):
                if f_lower.endswith(extensions_or_suffix): is_match = True
                
            if is_match:
                candidates.append(os.path.join(folder, f))
        
        # Return first match or None
        return candidates[0] if candidates else None

    # --- 1. VIDEO PATH ---
    @property
    def video_path(self):
        return self._find_file(self._get_active_input_dir(), ('.mp4', '.avi', '.mov', '.mkv'))
    
    @video_path.setter
    def video_path(self, val): 
        # Read-only property derived from file system state.
        pass

    # --- 2. GAZE DATA (Tobii .gz) ---
    @property
    def gaze_data_path(self):
        # Must be .gz but NOT _yolo.json.gz
        return self._find_file(self._get_active_input_dir(), '.gz', exclude='_yolo')
        
    @gaze_data_path.setter
    def gaze_data_path(self, val): pass

    # --- 3. POSE DATA (YOLO Output) ---
    @property
    def pose_data_path(self):
        # Prioritize Output folder, fallback to Input (if imported manually)
        out_f = self._find_file(self._get_active_output_dir(), '_yolo.json.gz')
        if out_f: return out_f
        return self._find_file(self._get_active_input_dir(), '_yolo.json.gz')

    @pose_data_path.setter
    def pose_data_path(self, val): pass

    # --- 4. IDENTITY MAP ---
    @property
    def identity_map_path(self):
        return self._find_file(self._get_active_output_dir(), '_identity.json')

    @identity_map_path.setter
    def identity_map_path(self, val): pass

    # --- 5. TOI FILE (Time Windows) ---
    @property
    def toi_path(self):
        # Look for generated _tois.tsv first
        out_f = self._find_file(self._get_active_output_dir(), '_tois.tsv')
        if out_f: return out_f
        # Fallback to any .tsv/.txt in input
        return self._find_file(self._get_active_input_dir(), ('.tsv', '.txt'))

    @toi_path.setter
    def toi_path(self, val): pass

    # --- 6. CSV OUTPUTS (AOI / Mapped) ---
    @property
    def aoi_csv_path(self):
        # Finds a CSV that is NOT "mapped" or "results" or "final"
        folder = self._get_active_output_dir()
        if not folder: return None
        for f in os.listdir(folder):
            lower = f.lower()
            if lower.endswith('.csv') and 'mapped' not in lower and 'results' not in lower and 'stats' not in lower:
                return os.path.join(folder, f)
        return None
        
    @aoi_csv_path.setter
    def aoi_csv_path(self, val): pass

    @property
    def mapped_csv_path(self):
        return self._find_file(self._get_active_output_dir(), '_mapped.csv')

    @mapped_csv_path.setter
    def mapped_csv_path(self, val): pass

    # --- PATHS DICTIONARY (Legacy Compatibility) ---
    @property
    def paths(self):
        """Returns path dict for modules expecting 'self.context.paths'."""
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

    def import_file_for_participant(self, pid, source_path):
        """
        Copies a file into the specific participant's INPUT folder.
        """
        if not self.project_root: return None
        dest_dir = os.path.join(self.project_root, "participants", pid, "input")
        if not os.path.exists(dest_dir): return None
        
        filename = os.path.basename(source_path)
        dest_path = os.path.join(dest_dir, filename)
        
        try:
            # If same path, skip
            if os.path.abspath(source_path) == os.path.abspath(dest_path):
                return dest_path
                
            shutil.copy2(source_path, dest_path)
            print(f"IMPORT: {filename} -> {pid}/input")
            return dest_path
        except Exception as e:
            print(f"IMPORT ERROR: {e}")
            return None