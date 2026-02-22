import datetime
import json
import os
import shutil

import torch


class AppContext:
    def __init__(self):
        # 1. Hardware Initialization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "No GPU"
        print(f"SYSTEM: Context initialized on {self.device} ({self.gpu_name})")

        # Global Persistence (File di configurazione generale dell'app)
        self.config_file = "hermes_global_config.json"
        self.default_config_file = "hermes_global_config.default.json"
        self.last_project = None
        self.recent_files: dict[str, list[str]] = {"video": [], "data": []}

        # 2. Project State (In Memory)
        self.project_root = None
        self.project_config = {}

        # 3. Participant Management
        self.participants = []  # List of IDs: ["P001", "P002"]
        self.current_participant = None  # Active ID: "P001"

        # 4. Global Settings (Shared across project)
        self.cast = {}
        self.yolo_model_path = None
        # Path overrides set by GUI/runtime (per participant, in-memory).
        # Structure: {participant_id_or_"__global__": {key: absolute_path}}
        self._manual_paths = {}

        # Load global app config on startup
        self.load_global_config()

    # ════════════════════════════════════════════════════════════════
    # GLOBAL APP CONFIG (Last opened project, etc.)
    # ════════════════════════════════════════════════════════════════

    def _default_global_config(self):
        return {"last_project": None, "recent_files": {"video": [], "data": []}}

    def _ensure_global_config_exists(self):
        if os.path.exists(self.config_file):
            return

        try:
            if os.path.exists(self.default_config_file):
                shutil.copy2(self.default_config_file, self.config_file)
                print(f"GLOBAL CONFIG: Created local config from '{self.default_config_file}'.")
            else:
                with open(self.config_file, "w", encoding="utf-8") as f:
                    json.dump(self._default_global_config(), f, indent=4)
                print("GLOBAL CONFIG: Template missing, created a minimal local config.")
        except Exception as e:
            print(f"Global Config Init Error: {e}")

    def load_global_config(self):
        self._ensure_global_config_exists()

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        raise ValueError("Invalid config format: expected JSON object.")

                    self.last_project = data.get("last_project")
                    self.recent_files = data.get("recent_files", {"video": [], "data": []})
                    if not isinstance(self.recent_files, dict):
                        self.recent_files = {"video": [], "data": []}
            except Exception as e:
                print(f"Global Config Load Error: {e}")
                self.last_project = None
                self.recent_files = {"video": [], "data": []}

    def save_global_config(self):
        data = {"last_project": self.project_root, "recent_files": self.recent_files}
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Global Config Save Error: {e}")

    # ════════════════════════════════════════════════════════════════
    # PROJECT MANAGEMENT (Hub & Spoke Structure)
    # ════════════════════════════════════════════════════════════════

    def create_project(self, parent_folder, name):
        """Creates the folder structure for a new multi-participant project."""
        self.project_root = os.path.join(parent_folder, name)
        self._manual_paths = {}

        if os.path.exists(self.project_root):
            raise FileExistsError(f"Folder '{name}' already exists in that location.")

        # Create Hierarchy
        os.makedirs(os.path.join(self.project_root, "assets", "models"), exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "assets", "profiles_aoi"), exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "assets", "profiles_toi"), exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "participants"), exist_ok=True)

        # --- AUTO-IMPORT MODELS ---
        # Copia i modelli _ready dalla cartella dell'applicazione al nuovo progetto
        app_dir = os.path.dirname(os.path.abspath(__file__))
        src_models_dir = os.path.join(app_dir, "Models")
        dst_models_dir = os.path.join(self.project_root, "assets", "models")

        if os.path.exists(src_models_dir):
            models_to_copy = ["resnet50_msmt17_ready.pt", "osnet_ain_x1_0_ready.pt"]
            for m in models_to_copy:
                src = os.path.join(src_models_dir, m)
                if os.path.exists(src):
                    try:
                        shutil.copy2(src, os.path.join(dst_models_dir, m))
                        print(f"PROJECT: Imported model {m}")
                    except Exception as e:
                        print(f"PROJECT: Failed to import {m}: {e}")

        # Initial Config
        self.project_config = {
            "name": name,
            "created_at": str(datetime.datetime.now()),
            "hermes_version": "1.0",
            "description": "Multi-participant eye-tracking project",
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
        self._manual_paths = {}
        with open(config_path) as f:
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
        if not self.project_root:
            return
        with open(os.path.join(self.project_root, "hermes_project.json"), "w") as f:
            json.dump(self.project_config, f, indent=4)

    # ════════════════════════════════════════════════════════════════
    # PARTICIPANT MANAGEMENT
    # ════════════════════════════════════════════════════════════════

    def add_participant(self, pid):
        """Adds a new participant folder structure."""
        if not self.project_root:
            return
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

    def _active_key(self):
        return self.current_participant or "__global__"

    def _get_manual_path(self, key):
        # Prefer active participant override, fallback to global override.
        active = self._active_key()
        active_map = self._manual_paths.get(active, {})
        if key in active_map:
            return active_map[key]
        global_map = self._manual_paths.get("__global__", {})
        return global_map.get(key)

    def _set_manual_path(self, key, val):
        active = self._active_key()
        if not val:
            if active in self._manual_paths and key in self._manual_paths[active]:
                del self._manual_paths[active][key]
                if not self._manual_paths[active]:
                    del self._manual_paths[active]
            return

        path = os.path.abspath(str(val))
        self._manual_paths.setdefault(active, {})[key] = path

    # ════════════════════════════════════════════════════════════════
    # DYNAMIC PATH RESOLUTION (The Magic Trick)
    # These properties allow modules (Human, Entity, etc.) to ask for
    # "video_path" and get the file for the ACTIVE participant automatically.
    # ════════════════════════════════════════════════════════════════

    def _get_active_input_dir(self):
        if not self.project_root or not self.current_participant:
            return None
        return os.path.join(self.project_root, "participants", self.current_participant, "input")

    def _get_active_output_dir(self):
        if not self.project_root or not self.current_participant:
            return None
        return os.path.join(self.project_root, "participants", self.current_participant, "output")

    def _find_file(self, folder, extensions_or_suffix, exclude=None):
        """Helper to find files in the active folders."""
        if not folder or not os.path.exists(folder):
            return None

        candidates = []
        for f in os.listdir(folder):
            f_lower = f.lower()

            # Check exclusions
            if exclude and exclude in f_lower:
                continue

            # Check match (extension tuple OR suffix string)
            is_match = False
            if isinstance(extensions_or_suffix, tuple):
                if f_lower.endswith(extensions_or_suffix):
                    is_match = True
            elif isinstance(extensions_or_suffix, str):
                if f_lower.endswith(extensions_or_suffix):
                    is_match = True

            if is_match:
                candidates.append(os.path.join(folder, f))

        # Return first match or None
        return candidates[0] if candidates else None

    # --- 1. VIDEO PATH ---
    @property
    def video_path(self):
        manual = self._get_manual_path("video_path")
        if manual:
            return manual
        return self._find_file(self._get_active_input_dir(), (".mp4", ".avi", ".mov", ".mkv"))

    @video_path.setter
    def video_path(self, val):
        self._set_manual_path("video_path", val)

    # --- 2. GAZE DATA (Tobii .gz) ---
    @property
    def gaze_data_path(self):
        manual = self._get_manual_path("gaze_data_path")
        if manual:
            return manual
        # Must be .gz but NOT _yolo.json.gz
        return self._find_file(self._get_active_input_dir(), ".gz", exclude="_yolo")

    @gaze_data_path.setter
    def gaze_data_path(self, val):
        self._set_manual_path("gaze_data_path", val)

    # --- 3. POSE DATA (YOLO Output) ---
    @property
    def pose_data_path(self):
        manual = self._get_manual_path("pose_data_path")
        if manual:
            return manual
        # Prioritize Output folder, fallback to Input (if imported manually)
        out_f = self._find_file(self._get_active_output_dir(), "_yolo.json.gz")
        if out_f:
            return out_f
        return self._find_file(self._get_active_input_dir(), "_yolo.json.gz")

    @pose_data_path.setter
    def pose_data_path(self, val):
        self._set_manual_path("pose_data_path", val)

    # --- 4. IDENTITY MAP ---
    @property
    def identity_map_path(self):
        manual = self._get_manual_path("identity_map_path")
        if manual:
            return manual
        return self._find_file(self._get_active_output_dir(), "_identity.json")

    @identity_map_path.setter
    def identity_map_path(self, val):
        self._set_manual_path("identity_map_path", val)

    # --- 5. TOI FILE (Time Windows) ---
    @property
    def toi_path(self):
        manual = self._get_manual_path("toi_path")
        if manual:
            return manual
        # Look for generated _tois.tsv first
        out_f = self._find_file(self._get_active_output_dir(), "_tois.tsv")
        if out_f:
            return out_f
        # Fallback to any .tsv/.txt in input
        return self._find_file(self._get_active_input_dir(), (".tsv", ".txt"))

    @toi_path.setter
    def toi_path(self, val):
        self._set_manual_path("toi_path", val)

    # --- 6. CSV OUTPUTS (AOI / Mapped) ---
    @property
    def aoi_csv_path(self):
        manual = self._get_manual_path("aoi_csv_path")
        if manual:
            return manual
        # Finds a CSV that is NOT "mapped" or "results" or "final"
        folder = self._get_active_output_dir()
        if not folder:
            return None
        for f in os.listdir(folder):
            lower = f.lower()
            if lower.endswith(".csv") and "mapped" not in lower and "results" not in lower and "stats" not in lower:
                return os.path.join(folder, f)
        return None

    @aoi_csv_path.setter
    def aoi_csv_path(self, val):
        self._set_manual_path("aoi_csv_path", val)

    @property
    def mapped_csv_path(self):
        manual = self._get_manual_path("mapped_csv_path")
        if manual:
            return manual
        return self._find_file(self._get_active_output_dir(), "_mapped.csv")

    @mapped_csv_path.setter
    def mapped_csv_path(self, val):
        self._set_manual_path("mapped_csv_path", val)

    # --- 7. FILTERED OUTPUTS (I-VT) ---
    @property
    def fixations_csv_path(self):
        manual = self._get_manual_path("fixations_csv_path")
        if manual:
            return manual
        return self._find_file(self._get_active_output_dir(), "_fixations.csv")

    @fixations_csv_path.setter
    def fixations_csv_path(self, val):
        self._set_manual_path("fixations_csv_path", val)

    @property
    def filtered_gaze_csv_path(self):
        manual = self._get_manual_path("filtered_gaze_csv_path")
        if manual:
            return manual
        return self._find_file(self._get_active_output_dir(), "_filtered.csv")

    @filtered_gaze_csv_path.setter
    def filtered_gaze_csv_path(self, val):
        self._set_manual_path("filtered_gaze_csv_path", val)

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
            "output": self._get_active_output_dir() or "",
        }

    # ════════════════════════════════════════════════════════════════
    # FILE IMPORT UTILITIES
    # ════════════════════════════════════════════════════════════════

    def import_file_for_participant(self, pid, source_path, rename_to=None):
        """
        Copies a file into the specific participant's INPUT folder.
        """
        if not self.project_root:
            return None
        dest_dir = os.path.join(self.project_root, "participants", pid, "input")
        if not os.path.exists(dest_dir):
            return None

        filename = rename_to if rename_to else os.path.basename(source_path)
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
        if not os.path.exists(dest_dir):
            return None

        filename = rename_to if rename_to else os.path.basename(source_path)
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
