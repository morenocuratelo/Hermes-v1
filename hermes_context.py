import torch
import os

class AppContext:
    def __init__(self):
        # 1. Hardware Initialization (Fatto una volta sola all'avvio)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "Nessuna GPU"
        print(f"SISTEMA: Context inizializzato su {self.device} ({self.gpu_name})")

        # 2. Shared Data (Accessibili da tutti i moduli)
        self.video_path = None
        self.pose_data_path = None      # .json.gz
        self.identity_map_path = None   # .json
        self.aoi_config_path = None     # .json
        
        # Mapping Ruoli/Colori condiviso
        self.cast = {} 

    def update_video(self, path):
        if path and os.path.exists(path):
            self.video_path = path
            print(f"CONTESTO: Video aggiornato -> {path}")