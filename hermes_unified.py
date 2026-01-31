import tkinter as tk
from tkinter import ttk, messagebox
import sys

# Importa il contesto
from hermes_context import AppContext

# IMPORT MODULI - GESTIONE NOMI AGGIORNATI (Versione Finale)
try:
    # 1. HUMAN
    from hermes_human import YoloView
    
    # 2. ENTITY
    from hermes_entity import IdentityView
    
    # 3. REGION (Aggiornato)
    from hermes_region import RegionView
    
    # 4. MASTER TOI (Aggiornato)
    from hermes_master_toi import TOIGeneratorView
    
    # 5. EYE (Aggiornato)
    from hermes_eye import GazeView
    
    # 6. STATS (Aggiornato)
    from hermes_stats import GazeStatsView

except ImportError as e:
    print(f"ERRORE CRITICO DI IMPORTAZIONE: {e}")
    print("Controlla che tutti i file (hermes_region.py, hermes_eye.py, ecc.) siano salvati con le nuove classi.")
    sys.exit(1)

class HermesUnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("H.E.R.M.E.S. Integrated | Lab Modigliani")
        self.root.geometry("1600x900")
        
        # Inizializza il cervello condiviso
        self.context = AppContext()
        
        self._setup_layout()
        
    def _setup_layout(self):
        # A. Sidebar (Menu Laterale)
        sidebar = tk.Frame(self.root, width=220, bg="#2c3e50")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Header Sidebar
        tk.Label(sidebar, text="HERMES", font=("Trajan Pro", 20, "bold"), fg="white", bg="#2c3e50").pack(pady=20)
        
        # Info GPU
        gpu_col = "#2ecc71" if self.context.device == "cuda" else "#e74c3c"
        tk.Label(sidebar, text=f"System: {self.context.device.upper()}", fg=gpu_col, bg="#2c3e50", font=("Consolas", 10)).pack(side=tk.BOTTOM, pady=10)

        # Container Bottoni
        self.btn_container = tk.Frame(sidebar, bg="#2c3e50")
        self.btn_container.pack(fill=tk.X)

        # B. Navigazione (Passiamo le nuove classi)
        self.add_nav("1. HUMAN (Yolo)", YoloView)
        self.add_nav("2. ENTITY (ID)", IdentityView)
        self.add_nav("3. REGION (AOI)", RegionView)
        
        # Separatore visivo
        tk.Frame(self.btn_container, height=2, bg="gray").pack(fill=tk.X, pady=10)
        
        self.add_nav("4. TOI BUILDER", TOIGeneratorView)
        self.add_nav("5. EYE MAPPING", GazeView)
        self.add_nav("6. STATS", GazeStatsView)

        # C. Area Contenuto (Centrale)
        self.content_area = tk.Frame(self.root, bg="#ecf0f1")
        self.content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Messaggio di benvenuto
        tk.Label(self.content_area, text="Benvenuto in HERMES.\nSeleziona un modulo a sinistra.", 
                 font=("Segoe UI", 16), fg="#95a5a6", bg="#ecf0f1").pack(expand=True)

    def add_nav(self, text, ViewClass):
        btn = tk.Button(self.btn_container, text=text, bg="#34495e", fg="white", font=("Segoe UI", 11),
                        relief="flat", anchor="w", padx=20,
                        command=lambda: self.switch_view(ViewClass))
        btn.pack(fill=tk.X, pady=2)

    def switch_view(self, ViewClass):
        # 1. Pulisci l'area centrale (distruggi la vista precedente)
        for widget in self.content_area.winfo_children():
            widget.destroy()
            
        # 2. Crea un nuovo frame contenitore bianco
        container = tk.Frame(self.content_area, bg="white")
        container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # 3. Inietta le dipendenze (Context)
        # Ora tutte le classi accettano (parent, context), quindi non serve più il try/except
        ViewClass(container, self.context)

if __name__ == "__main__":
    root = tk.Tk()
    app = HermesUnifiedApp(root)
    root.mainloop()