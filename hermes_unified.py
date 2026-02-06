import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys

# THEME INTEGRATION
try:
    import sv_ttk  # type: ignore
except ImportError:
    sv_ttk = None

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
    print(f"CRITICAL IMPORT ERROR: {e}")
    print("Check that all files (hermes_region.py, hermes_eye.py, etc.) are saved with the new classes.")
    sys.exit(1)

class ProjectWizard:
    def __init__(self, root, context, on_complete):
        self.root = root
        self.context = context
        self.on_complete = on_complete
        
        self.win = tk.Toplevel(root)
        self.win.title("HERMES - Project Setup")
        self.win.geometry("500x400")
        self.win.protocol("WM_DELETE_WINDOW", sys.exit) # Se chiudi, esci dall'app
        
        ttk.Label(self.win, text="Welcome to HERMES", font=("Segoe UI", 20, "bold")).pack(pady=20)
        ttk.Label(self.win, text="To start, select a working folder.\nAll files (video, output, profiles) will be saved there.", 
                 justify="center").pack(pady=10)
        
        ttk.Button(self.win, text="📂 Open / Create Project Folder", command=self.select_folder).pack(fill=tk.X, padx=50, pady=20)
        
        self.lbl_path = ttk.Label(self.win, text="No folder selected", foreground="gray")
        self.lbl_path.pack()

    def select_folder(self):
        path = filedialog.askdirectory(title="Select Project Folder")
        if path:
            self.lbl_path.config(text=path, foreground="black")
            # Inizializza struttura
            self.context.initialize_project(path)
            
            # Chiedi se importare file
            if messagebox.askyesno("Quick Import", "Do you want to import source files (Video, Tobii, Matlab) now?"):
                self.run_import_wizard()
            
            self.win.destroy()
            self.on_complete() # Avvia l'app principale

    def run_import_wizard(self):
        # Import Video
        v = filedialog.askopenfilename(title="Select VIDEO", filetypes=[("Video", "*.mp4 *.avi")])
        if v: 
            dest = self.context.import_file(v)
            self.context.update_video(dest)
        
        # Import Matlab
        m = filedialog.askopenfilename(title="Select MATLAB/CSV", filetypes=[("Data", "*.mat *.csv")])
        if m: self.context.import_file(m)

        # Import Tobii JSON
        t = filedialog.askopenfilename(title="Select TOBII JSON", filetypes=[("JSON", "*.json")])
        if t: self.context.import_file(t)
        
        messagebox.showinfo("Done", "Files copied to 'Input' folder.")


class HermesUnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.withdraw() # Nascondi la finestra principale all'inizio

        # --- THEME SETUP ---
        if sv_ttk:
            sv_ttk.set_theme("dark") # Modern scientific dark theme
            print("THEME: sv_ttk applied (Dark Mode)")
        
        self.context = AppContext()
        
        # --- AUTO-LOAD CHECK ---
        if self.context.last_project and os.path.exists(self.context.last_project):
            if messagebox.askyesno("Resume Session", f"Resume last project?\n{self.context.last_project}"):
                self.context.initialize_project(self.context.last_project)
                self.start_main_ui()
                return

        # Lancia il Wizard
        ProjectWizard(self.root, self.context, self.start_main_ui)
        
    def start_main_ui(self):
        self.root.deiconify() # Mostra finestra principale
        self.root.title(f"H.E.R.M.E.S. | Progetto: {os.path.basename(self.context.project_path or '')}")
        self.root.geometry("1600x900")
        self._setup_layout()

    def _setup_layout(self):
        # A. Sidebar (Menu Laterale)
        sidebar = ttk.Frame(self.root, width=220)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Header Sidebar
        ttk.Label(sidebar, text="HERMES", font=("Trajan Pro", 20, "bold")).pack(pady=20)
        
        # Info GPU
        gpu_col = "green" if self.context.device == "cuda" else "red"
        ttk.Label(sidebar, text=f"System: {self.context.device.upper()}", foreground=gpu_col, font=("Consolas", 10)).pack(side=tk.BOTTOM, pady=10)

        # Container Bottoni
        self.btn_container = ttk.Frame(sidebar)
        self.btn_container.pack(fill=tk.X)

        # B. Navigazione (Passiamo le nuove classi)
        # FASE 1: DATI GREZZI & PREPARAZIONE
        self.add_nav("1. HUMAN (Kinematic Extraction)", YoloView)
        self.add_nav("2. MASTER TOI (Cut)", TOIGeneratorView) # <--- SPOSTATO QUI (Era il 4)
        
        # Separatore
        ttk.Separator(self.btn_container, orient='horizontal').pack(fill=tk.X, pady=10, padx=10)
        
        # FASE 2: ANALISI VISIVA (Entity & Region lavorano sul CSV Tagliato)
        self.add_nav("3. ENTITY (ID)", IdentityView)
        self.add_nav("4. REGION (AOI)", RegionView)
        
        # Separatore
        ttk.Separator(self.btn_container, orient='horizontal').pack(fill=tk.X, pady=10, padx=10)

        # FASE 3: EYE TRACKING & REPORT
        self.add_nav("5. EYE MAPPING", GazeView)
        self.add_nav("6. ANALYTICS & REPORTING", GazeStatsView)

        # C. Area Contenuto (Centrale)
        self.content_area = ttk.Frame(self.root)
        self.content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Messaggio di benvenuto
        ttk.Label(self.content_area, text="Welcome to HERMES.\nSelect a module on the left.", 
                 font=("Segoe UI", 16), foreground="gray").pack(expand=True)

    def add_nav(self, text, ViewClass):
        btn = ttk.Button(self.btn_container, text=text, command=lambda: self.switch_view(ViewClass))
        btn.pack(fill=tk.X, pady=2, padx=10)

    def switch_view(self, ViewClass):
        # 1. Pulisci l'area centrale (distruggi la vista precedente)
        for widget in self.content_area.winfo_children():
            widget.destroy()
            
        # 2. Crea un nuovo frame contenitore bianco
        container = ttk.Frame(self.content_area)
        container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # 3. Inietta le dipendenze (Context)
        # Ora tutte le classi accettano (parent, context), quindi non serve più il try/except
        ViewClass(container, self.context)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = HermesUnifiedApp(root)
    root.mainloop()