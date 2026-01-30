import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk  # Assicurati di avere pillow installato: pip install pillow
import os
import sys

# --- IMPORTAZIONE MODULI (Dynamic Import per gestire errori se mancano file) ---
def safe_import(module_name, class_name):
    try:
        mod = __import__(module_name)
        return getattr(mod, class_name)
    except ImportError:
        return None
    except AttributeError:
        return None

# Importiamo le classi dai file rinominati
# H - Human
YoloLauncherGUI = safe_import("hermes_human", "YoloLauncherGUI")
# E - Entity
IdentityMapperV7 = safe_import("hermes_entity", "IdentityMapperV7")
# R - Region
AOIBuilderDebug = safe_import("hermes_region", "AOIBuilderDebug")
# M - Master (Profile & Sync)
ProfileWizard = safe_import("hermes_master_prof", "ProfileWizard")
TOIGeneratorApp = safe_import("hermes_master_toi", "App") # Nota: la classe in 4b si chiamava 'App'
# E - Eye
GazeMapper = safe_import("hermes_eye", "GazeMapper")
# S - Stats
GazeAnalyzer = safe_import("hermes_stats", "GazeAnalyzer")

class HermesDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("H.E.R.M.E.S. - Lab Modigliani")
        self.root.geometry("1100x700")
        self.root.configure(bg="#f0f2f5")

        # Header
        header = tk.Frame(root, bg="#2c3e50", pady=20)
        header.pack(fill=tk.X)
        
        lbl_title = tk.Label(header, text="H.E.R.M.E.S.", font=("Trajan Pro", 36, "bold"), fg="white", bg="#2c3e50")
        lbl_title.pack()
        lbl_sub = tk.Label(header, text="Human Extraction, Recognition, Mapping & Experimental Sync", font=("Segoe UI", 12), fg="#bdc3c7", bg="#2c3e50")
        lbl_sub.pack()

        # Main Grid
        container = tk.Frame(root, bg="#f0f2f5")
        container.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Configurazione Griglia (2 righe, 3 colonne)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(2, weight=1)
        container.rowconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        # Creazione Pulsanti Modulari
        self.create_card(container, 0, 0, "H", "Human Pose", "Extract raw skeletons from video (YOLO)", 
                         "#e74c3c", YoloLauncherGUI)
        
        self.create_card(container, 0, 1, "E", "Entity ID", "Assign roles (Target/Confederate)", 
                         "#e67e22", IdentityMapperV7)
        
        self.create_card(container, 0, 2, "R", "Regions (AOI)", "Define Geometry (Head, Body, etc.)", 
                         "#f1c40f", AOIBuilderDebug)

        # M gestisce due tool, quindi useremo un menu a tendina o logica custom
        self.create_card_multi(container, 1, 0, "M", "Master Sync", "Profile Wizard & TOI Cutting", 
                               "#27ae60", [("1. Wizard Profilo", ProfileWizard), ("2. TOI Builder", TOIGeneratorApp)])
        
        self.create_card(container, 1, 1, "E", "Eye Mapping", "Map Gaze data onto AOIs", 
                         "#2980b9", GazeMapper)
        
        self.create_card(container, 1, 2, "S", "Statistics", "Generate Final CSV Reports", 
                         "#8e44ad", GazeAnalyzer)

        # Footer
        footer = tk.Label(root, text="v1.0 - Lab Modigliani Internal Tool", bg="#f0f2f5", fg="#7f8c8d")
        footer.pack(side=tk.BOTTOM, pady=10)

    def create_card(self, parent, r, c, letter, title, desc, color, module_class):
        frame = tk.Frame(parent, bg="white", relief=tk.RAISED, bd=1)
        frame.grid(row=r, column=c, padx=15, pady=15, sticky="nsew")
        frame.bind("<Button-1>", lambda e: self.launch_module(module_class, title))
        
        # Banda Colorata Laterale
        strip = tk.Frame(frame, bg=color, width=10)
        strip.pack(side=tk.LEFT, fill=tk.Y)
        
        # Contenuto
        content = tk.Frame(frame, bg="white", padx=20, pady=20)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        lbl_let = tk.Label(content, text=letter, font=("Times New Roman", 48, "bold"), fg=color, bg="white")
        lbl_let.pack(anchor="nw")
        
        tk.Label(content, text=title, font=("Segoe UI", 16, "bold"), bg="white", fg="#34495e").pack(anchor="w", pady=(5,0))
        tk.Label(content, text=desc, font=("Segoe UI", 10), bg="white", fg="#7f8c8d", wraplength=200, justify="left").pack(anchor="w")

        # Click Binding su tutti i widget interni
        for w in [strip, content, lbl_let]:
            w.bind("<Button-1>", lambda e: self.launch_module(module_class, title))

    def create_card_multi(self, parent, r, c, letter, title, desc, color, modules_list):
        # Variante per la "M" che ha due sottomoduli
        frame = tk.Frame(parent, bg="white", relief=tk.RAISED, bd=1)
        frame.grid(row=r, column=c, padx=15, pady=15, sticky="nsew")
        
        strip = tk.Frame(frame, bg=color, width=10)
        strip.pack(side=tk.LEFT, fill=tk.Y)
        
        content = tk.Frame(frame, bg="white", padx=20, pady=20)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        lbl_let = tk.Label(content, text=letter, font=("Times New Roman", 48, "bold"), fg=color, bg="white")
        lbl_let.pack(anchor="nw")
        
        tk.Label(content, text=title, font=("Segoe UI", 16, "bold"), bg="white", fg="#34495e").pack(anchor="w", pady=(5,0))
        
        # Bottoni interni per sottoscelte
        btn_frame = tk.Frame(content, bg="white")
        btn_frame.pack(anchor="w", pady=10)
        
        for sub_title, sub_class in modules_list:
            btn = tk.Button(btn_frame, text=sub_title, bg="#ecf0f1", relief=tk.FLAT, 
                            command=lambda C=sub_class, T=sub_title: self.launch_module(C, T))
            btn.pack(fill=tk.X, pady=2)

    def launch_module(self, module_class, title):
        if module_class is None:
            messagebox.showerror("Errore Modulo", f"Impossibile trovare il file per: {title}.\nControlla di aver rinominato i file correttamente (es. hermes_human.py).")
            return
            
        # Crea una nuova finestra (Toplevel) invece di Tk()
        window = tk.Toplevel(self.root)
        window.title(f"HERMES - {title}")
        
        # Istanzia la classe GUI passando la nuova finestra come root
        try:
            app = module_class(window)
        except Exception as e:
            messagebox.showerror("Errore Avvio", f"Errore nell'avvio del modulo {title}:\n{str(e)}")
            window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HermesDashboard(root)
    root.mainloop()