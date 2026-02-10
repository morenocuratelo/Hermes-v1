import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import os
import sys
import multiprocessing

# Tentativo di importare un tema moderno (opzionale)
try:
    import sv_ttk
except ImportError:
    sv_ttk = None

# Importa il contesto (Hub)
from hermes_context import AppContext

# ════════════════════════════════════════════════════════════════
# LAZY IMPORTS (Per evitare crash se un sottomodulo ha errori)
# ════════════════════════════════════════════════════════════════

def get_module_class(module_name, class_name):
    """Importa dinamicamente una classe da un modulo."""
    try:
        mod = __import__(module_name, fromlist=[class_name])
        return getattr(mod, class_name)
    except ImportError as e:
        return f"Import Error: {e}"
    except AttributeError:
        return f"Class '{class_name}' not found in '{module_name}'"
    except Exception as e:
        return f"Error loading module: {e}"

# Mappa dei moduli (Nome visualizzato -> (File, Classe))
MODULES_MAP = {
    "1. HUMAN (Kinematic Extraction)": ("hermes_human", "YoloView"),
    "2. MASTER TOI (Cut & Sync)":      ("hermes_master_toi", "TOIGeneratorView"),
    "3. ENTITY (ID Assignment)":       ("hermes_entity", "IdentityView"),
    "4. REGION (AOI Definition)":      ("hermes_region", "RegionView"),
    "5. EYE MAPPING":                  ("hermes_eye", "GazeView"),
    "6. ANALYTICS & REPORTING":        ("hermes_stats", "GazeStatsView"),
    "7. DATA REVIEWER":                ("hermes_reviewer", "ReviewerView") # Futuro
}

# ════════════════════════════════════════════════════════════════
# PROJECT WIZARD (Creazione/Apertura Progetto)
# ════════════════════════════════════════════════════════════════

class ProjectWizard:
    def __init__(self, root, context, on_complete):
        self.root = root
        self.context = context
        self.on_complete = on_complete
        
        self.win = tk.Toplevel(root)
        self.win.title("HERMES - Project Setup")
        self.win.geometry("600x450")
        self.win.protocol("WM_DELETE_WINDOW", sys.exit) # Chiude tutto se si esce qui
        
        # Center window
        self.win.update_idletasks()
        w, h = self.win.winfo_width(), self.win.winfo_height()
        x = (self.win.winfo_screenwidth() // 2) - (w // 2)
        y = (self.win.winfo_screenheight() // 2) - (h // 2)
        self.win.geometry(f"+{x}+{y}")

        ttk.Label(self.win, text="Welcome to HERMES", font=("Segoe UI", 24, "bold")).pack(pady=(40, 10))
        ttk.Label(self.win, text="Scientific Eye-Tracking & Motion Analysis Suite", font=("Segoe UI", 12)).pack(pady=(0, 40))
        
        f_btns = ttk.Frame(self.win)
        f_btns.pack(fill=tk.X, padx=50)
        
        ttk.Button(f_btns, text="✨ Create New Project", command=self.create_project).pack(fill=tk.X, pady=5, ipady=10)
        ttk.Button(f_btns, text="📂 Open Existing Project", command=self.open_project).pack(fill=tk.X, pady=5, ipady=10)
        
        # Recent Projects (se esistono)
        if self.context.last_project and os.path.exists(self.context.last_project):
            name = os.path.basename(self.context.last_project)
            ttk.Label(self.win, text="Recent:", foreground="gray").pack(pady=(30, 5))
            ttk.Button(self.win, text=f"Resume: {name}", 
                      command=lambda: self.load_and_start(self.context.last_project)).pack(fill=tk.X, padx=100)

    def create_project(self):
        parent_dir = filedialog.askdirectory(title="Select Parent Folder")
        if not parent_dir: return
        
        name = simpledialog.askstring("New Project", "Project Name:")
        if not name: return
        
        try:
            self.context.create_project(parent_dir, name)
            # Aggiunge subito un partecipante di default
            if messagebox.askyesno("Setup", "Create first participant now?"):
                pid = simpledialog.askstring("Participant", "ID (e.g. P001):", initialvalue="P001")
                if pid: self.context.add_participant(pid)
            
            self.win.destroy()
            self.on_complete()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def open_project(self):
        path = filedialog.askdirectory(title="Select Project Folder")
        if path:
            self.load_and_start(path)

    def load_and_start(self, path):
        try:
            self.context.load_project(path)
            self.win.destroy()
            self.on_complete()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

# ════════════════════════════════════════════════════════════════
# MAIN APP (Unified Interface)
# ════════════════════════════════════════════════════════════════

class HermesUnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.withdraw() # Hide root during wizard
        
        # Apply Theme
        if sv_ttk:
            sv_ttk.set_theme("light") # O "dark"
        
        self.context = AppContext()
        self.current_view = None
        
        # Launch Wizard
        ProjectWizard(self.root, self.context, self.launch_main_ui)

    def launch_main_ui(self):
        self.root.deiconify()
        self.root.geometry("1400x900")
        self.update_title()
        
        self._setup_layout()
        self._refresh_sidebar()

    def update_title(self):
        p_name = os.path.basename(self.context.project_root) if self.context.project_root else "No Project"
        self.root.title(f"H.E.R.M.E.S. | {p_name}")

    def _setup_layout(self):
        # 1. Sidebar (Left)
        self.sidebar = ttk.Frame(self.root, width=260)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False) # Fixed width
        
        # Separator
        ttk.Separator(self.root, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y)
        
        # 2. Content (Right)
        self.content_area = ttk.Frame(self.root)
        self.content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Welcome Screen
        self._show_welcome()

    def _refresh_sidebar(self):
        # Clear sidebar
        for w in self.sidebar.winfo_children(): w.destroy()
        
        # A. Header
        f_head = ttk.Frame(self.sidebar, padding=15)
        f_head.pack(fill=tk.X)
        ttk.Label(f_head, text="HERMES", font=("Trajan Pro", 24, "bold")).pack(anchor="w")
        ttk.Label(f_head, text="Research Suite", font=("Segoe UI", 10), foreground="gray").pack(anchor="w")
        
        # B. Participant Selector
        lf_part = ttk.LabelFrame(self.sidebar, text="Active Participant", padding=10)
        lf_part.pack(fill=tk.X, padx=10, pady=10)
        
        self.cb_part = ttk.Combobox(lf_part, values=self.context.participants, state="readonly")
        self.cb_part.pack(fill=tk.X, pady=(0, 5))
        self.cb_part.bind("<<ComboboxSelected>>", self._on_participant_change)
        
        # Set active if exists
        if self.context.current_participant:
            self.cb_part.set(self.context.current_participant)
        elif self.context.participants:
            self.cb_part.current(0)
            self._on_participant_change(None)
            
        f_p_btns = ttk.Frame(lf_part)
        f_p_btns.pack(fill=tk.X)
        ttk.Button(f_p_btns, text="➕ New", width=6, command=self._add_participant).pack(side=tk.LEFT)
        ttk.Button(f_p_btns, text="📂 Input Folder", command=self._open_input_folder).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5,0))

        # C. Modules Navigation
        ttk.Label(self.sidebar, text="WORKFLOW MODULES", font=("Segoe UI", 9, "bold"), foreground="#555").pack(anchor="w", padx=15, pady=(20, 5))
        
        for label, (mod_name, cls_name) in MODULES_MAP.items():
            btn = ttk.Button(self.sidebar, text=label, command=lambda m=mod_name, c=cls_name: self._load_module(m, c))
            btn.pack(fill=tk.X, padx=10, pady=2)

        # D. Footer
        f_foot = ttk.Frame(self.sidebar, padding=10)
        f_foot.pack(side=tk.BOTTOM, fill=tk.X)
        device_col = "green" if self.context.device == "cuda" else "orange"
        ttk.Label(f_foot, text=f"Engine: {self.context.device.upper()}", foreground=device_col, font=("Consolas", 8)).pack(anchor="w")

    # ════════════════════════════════════════════════════════════════
    # LOGIC
    # ════════════════════════════════════════════════════════════════

    def _on_participant_change(self, event):
        pid = self.cb_part.get()
        if pid:
            self.context.set_active_participant(pid)
            # Se c'è un modulo aperto, ricaricalo per mostrare i dati del nuovo partecipante
            if self.current_view:
                # Hack: simuliamo un reload del modulo corrente
                # In un sistema perfetto, le View avrebbero un metodo .reload()
                # Qui ci limitiamo a ricaricare la classe della vista.
                self._reload_current_view()

    def _add_participant(self):
        pid = simpledialog.askstring("New Participant", "ID (e.g. P002):")
        if pid:
            self.context.add_participant(pid)
            self.cb_part['values'] = self.context.participants
            self.cb_part.set(pid)
            self._on_participant_change(None)
# Line 225-229: add guard for project_root
    def _open_input_folder(self):
        if not self.context.current_participant or not self.context.project_root:
            return
        path = os.path.join(self.context.project_root, "participants", self.context.current_participant, "input")
        os.startfile(path) if os.name == 'nt' else os.system(f'open "{path}"')

    def _load_module(self, mod_name, cls_name):
        if not self.context.current_participant:
            messagebox.showwarning("Attention", "Please create or select a Participant first.")
            return

        # Lazy Import
        ViewClass = get_module_class(mod_name, cls_name)
        
        # Check errors
        if isinstance(ViewClass, str):
            self._show_error(f"Failed to load module {mod_name}:\n{ViewClass}")
            return

        # Clear Content
        for w in self.content_area.winfo_children(): w.destroy()
        
        # Instantiate View
        try:
            # Container per padding
            container = ttk.Frame(self.content_area)
            container.pack(fill=tk.BOTH, expand=True)
            
            # Init Module
            self.current_view = ViewClass(container, self.context)
            
            # Salva info per reload
            self.current_view_info = (mod_name, cls_name)
            
        except Exception as e:
            import traceback
            self._show_error(f"Error initializing {cls_name}:\n{e}\n\n{traceback.format_exc()}")

    def _reload_current_view(self):
        if hasattr(self, 'current_view_info'):
            mod, cls = self.current_view_info
            self._load_module(mod, cls)

    def _show_welcome(self):
        for w in self.content_area.winfo_children(): w.destroy()
        f = ttk.Frame(self.content_area)
        f.place(relx=0.5, rely=0.5, anchor="center")
        ttk.Label(f, text="HERMES Project Loaded", font=("Segoe UI", 20)).pack()
        ttk.Label(f, text=f"Root: {self.context.project_root}", foreground="gray").pack(pady=5)
        ttk.Label(f, text="Select a Participant and a Module from the sidebar to begin.", foreground="gray").pack(pady=20)

    def _show_error(self, msg):
        for w in self.content_area.winfo_children(): w.destroy()
        f = ttk.Frame(self.content_area)
        f.place(relx=0.5, rely=0.5, anchor="center")
        ttk.Label(f, text="⚠️ Module Error", font=("Segoe UI", 20), foreground="red").pack()
        text = tk.Text(f, width=60, height=15, wrap="word", bg="#f0f0f0", relief=tk.FLAT)
        text.insert("1.0", msg)
        text.config(state="disabled")
        text.pack(pady=20)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = HermesUnifiedApp(root)
    root.mainloop()