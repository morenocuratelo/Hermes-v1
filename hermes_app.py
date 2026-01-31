import tkinter as tk
from tkinter import ttk, messagebox
import sys
import torch

# --- 1. SHARED STATE CLASS ---
# This acts as the memory of your application.
class AppContext:
    def __init__(self):
        self.project_name = "Untitled"
        
        # Centralized Paths (Data Flow)
        self.video_path = None
        self.pose_data_path = None
        self.identity_path = None
        self.aoi_config_path = None
        
        # Hardware Status
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "None"

# --- 2. MAIN APPLICATION ---
class HermesUnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("H.E.R.M.E.S. Integrated | Lab Modigliani")
        self.root.geometry("1400x900")
        
        # Initialize Context
        self.context = AppContext()
        print(f"System Init: Running on {self.context.device} ({self.context.gpu_name})")

        self._build_layout()

    def _build_layout(self):
        # A. Sidebar (Navigation)
        sidebar = tk.Frame(self.root, width=200, bg="#2c3e50")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False) # Force width

        # Title
        tk.Label(sidebar, text="HERMES", font=("Trajan Pro", 20, "bold"), fg="white", bg="#2c3e50").pack(pady=20)
        
        # Status GPU
        gpu_color = "#2ecc71" if self.context.device == "cuda" else "#e74c3c"
        tk.Label(sidebar, text=f"GPU: {self.context.device.upper()}", fg=gpu_color, bg="#2c3e50", font=("Consolas", 10)).pack(side=tk.BOTTOM, pady=10)

        # B. Main Content Area
        self.content_area = tk.Frame(self.root, bg="#ecf0f1")
        self.content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # C. Navigation Buttons
        # We pass the class reference, not the instance
        self.add_nav_btn(sidebar, "1. HUMAN (Yolo)", self.show_human_view)
        self.add_nav_btn(sidebar, "2. ENTITY (ID)", self.show_entity_view)
        self.add_nav_btn(sidebar, "3. REGION (AOI)", self.show_region_view)
        # ... add others ...

        # D. View Container (where we inject the modules)
        self.current_view = None

    def add_nav_btn(self, parent, text, command):
        btn = tk.Button(parent, text=text, bg="#34495e", fg="white", font=("Segoe UI", 11),
                        relief="flat", anchor="w", padx=20, command=command)
        btn.pack(fill=tk.X, pady=2)

    # --- VIEW SWITCHING LOGIC ---
    def switch_view(self, ViewClass):
        # 1. Destroy current view
        if self.current_view:
            self.current_view.destroy()
        
        # 2. Create container
        self.current_view = tk.Frame(self.content_area, bg="white")
        self.current_view.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 3. Inject Dependency (Context)
        # The ViewClass must accept (parent, context)
        ViewClass(self.current_view, self.context)

    # Wrappers for specific modules (Import these at top of file)
    def show_human_view(self):
        from hermes_human import YoloView # Note the name change
        self.switch_view(YoloView)

    def show_entity_view(self):
        from hermes_entity import IdentityView
        self.switch_view(IdentityView)

    def show_region_view(self):
        # Placeholder for other modules
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = HermesUnifiedApp(root)
    root.mainloop()