import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import json
import gzip
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageTk

# Mappatura Standard YOLO
KEYPOINTS_MAP = {
    0: "Nose", 1: "L_Eye", 2: "R_Eye", 3: "L_Ear", 4: "R_Ear",
    5: "L_Shoulder", 6: "R_Shoulder", 7: "L_Elbow", 8: "R_Elbow",
    9: "L_Wrist", 10: "R_Wrist", 11: "L_Hip", 12: "R_Hip",
    13: "L_Knee", 14: "R_Knee", 15: "L_Ankle", 16: "R_Ankle"
}

class AOIProfileManager:
    def __init__(self, folder="profiles_aoi"):
        self.folder = folder
        
    def create_default_profile(self):
        profile = {
            "name": "BW Invasion Granular",
            "roles": {
                "Target": [
                    {
                        "name": "Face",
                        "kps": [0, 1, 2, 3, 4],
                        "margin_px": 30,
                        "expand_factor": 1.0
                    },
                    {
                        "name": "Torso",
                        "kps": [5, 6, 11, 12],
                        "margin_px": 20,
                        "expand_factor": 1.0
                    },
                    {
                        "name": "Arms",
                        "kps": [7, 8, 9, 10],
                        "margin_px": 20,
                        "expand_factor": 1.0
                    },
                    {
                        "name": "Legs",
                        "kps": [13, 14, 15, 16],
                        "margin_px": 20,
                        "expand_factor": 1.0
                    },
                    {
                        "name": "Peripersonal",
                        # Somma di: Head(0-4) + Torso(5-10) + Legs(11-14) + Feet(15-16) = Tutto il corpo (0-16)
                        "kps": list(range(17)), 
                        "margin_px": 0,
                        # Moltiplicatore alto per creare l'area di prossimità estesa
                        "expand_factor": 3.0 
                    }
                ],
                "DEFAULT": [
                    # Include tutti i 17 punti (0-16)
                    {"name": "FullBody", "kps": list(range(17)), "margin_px": 20, "expand_factor": 1.0}
                ]
            }
        }
        self.save_profile("default_invasion.json", profile)
        return profile

    def load_profile(self, name):
        try:
            with open(os.path.join(self.folder, name), 'r') as f: return json.load(f)
        except Exception: return {}

    def save_profile(self, name, data):
        with open(os.path.join(self.folder, name), 'w') as f: json.dump(data, f, indent=4)
            
    def list_profiles(self):
        if not os.path.exists(self.folder): return []
        return [f for f in os.listdir(self.folder) if f.endswith(".json")]

class RegionView:
    def __init__(self, parent, context):
        self.parent = parent
        self.context = context
        
        # --- FIX IMPORTANTE: Usa il path dal progetto ---
        if self.context.paths["profiles_aoi"]:
            profile_dir = self.context.paths["profiles_aoi"]
        else:
            profile_dir = "profiles_aoi_fallback"
            if not os.path.exists(profile_dir): os.makedirs(profile_dir)
            
        self.pm = AOIProfileManager(folder=profile_dir)
        
        # Crea default se vuoto
        if not self.pm.list_profiles(): 
            self.pm.create_default_profile()
        
        self.video_path = None
        self.pose_data = {} 
        self.identity_map = {} 
        
        # Carica il primo profilo disponibile
        profs = self.pm.list_profiles()
        self.current_profile = self.pm.load_profile(profs[0]) if profs else {}
        
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0
        self.is_playing = False
        self.kp_conf_thresh = tk.DoubleVar(value=0.3) # Default originale era 0.3
        
        self._setup_ui()
        self._setup_hotkeys()

        # --- AUTO-LOAD DAL CONTEXT ---
        if self.context.video_path:
            self.load_video_direct(self.context.video_path)
            
        if self.context.pose_data_path:
            self.load_pose_direct(self.context.pose_data_path)
            
        if self.context.identity_map_path:
            self.load_identity_direct(self.context.identity_map_path)

    def _setup_ui(self):
        # Header visivo
        tk.Label(self.parent, text="3. Spatial Area of Interest (AOI) Definition", font=("Segoe UI", 18, "bold"), bg="white").pack(pady=(0, 10), anchor="w")

        main = tk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)
        
        # SX: Video
        left = tk.Frame(main, bg="black")
        main.add(left, minsize=900)
        self.lbl_video = tk.Label(left, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)
        
        ctrl = tk.Frame(left)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        self.slider = ttk.Scale(ctrl, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_seek)
        self.slider.pack(fill=tk.X, padx=5)
        
        btns = tk.Frame(ctrl)
        btns.pack(pady=5)
        tk.Button(btns, text="1. Video Source", command=self.browse_video).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="2. Pose Data (.gz)", command=self.browse_pose).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="3. Identity Map (.json)", command=self.browse_identity).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="⏯ Playback", command=self.toggle_play).pack(side=tk.LEFT, padx=20)
        
        # TASTO DIAGNOSTICA
        tk.Button(btns, text="🔍 FRAME DIAGNOSTICS", bg="red", fg="white", font=("Bold", 10), command=self.run_diagnostics).pack(side=tk.RIGHT, padx=20)
        
        # DX: Config
        right = tk.Frame(main, padx=10, pady=10)
        main.add(right, minsize=400)
        
        tk.Label(right, text="AOI Configuration", font=("Bold", 14)).pack(pady=10)
        
        f_prof = tk.Frame(right)
        f_prof.pack(fill=tk.X)
        tk.Label(f_prof, text="Profile:").pack(side=tk.LEFT)
        
        self.cb_profile = ttk.Combobox(f_prof, values=self.pm.list_profiles(), state="readonly")
        self.cb_profile.pack(side=tk.LEFT, padx=5)
        if self.pm.list_profiles(): self.cb_profile.current(0)
        self.cb_profile.bind("<<ComboboxSelected>>", self.on_profile_change)
        
        tk.Button(f_prof, text="✨ New (Wizard)", command=self.open_profile_wizard, bg="#e1f5fe").pack(side=tk.LEFT, padx=5)
        
        # --- SEZIONE SLIDER (Corretta) ---
        lf_conf = tk.LabelFrame(right, text="Detection Sensitivity / Confidence Threshold", padx=5, pady=5)
        lf_conf.pack(fill=tk.X, pady=10)
        
        tk.Label(lf_conf, text="Keypoint Confidence Threshold (0.0 - 1.0):").pack(anchor="w")
        s_conf = tk.Scale(lf_conf, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, variable=self.kp_conf_thresh, command=lambda v: self.show_frame())
        s_conf.pack(fill=tk.X)
        tk.Label(lf_conf, text="(Lower to recover missing limbs, Raise to reduce noise)", fg="gray", font=("Arial", 8)).pack(anchor="w")
        # ---------------------------------

        # --- SEZIONE NOTEBOOK (Corretta: Una sola istanza) ---
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.frame_target = tk.Frame(self.notebook)
        self.notebook.add(self.frame_target, text="Target Rules")
        self.frame_others = tk.Frame(self.notebook)
        self.notebook.add(self.frame_others, text="Non-Target Rules")
        
        self.refresh_editors()
        tk.Button(right, text="GENERATE & EXPORT AOI CSV", bg="#4CAF50", fg="white", font=("Bold", 12), height=2, command=self.export_data).pack(side=tk.BOTTOM, fill=tk.X, pady=20)

    def _setup_hotkeys(self):
        root = self.parent.winfo_toplevel()
        root.bind("<Space>", self._on_space)
        root.bind("<Left>", self._on_left)
        root.bind("<Right>", self._on_right)
        root.bind("<Shift-Left>", self._on_shift_left)
        root.bind("<Shift-Right>", self._on_shift_right)

    def _is_hotkey_safe(self):
        # Ensure view is visible and user is not typing
        if not self.parent.winfo_viewable(): return False
        focused = self.parent.focus_get()
        if focused and focused.winfo_class() in ['Entry', 'Text', 'Spinbox', 'TEntry']:
            return False
        return True

    def _on_space(self, event):
        if self._is_hotkey_safe(): self.toggle_play()

    def _on_left(self, event):
        if self._is_hotkey_safe(): self.seek_relative(-1)

    def _on_right(self, event):
        if self._is_hotkey_safe(): self.seek_relative(1)

    def _on_shift_left(self, event):
        if self._is_hotkey_safe(): self.seek_relative(-10)

    def _on_shift_right(self, event):
        if self._is_hotkey_safe(): self.seek_relative(10)

    def seek_relative(self, delta):
        if not self.cap: return
        self.current_frame = max(0, min(self.total_frames - 1, self.current_frame + delta))
        self.slider.set(self.current_frame)
        self.show_frame()

    def open_profile_wizard(self):
        win = tk.Toplevel(self.parent)
        win.title("Advanced Profile Wizard")
        win.geometry("500x750")
        
        # --- Variabili ---
        v_name = tk.StringVar(value="New_Strategy_Profile")
        
        # Parametri Numerici
        v_head_m = tk.IntVar(value=30)
        v_body_m = tk.IntVar(value=20)
        v_feet_m = tk.IntVar(value=20)
        v_feet_off = tk.IntVar(value=30)
        v_peri_exp = tk.DoubleVar(value=2.5)
        
        # Dizionario per salvare le variabili delle scelte (Role -> IntVar)
        self.strat_vars = {} 

        # --- UI Layout ---
        # SEZIONE 1: NOME
        tk.Label(win, text="1. Profile Name", font=("Bold", 12)).pack(pady=(10, 5))
        
        f_name = tk.Frame(win); f_name.pack(fill=tk.X, padx=20)
        tk.Label(f_name, text="Filename:").pack(side=tk.LEFT)
        tk.Entry(f_name, textvariable=v_name).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # SEZIONE 2: RUOLI
        tk.Label(win, text="2. Role Configuration", font=("Bold", 12)).pack(pady=(15, 5))
        
        # Bottone Carica Identity
        def load_identity_wiz():
            path = filedialog.askopenfilename(filetypes=[("Identity JSON", "*.json")])
            if path:
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        roles = set(data.values())
                        refresh_roles_ui(roles)
                        messagebox.showinfo("Info", f"Loaded {len(roles)} roles from file.")
                except Exception as e:
                    messagebox.showerror("Error", str(e))

        tk.Button(win, text="📂 Load Identity JSON (Refresh List)", command=load_identity_wiz).pack(fill=tk.X, padx=20, pady=5)

        lf_strat = tk.LabelFrame(win, text="Visualization Mode", padx=10, pady=10)
        lf_strat.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        # Canvas per scrollare se ci sono molti ruoli
        canvas = tk.Canvas(lf_strat, height=200)
        sb = ttk.Scrollbar(lf_strat, orient="vertical", command=canvas.yview)
        frame_roles = tk.Frame(canvas)
        
        frame_roles.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame_roles, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        def refresh_roles_ui(roles_set):
            # Pulisci widget esistenti
            for w in frame_roles.winfo_children(): w.destroy()
            self.strat_vars.clear()
            
            # Filtri
            roles_set.discard("Ignore")
            roles_set.discard("Noise")
            roles_set.discard("Unknown")
            
            if not roles_set: roles_set = {"Target"}
            roles_set.add("DEFAULT")

            # Header
            tk.Label(frame_roles, text="Role", font=("Bold", 9)).grid(row=0, column=0, sticky="w", padx=5)
            tk.Label(frame_roles, text="Strategy", font=("Bold", 9)).grid(row=0, column=1, sticky="w", padx=5)
            
            r_idx = 1
            for role in sorted(list(roles_set)):
                tk.Label(frame_roles, text=f"{role}:").grid(row=r_idx, column=0, sticky="w", padx=5, pady=2)
                
                val = 1 if role == "Target" else 2
                v = tk.IntVar(value=val)
                self.strat_vars[role] = v
                
                fr = tk.Frame(frame_roles)
                fr.grid(row=r_idx, column=1, sticky="w", padx=5)
                tk.Radiobutton(fr, text="AOI", variable=v, value=1).pack(side=tk.LEFT)
                tk.Radiobutton(fr, text="Box", variable=v, value=2).pack(side=tk.LEFT)
                tk.Radiobutton(fr, text="Hide", variable=v, value=0).pack(side=tk.LEFT)
                r_idx += 1

        # Inizializza con i dati correnti
        current_roles = set(self.identity_map.values()) if self.identity_map else set()
        refresh_roles_ui(current_roles)

        # SEZIONE 3: PARAMETRI GEOMETRICI
        tk.Label(win, text="3. Geometric Parameters (AOI)", font=("Bold", 12)).pack(pady=(15, 5))
        
        def add_field(p, lbl, var):
            f = tk.Frame(p); f.pack(fill=tk.X, padx=30, pady=2)
            tk.Label(f, text=lbl).pack(side=tk.LEFT)
            tk.Spinbox(f, from_=0, to=500, textvariable=var, width=8).pack(side=tk.RIGHT)

        add_field(win, "Head Margin (px):", v_head_m)
        add_field(win, "Body Margin (px):", v_body_m)
        add_field(win, "Feet Margin (px):", v_feet_m)
        add_field(win, "Feet Bottom Offset (px):", v_feet_off)
        
        f = tk.Frame(win); f.pack(fill=tk.X, padx=30, pady=2)
        tk.Label(f, text="Peripersonal Expansion (x):").pack(side=tk.LEFT)
        tk.Spinbox(f, from_=1.0, to=5.0, increment=0.1, textvariable=v_peri_exp, width=8).pack(side=tk.RIGHT)

        # --- Logica Salvataggio ---
        def save_wiz():
            name = v_name.get().strip()
            if not name.endswith(".json"): name += ".json"
            
            # Funzione helper (rimane uguale a prima...)
            def build_rules(strategy_code):
                if strategy_code == 1: # AOI Completi
                    return [
                        {"name": "Head", "kps": [0,1,2,3,4], "margin_px": v_head_m.get(), "expand_factor": 1.0},
                        {"name": "Body", "kps": [5,6,7,8,9,10,11,12,13,14], "margin_px": v_body_m.get(), "expand_factor": 1.0},
                        {"name": "Feet", "kps": [15,16], "margin_px": v_feet_m.get(), "expand_factor": 1.0, "offset_y_bottom": v_feet_off.get()},
                        {"name": "Peripersonal", "kps": [5,6,11,12], "margin_px": 0, "expand_factor": v_peri_exp.get()}
                    ]
                elif strategy_code == 2: # Solo Box
                    return [{"name": "FullBody", "kps": list(range(17)), "margin_px": 20, "expand_factor": 1.0}]
                else: return [] # Nascondi (0)

            # Costruzione dinamica dei ruoli
            roles_config = {}
            for role_name, var in self.strat_vars.items():
                roles_config[role_name] = build_rules(var.get())

            new_profile = {
                "name": name.replace(".json", ""),
                "roles": roles_config
            }
            
            self.pm.save_profile(name, new_profile)
            messagebox.showinfo("Success", f"Profile '{name}' saved!")
            win.destroy()
            
            self.cb_profile['values'] = self.pm.list_profiles()
            self.cb_profile.set(name)
            self.on_profile_change(None)

        tk.Button(win, text="💾 GENERATE PROFILE", bg="#4CAF50", fg="white", font=("Bold", 12), command=save_wiz).pack(side=tk.BOTTOM, fill=tk.X, pady=20)

    def refresh_editors(self):
        for widget in self.frame_target.winfo_children(): widget.destroy()
        for widget in self.frame_others.winfo_children(): widget.destroy()
        self._build_role_editor(self.frame_target, "Target")
        self._build_role_editor(self.frame_others, "DEFAULT")

    def _build_role_editor(self, parent, role_key):
        rules = self.current_profile["roles"].get(role_key, [])
        canvas = tk.Canvas(parent); scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True); scroll.pack(side="right", fill="y")
        
        for idx, rule in enumerate(rules):
            lf = tk.LabelFrame(frame, text=f"AOI: {rule['name']}", pady=5, padx=5)
            lf.pack(fill=tk.X, pady=5)
            
            # 1. Margine
            tk.Label(lf, text="Margin (px):").grid(row=0, column=0)
            s_margin = tk.Scale(lf, from_=0, to=100, orient=tk.HORIZONTAL)
            s_margin.set(rule.get("margin_px", 0))
            s_margin.grid(row=0, column=1, sticky="ew")
            s_margin.bind("<ButtonRelease-1>", lambda e, r=role_key, i=idx, s=s_margin: self.update_rule_val(r, i, "margin_px", s.get()))
            
            # 2. Espansione
            tk.Label(lf, text="Expansion (x):").grid(row=1, column=0)
            s_exp = tk.Scale(lf, from_=1.0, to=4.0, resolution=0.1, orient=tk.HORIZONTAL)
            s_exp.set(rule.get("expand_factor", 1.0))
            s_exp.grid(row=1, column=1, sticky="ew")
            s_exp.bind("<ButtonRelease-1>", lambda e, r=role_key, i=idx, s=s_exp: self.update_rule_val(r, i, "expand_factor", s.get()))

            # 3. (NUOVO) Offset Fondo - Solo se la regola lo prevede (es. Feet)
            if "offset_y_bottom" in rule:
                tk.Label(lf, text="Bottom Extension:", fg="blue").grid(row=2, column=0)
                s_off = tk.Scale(lf, from_=0, to=100, orient=tk.HORIZONTAL, fg="blue")
                s_off.set(rule.get("offset_y_bottom", 0))
                s_off.grid(row=2, column=1, sticky="ew")
                s_off.bind("<ButtonRelease-1>", lambda e, r=role_key, i=idx, s=s_off: self.update_rule_val(r, i, "offset_y_bottom", s.get()))

    def update_rule_val(self, role, idx, key, val):
        self.current_profile["roles"][role][idx][key] = val
        self.show_frame()

    def on_profile_change(self, e):
        self.current_profile = self.pm.load_profile(self.cb_profile.get())
        self.refresh_editors()
        self.show_frame()

    # --- DIAGNOSTICA (IL CUORE DEL DEBUG) ---
    def run_diagnostics(self):
        print("\n" + "="*40)
        print(f"FRAME DIAGNOSTICS {self.current_frame}")
        print("="*40)
        
        # 1. Controllo Dati Pose
        if self.current_frame not in self.pose_data:
            print(f"❌ NO POSE found for frame {self.current_frame}.")
            print("Verify video and JSON alignment.")
            return
        
        frame_poses = self.pose_data[self.current_frame]
        print(f"✅ Found {len(frame_poses)} Tracked IDs in this frame: {list(frame_poses.keys())}")
        
        for tid, kps in frame_poses.items():
            print(f"\n--- ID Analysis {tid} ---")
            
            # 2. Controllo Identità
            role = self.identity_map.get(str(tid), "Unknown")
            print(f"   Mapped Role (Identity): '{role}'")
            
            if role == "Ignore" or role == "Noise":
                print("   ⛔ SKIPPED: Role is Ignore or Noise.")
                continue
            if role == "Unknown":
                print("   ⚠️ SKIPPED: Role is Unknown (unmapped).")
                continue
                
            # 3. Controllo Regole
            # Cerca il ruolo specifico, se non esiste usa DEFAULT, se non esiste lista vuota
            rules = self.current_profile['roles'].get(role, self.current_profile['roles'].get("DEFAULT", []))
            if role not in self.current_profile['roles']:
                print(f"   ℹ️ INFO: Role '{role}' not defined in AOI Profile. Using 'DEFAULT' rules.")
            
            # 4. Controllo Calcolo Box
            for rule in rules:
                box = self.calculate_box(kps, rule)
                if box:
                    print(f"   ✅ AOI '{rule['name']}': Box calculated {box}")
                else:
                    print(f"   ❌ AOI '{rule['name']}': Failed (Insufficient keypoints or low conf)")
                    # Debug Punti
                    indices = rule['kps']
                    valid_pts = 0
                    thresh = self.kp_conf_thresh.get()
                    print(f"      (Current threshold: {thresh})")
                    for i in indices:
                        if i < len(kps):
                            pt = kps[i]
                            conf = pt[2] if len(pt) > 2 else 0
                            if conf > thresh: valid_pts += 1 # ERA: if conf > 0.3
                    print(f"      Valid points found: {valid_pts}/{len(indices)}")
                    
        print("="*40 + "\n")

    # --- DATA LOADING ---
    def browse_video(self): # Collegato al bottone
        f = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi")])
        if f: self.load_video_direct(f)

    def load_video_direct(self, path): 
        if not os.path.exists(path): return
        self.video_path = path
        self.context.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.slider.config(to=self.total_frames-1)
        self.show_frame()

    def browse_pose(self):
        f = filedialog.askopenfilename(filetypes=[("Pose JSON", "*.json.gz")])
        if f: self.load_pose_direct(f)

    def load_pose_direct(self, path):
        if not os.path.exists(path): return
        self.context.pose_data_path = path 
        self.pose_data = {}
        print(f"--- Loading Pose Data: {os.path.basename(path)} ---")
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as file:
                for line in file:
                    d = json.loads(line)
                    f_idx = d['f_idx']
                    self.pose_data[f_idx] = {}
                    
                    # --- FIX CRITICO: Enumerate serve per replicare il calcolo dell'ID sintetico ---
                    for i, det in enumerate(d['det']):
                        if 'keypoints' in det:
                            # Gestione ID: Se manca o è -1, usa la logica sintetica di Entity
                            raw_tid = det.get('track_id', -1)
                            if raw_tid is None: raw_tid = -1
                            tid = int(raw_tid)
                            
                            # Replicazione logica Entity per ID non tracciati
                            if tid == -1:
                                tid = 9000000 + (f_idx * 1000) + i
                            
                            # Lettura Keypoints (codice originale)
                            raw_kps = det['keypoints']
                            final_kps = []
                            if isinstance(raw_kps, dict) and 'x' in raw_kps:
                                xs, ys = raw_kps['x'], raw_kps['y']
                                confs = raw_kps.get('visible', raw_kps.get('confidence', [1.0]*len(xs)))
                                for k in range(len(xs)): final_kps.append([xs[k], ys[k], confs[k] if k<len(confs) else 0])
                            elif isinstance(raw_kps, list):
                                final_kps = raw_kps
                            
                            self.pose_data[f_idx][tid] = final_kps
            print(f"Poses loaded: {len(self.pose_data)} frames.")
        except Exception as e: 
            messagebox.showerror("Error", f"Error loading poses: {str(e)}")
            import traceback
            traceback.print_exc()
        self.show_frame()

    def browse_identity(self):
        f = filedialog.askopenfilename(filetypes=[("Identity", "*.json")])
        if f: self.load_identity_direct(f)

    def load_identity_direct(self, path):
        if not os.path.exists(path): return
        self.context.identity_map_path = path # <--- AGGIORNA CONTEXT
        with open(path, 'r') as file: self.identity_map = json.load(file)
        print(f"Identities loaded: {len(self.identity_map)} IDs.")
        self.show_frame()

    def calculate_box(self, kps_data, rule):
        indices = rule['kps']
        xs, ys = [], []
        for i in indices:
            if i >= len(kps_data): continue
            pt = kps_data[i]
            x, y, conf = 0, 0, 0
            if isinstance(pt, list):
                if len(pt)>=2: x, y = pt[0], pt[1]
                if len(pt)>=3: conf = pt[2]
                else: conf = 1.0
            thresh = self.kp_conf_thresh.get()
            if conf > thresh and x > 1 and y > 1:
                xs.append(x); ys.append(y)
        
        if not xs: return None
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        m = int(rule.get('margin_px', 0))
        min_x -= m; max_x += m; min_y -= m; max_y += m
        if 'offset_y_bottom' in rule: max_y += int(rule['offset_y_bottom'])
        f = float(rule.get('expand_factor', 1.0))
        if f != 1.0:
            w = max_x - min_x; h = max_y - min_y
            cx, cy = min_x + w/2, min_y + h/2
            min_x = cx - (w*f)/2; max_x = cx + (w*f)/2
            min_y = cy - (h*f)/2; max_y = cy + (h*f)/2
        return (int(min_x), int(min_y), int(max_x), int(max_y))

    def show_frame(self):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret: return
        
        if self.current_frame in self.pose_data:
            for tid, kps in self.pose_data[self.current_frame].items():
                role = self.identity_map.get(str(tid), "Unknown")
                if role in ["Ignore", "Noise", "Unknown"]: continue
                
                rules = self.current_profile['roles'].get(role, self.current_profile['roles'].get("DEFAULT", []))
                
                for rule in rules:
                    box = self.calculate_box(kps, rule)
                    if box:
                        x1, y1, x2, y2 = box
                        c = (0, 255, 255)
                        if rule['name']=="Peripersonal": c=(255,0,255)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), c, 2)
                        cv2.putText(frame, f"{role}:{rule['name']}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        w, h = self.lbl_video.winfo_width(), self.lbl_video.winfo_height()
        if w<10: w=800; h=600
        img.thumbnail((w, h))
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.lbl_video.config(image=self.tk_img)

    def on_seek(self, v): self.current_frame = int(float(v)); self.show_frame()
    def toggle_play(self): 
        self.is_playing = not self.is_playing
        if self.is_playing: self.play_loop()
    def play_loop(self):
        if self.is_playing and self.cap:
            self.current_frame += 1
            if self.current_frame >= self.total_frames: self.is_playing=False
            self.slider.set(self.current_frame); self.show_frame(); self.parent.after(30, self.play_loop)
    def export_data(self):
        if not self.pose_data: return
        out = filedialog.asksaveasfilename(defaultextension=".csv")
        if not out: return
        self.context.export_path = out # <--- AGGIORNA CONTEXT
        rows = []
        for f, d in self.pose_data.items():
            for tid, kps in d.items():
                role = self.identity_map.get(str(tid), "Unknown")
                if role in ["Ignore", "Noise", "Unknown"]: continue
                rules = self.current_profile['roles'].get(role, self.current_profile['roles'].get("DEFAULT", []))
                for r in rules:
                    b = self.calculate_box(kps, r)
                    if b: rows.append({
                        "Frame": f,
                        "Timestamp": round(f / self.fps, 4),
                        "TrackID": tid,
                        "Role": role,
                        "AOI": r['name'],
                        "x1": b[0],
                        "y1": b[1],
                        "x2": b[2],
                        "y2": b[3]
                    })
        pd.DataFrame(rows).to_csv(out, index=False)
        self.context.aoi_csv_path = out
        messagebox.showinfo("OK", "Export complete.")
