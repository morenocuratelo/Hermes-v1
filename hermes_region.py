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
    0: "Naso", 1: "Occhio SX", 2: "Occhio DX", 3: "Orecchio SX", 4: "Orecchio DX",
    5: "Spalla SX", 6: "Spalla DX", 7: "Gomito SX", 8: "Gomito DX",
    9: "Polso SX", 10: "Polso DX", 11: "Anca SX", 12: "Anca DX",
    13: "Ginocchio SX", 14: "Ginocchio DX", 15: "Caviglia SX", 16: "Caviglia DX"
}

class AOIProfileManager:
    def __init__(self, folder="profiles_aoi"):
        self.folder = folder
        if not os.path.exists(folder): os.makedirs(folder)
        
    def create_default_profile(self):
        profile = {
            "name": "BW Invasion Standard",
            "roles": {
                "Target": [
                    {"name": "Head", "kps": [0, 1, 2, 3, 4], "margin_px": 30, "expand_factor": 1.0},
                    {"name": "Body", "kps": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "margin_px": 20, "expand_factor": 1.0},
                    {"name": "Feet", "kps": [15, 16], "margin_px": 20, "expand_factor": 1.0, "offset_y_bottom": 30},
                    {"name": "Peripersonal", "kps": [5, 6, 11, 12], "margin_px": 0, "expand_factor": 2.5}
                ],
                "DEFAULT": [
                    {"name": "FullBody", "kps": list(range(17)), "margin_px": 20, "expand_factor": 1.0}
                ]
            }
        }
        self.save_profile("default_invasion.json", profile)
        return profile

    def load_profile(self, name):
        with open(os.path.join(self.folder, name), 'r') as f: return json.load(f)

    def save_profile(self, name, data):
        with open(os.path.join(self.folder, name), 'w') as f: json.dump(data, f, indent=4)
            
    def list_profiles(self):
        if not os.path.exists(self.folder): self.create_default_profile()
        return [f for f in os.listdir(self.folder) if f.endswith(".json")]

class RegionView: # <--- NOME CAMBIATO
    def __init__(self, parent, context): # <--- NUOVI ARGOMENTI
        self.parent = parent
        self.context = context
        
        # Rimosso self.root.title/geometry (gestiti dal main)
        
        self.pm = AOIProfileManager()
        if not self.pm.list_profiles(): self.pm.create_default_profile()
        
        self.video_path = None
        self.pose_data = {} 
        self.identity_map = {} 
        self.current_profile = self.pm.load_profile(self.pm.list_profiles()[0])
        
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        
        self._setup_ui()

        # --- AUTO-LOAD DAL CONTEXT ---
        # Se i dati esistono già nel "cervello" condiviso, caricali subito
        if self.context.video_path:
            self.load_video_direct(self.context.video_path)
            
        if self.context.pose_data_path:
            self.load_pose_direct(self.context.pose_data_path)
            
        if self.context.identity_path:
            self.load_identity_direct(self.context.identity_path)

    def _setup_ui(self):
        # Header visivo
        tk.Label(self.parent, text="3. Region Definition (AOI)", font=("Segoe UI", 18, "bold"), bg="white").pack(pady=(0, 10), anchor="w")

        main = tk.PanedWindow(self.parent, orient=tk.HORIZONTAL) # <--- CORRETTO: self.parent
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
        tk.Button(btns, text="1. Video", command=self.browse_video).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="2. Pose (.gz)", command=self.browse_pose).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="3. Identità (.json)", command=self.browse_identity).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="⏯ Play", command=self.toggle_play).pack(side=tk.LEFT, padx=20)
        
        # TASTO DIAGNOSTICA
        tk.Button(btns, text="🔍 DIAGNOSTICA FRAME", bg="red", fg="white", font=("Bold", 10), command=self.run_diagnostics).pack(side=tk.RIGHT, padx=20)
        
        # DX: Config
        right = tk.Frame(main, padx=10, pady=10)
        main.add(right, minsize=400)
        
        tk.Label(right, text="Configurazione AOI", font=("Bold", 14)).pack(pady=10)
        
        f_prof = tk.Frame(right)
        f_prof.pack(fill=tk.X)
        tk.Label(f_prof, text="Profilo:").pack(side=tk.LEFT)
        
        self.cb_profile = ttk.Combobox(f_prof, values=self.pm.list_profiles(), state="readonly")
        self.cb_profile.pack(side=tk.LEFT, padx=5)
        if self.pm.list_profiles(): self.cb_profile.current(0)
        self.cb_profile.bind("<<ComboboxSelected>>", self.on_profile_change)
        
        # --- NUOVO: Bottone Wizard ---
        tk.Button(f_prof, text="✨ Nuovo (Wizard)", command=self.open_profile_wizard, bg="#e1f5fe").pack(side=tk.LEFT, padx=5)
        # -----------------------------
        
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.frame_target = tk.Frame(self.notebook)
        self.notebook.add(self.frame_target, text="Regole Target")
        self.frame_others = tk.Frame(self.notebook)
        self.notebook.add(self.frame_others, text="Regole Altri")
        
        self.refresh_editors()
        tk.Button(right, text="GENERA E ESPORTA CSV AOI", bg="#4CAF50", fg="white", font=("Bold", 12), height=2, command=self.export_data).pack(side=tk.BOTTOM, fill=tk.X, pady=20)

    def open_profile_wizard(self):
        win = tk.Toplevel(self.parent)
        win.title("Wizard Profilo Avanzato")
        win.geometry("450x650")
        
        # --- Variabili ---
        v_name = tk.StringVar(value="New_Strategy_Profile")
        
        # Parametri Numerici
        v_head_m = tk.IntVar(value=30)
        v_body_m = tk.IntVar(value=20)
        v_feet_m = tk.IntVar(value=20)
        v_feet_off = tk.IntVar(value=30)
        v_peri_exp = tk.DoubleVar(value=2.5)
        
        # Strategie (1=AOI Completi, 2=Solo Box, 0=Nascondi)
        v_strat_target = tk.IntVar(value=1)
        v_strat_others = tk.IntVar(value=2)
        
        # --- UI Layout ---
        # SEZIONE 1: NOME E STRATEGIE
        tk.Label(win, text="1. Nome e Strategie", font=("Bold", 12)).pack(pady=10)
        
        f_name = tk.Frame(win); f_name.pack(fill=tk.X, padx=20)
        tk.Label(f_name, text="Nome File:").pack(side=tk.LEFT)
        tk.Entry(f_name, textvariable=v_name).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        lf_strat = tk.LabelFrame(win, text="Modalità Visualizzazione", padx=10, pady=10)
        lf_strat.pack(fill=tk.X, padx=20, pady=10)
        
        # Target Strategy
        tk.Label(lf_strat, text="Ruolo 'Target':", font=("Bold", 9)).grid(row=0, column=0, sticky="w")
        tk.Radiobutton(lf_strat, text="AOI Completi (Testa/Corpo...)", variable=v_strat_target, value=1).grid(row=0, column=1, sticky="w")
        tk.Radiobutton(lf_strat, text="Solo Box Intero", variable=v_strat_target, value=2).grid(row=1, column=1, sticky="w")
        
        # --- CORREZIONE QUI: Usa ttk.Separator invece di tk.Separator ---
        ttk.Separator(lf_strat, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        # ----------------------------------------------------------------
        
        # Others Strategy
        tk.Label(lf_strat, text="Altri ID (Default):", font=("Bold", 9)).grid(row=3, column=0, sticky="w")
        tk.Radiobutton(lf_strat, text="AOI Completi", variable=v_strat_others, value=1).grid(row=3, column=1, sticky="w")
        tk.Radiobutton(lf_strat, text="Solo Box Intero", variable=v_strat_others, value=2).grid(row=4, column=1, sticky="w")
        tk.Radiobutton(lf_strat, text="Nascondi (Nessun Box)", variable=v_strat_others, value=0).grid(row=5, column=1, sticky="w")

        # SEZIONE 2: PARAMETRI DIMENSIONI
        tk.Label(win, text="2. Configurazione Dimensioni", font=("Bold", 12)).pack(pady=(15,5))
        
        # Definizione funzione helper per i campi
        def add_field(p, lbl, var):
            f = tk.Frame(p); f.pack(fill=tk.X, padx=30, pady=2)
            tk.Label(f, text=lbl).pack(side=tk.LEFT)
            tk.Spinbox(f, from_=0, to=500, textvariable=var, width=8).pack(side=tk.RIGHT)

        add_field(win, "Head Margin (px):", v_head_m)
        add_field(win, "Body Margin (px):", v_body_m)
        add_field(win, "Feet Margin (px):", v_feet_m)
        add_field(win, "Feet Bottom Offset (px):", v_feet_off)
        
        f = tk.Frame(win); f.pack(fill=tk.X, padx=30, pady=2)
        tk.Label(f, text="Peripersonal Expand (x):").pack(side=tk.LEFT)
        tk.Spinbox(f, from_=1.0, to=5.0, increment=0.1, textvariable=v_peri_exp, width=8).pack(side=tk.RIGHT)

        # --- Logica Salvataggio ---
        def save_wiz():
            name = v_name.get().strip()
            if not name.endswith(".json"): name += ".json"
            
            def build_rules(strategy_code):
                if strategy_code == 1: # AOI Completi
                    return [
                        {"name": "Head", "kps": [0,1,2,3,4], "margin_px": v_head_m.get(), "expand_factor": 1.0},
                        {"name": "Body", "kps": [5,6,7,8,9,10,11,12,13,14], "margin_px": v_body_m.get(), "expand_factor": 1.0},
                        {"name": "Feet", "kps": [15,16], "margin_px": v_feet_m.get(), "expand_factor": 1.0, "offset_y_bottom": v_feet_off.get()},
                        {"name": "Peripersonal", "kps": [5,6,11,12], "margin_px": 0, "expand_factor": v_peri_exp.get()}
                    ]
                elif strategy_code == 2: # Solo Box
                    return [
                        {"name": "FullBody", "kps": list(range(17)), "margin_px": 20, "expand_factor": 1.0}
                    ]
                else: # Nascondi (0)
                    return []

            new_profile = {
                "name": name.replace(".json", ""),
                "roles": {
                    "Target": build_rules(v_strat_target.get()),
                    "DEFAULT": build_rules(v_strat_others.get())
                }
            }
            
            self.pm.save_profile(name, new_profile)
            messagebox.showinfo("Successo", f"Profilo '{name}' salvato!")
            win.destroy()
            
            self.cb_profile['values'] = self.pm.list_profiles()
            self.cb_profile.set(name)
            self.on_profile_change(None)

        tk.Button(win, text="💾 GENERA PROFILO", bg="#4CAF50", fg="white", font=("Bold", 12), command=save_wiz).pack(side=tk.BOTTOM, fill=tk.X, pady=20)

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
            tk.Label(lf, text="Margine (px):").grid(row=0, column=0)
            s_margin = tk.Scale(lf, from_=0, to=100, orient=tk.HORIZONTAL)
            s_margin.set(rule.get("margin_px", 0))
            s_margin.grid(row=0, column=1, sticky="ew")
            s_margin.bind("<ButtonRelease-1>", lambda e, r=role_key, i=idx, s=s_margin: self.update_rule_val(r, i, "margin_px", s.get()))
            
            # 2. Espansione
            tk.Label(lf, text="Espansione (x):").grid(row=1, column=0)
            s_exp = tk.Scale(lf, from_=1.0, to=4.0, resolution=0.1, orient=tk.HORIZONTAL)
            s_exp.set(rule.get("expand_factor", 1.0))
            s_exp.grid(row=1, column=1, sticky="ew")
            s_exp.bind("<ButtonRelease-1>", lambda e, r=role_key, i=idx, s=s_exp: self.update_rule_val(r, i, "expand_factor", s.get()))

            # 3. (NUOVO) Offset Fondo - Solo se la regola lo prevede (es. Feet)
            if "offset_y_bottom" in rule:
                tk.Label(lf, text="Estensione Fondo:", fg="blue").grid(row=2, column=0)
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
        print(f"DIAGNOSTICA FRAME {self.current_frame}")
        print("="*40)
        
        # 1. Controllo Dati Pose
        if self.current_frame not in self.pose_data:
            print(f"❌ NESSUNA POSA trovata per il frame {self.current_frame}.")
            print("Verifica che il video e il file JSON siano allineati.")
            return
        
        frame_poses = self.pose_data[self.current_frame]
        print(f"✅ Trovati {len(frame_poses)} ID Tracked in questo frame: {list(frame_poses.keys())}")
        
        for tid, kps in frame_poses.items():
            print(f"\n--- Analisi ID {tid} ---")
            
            # 2. Controllo Identità
            role = self.identity_map.get(str(tid), "Unknown")
            print(f"   Ruolo Mappato (Identity): '{role}'")
            
            if role == "Ignore" or role == "Noise":
                print("   ⛔ SKIPPED: Ruolo è Ignore o Noise.")
                continue
            if role == "Unknown":
                print("   ⚠️ SKIPPED: Ruolo è Unknown (non mappato).")
                continue
                
            # 3. Controllo Regole
            rules = []
            if role == "Target":
                print("   Applicazione: Regole 'Target'")
                rules = self.current_profile['roles'].get("Target", [])
            else:
                print("   Applicazione: Regole 'DEFAULT' (per Confederati/Altri)")
                rules = self.current_profile['roles'].get("DEFAULT", [])
            
            if not rules:
                print("   ❌ ERRORE: Nessuna regola trovata nel profilo per questo ruolo!")
                continue
            
            # 4. Controllo Calcolo Box
            for rule in rules:
                box = self.calculate_box(kps, rule)
                if box:
                    print(f"   ✅ AOI '{rule['name']}': Box calcolato {box}")
                else:
                    print(f"   ❌ AOI '{rule['name']}': Fallito (Keypoints insufficienti o conf bassa)")
                    # Debug Punti
                    indices = rule['kps']
                    valid_pts = 0
                    for i in indices:
                        if i < len(kps):
                            pt = kps[i]
                            conf = pt[2] if len(pt) > 2 else 0
                            if conf > 0.3: valid_pts += 1
                    print(f"      Punti validi trovati: {valid_pts}/{len(indices)}")
                    
        print("="*40 + "\n")

    # --- DATA LOADING ---
    def browse_video(self): # Collegato al bottone
        f = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi")])
        if f: self.load_video_direct(f)

    def load_video_direct(self, path): # Logica pura
        if not os.path.exists(path): return
        self.video_path = path
        self.context.video_path = path # <--- AGGIORNA CONTEXT
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.config(to=self.total_frames-1)
        self.show_frame()

    def browse_pose(self):
        f = filedialog.askopenfilename(filetypes=[("Pose JSON", "*.json.gz")])
        if f: self.load_pose_direct(f)

    def load_pose_direct(self, path):
        if not os.path.exists(path): return
        self.context.pose_data_path = path # <--- AGGIORNA CONTEXT
        self.pose_data = {}
        print(f"--- Caricamento Pose: {os.path.basename(path)} ---")
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as file:
                for line in file:
                    d = json.loads(line)
                    f_idx = d['f_idx']
                    self.pose_data[f_idx] = {}
                    for det in d['det']:
                        if 'keypoints' in det and 'track_id' in det:
                            tid = int(det['track_id'])
                            raw_kps = det['keypoints']
                            final_kps = []
                            # Gestione formati YOLO vari
                            if isinstance(raw_kps, dict) and 'x' in raw_kps:
                                xs, ys = raw_kps['x'], raw_kps['y']
                                confs = raw_kps.get('visible', raw_kps.get('confidence', [1.0]*len(xs)))
                                for i in range(len(xs)): final_kps.append([xs[i], ys[i], confs[i] if i<len(confs) else 0])
                            elif isinstance(raw_kps, list):
                                final_kps = raw_kps
                            self.pose_data[f_idx][tid] = final_kps
            print(f"Pose caricate: {len(self.pose_data)} frames.")
        except Exception as e: messagebox.showerror("Err", str(e))
        self.show_frame()

    def browse_identity(self):
        f = filedialog.askopenfilename(filetypes=[("Identity", "*.json")])
        if f: self.load_identity_direct(f)

    def load_identity_direct(self, path):
        if not os.path.exists(path): return
        self.context.identity_path = path # <--- AGGIORNA CONTEXT
        with open(path, 'r') as file: self.identity_map = json.load(file)
        print(f"Identità caricate: {len(self.identity_map)} ID.")
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
            if conf > 0.3 and x > 1 and y > 1:
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
                
                rules = self.current_profile['roles'].get("Target", []) if role == "Target" else self.current_profile['roles'].get("DEFAULT", [])
                
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
                rules = self.current_profile['roles'].get("Target", []) if role == "Target" else self.current_profile['roles'].get("DEFAULT", [])
                for r in rules:
                    b = self.calculate_box(kps, r)
                    if b: rows.append({"Frame":f, "ID":tid, "Role":role, "AOI":r['name'], "x1":b[0], "y1":b[1], "x2":b[2], "y2":b[3]})
        pd.DataFrame(rows).to_csv(out, index=False)
        self.context.aoi_csv_path = out
        messagebox.showinfo("OK", "Export completo.")

