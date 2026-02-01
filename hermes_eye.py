import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import json
import gzip
import os
import math
import cv2
from PIL import Image, ImageTk

class GazeView: # <--- NOME CAMBIATO
    def __init__(self, parent, context): # <--- ACCETTA CONTEXT
        self.parent = parent
        self.context = context
        
        # Variabili
        self.aoi_path = tk.StringVar()
        self.gaze_path = tk.StringVar()
        
        # Parametri Video
        self.video_res_w = tk.IntVar(value=1920)
        self.video_res_h = tk.IntVar(value=1080)
        self.fps = tk.DoubleVar(value=25.0) 
        self.sync_offset = tk.DoubleVar(value=0.0) 
        
        self._build_ui()
        
        # --- AUTO-LOAD DAL CONTEXT ---
        # 1. Cerca il file AOI generato dal modulo Region
        if self.context.aoi_csv_path and os.path.exists(self.context.aoi_csv_path):
            self.aoi_path.set(self.context.aoi_csv_path)
            
        # 2. Cerca file Gaze (se impostato precedentemente o in setup futuri)
        if hasattr(self.context, 'gaze_data_path') and self.context.gaze_data_path:
             self.gaze_path.set(self.context.gaze_data_path)

    def _build_ui(self):
        # Header visivo
        tk.Label(self.parent, text="5. Eye Mapping (Gaze -> AOI)", font=("Segoe UI", 18, "bold"), bg="white").pack(pady=(0, 10), anchor="w")

        main = tk.Frame(self.parent, padx=20, pady=20, bg="white") # <--- CORRETTO: self.parent
        main.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main, text="Gaze Mapper: Eye Tracking + AOI", font=("Segoe UI", 16, "bold")).pack(pady=(0,20))

        # 1. File Input
        lf_files = tk.LabelFrame(main, text="1. Input Files", padx=10, pady=10)
        lf_files.pack(fill=tk.X, pady=5)
        
        self._add_file_picker(lf_files, "AOI File (.csv):", self.aoi_path, "*.csv")
        self._add_file_picker(lf_files, "Tobii Gaze Data (.gz):", self.gaze_path, "*.gz")

        # 2. Parametri Sincronizzazione
        lf_params = tk.LabelFrame(main, text="2. Parametri Video & Sync", padx=10, pady=10)
        lf_params.pack(fill=tk.X, pady=5)
        
        grid_f = tk.Frame(lf_params)
        grid_f.pack(fill=tk.X)
        
        tk.Label(grid_f, text="Risoluzione Video (WxH):").grid(row=0, column=0, sticky="w")
        tk.Entry(grid_f, textvariable=self.video_res_w, width=8).grid(row=0, column=1)
        tk.Label(grid_f, text="x").grid(row=0, column=2)
        tk.Entry(grid_f, textvariable=self.video_res_h, width=8).grid(row=0, column=3)
        
        tk.Label(grid_f, text="Frame Rate (FPS):").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(grid_f, textvariable=self.fps, width=8).grid(row=1, column=1, pady=5)
        
        tk.Label(grid_f, text="Sync Offset (sec):").grid(row=2, column=0, sticky="w")
        tk.Entry(grid_f, textvariable=self.sync_offset, width=8).grid(row=2, column=1)
        tk.Label(grid_f, text="(Usa valori negativi se il Gaze inizia DOPO il video)", fg="gray", font=("Arial", 8)).grid(row=2, column=2, columnspan=3, sticky="w")

        # 3. Process
        tk.Button(main, text="ELABORA E MAPPA", bg="#007ACC", fg="white", font=("Bold", 12), height=2, command=self.run_process).pack(fill=tk.X, pady=20)
        
        # --- NUOVO BOTTONE PLAYER ---
        tk.Button(main, text="📺 VISUALIZZA PLAYER RISULTATI", command=self.open_player).pack(fill=tk.X, pady=5)
        
        self.progress = ttk.Progressbar(main, orient=tk.HORIZONTAL, mode='indeterminate')
        self.progress.pack(fill=tk.X)
        
        self.lbl_status = tk.Label(main, text="Pronto.")
        self.lbl_status.pack()

    def _add_file_picker(self, parent, label, var, filetype):
        f = tk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=label, width=20, anchor="w").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(f, text="...", width=3, command=lambda: self.browse(var, filetype)).pack(side=tk.LEFT)

    def browse(self, var, ft):
        path = filedialog.askopenfilename(filetypes=[("File", ft)])
        if path: 
            var.set(path)
            # Se è il file Gaze, salvalo nel context per il futuro
            if var == self.gaze_path:
                self.context.gaze_data_path = path

    def run_process(self):
        if not self.aoi_path.get() or not self.gaze_path.get():
            messagebox.showwarning("Mancano File", "Seleziona sia il file AOI che il file GazeData.")
            return
            
        self.lbl_status.config(text="Caricamento AOI in memoria...")
        self.parent.update()
        
        try:
            # 1. Carica AOI
            df_aoi = pd.read_csv(self.aoi_path.get())
            
            # --- DEBUG COLONNE ---
            # Controlliamo se le colonne essenziali esistono
            required_cols = ['Frame', 'Role', 'AOI', 'x1', 'y1', 'x2', 'y2']
            # Gestione flessibile ID/TrackID
            id_col_name = 'ID' if 'ID' in df_aoi.columns else 'TrackID'
            if id_col_name not in df_aoi.columns:
                 raise ValueError(f"Colonna ID mancante nel CSV. Colonne trovate: {list(df_aoi.columns)}")
            
            aoi_lookup = {}
            for frame, group in df_aoi.groupby('Frame'):
                aoi_lookup[frame] = group.to_dict('records')
            
            self.lbl_status.config(text=f"AOI Indicizzate ({len(aoi_lookup)} frame). Elaborazione Gaze...")
            self.progress.start(10)
            self.parent.update()
            
            # Parametri
            W, H = self.video_res_w.get(), self.video_res_h.get()
            FPS = self.fps.get()
            OFFSET = self.sync_offset.get()
            
            output_rows = []
            
            # 2. Streaming Gaze Data
            with gzip.open(self.gaze_path.get(), 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        gaze_pkg = json.loads(line)
                        if 'data' not in gaze_pkg or 'gaze2d' not in gaze_pkg['data']: continue
                            
                        ts = gaze_pkg.get('timestamp', 0)
                        g2d = gaze_pkg['data']['gaze2d']
                        if not g2d: continue 
                        gx, gy = g2d[0], g2d[1]
                        
                        frame_idx = int((ts - OFFSET) * FPS)
                        if frame_idx < 0: continue
                        
                        px = gx * W
                        py = gy * H
                        
                        active_aois = aoi_lookup.get(frame_idx, [])
                        
                        hit_info = {
                            "Hit_Role": "None", "Hit_AOI": "None", "Hit_TrackID": -1, "Hit_Area": 999999
                        }
                        
                        hits = []
                        for aoi in active_aois:
                            x1, y1, x2, y2 = aoi['x1'], aoi['y1'], aoi['x2'], aoi['y2']
                            if x1 <= px <= x2 and y1 <= py <= y2:
                                area = (x2-x1) * (y2-y1)
                                hits.append({
                                    "role": aoi['Role'],
                                    "aoi": aoi['AOI'],
                                    "tid": aoi[id_col_name], # <--- CORREZIONE: Usa il nome colonna dinamico
                                    "area": area
                                })
                        
                        if hits:
                            best_hit = min(hits, key=lambda x: x['area'])
                            hit_info["Hit_Role"] = best_hit['role']
                            hit_info["Hit_AOI"] = best_hit['aoi']
                            hit_info["Hit_TrackID"] = best_hit['tid']
                            hit_info["Hit_Area"] = best_hit['area']
                        
                        output_rows.append({
                            "Timestamp": ts,
                            "Frame_Est": frame_idx,
                            "Gaze_X": px,
                            "Gaze_Y": py,
                            "Hit_Role": hit_info["Hit_Role"],
                            "Hit_AOI": hit_info["Hit_AOI"],
                            "Hit_TrackID": hit_info["Hit_TrackID"],
                            "Raw_Gaze2D_X": gx,
                            "Raw_Gaze2D_Y": gy
                        })
                        
                    except json.JSONDecodeError: continue

            self.progress.stop()
            self.lbl_status.config(text="Salvataggio CSV...")
            
            # 3. Export
            out_path = self.gaze_path.get().replace(".gz", "_MAPPED.csv")
            if out_path == self.gaze_path.get(): out_path += "_mapped.csv"
            
            df_out = pd.DataFrame(output_rows)
            self.context.mapped_csv_path = out_path
            df_out.to_csv(out_path, index=False)
            
            messagebox.showinfo("Successo", f"Mapping completato!\nFile salvato in:\n{out_path}\n\nRighe totali: {len(df_out)}")
            self.lbl_status.config(text="Fatto.")
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Errore Critico", str(e))
            self.lbl_status.config(text="Errore.")

    def open_player(self):
        vid = self.context.video_path
        csv = getattr(self.context, 'mapped_csv_path', None)
        
        # Fallback se il context non ha il csv ma il path è nella entry
        if not csv and self.gaze_path.get():
             candidate = self.gaze_path.get().replace(".gz", "_MAPPED.csv")
             if os.path.exists(candidate):
                 csv = candidate
        
        if not vid or not os.path.exists(vid):
            messagebox.showerror("Errore", "Video non trovato nel contesto.\nCarica un video in Human o Entity prima.")
            return
            
        if not csv or not os.path.exists(csv):
            messagebox.showerror("Errore", "File Mapped CSV non trovato.\nEsegui prima il mapping.")
            return
            
        GazeResultPlayer(self.parent, vid, csv)

class GazeResultPlayer:
    def __init__(self, parent, video_path, csv_path):
        self.win = tk.Toplevel(parent)
        self.win.title("Gaze Mapping Player")
        self.win.geometry("1000x700")
        
        self.video_path = video_path
        self.csv_path = csv_path
        
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Load Data
        try:
            self.df = pd.read_csv(csv_path)
            self.data_map = {}
            for _, row in self.df.iterrows():
                f_idx = int(row['Frame_Est'])
                if f_idx not in self.data_map: self.data_map[f_idx] = []
                self.data_map[f_idx].append(row)
        except Exception as e:
            messagebox.showerror("Errore CSV", str(e))
            self.win.destroy()
            return

        self.is_playing = False
        self.current_frame = 0
        
        self.lbl_video = tk.Label(self.win, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)
        
        ctrl = tk.Frame(self.win, bg="#eee", pady=5)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.slider = tk.Scale(ctrl, from_=0, to=self.total_frames-1, orient=tk.HORIZONTAL, command=self.on_seek, showvalue=False, bg="#eee", highlightthickness=0)
        self.slider.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        btn_frame = tk.Frame(ctrl, bg="#eee")
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="⏮ -10", command=lambda: self.step(-10), width=6).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="⏴ -1", command=lambda: self.step(-1), width=6).pack(side=tk.LEFT, padx=2)
        
        self.btn_play = tk.Button(btn_frame, text="▶ Play", command=self.toggle_play, font=("Bold", 10), width=10, bg="#ddd")
        self.btn_play.pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_frame, text="+1 ⏵", command=lambda: self.step(1), width=6).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="+10 ⏭", command=lambda: self.step(10), width=6).pack(side=tk.LEFT, padx=2)
        
        self.lbl_info = tk.Label(ctrl, text="Ready", font=("Consolas", 10), bg="#eee")
        self.lbl_info.pack(pady=2)

        self.win.update_idletasks()
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_frame()

    def on_close(self):
        self.is_playing = False
        if self.cap: self.cap.release()
        self.win.destroy()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="⏸ Pause")
            self.play_loop()
        else:
            self.btn_play.config(text="▶ Play")

    def step(self, delta):
        self.is_playing = False
        self.btn_play.config(text="▶ Play")
        new_frame = self.current_frame + delta
        self.current_frame = max(0, min(new_frame, self.total_frames - 1))
        self.slider.set(self.current_frame)
        self.show_frame()

    def play_loop(self):
        if not self.is_playing: return
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.is_playing = False
            self.btn_play.config(text="▶ Play")
            return
        self.slider.set(self.current_frame)
        self.show_frame()
        self.win.after(int(1000/self.fps), self.play_loop)

    def on_seek(self, val):
        self.current_frame = int(val)
        self.show_frame()

    def show_frame(self):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret: return
        
        # --- LOGICA DISEGNO (Invariata) ---
        if self.current_frame in self.data_map:
            samples = self.data_map[self.current_frame]
            hit_samples = [s for s in samples if s['Hit_Role'] != 'None' and pd.notna(s['Hit_Role'])]
            best_sample = hit_samples[0] if hit_samples else samples[0]
            gx, gy = int(best_sample['Gaze_X']), int(best_sample['Gaze_Y'])
            role = best_sample['Hit_Role']
            aoi = best_sample['Hit_AOI']
            color = (0, 0, 255) if (role != 'None' and pd.notna(role)) else (0, 255, 255)
            if role != 'None' and pd.notna(role):
                cv2.putText(frame, f"HIT: {role} ({aoi})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.circle(frame, (gx, gy), 10, color, 2)
            cv2.line(frame, (gx-15, gy), (gx+15, gy), color, 2)
            cv2.line(frame, (gx, gy-15), (gx, gy+15), color, 2)
            self.lbl_info.config(text=f"Frame: {self.current_frame} | Hit: {role} - {aoi}")
        else:
            self.lbl_info.config(text=f"Frame: {self.current_frame} | No Gaze Data")

        # --- CORREZIONE RENDERING ---
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        # Ottieni dimensioni attuali della label
        w = self.lbl_video.winfo_width()
        h = self.lbl_video.winfo_height()
        
        # Fallback se non ancora renderizzata, altrimenti usa dimensioni reali
        if w < 50 or h < 50:
            render_w, render_h = 800, 600
        else:
            render_w, render_h = w, h

        img.thumbnail((render_w, render_h), Image.Resampling.LANCZOS)
        
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.lbl_video.configure(image=self.tk_img)
        self.lbl_video.image = self.tk_img  # type: ignore # <--- ANCORAGGIO DI SICUREZZA
