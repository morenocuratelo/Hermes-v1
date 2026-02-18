import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import pandas as pd
import bisect
import os
import math
from PIL import Image, ImageTk

class TimelineWidget(tk.Canvas):
    """Timeline visualization for TOIs."""
    def __init__(self, parent, command_seek, **kwargs):
        super().__init__(parent, **kwargs)
        self.command_seek = command_seek
        self.duration = 0.0
        self.tois = []
        self.cursor_x = 0
        self.bind("<Button-1>", self.on_click)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<Configure>", lambda e: self.redraw())

    def set_data(self, duration, df_tois):
        self.duration = float(duration)
        self.tois = []
        if df_tois is not None and not df_tois.empty:
            # Cyclic colors
            colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00']
            cond_map = {}
            for _, row in df_tois.iterrows():
                cond = str(row.get('Condition', row.get('Phase', 'Base')))
                if cond not in cond_map:
                    cond_map[cond] = colors[len(cond_map) % len(colors)]
                
                try:
                    s = float(row['Start'])
                    e = float(row['End'])
                    name = str(row.get('Name', ''))
                    self.tois.append({'s': s, 'e': e, 'c': cond_map[cond], 'n': name})
                except ValueError:
                    continue
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if self.duration <= 0 or w <= 1:
            return
        
        # Draw TOI blocks
        for t in self.tois:
            x1 = (t['s'] / self.duration) * w
            x2 = (t['e'] / self.duration) * w
            if x2 > x1:
                self.create_rectangle(x1, 2, x2, h-2, fill=t['c'], outline="gray", tags="toi")
        
        # Draw Cursor
        self.create_line(self.cursor_x, 0, self.cursor_x, h, fill="red", width=2, tags="cursor")

    def update_cursor(self, current_sec):
        if self.duration > 0:
            w = self.winfo_width()
            self.cursor_x = (current_sec / self.duration) * w
            self.coords("cursor", self.cursor_x, 0, self.cursor_x, self.winfo_height())

    def on_click(self, event):
        self._seek_event(event)

    def on_drag(self, event):
        self._seek_event(event)

    def _seek_event(self, event):
        if self.duration > 0:
            w = self.winfo_width()
            perc = max(0.0, min(1.0, event.x / w))
            sec = perc * self.duration
            self.command_seek(sec)

class ReviewerLogic:
    """Separated logic for Reviewer state and data management."""
    def __init__(self):
        self.cap = None
        self.df_gaze = None
        self.df_tois = None
        
        self.fps = 30.0
        self.total_duration = 0.0
        self.total_frames = 0
        self.current_frame = 0
        
        # Gaze Data Cache
        self.gaze_t = []
        self.gaze_x = []
        self.gaze_y = []

    def load_video(self, path):
        if not os.path.exists(path): return False
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): return False
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_duration = self.total_frames / self.fps
        return True

    def load_tois(self, path):
        if not os.path.exists(path): return False
        try:
            sep = '\t' if path.endswith('.tsv') or path.endswith('.txt') else ','
            self.df_tois = pd.read_csv(path, sep=sep)
            if 'Start' in self.df_tois.columns:
                self.df_tois = self.df_tois.sort_values('Start').reset_index(drop=True)
            return True
        except Exception:
            return False

    def load_gaze(self, path):
        if not os.path.exists(path): return False
        try:
            df = pd.read_csv(path)
            req = ['Timestamp', 'Gaze_X', 'Gaze_Y']
            if not all(c in df.columns for c in req):
                if 'gaze2d_x' in df.columns:
                    df.rename(columns={'gaze2d_x': 'Gaze_X', 'gaze2d_y': 'Gaze_Y', 'timestamp': 'Timestamp'}, inplace=True)
                else:
                    return False
            
            self.df_gaze = df.sort_values('Timestamp').reset_index(drop=True)
            self.gaze_t = self.df_gaze['Timestamp'].values
            self.gaze_x = self.df_gaze['Gaze_X'].values
            self.gaze_y = self.df_gaze['Gaze_Y'].values
            return True
        except Exception:
            return False

    def get_frame_image(self):
        if not self.cap: return False, None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        return self.cap.read()

class ReviewerView:
    def __init__(self, parent, context=None):
        self.parent = parent
        self.context = context
        self.logic = ReviewerLogic()
        self.is_playing = False
        
        # UI Variables
        self.video_path_var = tk.StringVar()
        self.toi_path_var = tk.StringVar()
        self.gaze_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready.")
        
        self._setup_ui()
        
        if self.context:
            self._auto_load_from_context()

    def _setup_ui(self):
        # Main Layout: PanedWindow
        self.paned = tk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)
        
        # --- LEFT PANEL: Player ---
        self.f_player = ttk.Frame(self.paned)
        self.paned.add(self.f_player, minsize=600, stretch="always")
        
        # Controls Area (Pack FIRST to ensure visibility at bottom)
        f_ctrl = ttk.Frame(self.f_player, padding=5)
        f_ctrl.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Timeline
        self.timeline = TimelineWidget(f_ctrl, command_seek=self.seek_seconds, height=30, bg="#e0e0e0")
        self.timeline.pack(fill=tk.X, pady=(0, 5))
        
        # Transport Buttons
        f_btns = ttk.Frame(f_ctrl)
        f_btns.pack(fill=tk.X)
        
        ttk.Button(f_btns, text="⏮ TOI", command=lambda: self.seek_toi(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_btns, text="⏪ -5s", command=lambda: self.seek_relative_sec(-5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_btns, text="⏴ -1f", command=lambda: self.seek_relative_frames(-1)).pack(side=tk.LEFT, padx=2)
        self.btn_play = ttk.Button(f_btns, text="▶ Play", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=10)
        ttk.Button(f_btns, text="+1f ⏵", command=lambda: self.seek_relative_frames(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_btns, text="+5s ⏩", command=lambda: self.seek_relative_sec(5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_btns, text="TOI ⏭", command=lambda: self.seek_toi(1)).pack(side=tk.LEFT, padx=2)
        
        self.lbl_time = ttk.Label(f_btns, text="00:00.00 / 00:00.00", font=("Consolas", 10))
        self.lbl_time.pack(side=tk.RIGHT)

        # Video Area (Pack LAST to fill remaining space)
        self.lbl_video = tk.Label(self.f_player, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

        # --- RIGHT PANEL: Settings & Info ---
        self.f_sidebar = ttk.Frame(self.paned, padding=10)
        self.paned.add(self.f_sidebar, minsize=300, stretch="always")
        
        ttk.Label(self.f_sidebar, text="7. Data Reviewer", font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 20))
        
        # File Loaders
        lf_files = ttk.LabelFrame(self.f_sidebar, text="Data Sources", padding=10)
        lf_files.pack(fill=tk.X, pady=5)
        
        self._add_file_picker(lf_files, "Video:", self.video_path_var, self.load_video, "*.mp4 *.avi")
        self._add_file_picker(lf_files, "TOI (.tsv):", self.toi_path_var, self.load_tois, "*.tsv *.txt *.csv")
        self._add_file_picker(lf_files, "Gaze Mapped (.csv):", self.gaze_path_var, self.load_gaze, "*.csv")
        
        # Info Box
        lf_info = ttk.LabelFrame(self.f_sidebar, text="Current Frame Info", padding=10)
        lf_info.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.txt_info = tk.Text(lf_info, height=15, width=30, state="disabled", font=("Consolas", 9), bg="#f4f4f4")
        self.txt_info.pack(fill=tk.BOTH, expand=True)

    def _add_file_picker(self, parent, label, var, callback, filetypes):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        ttk.Label(f, text=label).pack(anchor="w")
        
        f_in = ttk.Frame(f)
        f_in.pack(fill=tk.X)
        ttk.Entry(f_in, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(f_in, text="📂", width=3, command=lambda: self._browse(var, callback, filetypes)).pack(side=tk.LEFT, padx=(2,0))

    def _browse(self, var, callback, filetypes):
        path = filedialog.askopenfilename(filetypes=[("Files", filetypes)])
        if path:
            var.set(path)
            callback(path)

    def _auto_load_from_context(self):
        if not self.context:
            return
        # Video
        if self.context.video_path and os.path.exists(self.context.video_path):
            self.video_path_var.set(self.context.video_path)
            self.load_video(self.context.video_path)
            
        # TOI
        if self.context.toi_path and os.path.exists(self.context.toi_path):
            self.toi_path_var.set(self.context.toi_path)
            self.load_tois(self.context.toi_path)
            
        # Gaze (Mapped)
        if hasattr(self.context, 'mapped_csv_path') and self.context.mapped_csv_path:
             if os.path.exists(self.context.mapped_csv_path):
                self.gaze_path_var.set(self.context.mapped_csv_path)
                self.load_gaze(self.context.mapped_csv_path)

    def load_video(self, path):
        if not self.logic.load_video(path):
            messagebox.showerror("Error", "Could not open video.")
            return
        
        self.timeline.set_data(self.logic.total_duration, self.logic.df_tois)
        self.show_frame()

    def load_tois(self, path):
        if self.logic.load_tois(path):
            if self.logic.cap:
                self.timeline.set_data(self.logic.total_duration, self.logic.df_tois)
        else:
            messagebox.showerror("TOI Error", "Could not load TOI file.")

    def load_gaze(self, path):
        if not self.logic.load_gaze(path):
            messagebox.showerror("Gaze Error", "Could not load Gaze file or missing columns.")

    # --- Playback Logic ---

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.config(text="⏸ Pause" if self.is_playing else "▶ Play")
        if self.is_playing:
            self.play_loop()

    def play_loop(self):
        if not self.is_playing or not self.logic.cap:
            return
        
        self.logic.current_frame += 1
        if self.logic.current_frame >= self.logic.total_frames:
            self.is_playing = False
            self.btn_play.config(text="▶ Play")
            return
            
        self.show_frame()
        delay = int(1000 / self.logic.fps)
        self.parent.after(delay, self.play_loop)

    def seek_seconds(self, sec):
        if not self.logic.cap: return
        self.logic.current_frame = int(sec * self.logic.fps)
        self.logic.current_frame = max(0, min(self.logic.total_frames - 1, self.logic.current_frame))
        self.show_frame()

    def seek_relative_sec(self, delta):
        self.seek_seconds((self.logic.current_frame / self.logic.fps) + delta)

    def seek_relative_frames(self, delta):
        if not self.logic.cap: return
        self.logic.current_frame = max(0, min(self.logic.total_frames - 1, self.logic.current_frame + delta))
        self.show_frame()

    def seek_toi(self, direction):
        if self.logic.df_tois is None or self.logic.df_tois.empty:
            return
        
        curr_sec = self.logic.current_frame / self.logic.fps
        
        if direction > 0:
            # Next TOI: Find first TOI starting after current time
            candidates = self.logic.df_tois[self.logic.df_tois['Start'] > (curr_sec + 0.1)]
            if not candidates.empty:
                self.seek_seconds(candidates.iloc[0]['Start'])
        else:
            # Prev TOI: Find last TOI starting before current time
            candidates = self.logic.df_tois[self.logic.df_tois['Start'] < (curr_sec - 0.1)]
            if not candidates.empty:
                self.seek_seconds(candidates.iloc[-1]['Start'])

    def show_frame(self):
        ret, frame = self.logic.get_frame_image()
        if not ret or frame is None: return
        
        curr_sec = self.logic.current_frame / self.logic.fps
        self.timeline.update_cursor(curr_sec)
        self.lbl_time.config(text=f"{self._fmt_time(curr_sec)} / {self._fmt_time(self.logic.total_duration)}")
        
        # Overlays
        info_lines = [f"Time: {curr_sec:.3f}s", f"Frame: {self.logic.current_frame}"]
        
        # 1. TOI Info
        active_toi = "None"
        if self.logic.df_tois is not None:
            matches = self.logic.df_tois[(self.logic.df_tois['Start'] <= curr_sec) & (self.logic.df_tois['End'] >= curr_sec)]
            if not matches.empty:
                r = matches.iloc[0]
                active_toi = r.get('Name', 'TOI')
                cond = r.get('Condition', '')
                phase = r.get('Phase', '')
                info_lines.append(f"TOI: {active_toi}")
                info_lines.append(f"Cond: {cond}")
                info_lines.append(f"Phase: {phase}")
                
                # Draw on video
                cv2.putText(frame, f"{active_toi} ({cond})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 2. Gaze Info
        if self.logic.df_gaze is not None:
            idx = bisect.bisect_left(self.logic.gaze_t, curr_sec)
            if idx < len(self.logic.gaze_t):
                # Check if sample is close enough (e.g. within 1 frame duration)
                sample_t = self.logic.gaze_t[idx]
                if abs(sample_t - curr_sec) < (1.0 / self.logic.fps):
                    gx, gy = int(self.logic.gaze_x[idx]), int(self.logic.gaze_y[idx])
                    
                    # Draw Crosshair
                    cv2.circle(frame, (gx, gy), 10, (0, 0, 255), 2)
                    cv2.line(frame, (gx-15, gy), (gx+15, gy), (0, 0, 255), 2)
                    cv2.line(frame, (gx, gy-15), (gx, gy+15), (0, 0, 255), 2)
                    
                    # Hit Info
                    row = self.logic.df_gaze.iloc[idx]
                    hit_role = row.get('Hit_Role', 'None')
                    hit_aoi = row.get('Hit_AOI', 'None')
                    info_lines.append("-" * 20)
                    info_lines.append(f"Gaze X,Y: {gx}, {gy}")
                    info_lines.append(f"Hit Role: {hit_role}")
                    info_lines.append(f"Hit AOI:  {hit_aoi}")
                    
                    if str(hit_role) != "None" and pd.notna(hit_role):
                        cv2.putText(frame, f"HIT: {hit_role}-{hit_aoi}", (gx + 20, gy - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Update Info Box
        self.txt_info.config(state="normal")
        self.txt_info.delete("1.0", tk.END)
        self.txt_info.insert("1.0", "\n".join(info_lines))
        self.txt_info.config(state="disabled")

        # Display Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        # Resize to fit label
        w = self.lbl_video.winfo_width()
        h = self.lbl_video.winfo_height()
        if w > 10 and h > 10:
            img.thumbnail((w, h), Image.Resampling.LANCZOS)
            
        imgtk = ImageTk.PhotoImage(image=img)
        self.lbl_video.imgtk = imgtk # type: ignore # Keep ref
        self.lbl_video.configure(image=imgtk)

    def _fmt_time(self, seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 100)
        return f"{m:02d}:{s:02d}.{ms:02d}"
