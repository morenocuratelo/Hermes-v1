import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import json
import gzip
import os
import threading
import cv2
from PIL import Image, ImageTk


# ═══════════════════════════════════════════════════════════════════
# MODEL — Pure logic, no tkinter
# ═══════════════════════════════════════════════════════════════════

class GazeLogic:
    """
    Pure computational logic for gaze-to-AOI mapping.
    No tkinter imports or UI references.
    """

    def __init__(self):
        self._cancel_flag = False

    def cancel(self):
        self._cancel_flag = True

    # ── AOI Loading ─────────────────────────────────────────────

    def load_aoi_data(self, csv_path: str) -> tuple[dict, str]:
        """
        Parses the AOI CSV into a dict indexed by frame number.

        Returns
        -------
        aoi_lookup : dict[int, list[dict]]
            { frame_idx: [ {x1, y1, x2, y2, Role, AOI, id_col, ...} ] }
        id_col_name : str
            The name of the identity column found ('ID' or 'TrackID').
        """
        df = pd.read_csv(csv_path)

        # Flexible ID column detection
        if 'ID' in df.columns:
            id_col_name = 'ID'
        elif 'TrackID' in df.columns:
            id_col_name = 'TrackID'
        else:
            raise ValueError(
                f"Missing ID column in AOI CSV. Found columns: {list(df.columns)}"
            )

        aoi_lookup: dict[int, list[dict]] = {}
        for frame, group in df.groupby('Frame'):
            aoi_lookup[int(frame)] = group.to_dict('records')  # type: ignore[arg-type]

        return aoi_lookup, id_col_name

    # ── Hit-Testing ─────────────────────────────────────────────

    @staticmethod
    def calculate_hit(gaze_x: float, gaze_y: float,
                      aois_in_frame: list[dict],
                      id_col_name: str) -> dict | None:
        """
        Pure geometric hit-test.

        Returns the AOI with the smallest area that contains the gaze
        point, or *None* if no AOI is hit.
        """
        hits = []
        for aoi in aois_in_frame:
            x1, y1, x2, y2 = aoi['x1'], aoi['y1'], aoi['x2'], aoi['y2']
            if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
                area = (x2 - x1) * (y2 - y1)
                hits.append({
                    "role": aoi['Role'],
                    "aoi":  aoi['AOI'],
                    "tid":  aoi[id_col_name],
                    "area": area,
                })

        if not hits:
            return None
        return min(hits, key=lambda h: h['area'])

    # ── Coordinate Conversion ───────────────────────────────────

    @staticmethod
    def normalised_to_pixel(gx: float, gy: float,
                            width: int, height: int) -> tuple[float, float]:
        """Convert normalised gaze (0.0-1.0) to pixel coordinates."""
        return gx * width, gy * height

    @staticmethod
    def timestamp_to_frame(timestamp: float, offset: float,
                           fps: float) -> int:
        """Convert a gaze timestamp (seconds) to a frame index."""
        return int((timestamp - offset) * fps)

    # ── Main Mapping Pipeline ───────────────────────────────────

    def run_mapping(self,
                    aoi_path: str,
                    gaze_path: str,
                    video_res: tuple[int, int],
                    fps: float,
                    offset: float,
                    progress_callback=None) -> tuple[str, int]:
        """
        End-to-end gaze → AOI mapping.

        Parameters
        ----------
        aoi_path : str
            Path to the AOI CSV produced by the Region module.
        gaze_path : str
            Path to the Tobii .gz gaze-data file.
        video_res : (width, height)
            Video resolution in pixels.
        fps : float
            Frame rate of the scene video.
        offset : float
            Sync offset in seconds (negative = gaze starts after video).
        progress_callback : callable(message: str) | None
            Optional UI feedback hook (called from worker thread).

        Returns
        -------
        out_path : str
            Path of the written _MAPPED.csv file.
        total_rows : int
            Number of gaze samples written.
        """
        self._cancel_flag = False
        W, H = video_res

        # 1. Load & index AOI
        if progress_callback:
            progress_callback("Loading AOI into memory...")
        aoi_lookup, id_col_name = self.load_aoi_data(aoi_path)
        if progress_callback:
            progress_callback(f"AOI indexed ({len(aoi_lookup)} frames). Streaming gaze…")

        # 2. Stream gaze data
        output_rows: list[dict] = []

        with gzip.open(gaze_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if self._cancel_flag:
                    raise InterruptedError("Mapping cancelled by user.")

                try:
                    gaze_pkg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if 'data' not in gaze_pkg or 'gaze2d' not in gaze_pkg['data']:
                    continue

                ts = gaze_pkg.get('timestamp', 0)
                g2d = gaze_pkg['data']['gaze2d']
                if not g2d:
                    continue

                gx, gy = g2d[0], g2d[1]

                frame_idx = self.timestamp_to_frame(ts, offset, fps)
                if frame_idx < 0:
                    continue

                px, py = self.normalised_to_pixel(gx, gy, W, H)

                # Hit-test
                active_aois = aoi_lookup.get(frame_idx, [])
                best = self.calculate_hit(px, py, active_aois, id_col_name)

                if best:
                    hit_role = best['role']
                    hit_aoi  = best['aoi']
                    hit_tid  = best['tid']
                else:
                    hit_role = "None"
                    hit_aoi  = "None"
                    hit_tid  = -1

                output_rows.append({
                    "Timestamp":    ts,
                    "Frame_Est":    frame_idx,
                    "Gaze_X":       px,
                    "Gaze_Y":       py,
                    "Hit_Role":     hit_role,
                    "Hit_AOI":      hit_aoi,
                    "Hit_TrackID":  hit_tid,
                    "Raw_Gaze2D_X": gx,
                    "Raw_Gaze2D_Y": gy,
                })

        # 3. Write output
        if progress_callback:
            progress_callback("Saving CSV…")

        out_path = gaze_path.replace(".gz", "_MAPPED.csv")
        if out_path == gaze_path:
            out_path += "_mapped.csv"

        df_out = pd.DataFrame(output_rows)
        df_out.to_csv(out_path, index=False)

        return out_path, len(df_out)


# ═══════════════════════════════════════════════════════════════════
# VIEW / CONTROLLER — UI only
# ═══════════════════════════════════════════════════════════════════

class GazeView:
    def __init__(self, parent, context):
        self.parent = parent
        self.context = context
        self.logic = GazeLogic()

        # UI variables
        self.aoi_path = tk.StringVar()
        self.gaze_path = tk.StringVar()

        self.video_res_w = tk.IntVar(value=1920)
        self.video_res_h = tk.IntVar(value=1080)
        self.fps = tk.DoubleVar(value=25.0)
        self.sync_offset = tk.DoubleVar(value=0.0)

        self._build_ui()

        # Auto-load from context
        if self.context.aoi_csv_path and os.path.exists(self.context.aoi_csv_path):
            self.aoi_path.set(self.context.aoi_csv_path)
        if hasattr(self.context, 'gaze_data_path') and self.context.gaze_data_path:
            self.gaze_path.set(self.context.gaze_data_path)

    # ── UI Construction ─────────────────────────────────────────

    def _build_ui(self):
        tk.Label(self.parent, text="5. Eye Mapping (Gaze → AOI)",
                 font=("Segoe UI", 18, "bold"), bg="white"
                 ).pack(pady=(0, 10), anchor="w")

        main = tk.Frame(self.parent, padx=20, pady=20, bg="white")
        main.pack(fill=tk.BOTH, expand=True)

        tk.Label(main, text="Gaze Mapper: Eye Tracking + AOI",
                 font=("Segoe UI", 16, "bold")).pack(pady=(0, 20))

        # 1. File inputs
        lf_files = tk.LabelFrame(main, text="1. Input Files", padx=10, pady=10)
        lf_files.pack(fill=tk.X, pady=5)

        self._add_file_picker(lf_files, "AOI File (.csv):", self.aoi_path, "*.csv")
        self._add_file_picker(lf_files, "Tobii Gaze Data (.gz):", self.gaze_path, "*.gz")

        # 2. Sync parameters
        lf_params = tk.LabelFrame(main, text="2. Video & Sync Parameters",
                                  padx=10, pady=10)
        lf_params.pack(fill=tk.X, pady=5)

        grid_f = tk.Frame(lf_params)
        grid_f.pack(fill=tk.X)

        tk.Label(grid_f, text="Video Resolution (WxH):").grid(
            row=0, column=0, sticky="w")
        tk.Entry(grid_f, textvariable=self.video_res_w, width=8).grid(
            row=0, column=1)
        tk.Label(grid_f, text="x").grid(row=0, column=2)
        tk.Entry(grid_f, textvariable=self.video_res_h, width=8).grid(
            row=0, column=3)

        tk.Label(grid_f, text="Frame Rate (FPS):").grid(
            row=1, column=0, sticky="w", pady=5)
        tk.Entry(grid_f, textvariable=self.fps, width=8).grid(
            row=1, column=1, pady=5)

        tk.Label(grid_f, text="Sync Offset (sec):").grid(
            row=2, column=0, sticky="w")
        tk.Entry(grid_f, textvariable=self.sync_offset, width=8).grid(
            row=2, column=1)
        tk.Label(grid_f,
                 text="(Use negative values if Gaze starts AFTER video)",
                 fg="gray", font=("Arial", 8)).grid(
            row=2, column=2, columnspan=3, sticky="w")

        # 3. Buttons
        self.btn_process = tk.Button(
            main, text="PROCESS AND MAP", bg="#007ACC", fg="white",
            font=("Bold", 12), height=2, command=self.run_process)
        self.btn_process.pack(fill=tk.X, pady=20)

        tk.Button(main, text="📺 VIEW RESULTS PLAYER",
                  command=self.open_player).pack(fill=tk.X, pady=5)

        self.progress = ttk.Progressbar(main, orient=tk.HORIZONTAL,
                                        mode='indeterminate')
        self.progress.pack(fill=tk.X)

        self.lbl_status = tk.Label(main, text="Ready.")
        self.lbl_status.pack()

    def _add_file_picker(self, parent, label, var, filetype):
        f = tk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=label, width=20, anchor="w").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(f, text="...", width=3,
                  command=lambda: self._browse(var, filetype)).pack(side=tk.LEFT)

    def _browse(self, var, ft):
        path = filedialog.askopenfilename(filetypes=[("File", ft)])
        if path:
            var.set(path)
            if var is self.gaze_path:
                self.context.gaze_data_path = path

    # ── Process Orchestration (threaded) ────────────────────────

    def run_process(self):
        if not self.aoi_path.get() or not self.gaze_path.get():
            messagebox.showwarning("Missing Files",
                                   "Select both AOI file and GazeData file.")
            return

        # Switch button to Cancel mode
        self.btn_process.config(
            text="⛔ STOP PROCESS", bg="#CC0000", fg="white",
            command=self._cancel_process)
        self.progress.start(10)
        self.lbl_status.config(text="Starting…")

        params = {
            "aoi_path":  self.aoi_path.get(),
            "gaze_path": self.gaze_path.get(),
            "video_res": (self.video_res_w.get(), self.video_res_h.get()),
            "fps":       self.fps.get(),
            "offset":    self.sync_offset.get(),
        }

        threading.Thread(target=self._thread_worker,
                         args=(params,), daemon=True).start()

    def _cancel_process(self):
        if messagebox.askyesno("Stop", "Abort processing?"):
            self.logic.cancel()
            self.lbl_status.config(text="Cancelling…")

    def _thread_worker(self, params):
        try:
            out_path, total_rows = self.logic.run_mapping(
                aoi_path=params["aoi_path"],
                gaze_path=params["gaze_path"],
                video_res=params["video_res"],
                fps=params["fps"],
                offset=params["offset"],
                progress_callback=self._on_progress,
            )
            self.context.mapped_csv_path = out_path
            self.parent.after(0, lambda: self._on_success(out_path, total_rows))
        except InterruptedError:
            self.parent.after(0, self._on_cancelled)
        except Exception as e:
            self.parent.after(0, lambda: self._on_error(e))

    # ── Thread-safe UI callbacks ────────────────────────────────

    def _on_progress(self, message: str):
        self.parent.after(0, lambda: self.lbl_status.config(text=message))

    def _restore_button(self):
        """Reset the process button to its default state."""
        self.btn_process.config(
            text="PROCESS AND MAP", bg="#007ACC", fg="white",
            state=tk.NORMAL, command=self.run_process)
        self.progress.stop()

    def _on_success(self, out_path: str, total_rows: int):
        self._restore_button()
        self.lbl_status.config(text="Done.")
        messagebox.showinfo(
            "Success",
            f"Mapping complete!\n"
            f"File saved in:\n{out_path}\n\n"
            f"Total rows: {total_rows}")

    def _on_cancelled(self):
        self._restore_button()
        self.lbl_status.config(text="Cancelled by user.")
        messagebox.showwarning("Cancelled", "Processing was stopped by the user.")

    def _on_error(self, exc: Exception):
        self._restore_button()
        self.lbl_status.config(text="Error.")
        messagebox.showerror("Critical Error", str(exc))

    # ── Player launcher ─────────────────────────────────────────

    def open_player(self):
        vid = self.context.video_path
        csv = getattr(self.context, 'mapped_csv_path', None)

        if not csv and self.gaze_path.get():
            candidate = self.gaze_path.get().replace(".gz", "_MAPPED.csv")
            if os.path.exists(candidate):
                csv = candidate

        if not vid or not os.path.exists(vid):
            messagebox.showerror(
                "Error",
                "Video not found in context.\n"
                "Load a video in Human or Entity first.")
            return
        if not csv or not os.path.exists(csv):
            messagebox.showerror(
                "Error",
                "Mapped CSV file not found.\nRun mapping first.")
            return

        GazeResultPlayer(self.parent, vid, csv)


# ═══════════════════════════════════════════════════════════════════
# VISUALISATION — Result Player
# ═══════════════════════════════════════════════════════════════════

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

        # Load mapped CSV
        try:
            self.df = pd.read_csv(csv_path)
            self.data_map: dict[int, list] = {}
            for _, row in self.df.iterrows():
                f_idx = int(row['Frame_Est'])
                if f_idx not in self.data_map:
                    self.data_map[f_idx] = []
                self.data_map[f_idx].append(row)
        except Exception as e:
            messagebox.showerror("CSV Error", str(e))
            self.win.destroy()
            return

        self.is_playing = False
        self.current_frame = 0

        # Video display
        self.lbl_video = tk.Label(self.win, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

        # Controls
        ctrl = tk.Frame(self.win, bg="#eee", pady=5)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM)

        self.slider = tk.Scale(
            ctrl, from_=0, to=self.total_frames - 1,
            orient=tk.HORIZONTAL, command=self.on_seek,
            showvalue=False, bg="#eee", highlightthickness=0)
        self.slider.pack(fill=tk.X, padx=10, pady=(0, 5))

        btn_frame = tk.Frame(ctrl, bg="#eee")
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="⏮ -10",
                  command=lambda: self.step(-10), width=6).pack(
            side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="⏴ -1",
                  command=lambda: self.step(-1), width=6).pack(
            side=tk.LEFT, padx=2)

        self.btn_play = tk.Button(
            btn_frame, text="▶ Play", command=self.toggle_play,
            font=("Bold", 10), width=10, bg="#ddd")
        self.btn_play.pack(side=tk.LEFT, padx=10)

        tk.Button(btn_frame, text="+1 ⏵",
                  command=lambda: self.step(1), width=6).pack(
            side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="+10 ⏭",
                  command=lambda: self.step(10), width=6).pack(
            side=tk.LEFT, padx=2)

        self.lbl_info = tk.Label(ctrl, text="Ready",
                                 font=("Consolas", 10), bg="#eee")
        self.lbl_info.pack(pady=2)

        self.win.update_idletasks()
        self._setup_hotkeys()
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_frame()

    # ── Lifecycle ───────────────────────────────────────────────

    def on_close(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()
        self.win.destroy()

    # ── Hotkeys ─────────────────────────────────────────────────

    def _setup_hotkeys(self):
        self.win.bind("<Space>", self._on_space)
        self.win.bind("<Left>", self._on_left)
        self.win.bind("<Right>", self._on_right)
        self.win.bind("<Shift-Left>", self._on_shift_left)
        self.win.bind("<Shift-Right>", self._on_shift_right)
        self.win.focus_set()

    def _is_hotkey_safe(self):
        focused = self.win.focus_get()
        return not (focused and focused.winfo_class() in ('Entry', 'Text'))

    def _on_space(self, event):
        if self._is_hotkey_safe():
            self.toggle_play()

    def _on_left(self, event):
        if self._is_hotkey_safe():
            self.step(-1)

    def _on_right(self, event):
        if self._is_hotkey_safe():
            self.step(1)

    def _on_shift_left(self, event):
        if self._is_hotkey_safe():
            self.step(-10)

    def _on_shift_right(self, event):
        if self._is_hotkey_safe():
            self.step(10)

    # ── Playback ────────────────────────────────────────────────

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
        self.current_frame = max(
            0, min(self.current_frame + delta, self.total_frames - 1))
        self.slider.set(self.current_frame)
        self.show_frame()

    def play_loop(self):
        if not self.is_playing:
            return
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.is_playing = False
            self.btn_play.config(text="▶ Play")
            return
        self.slider.set(self.current_frame)
        self.show_frame()
        self.win.after(int(1000 / self.fps), self.play_loop)

    def on_seek(self, val):
        self.current_frame = int(val)
        self.show_frame()

    # ── Rendering ───────────────────────────────────────────────

    def show_frame(self):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return

        # Overlay gaze data
        if self.current_frame in self.data_map:
            samples = self.data_map[self.current_frame]
            hit_samples = [
                s for s in samples
                if s['Hit_Role'] != 'None' and pd.notna(s['Hit_Role'])
            ]
            best_sample = hit_samples[0] if hit_samples else samples[0]
            gx, gy = int(best_sample['Gaze_X']), int(best_sample['Gaze_Y'])
            role = best_sample['Hit_Role']
            aoi = best_sample['Hit_AOI']

            is_hit = role != 'None' and pd.notna(role)
            color = (0, 0, 255) if is_hit else (0, 255, 255)

            if is_hit:
                cv2.putText(frame, f"HIT: {role} ({aoi})",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 2)
            cv2.circle(frame, (gx, gy), 10, color, 2)
            cv2.line(frame, (gx - 15, gy), (gx + 15, gy), color, 2)
            cv2.line(frame, (gx, gy - 15), (gx, gy + 15), color, 2)

            self.lbl_info.config(
                text=f"Frame: {self.current_frame} | Hit: {role} - {aoi}")
        else:
            self.lbl_info.config(
                text=f"Frame: {self.current_frame} | No Gaze Data")

        # Convert & resize deterministically
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        w = self.lbl_video.winfo_width()
        h = self.lbl_video.winfo_height()
        render_w = w if w >= 50 else 800
        render_h = h if h >= 50 else 600

        img.thumbnail((render_w, render_h), Image.Resampling.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(image=img)
        self.lbl_video.configure(image=self.tk_img)
        self.lbl_video.image = self.tk_img  # type: ignore[attr-defined]  # prevent GC