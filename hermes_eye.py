import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import json
import gzip
import os
import math
import threading
import cv2
from PIL import Image, ImageTk


# ═══════════════════════════════════════════════════════════════════
# MODEL — Pure logic
# ═══════════════════════════════════════════════════════════════════

class GazeLogic:
    """
    Pure computational logic for gaze-to-AOI mapping.
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
        # DataFrame, similar to a table, with columns and rows. 
        # Each column has a name and a type, e.g. 'Frame' (int), 'x1' (float), 'Role' (str), etc.
        df = pd.read_csv(csv_path)
        # Flexible ID column detection
        if 'ID' in df.columns:
            id_col_name = 'ID'
        elif 'TrackID' in df.columns:
            id_col_name = 'TrackID'
        else:
            # 2. Flexible research: looks for any column that contains 'id' (case-insensitive) in its name
            #for col in df.columns:
                # Sostituisce _ e - con spazi, converte in minuscolo e spezza in lista
                # Es: "id_track" -> "id track" -> ['id', 'track'] -> TROVATO
                # Es: "valido" -> "valido" -> ['valido'] -> IGNORATO
                #clean_parts = col.lower().replace('_', ' ').replace('-', ' ').split()
                #if 'id' in clean_parts:
                    #id_col_name = col
                    #break
            raise ValueError(
                f"Missing ID column in AOI CSV. Found columns: {list(df.columns)}"
            )
        
        aoi_lookup: dict[int, list[dict]] = {}
        for frame, group in df.groupby('Frame'):
            # aoi_lookup is a dict where the keys are frame numbers (int) and the values are lists of dicts.
            # Each dict represents an AOI in that frame with its properties (x1, y1, x2, y2, Role, AOI, ID/TrackID, etc.)
            aoi_lookup[int(frame)] = group.to_dict('records')  # type: ignore[arg-type]
        return aoi_lookup, id_col_name

    # ── Hit-Testing ─────────────────────────────────────────────

    @staticmethod
    def _point_in_polygon(px: float, py: float, points: list[tuple[float, float]]) -> bool:
        if len(points) < 3:
            return False
        inside = False
        j = len(points) - 1
        for i in range(len(points)):
            xi, yi = points[i]
            xj, yj = points[j]
            intersects = ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / max(1e-9, (yj - yi)) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _polygon_area(points: list[tuple[float, float]]) -> float:
        if len(points) < 3:
            return 0.0
        area = 0.0
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            area += (x1 * y2) - (x2 * y1)
        return abs(area) / 2.0

    @staticmethod
    def _parse_shape_points(raw) -> list[tuple[float, float]]:
        if raw is None:
            return []
        if isinstance(raw, list):
            seq = raw
        else:
            txt = str(raw).strip()
            if txt == "" or txt.lower() == "nan":
                return []
            try:
                seq = json.loads(txt)
            except Exception:
                return []
        pts: list[tuple[float, float]] = []
        if isinstance(seq, list):
            for p in seq:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    try:
                        pts.append((float(p[0]), float(p[1])))
                    except (TypeError, ValueError):
                        continue
        return pts

    @staticmethod
    def _shape_hit_and_area(gaze_x: float, gaze_y: float, aoi: dict) -> tuple[bool, float, str]:
        x1 = float(aoi.get('x1', 0))
        y1 = float(aoi.get('y1', 0))
        x2 = float(aoi.get('x2', 0))
        y2 = float(aoi.get('y2', 0))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        box_area = max(1.0, (x2 - x1) * (y2 - y1))

        shape = str(aoi.get("ShapeType", "box")).lower()
        if shape == "circle":
            cx = float(aoi.get("CenterX", (x1 + x2) / 2))
            cy = float(aoi.get("CenterY", (y1 + y2) / 2))
            r = float(aoi.get("Radius", min(x2 - x1, y2 - y1) / 2))
            r = max(1.0, r)
            dx = gaze_x - cx
            dy = gaze_y - cy
            return (dx * dx + dy * dy) <= (r * r), max(1.0, math.pi * r * r), "circle"

        if shape in ("oval", "ellipse"):
            cx = float(aoi.get("CenterX", (x1 + x2) / 2))
            cy = float(aoi.get("CenterY", (y1 + y2) / 2))
            rx = float(aoi.get("RadiusX", (x2 - x1) / 2))
            ry = float(aoi.get("RadiusY", (y2 - y1) / 2))
            rx = max(1.0, rx)
            ry = max(1.0, ry)
            nx = (gaze_x - cx) / rx
            ny = (gaze_y - cy) / ry
            return (nx * nx + ny * ny) <= 1.0, max(1.0, math.pi * rx * ry), "oval"

        if shape == "polygon":
            points = GazeLogic._parse_shape_points(aoi.get("ShapePoints", ""))
            if len(points) >= 3:
                hit = GazeLogic._point_in_polygon(gaze_x, gaze_y, points)
                area = max(1.0, GazeLogic._polygon_area(points))
                return hit, area, "polygon"

        hit = (x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2)
        return hit, box_area, "box"

    @staticmethod
    def calculate_hit(gaze_x: float, gaze_y: float,
                      aois_in_frame: list[dict],
                      id_col_name: str) -> dict | None:
        """
        Follows a pure geometric hit-test.
        Returns the AOI with the smallest area that contains the gaze point, or None if no AOI is hit.
        """
        # It takes the gaze coordinates (x, y), the list of all AOIs present in the current frame, 
        # and the name of the column used for IDs (e.g., "TrackID").
        # It returns a dictionary (the winning AOI) or None if the user is looking at nothing/background.

        hits = []
        # It iterates through every potential target in the frame (e.g., Person A's Hand, Person B's Head).
        for aoi in aois_in_frame:
            # And extracts the coordinates of the bounding box.
            x1, y1, x2, y2 = aoi['x1'], aoi['y1'], aoi['x2'], aoi['y2']
            
            # If the gaze point is within the bounding box, it calculates the area of the AOI (width * height) 
            # and adds it to a list of hits.
            is_hit, area, shape = GazeLogic._shape_hit_and_area(gaze_x, gaze_y, aoi)
            if is_hit:
                
                # The dictionary for each hit contains the role (e.g., "Hand"), the AOI name (e.g., "Person A Hand"), 
                # the track ID, and the area of the AOI.
                hits.append({
                    "role": aoi['Role'],
                    "aoi":  aoi['AOI'],
                    "tid":  aoi[id_col_name],
                    "area": area,
                    "shape": shape,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })

        # If the loop finishes and the list is empty, the user was looking at the background. 
        # In this case, the function returns None.
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
        # The timestamp_to_frame function takes a gaze timestamp (in seconds), applies the synchronization offset 
        # (also in seconds), and multiplies by the frame rate (frames per second) to get the corresponding frame index.
        # The offset allows for correcting any temporal misalignment between the gaze data and the video.
        return int((timestamp - offset) * fps)

    # ── Main Mapping Pipeline ───────────────────────────────────
    
    def run_mapping(self,
                    aoi_path: str,
                    gaze_path: str,
                    video_res: tuple[int, int],
                    fps: float,
                    offset: float,
                    output_dir: str | None = None,  # <--- Nuovo Parametro
                    progress_callback=None) -> tuple[str, int]:
        
        self._cancel_flag = False
        W, H = video_res

        # 1. Load & index AOI
        if progress_callback:
            progress_callback("Loading AOI into memory...")
        
        # The load_aoi_data function reads the AOI CSV file and organizes the AOI information into a dictionary.
        # It also detects which column in the CSV contains the unique identifier for each AOI.
        aoi_lookup, id_col_name = self.load_aoi_data(aoi_path)
        
        if progress_callback:
            progress_callback(f"AOI indexed ({len(aoi_lookup)} frames). Streaming gaze…")

        # 2. Stream gaze data 
        output_rows: list[dict] = []

        # With the gazedata.gz file open, it reads it line by line. 
        # (Instead of loading the whole file in memory, it reads line by line).
        with gzip.open(gaze_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if self._cancel_flag:
                    raise InterruptedError("Mapping cancelled by user.")
                
                # The line read from the file is just a text string (e.g., '{"timestamp": 123, ...}'). 
                # This function converts that string into a real Python dictionary.
                try:
                    gaze_pkg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # "Does this packet actually contain data? And specifically, does it contain 2D gaze coordinates?" 
                # If not, skip it. This check is to avoid "Syncport" beacons or non-gaze data.
                if 'data' not in gaze_pkg or 'gaze2d' not in gaze_pkg['data']:
                    continue

                # The timestamp is extracted from the gaze package.
                ts = gaze_pkg.get('timestamp', 0)
                
                # The normalized gaze coordinates (gaze2d) are extracted. 
                # These are typically in the range [0.0, 1.0].
                g2d = gaze_pkg['data']['gaze2d']
                if not g2d:
                    continue

                # Extract 3D Gaze (HUCS mm) if available - Useful for I-VT
                g3d = gaze_pkg['data'].get('gaze3d')
                g3d_x, g3d_y, g3d_z = (g3d[0], g3d[1], g3d[2]) if g3d and len(g3d) >= 3 else (None, None, None)

                # The normalized gaze coordinates are unpacked into gx and gy for easier access.
                gx, gy = g2d[0], g2d[1]
                
                # The timestamp is converted to a frame index using the timestamp_to_frame function.
                frame_idx = self.timestamp_to_frame(ts, offset, fps)
                
                # If the resulting frame index is negative, it means the gaze point occurs before the start of the video.
                if frame_idx < 0:
                    continue

                px, py = self.normalised_to_pixel(gx, gy, W, H)
                # The normalized gaze coordinates (gx, gy) are converted to pixel coordinates (px, py) using the normalised_to_pixel function. This function multiplies the normalized values by the video resolution (width and height) to get the actual pixel position of the gaze on the video frame.

                # Hit-test
                # For the current frame index, it retrieves the list of active AOIs from the aoi_lookup dictionary.
                active_aois = aoi_lookup.get(frame_idx, [])
                
                # The calculate_hit function is called to determine if the gaze point (px, py) hits any of the active AOIs in the current frame. 
                # It checks if the gaze point is within the bounding box of each AOI and returns the one with the smallest area that contains the gaze point. 
                best = self.calculate_hit(px, py, active_aois, id_col_name)

                if best:
                    hit_role, hit_aoi, hit_tid = best['role'], best['aoi'], best['tid']
                    hit_shape = best.get('shape', 'box')
                    hit_x1, hit_y1, hit_x2, hit_y2 = best['x1'], best['y1'], best['x2'], best['y2']
                else:
                    hit_role, hit_aoi, hit_tid = "None", "None", -1
                    hit_shape = "None"
                    hit_x1, hit_y1, hit_x2, hit_y2 = -1, -1, -1, -1


                output_rows.append({
                    "Timestamp":    ts,
                    "Frame_Est":    frame_idx,
                    "Gaze_X":       px,
                    "Gaze_Y":       py,
                    "Hit_Role":     hit_role,
                    "Hit_AOI":      hit_aoi,
                    "Hit_TrackID":  hit_tid,
                    "Hit_Shape":    hit_shape,
                    "Hit_x1":       hit_x1,
                    "Hit_y1":       hit_y1,
                    "Hit_x2":       hit_x2,
                    "Hit_y2":       hit_y2,
                    "Raw_Gaze2D_X": gx,
                    "Raw_Gaze2D_Y": gy,
                    "Gaze_3D_X":    g3d_x,
                    "Gaze_3D_Y":    g3d_y,
                    "Gaze_3D_Z":    g3d_z,
                })

        # 3. Write output
        if progress_callback:
            progress_callback("Saving CSV…")

        base_name = os.path.basename(gaze_path).replace(".gz", "")
        filename = f"{base_name}_MAPPED.csv"

        # Se output_dir esiste, usalo. Altrimenti usa la cartella del file input.
        if output_dir and os.path.exists(output_dir):
            target_dir = output_dir
        else:
            target_dir = os.path.dirname(gaze_path)
            
        out_path = os.path.join(target_dir, filename)

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
        self.output_dir = tk.StringVar()

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
        if hasattr(self.context, 'output_dir') and self.context.output_dir:
            self.output_dir.set(self.context.output_dir)
            
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
        self._add_dir_picker(lf_files, "Output Folder:", self.output_dir)

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
    
    def _add_dir_picker(self, parent, label, var):
        f = tk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=label, width=20, anchor="w").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(f, text="...", width=3, 
                  command=lambda: var.set(filedialog.askdirectory())).pack(side=tk.LEFT)

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
            "output_dir": self.output_dir.get(),
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
                output_dir=params["output_dir"],
                progress_callback=self._on_progress,
            )
            self.context.mapped_csv_path = out_path
            self.parent.after(0, lambda: self._on_success(out_path, total_rows))
        except InterruptedError:
            self.parent.after(0, self._on_cancelled)
        except Exception as e:
            self.parent.after(0, lambda e=e: self._on_error(e))

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
