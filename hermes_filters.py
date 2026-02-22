import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
# MODEL — Filter Logic (I-VT Algorithm)
# ═══════════════════════════════════════════════════════════════════


class FilterLogic:
    def __init__(self):
        self.df = None
        self.fixations = []
        self.processed_df = None
        self.sample_rate = 0.0

    def load_data(self, path):
        try:
            self.df = pd.read_csv(path)
            if "Timestamp" in self.df.columns:
                self.df = self.df.sort_values("Timestamp").reset_index(drop=True)
            return True
        except Exception as e:
            print(f"Load Error: {e}")
            return False

    def run_ivt(self, params, progress_callback=None):
        """
        Esegue l'algoritmo I-VT (Velocity-Threshold Identification) completo.
        Replica la pipeline di Tobii Pro Lab:
        1. Gap Fill (Interpolazione)
        2. Noise Reduction (Moving Median/Average)
        3. Velocity Calculation (Angular)
        4. Classification (Fixation vs Saccade)
        5. Merge Adjacent Fixations
        6. Discard Short Fixations
        """
        if self.df is None:
            raise ValueError("No data loaded")

        if progress_callback:
            progress_callback("Preprocessing (Gap Fill & Noise)...")

        # Lavoriamo su una copia per non distruggere i dati originali in memoria
        df = self.df.copy()

        # 1. Stima Sampling Rate (per convertire ms in samples)
        diffs = df["Timestamp"].diff()
        avg_diff = diffs[diffs < 0.1].mean()  # Ignora gap > 100ms
        self.sample_rate = 1.0 / avg_diff if avg_diff > 0 else 50.0

        # 2. Gap Fill (Interpolation)
        if params["gap_fill"]:
            max_gap_samples = int((params["max_gap_ms"] / 1000.0) * self.sample_rate)
            # Interpolazione lineare limitata
            df["Gaze_X"] = df["Gaze_X"].interpolate(method="linear", limit=max_gap_samples, limit_direction="both")
            df["Gaze_Y"] = df["Gaze_Y"].interpolate(method="linear", limit=max_gap_samples, limit_direction="both")

        # 3. Noise Reduction
        if params["noise_reduction"] != "None":
            window = int(params["noise_window"])
            if window % 2 == 0:
                window += 1  # Deve essere dispari

            if params["noise_reduction"] == "Moving Median":
                df["Gaze_X"] = df["Gaze_X"].rolling(window=window, center=True, min_periods=1).median()
                df["Gaze_Y"] = df["Gaze_Y"].rolling(window=window, center=True, min_periods=1).median()
            elif params["noise_reduction"] == "Moving Average":
                df["Gaze_X"] = df["Gaze_X"].rolling(window=window, center=True, min_periods=1).mean()
                df["Gaze_Y"] = df["Gaze_Y"].rolling(window=window, center=True, min_periods=1).mean()

        if progress_callback:
            progress_callback("Calculating Angular Velocity...")

        # 4. Velocity Calculation (Angular)
        dt = df["Timestamp"].diff()

        # Common params
        res_w = params["res_w"]
        res_h = params["res_h"]

        # Check for 3D data availability (Preferred for Wearable/Glasses 3)
        has_3d = "Gaze_3D_X" in df.columns and "Gaze_3D_Y" in df.columns and "Gaze_3D_Z" in df.columns

        if has_3d and params.get("tracker_type") == "Wearable":
            # --- WEARABLE 3D (Accurate) ---
            # Use 3D vectors for accurate angular velocity (HUCS)
            # Vectors from origin (0,0,0) to Gaze3D point
            v = df[["Gaze_3D_X", "Gaze_3D_Y", "Gaze_3D_Z"]].fillna(0).to_numpy()

            # Normalize to unit vectors (Direction)
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            v_norm = v / norms

            # Calculate angle between consecutive samples (v_i and v_{i-1})
            # Dot product: a . b = |a||b|cos(theta) -> cos(theta) = a . b (since unit vectors)
            v_prev = np.roll(v_norm, 1, axis=0)

            # Compute dot product row-wise
            dot_prod = np.sum(v_norm * v_prev, axis=1)

            # Clip to domain [-1, 1] to avoid NaN in arccos due to float errors
            dot_prod = np.clip(dot_prod, -1.0, 1.0)

            angles_rad = np.arccos(dot_prod)
            angles_deg = np.degrees(angles_rad)

            # First sample has no previous, set to 0
            angles_deg[0] = 0.0

        elif params.get("tracker_type") == "Wearable":
            # --- WEARABLE 2D (Fallback) ---
            # Usa FOV della Scene Camera per stimare velocità angolare sui pixel video
            fov_h = params.get("fov_h", 95.0)
            fov_v = params.get("fov_v", 63.0)

            # Gradi per pixel (approssimazione lineare valida per piccoli delta)
            deg_per_px_h = fov_h / res_w
            deg_per_px_v = fov_v / res_h

            dx_deg = df["Gaze_X"].diff() * deg_per_px_h
            dy_deg = df["Gaze_Y"].diff() * deg_per_px_v

            # Distanza angolare
            dist_deg = np.sqrt(dx_deg**2 + dy_deg**2)
            angles_deg = dist_deg

        else:
            # --- SCREEN-BASED ---
            # Necessita geometria dello schermo per convertire pixel -> gradi
            screen_w_cm = params["screen_w_cm"]
            screen_h_cm = params["screen_h_cm"]
            distance_cm = params["distance_cm"]

            # Conversione Pixel -> CM
            x_cm = df["Gaze_X"] * (screen_w_cm / res_w)
            y_cm = df["Gaze_Y"] * (screen_h_cm / res_h)

            # Distanza Euclidea tra campioni consecutivi (in cm)
            dx = x_cm.diff()
            dy = y_cm.diff()
            dist_cm = np.sqrt(dx**2 + dy**2)

            # Conversione CM -> Gradi Visivi
            # Formula: theta = 2 * atan(dist / (2 * distance))
            angles_rad = 2 * np.arctan(dist_cm / (2 * distance_cm))
            angles_deg = np.degrees(angles_rad)

        # Velocità (°/s)
        velocity = angles_deg / dt
        df["Velocity"] = velocity.fillna(0)

        if progress_callback:
            progress_callback("Classifying Fixations...")

        # 5. I-VT Classification
        threshold = params["velocity_threshold"]
        # Classificazione iniziale
        df["EventType"] = np.where(df["Velocity"] < threshold, "Fixation", "Saccade")
        # I gap (NaN) diventano Unknown
        df.loc[df["Gaze_X"].isna(), "EventType"] = "Unknown"

        # 6. Raggruppamento in Eventi (Fixation Candidates)
        # Crea un ID univoco per ogni gruppo consecutivo di eventi uguali
        df["grp"] = (df["EventType"] != df["EventType"].shift()).cumsum()

        # Estrai solo i gruppi 'Fixation'
        fix_groups = df[df["EventType"] == "Fixation"].groupby("grp")

        raw_fixations = []
        for _, group in fix_groups:
            if group.empty:
                continue
            start_t = group["Timestamp"].min()
            end_t = group["Timestamp"].max()
            dur_ms = (end_t - start_t) * 1000

            raw_fixations.append(
                {
                    "start": start_t,
                    "end": end_t,
                    "duration": dur_ms,
                    "x": group["Gaze_X"].mean(),
                    "y": group["Gaze_Y"].mean(),
                    "count": len(group),
                }
            )

        if progress_callback:
            progress_callback(f"Merging {len(raw_fixations)} candidates...")

        # 7. Merge Adjacent Fixations
        if params["merge_adjacent"]:
            max_time_s = params["merge_max_time_ms"] / 1000.0
            max_angle = params["merge_max_angle"]

            merged = []
            if raw_fixations:
                curr = raw_fixations[0]
                for i in range(1, len(raw_fixations)):
                    next_f = raw_fixations[i]

                    t_gap = next_f["start"] - curr["end"]

                    # Calcolo angolo tra i centroidi delle due fissazioni
                    if params.get("tracker_type") == "Wearable":
                        fov_h = params.get("fov_h", 95.0)
                        fov_v = params.get("fov_v", 63.0)
                        dx_deg = (next_f["x"] - curr["x"]) * (fov_h / res_w)
                        dy_deg = (next_f["y"] - curr["y"]) * (fov_v / res_h)
                        angle = np.sqrt(dx_deg**2 + dy_deg**2)
                    else:
                        screen_w_cm = params["screen_w_cm"]
                        screen_h_cm = params["screen_h_cm"]
                        distance_cm = params["distance_cm"]
                        dx_cm = (next_f["x"] - curr["x"]) * (screen_w_cm / res_w)
                        dy_cm = (next_f["y"] - curr["y"]) * (screen_h_cm / res_h)
                        d_cm = np.sqrt(dx_cm**2 + dy_cm**2)
                        angle = np.degrees(2 * np.arctan(d_cm / (2 * distance_cm)))

                    if t_gap <= max_time_s and angle <= max_angle:
                        # Merge: Media ponderata delle coordinate
                        total_n = curr["count"] + next_f["count"]
                        curr["x"] = (curr["x"] * curr["count"] + next_f["x"] * next_f["count"]) / total_n
                        curr["y"] = (curr["y"] * curr["count"] + next_f["y"] * next_f["count"]) / total_n
                        curr["end"] = next_f["end"]
                        curr["duration"] = (curr["end"] - curr["start"]) * 1000
                        curr["count"] = total_n
                    else:
                        merged.append(curr)
                        curr = next_f
                merged.append(curr)
                raw_fixations = merged

        # 8. Discard Short Fixations
        if params["discard_short"]:
            min_dur = params["discard_min_dur_ms"]
            raw_fixations = [f for f in raw_fixations if f["duration"] >= min_dur]

        self.fixations = raw_fixations
        self.processed_df = df

        return len(self.fixations)

    def save_fixations(self, path):
        if not self.fixations:
            return
        pd.DataFrame(self.fixations).to_csv(path, index=False)

    def save_filtered_data(self, path):
        if self.processed_df is not None:
            self.processed_df.to_csv(path, index=False)


# ═══════════════════════════════════════════════════════════════════
# VIEW — UI Configuration
# ═══════════════════════════════════════════════════════════════════


class FilterView:
    def __init__(self, parent, context):
        self.parent = parent
        self.context = context
        self.logic = FilterLogic()

        # Variables
        self.input_path = tk.StringVar()
        self.output_fix_path = tk.StringVar()
        self.output_data_path = tk.StringVar()

        # Parameters (Defaults from Tobii Pro Lab I-VT Fixation)
        self.gap_fill = tk.BooleanVar(value=True)
        self.max_gap_ms = tk.IntVar(value=75)

        self.noise_reduction = tk.StringVar(value="Moving Median")
        self.noise_window = tk.IntVar(value=3)

        # Defaults for Tobii Glasses 3 (Attention Filter)
        self.velocity_threshold = tk.DoubleVar(value=100.0)

        self.tracker_type = tk.StringVar(value="Wearable")
        self.fov_h = tk.DoubleVar(value=95.0)
        self.fov_v = tk.DoubleVar(value=63.0)

        self.merge_adjacent = tk.BooleanVar(value=True)
        self.merge_max_time = tk.IntVar(value=75)
        self.merge_max_angle = tk.DoubleVar(value=0.5)

        self.discard_short = tk.BooleanVar(value=True)
        self.discard_min_dur = tk.IntVar(value=60)

        # Geometry (Defaults for standard screen setup)
        self.screen_w = tk.DoubleVar(value=50.0)  # cm
        self.screen_h = tk.DoubleVar(value=30.0)  # cm
        self.distance = tk.DoubleVar(value=60.0)  # cm
        self.res_w = tk.IntVar(value=1920)
        self.res_h = tk.IntVar(value=1080)

        self._setup_ui()

        # Auto-load from context
        if hasattr(self.context, "mapped_csv_path") and self.context.mapped_csv_path:
            self.input_path.set(self.context.mapped_csv_path)
            self._suggest_outputs(self.context.mapped_csv_path)

    def _setup_ui(self):
        tk.Label(
            self.parent,
            text="6. Gaze Filters (I-VT)",
            font=("Segoe UI", 18, "bold"),
            bg="white",
        ).pack(pady=(0, 10), anchor="w")

        main = tk.Frame(self.parent, padx=20, pady=20, bg="white")
        main.pack(fill=tk.BOTH, expand=True)

        # 1. Files
        lf_files = tk.LabelFrame(main, text="1. Input & Output", padx=10, pady=10, bg="white")
        lf_files.pack(fill=tk.X, pady=5)

        self._add_file_picker(lf_files, "Input Mapped CSV:", self.input_path, "*.csv")
        self._add_file_picker(lf_files, "Output Fixations (.csv):", self.output_fix_path, "*.csv", save=True)
        self._add_file_picker(lf_files, "Output Filtered Data (.csv):", self.output_data_path, "*.csv", save=True)

        # 2. Geometry
        lf_geo = tk.LabelFrame(main, text="2. Tracker Geometry (Angular Velocity)", padx=10, pady=10, bg="white")
        lf_geo.pack(fill=tk.X, pady=5)

        # Tracker Type Selector
        self.f_type = tk.Frame(lf_geo, bg="white")
        self.f_type.pack(fill=tk.X, pady=(0, 5))
        tk.Label(self.f_type, text="Tracker Type:", bg="white", font=("Bold", 9)).pack(side=tk.LEFT)
        ttk.Combobox(
            self.f_type, textvariable=self.tracker_type, values=["Wearable", "Screen-Based"], state="readonly", width=15
        ).pack(side=tk.LEFT, padx=5)
        self.tracker_type.trace_add("write", self._update_geo_ui)

        # Wearable Frame (FOV)
        self.f_wearable = tk.Frame(lf_geo, bg="white")
        self.f_wearable.pack(fill=tk.X)
        tk.Label(
            self.f_wearable,
            text="Tobii Glasses 3 Defaults:",
            fg="gray",
            bg="white",
        ).pack(side=tk.LEFT, padx=(0, 10))
        self._add_entry(self.f_wearable, "H-FOV (°):", self.fov_h)
        self._add_entry(self.f_wearable, "V-FOV (°):", self.fov_v)

        # Screen Frame (Physical dims)
        self.f_screen = tk.Frame(lf_geo, bg="white")
        # self.f_screen.pack(fill=tk.X) # Hidden by default
        self._add_entry(self.f_screen, "Screen W (cm):", self.screen_w)
        self._add_entry(self.f_screen, "Screen H (cm):", self.screen_h)
        self._add_entry(self.f_screen, "Dist (cm):", self.distance)

        # Resolution (Common)
        f_res = tk.Frame(lf_geo, bg="white")
        f_res.pack(fill=tk.X, pady=(5, 0))
        self._add_entry(f_res, "Video Res W:", self.res_w)
        self._add_entry(f_res, "Video Res H:", self.res_h)

        # 3. Filter Settings
        lf_filt = tk.LabelFrame(main, text="3. I-VT Settings (Tobii Pro Lab defaults)", padx=10, pady=10, bg="white")
        lf_filt.pack(fill=tk.X, pady=5)

        # Pre-processing
        f_pre = tk.Frame(lf_filt, bg="white")
        f_pre.pack(fill=tk.X, pady=2)
        tk.Checkbutton(f_pre, text="Gap Fill (Interp)", variable=self.gap_fill, bg="white").pack(side=tk.LEFT)
        tk.Label(f_pre, text="Max (ms):", bg="white").pack(side=tk.LEFT, padx=(5, 0))
        tk.Entry(f_pre, textvariable=self.max_gap_ms, width=5).pack(side=tk.LEFT)

        tk.Label(f_pre, text="| Noise Red:", bg="white").pack(side=tk.LEFT, padx=(15, 0))
        ttk.Combobox(
            f_pre,
            textvariable=self.noise_reduction,
            values=["None", "Moving Median", "Moving Average"],
            state="readonly",
            width=15,
        ).pack(side=tk.LEFT)
        tk.Label(f_pre, text="Win:", bg="white").pack(side=tk.LEFT)
        tk.Entry(f_pre, textvariable=self.noise_window, width=3).pack(side=tk.LEFT)

        # I-VT
        f_ivt = tk.Frame(lf_filt, bg="white")
        f_ivt.pack(fill=tk.X, pady=5)
        tk.Label(f_ivt, text="Velocity Threshold (°/s):", font=("Bold", 10), bg="white").pack(side=tk.LEFT)
        tk.Entry(f_ivt, textvariable=self.velocity_threshold, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(f_ivt, text="(30=Fixation, 100=Attention)", fg="gray", bg="white").pack(side=tk.LEFT)

        # Post-processing
        f_post = tk.Frame(lf_filt, bg="white")
        f_post.pack(fill=tk.X, pady=2)

        tk.Checkbutton(f_post, text="Merge Adjacent", variable=self.merge_adjacent, bg="white").pack(side=tk.LEFT)
        tk.Label(f_post, text="Max Time(ms):", bg="white").pack(side=tk.LEFT)
        tk.Entry(f_post, textvariable=self.merge_max_time, width=4).pack(side=tk.LEFT)
        tk.Label(f_post, text="Max Angle(°):", bg="white").pack(side=tk.LEFT)
        tk.Entry(f_post, textvariable=self.merge_max_angle, width=4).pack(side=tk.LEFT)

        tk.Checkbutton(
            f_post,
            text="Discard Short",
            variable=self.discard_short,
            bg="white",
        ).pack(side=tk.LEFT, padx=(15, 0))
        tk.Label(f_post, text="Min Dur(ms):", bg="white").pack(side=tk.LEFT)
        tk.Entry(f_post, textvariable=self.discard_min_dur, width=4).pack(side=tk.LEFT)

        # 4. Run
        self.btn_run = tk.Button(
            main,
            text="RUN I-VT FILTER",
            bg="#007ACC",
            fg="white",
            font=("Bold", 12),
            height=2,
            command=self.run_process,
        )
        self.btn_run.pack(fill=tk.X, pady=20)

        self.progress = ttk.Progressbar(main, mode="indeterminate")
        self.progress.pack(fill=tk.X)
        self.lbl_status = tk.Label(main, text="Ready.", bg="white")
        self.lbl_status.pack()

        self._update_geo_ui()

    def _update_geo_ui(self, *args):
        is_wearable = self.tracker_type.get() == "Wearable"
        if is_wearable:
            self.f_screen.pack_forget()
            self.f_wearable.pack(fill=tk.X, after=self.f_type)
        else:
            self.f_wearable.pack_forget()
            self.f_screen.pack(fill=tk.X, after=self.f_type)

    def _add_file_picker(self, parent, label, var, ft, save=False):
        f = tk.Frame(parent, bg="white")
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=label, width=20, anchor="w", bg="white").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        def cmd():
            if save:
                self._browse_save(var, ft)
            else:
                self._browse_open(var, ft)

        tk.Button(f, text="...", width=3, command=cmd).pack(side=tk.LEFT)

    def _add_entry(self, parent, label, var):
        tk.Label(parent, text=label, bg="white").pack(side=tk.LEFT, padx=(10, 2))
        tk.Entry(parent, textvariable=var, width=6).pack(side=tk.LEFT)

    def _browse_open(self, var, ft):
        f = filedialog.askopenfilename(filetypes=[("File", ft)])
        if f:
            var.set(f)
            if var == self.input_path:
                self._suggest_outputs(f)

    def _browse_save(self, var, ft):
        f = filedialog.asksaveasfilename(filetypes=[("File", ft)], defaultextension=".csv")
        if f:
            var.set(f)

    def _suggest_outputs(self, input_path):
        base = os.path.splitext(input_path)[0].replace("_MAPPED", "")
        self.output_fix_path.set(f"{base}_FIXATIONS.csv")
        self.output_data_path.set(f"{base}_FILTERED.csv")

    def run_process(self):
        inp = self.input_path.get()
        if not inp or not os.path.exists(inp):
            messagebox.showerror("Error", "Invalid input file.")
            return

        params = {
            "gap_fill": self.gap_fill.get(),
            "max_gap_ms": self.max_gap_ms.get(),
            "noise_reduction": self.noise_reduction.get(),
            "noise_window": self.noise_window.get(),
            "tracker_type": self.tracker_type.get(),
            "fov_h": self.fov_h.get(),
            "fov_v": self.fov_v.get(),
            "velocity_threshold": self.velocity_threshold.get(),
            "merge_adjacent": self.merge_adjacent.get(),
            "merge_max_time_ms": self.merge_max_time.get(),
            "merge_max_angle": self.merge_max_angle.get(),
            "discard_short": self.discard_short.get(),
            "discard_min_dur_ms": self.discard_min_dur.get(),
            "screen_w_cm": self.screen_w.get(),
            "screen_h_cm": self.screen_h.get(),
            "distance_cm": self.distance.get(),
            "res_w": self.res_w.get(),
            "res_h": self.res_h.get(),
        }

        self.btn_run.config(state="disabled")
        self.progress.start(10)

        def worker():
            try:
                self.logic.load_data(inp)
                count = self.logic.run_ivt(params, progress_callback=self._update_status)

                if self.output_fix_path.get():
                    self.logic.save_fixations(self.output_fix_path.get())
                if self.output_data_path.get():
                    self.logic.save_filtered_data(self.output_data_path.get())

                self.parent.after(0, lambda: self._on_success(count))
            except Exception as e:
                err_msg = str(e)
                self.parent.after(0, lambda: self._on_error(err_msg))

        threading.Thread(target=worker, daemon=True).start()

    def _update_status(self, msg):
        self.parent.after(0, lambda: self.lbl_status.config(text=msg))

    def _on_success(self, count):
        self.progress.stop()
        self.btn_run.config(state="normal")
        self.lbl_status.config(text="Done.")
        messagebox.showinfo("Success", f"I-VT Filter Complete.\nIdentified {count} fixations.")

    def _on_error(self, msg):
        self.progress.stop()
        self.btn_run.config(state="normal")
        self.lbl_status.config(text="Error.")
        messagebox.showerror("Error", msg)
