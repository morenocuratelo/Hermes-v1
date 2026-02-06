import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import torch
import cv2
import json
import gzip
import os
import threading
import sys
import random
import numpy as np
import requests
import csv
from ultralytics import YOLO # type: ignore

# --- RESEARCH PARAMETERS & HEURISTICS (CONSTANTS) ---
# Globally exposed for reproducibility and tuning.
# CONF_THRESHOLD: Conservative threshold to balance Precision and Recall.
# IOU_THRESHOLD: Threshold for Non-Maximum Suppression (NMS).
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7
RANDOM_SEED = 42
ULTRALYTICS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"

def set_determinism(seed=42):
    """
    Sets the seed to ensure scientific reproducibility of results.
    Locks CUDNN optimization heuristics that could introduce
    hardware non-determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinismo assoluto a scapito di leggera performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- REDIRECT PRINT TO GUI ---
class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        try:
            self.widget.configure(state="normal")
            self.widget.insert("end", str, (self.tag,))
            self.widget.see("end")
            self.widget.configure(state="disabled")
            self.widget.update_idletasks()
        except tk.TclError:
            pass

    def flush(self):
        pass

# --- MAIN VIEW CLASS ---
class YoloView:
    def __init__(self, parent, context):
        self.parent = parent    
        self.context = context 
        
        # Variabili Locali
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_name = tk.StringVar(value="yolo26x-pose.pt") 
        self.is_running = False
        self.tracker_type = tk.StringVar(value="botsort")
        self.conf_threshold = tk.DoubleVar(value=CONF_THRESHOLD)
        self.iou_threshold = tk.DoubleVar(value=IOU_THRESHOLD)
        self.match_threshold = tk.DoubleVar(value=0.7) #pronome
        self.track_buffer = tk.IntVar(value=30) #Buffer da capire
        
        # --- PARAMETRI AVANZATI TRACKER ---
        self.track_low_thresh = tk.DoubleVar(value=0.1)
        self.proximity_thresh = tk.DoubleVar(value=0.5)
        self.appearance_thresh = tk.DoubleVar(value=0.25)
        self.with_reid = tk.BooleanVar(value=False)
        
        # --- SYNC CONTEXT (LETTURA) ---
        if self.context.video_path:
            self.video_path.set(self.context.video_path)
            self._suggest_output_name(self.context.video_path)

        self._build_ui()
        self._check_hardware_from_context()

    def _build_ui(self):
        tk.Label(self.parent, text="1. Human Pose Estimation & Tracking", font=("Segoe UI", 18, "bold"), bg="white").pack(pady=(0, 20), anchor="w")

        # 1. Hardware Info
        self.lbl_hw = tk.Label(self.parent, text="Initializing...", bg="white", font=("Consolas", 10))
        self.lbl_hw.pack(pady=5, anchor="w")

        # 2. Selezione File
        lf_files = tk.LabelFrame(self.parent, text="Input & Output", padx=10, pady=10, bg="white")
        lf_files.pack(fill=tk.X, pady=5)
        
        self._add_picker(lf_files, "Video Input:", self.video_path, "*.mp4 *.avi *.mov")
        self._add_picker(lf_files, "Output JSON (.gz):", self.output_path, "*.json.gz", save=True) #capire altri formati

        # 3. Configurazione Modello
        lf_conf = tk.LabelFrame(self.parent, text="AI Model Configuration", padx=10, pady=10, bg="white")
        lf_conf.pack(fill=tk.X, pady=5)
        
        tk.Label(lf_conf, text="YOLO Model:", bg="white").pack(side=tk.LEFT)
        
        # LISTA MODELLI
        models = [
            "yolo26x-pose.pt", "yolo26l-pose.pt", "yolo26m-pose.pt", "yolo26s-pose.pt", "yolo26n-pose.pt",
        ]
        self.cb_model = ttk.Combobox(lf_conf, textvariable=self.model_name, values=models, state="readonly", width=25)
        self.cb_model.pack(side=tk.LEFT, padx=10)
        tk.Label(lf_conf, text="(Saved in: Project/Models)", fg="gray", bg="white").pack(side=tk.LEFT)

        # 3b. Configurazione Tracking
        lf_track = tk.LabelFrame(self.parent, text="Tracking Parameters", padx=10, pady=10, bg="white")
        lf_track.pack(fill=tk.X, pady=5)
        
        tk.Label(lf_track, text="Tracker:", bg="white").pack(side=tk.LEFT, padx=5)
        trackers = ["none", "botsort", "bytetrack", "deepocsort", "ocsort"]
        self.cb_tracker = ttk.Combobox(lf_track, textvariable=self.tracker_type, values=trackers, state="readonly", width=12)
        self.cb_tracker.pack(side=tk.LEFT, padx=5)
        
        tk.Label(lf_track, text="Confidence:", bg="white").pack(side=tk.LEFT, padx=(20, 5))
        tk.Scale(lf_track, from_=0.1, to=0.95, resolution=0.05, orient=tk.HORIZONTAL, variable=self.conf_threshold, bg="white", length=100).pack(side=tk.LEFT)

        tk.Label(lf_track, text="IoU:", bg="white").pack(side=tk.LEFT, padx=(5, 5))
        tk.Scale(lf_track, from_=0.1, to=0.95, resolution=0.05, orient=tk.HORIZONTAL, variable=self.iou_threshold, bg="white", length=100).pack(side=tk.LEFT)

        tk.Label(lf_track, text="Match Thresh:", bg="white").pack(side=tk.LEFT, padx=(10, 5))
        tk.Scale(lf_track, from_=0.1, to=0.95, resolution=0.05, orient=tk.HORIZONTAL, variable=self.match_threshold, bg="white", length=100).pack(side=tk.LEFT)

        tk.Label(lf_track, text="Buffer:", bg="white").pack(side=tk.LEFT, padx=(10, 5))
        tk.Scale(lf_track, from_=1, to=120, resolution=1, orient=tk.HORIZONTAL, variable=self.track_buffer, bg="white", length=100).pack(side=tk.LEFT)

        # Bottone Avanzate
        tk.Button(lf_track, text="⚙ Advanced", command=self.open_tracker_settings).pack(side=tk.LEFT, padx=10)

        # 4. Progress & Log
        self.progress = ttk.Progressbar(self.parent, orient=tk.HORIZONTAL, length=100, mode='determinate') 
        self.progress.pack(fill=tk.X, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(self.parent, height=12, state='disabled', font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        sys.stdout = TextRedirector(self.log_text, "stdout")
        sys.stderr = TextRedirector(self.log_text, "stderr")
        self.log_text.tag_config("stderr", foreground="red")

        # 5. Buttons
        self.btn_run = tk.Button(self.parent, text="START GPU ANALYSIS", bg="#007ACC", fg="white", font=("Bold", 12), height=2, command=self.start_thread)
        self.btn_run.pack(fill=tk.X, pady=10)

    def _check_hardware_from_context(self):
        if self.context.device == "cuda":
            self.lbl_hw.config(text=f"✅ ACCELERATION ACTIVE: {self.context.gpu_name}", fg="green")
        else:
            self.lbl_hw.config(text=f"⚠️ WARNING: No GPU detected. CPU Mode ({self.context.device}).", fg="orange")

    def _add_picker(self, p, lbl, var, ft, save=False):
        f = tk.Frame(p, bg="white"); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=lbl, width=15, anchor="w", bg="white").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        cmd = lambda: self.browse_save(var, ft) if save else self.browse_open(var, ft)
        tk.Button(f, text="...", width=3, command=cmd).pack(side=tk.LEFT)

    def _suggest_output_name(self, video_path):
        if not video_path: return
        base = os.path.splitext(video_path)[0]
        if self.context.paths["output"] and os.path.exists(self.context.paths["output"]):
            filename = os.path.basename(base) + "_yolo.json.gz"
            suggested = os.path.join(self.context.paths["output"], filename)
        else:
            suggested = base + "_yolo.json.gz"
        self.output_path.set(suggested)
        self.context.pose_data_path = suggested

    def browse_open(self, var, ft):
        f = filedialog.askopenfilename(filetypes=[("Video", ft)])
        if f: 
            var.set(f)
            self.context.update_video(f) 
            self._suggest_output_name(f)

    def browse_save(self, var, ft):
        f = filedialog.asksaveasfilename(filetypes=[("JSON GZ", ft)], defaultextension=".json.gz")
        if f: 
            var.set(f)
            self.context.pose_data_path = f

    def open_tracker_settings(self):
        """Apre una finestra per i parametri nascosti del tracker."""
        win = tk.Toplevel(self.parent)
        win.title("Advanced Tracker Settings")
        win.geometry("350x400")
        
        tk.Label(win, text="ByteTrack / BoT-SORT Settings", font=("Segoe UI", 10, "bold")).pack(pady=10)
        
        def add_scale(lbl, var, from_, to_, res):
            f = tk.Frame(win); f.pack(fill=tk.X, padx=15, pady=5)
            tk.Label(f, text=lbl).pack(anchor="w")
            tk.Scale(f, from_=from_, to=to_, resolution=res, orient=tk.HORIZONTAL, variable=var).pack(fill=tk.X)

        add_scale("Track Low Threshold (Low-confidence track recovery):", self.track_low_thresh, 0.01, 0.6, 0.01)
        add_scale("Proximity Threshold (BoT-SORT):", self.proximity_thresh, 0.1, 1.0, 0.05)
        add_scale("Appearance Threshold (BoT-SORT):", self.appearance_thresh, 0.1, 1.0, 0.05)
        
        f_chk = tk.Frame(win); f_chk.pack(fill=tk.X, padx=15, pady=15)
        tk.Checkbutton(f_chk, text="Enable Re-Identification (ReID)", variable=self.with_reid).pack(anchor="w")
        tk.Label(f_chk, text="(Requires automatic download of extra ReID weights)", fg="gray", font=("Arial", 8)).pack(anchor="w")
        
        tk.Button(win, text="Close", command=win.destroy, width=15).pack(pady=10)

    def start_thread(self):
        if self.is_running: return
        if not self.video_path.get() or not self.output_path.get():
            messagebox.showwarning("Missing Data", "Select video and output.")
            return
            
        self.is_running = True
        self.btn_run.config(state="disabled", text="INITIALIZING YOLO...")
        t = threading.Thread(target=self.run_yolo_process)
        t.start()

    def _download_model_manual(self, model_name, dest_path):
        print(f"📥 Downloading model: {model_name}...")
        # URL ufficiali Ultralytics Assets (es. v8.3.0)
        url = ULTRALYTICS_URL + model_name
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192): #capire 8192
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            perc = int((downloaded / total_size) * 50)
                            self.parent.after(0, lambda v=perc: self.progress.config(value=v))
            print("✅ Download complete.")
            return True
        except Exception as e:
            print(f"❌ Download error: {e}")
            if os.path.exists(dest_path): os.remove(dest_path)
            return False

    def _generate_tracker_config(self, tracker_name, reid_weights=None):
        filename = f"custom_{tracker_name}.yaml"
        conf = self.conf_threshold.get()
        match = self.match_threshold.get()
        buf = self.track_buffer.get()
        
        # Parametri Avanzati
        low_thresh = self.track_low_thresh.get()
        prox = self.proximity_thresh.get()
        app = self.appearance_thresh.get()
        reid = self.with_reid.get()
        
        lines = [
            f"tracker_type: {tracker_name}",
            f"track_high_thresh: {conf}",
            f"track_low_thresh: {low_thresh}",
            f"new_track_thresh: {conf}",
            f"track_buffer: {buf}",
            f"match_thresh: {match}"
        ]
        
        if tracker_name in ('botsort', 'bytetrack'):
            lines.append("fuse_score: True")

        if tracker_name == 'botsort':
            lines.append("gmc_method: sparseOptFlow")
            lines.append(f"proximity_thresh: {prox}")
            lines.append(f"appearance_thresh: {app}")
            
            if reid and reid_weights:
                lines.append("with_reid: True")
                lines.append(f"model: {reid_weights}")
            elif reid:
                # Fallback legacy (assumes file in CWD)
                lines.append("with_reid: True")
                lines.append("model: osnet_x0_25_msmt17.pt")
            else:
                lines.append("with_reid: False")
            
        with open(filename, 'w') as f:
            f.write("\n".join(lines))
        return filename

    # --- NUOVO METODO HELPER PER CSV (LONG FORMAT) ---
    def _export_to_csv_flat(self, json_gz_path):
        """
        Converts hierarchical JSON to 'flat' CSV (Long Format).
        Each row = One person in a frame. Strict for statistical analysis (Tidy Data).
        """
        try:
            print("Converting output to Flattened CSV...")
            csv_path = json_gz_path.replace(".json.gz", ".csv")
            
            # 1. Definizione Header (59 Colonne per modello Pose standard)
            # COCO Keypoints Order: 0:Nose, 1:L_Eye, 2:R_Eye, 3:L_Ear, 4:R_Ear...
            kp_names = [
                "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", 
                "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", 
                "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", 
                "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
            ]
            
            header = ["Frame", "Timestamp", "TrackID", "Conf", "Box_X1", "Box_Y1", "Box_X2", "Box_Y2"]
            for kp in kp_names:
                # Per ogni punto salviamo X, Y e Confidenza (C)
                header.extend([f"{kp}_X", f"{kp}_Y", f"{kp}_C"])

            # 2. Scrittura Stream
            with open(csv_path, mode='w', newline='') as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(header)
                
                with gzip.open(json_gz_path, 'rt', encoding='utf-8') as f_json:
                    for line in f_json:
                        frame = json.loads(line)
                        f_idx = frame['f_idx']
                        ts = frame['ts']
                        
                        for det in frame.get('det', []):
                            # Dati Base
                            row = [
                                f_idx, 
                                ts, 
                                det.get('track_id', -1), 
                                det.get('conf', 0)
                            ]
                            
                            # Box
                            b = det.get('box', {})
                            row.extend([b.get('x1',0), b.get('y1',0), b.get('x2',0), b.get('y2',0)])
                            
                            # Keypoints (Appiattimento 17 punti x 3 valori)
                            kps = det.get('keypoints', [])
                            # kps è una lista di liste [[x,y,c], [x,y,c]...]
                            
                            if not kps:
                                # Se manca lo scheletro, riempiamo di zeri per mantenere l'allineamento colonne
                                row.extend([0] * (17 * 3))
                            else:
                                for point in kps:
                                    # Aggiunge X, Y. Se manca C (confidence), mette 0
                                    row.extend(point) 
                                    if len(point) < 3: row.append(0) 
                            
                            writer.writerow(row)
                            
            print(f"✅ CSV Export complete: {csv_path}")
            return True
        except Exception as e:
            print(f"⚠️ CSV Export error: {e}")
            return False
    # --------------------------------------

    def run_yolo_process(self):
        video_file = self.video_path.get()
        out_file = self.output_path.get()
        model_name = self.model_name.get()
        
        conf_value = self.conf_threshold.get()
        iou_value = self.iou_threshold.get()
        
        try:
            print(f"--- Analysis Started ---")
            
            # 0. GESTIONE REID (Download & Path Assoluto)
            reid_path = None
            if self.with_reid.get() and self.tracker_type.get() == 'botsort':
                reid_name = "osnet_x0_25_msmt17.pt"
                reid_path = os.path.join(self.context.paths["models"], reid_name)
                
                if not os.path.exists(reid_path):
                    print(f"Missing ReID model, starting download: {reid_name}")
                    if not self._download_model_manual(reid_name, reid_path):
                        print("⚠️ ReID download failed. Disabling ReID.")
                        self.with_reid.set(False)
                        reid_path = None
                
                if reid_path: reid_path = reid_path.replace("\\", "/") # Fix YAML Windows paths

            tracker_config = self._generate_tracker_config(self.tracker_type.get(), reid_path)

            # Controllo esistenza file configurazione tracker
            if not os.path.exists(tracker_config):
                print(f"⚠️ WARNING: '{tracker_config}' not found in working directory.")
                print("   YOLO will use default parameters (may not be optimized for Human Pose).")
            else:
                print(f"✅ Tracker configuration generated: {tracker_config}")

            print(f"Reproducibility seed: {RANDOM_SEED}")
            set_determinism(RANDOM_SEED)

            # 1. GESTIONE MODELLO
            model_path = os.path.join(self.context.paths["models"], model_name)
            if not os.path.exists(model_path):
                print(f"Missing model, starting download: {model_name}")
                success = self._download_model_manual(model_name, model_path)
                if not success:
                    raise Exception("Unable to download model.")
            else:
                self.progress.config(value=50)

            # 2. CARICAMENTO
            print("Allocating YOLO weights to VRAM...")
            model = YOLO(model_path) 
            
            # 3. PREPARAZIONE METADATI
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise IOError(f"Unable to open video: {video_file}")
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            self.parent.after(0, lambda: self.progress.configure(maximum=total_frames + total_frames)) 
            
            print(f"Starting tracking {tracker_config} (Conf: {conf_value}, IoU: {iou_value})...")
            
            # ---------------------------------------------------------
            # [SECTION: COMPUTER VISION PIPELINE]
            # Questa sezione incapsula la logica di inferenza.
            # ---------------------------------------------------------
            # 1. GENERATORE STREAMING:
            #    L'argomento stream=True crea un iteratore lazy. I frame 
            #    vengono decodificati e passati alla GPU uno alla volta, 
            #    prevenendo overflow di memoria su video lunghi.
            #
            # 2. BYTETRACK ASSOCIATION (persist=True):
            #    A differenza di DeepSORT, ByteTrack associa detection ad 
            #    bassa confidenza se coerenti con la traiettoria del filtro 
            #    di Kalman. Questo riduce la frammentazione degli ID in 
            #    caso di occlusioni parziali.
            # ---------------------------------------------------------
            
            # Verifica se il tracker è attivo
            is_tracker_enabled = self.tracker_type.get() != "none"
            
            if is_tracker_enabled:
                results = model.track(
                    source=video_file,
                    stream=True,
                    persist=True,
                    tracker=tracker_config,
                    verbose=False,
                    conf=conf_value,
                    iou=iou_value,
                    device=0 if self.context.device == "cuda" else "cpu"
                )
            else:
                print("Starting analysis WITHOUT tracker (detection only)...")
                results = model.predict(
                    source=video_file,
                    stream=True,
                    verbose=False,
                    conf=conf_value,
                    iou=iou_value,
                    device=0 if self.context.device == "cuda" else "cpu"
                )
            
            with gzip.open(out_file, 'wt', encoding='utf-8') as f:
                for i, result in enumerate(results):
                    # -----------------------------------------------------
                    # [DATA EXTRACTION LAYER]
                    # Estrazione diretta dai tensori CUDA per evitare overhead.
                    # -----------------------------------------------------
                    # result.boxes.xywh: Coordinate normalizzate centro-dimensione
                    # result.boxes.id: Identificativo univoco tracciamento
                    # result.keypoints: Coordinate scheletriche (Pose estimation)
                    
                    # Spostamento tensori da VRAM a RAM
                    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else np.array([])  # type: ignore
                    ids = result.boxes.id.cpu().numpy() if result.boxes and result.boxes.id is not None else np.array([])  # type: ignore
                    confs = result.boxes.conf.cpu().numpy() if result.boxes else np.array([])  # type: ignore
                    
                    # Keypoints: [N, 17, 3] -> (x, y, visibility)
                    keypoints = result.keypoints.data.cpu().numpy() if result.keypoints else np.array([])  # type: ignore

                    det_list = []
                    # Iterazione sulle detection del singolo frame
                    for j in range(len(boxes)):
                        track_id = int(ids[j]) if len(ids) > 0 else -1
                        
                        # Format box for hermes_entity (x1, y1, x2, y2)
                        b = boxes[j].tolist()
                        
                        # Serializzazione ottimizzata
                        det_data = {
                            "track_id": track_id,
                            "box": {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]},
                            "conf": float(confs[j]),
                            "keypoints": keypoints[j].tolist() if len(keypoints) > 0 else []
                        }
                        det_list.append(det_data)

                    # Struttura dati finale per il frame
                    frame_data = {
                        "f_idx": i,
                        "ts": round(i / fps, 4) if fps > 0 else 0, # Timestamp assoluto 4 cifre decimali
                        "det": det_list
                    }
                    
                    # Scrittura riga JSONL (line-delimited)
                    f.write(json.dumps(frame_data) + "\n")
                    
                    # -----------------------------------------------------
                    # UI UPDATE (Non-Blocking)
                    # -----------------------------------------------------
                    if i % 10 == 0:
                        current_val = total_frames + i 
                        self.parent.after(0, lambda v=current_val: self.progress.config(value=v))
                        if i % 100 == 0:
                            print(f"Processed Frame: {i}/{total_frames} | Tracked objects: {len(det_list)}")

            print(f"✅ YOLO analysis complete. JSON output saved.")
            
            # Avvia conversione Matlab immediata
            self._export_to_csv_flat(out_file)
            
            self.context.pose_data_path = out_file
            messagebox.showinfo("Finished", "Analysis complete.")
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR: {str(e)}")
            import traceback
            traceback.print_exc() # Stampa stack trace per debug profondo
            messagebox.showerror("Error", f"Error during analysis:\n{str(e)}")
            
        finally:
            self.is_running = False
            self.parent.after(0, self._reset_btn)

    def _reset_btn(self):
        self.btn_run.config(state="normal", text="START GPU ANALYSIS")
        self.progress.config(value=0)
