import csv
import datetime
import gzip
import json
import os
import queue
import random
import sys
import threading
import tkinter as tk

# Logic Imports
import traceback
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
import numpy as np
import requests  # type: ignore
import torch
from ultralytics import YOLO  # type: ignore

# --- RESEARCH PARAMETERS & HEURISTICS (CONSTANTS) ---
# Globally exposed for reproducibility and tuning. Here are
# initialised, but can be adjusted by the user via the UI sliders.
# CONF_THRESHOLD: Conservative threshold to balance Precision and Recall.
# IOU_THRESHOLD: Threshold for Non-Maximum Suppression (NMS). Older versions of YOLO.
# MATCH_THRESHOLD: Specific for tracking association (e.g.,
# BoT-SORT), determines how strictly detections are matched
# to existing tracks.
CONF_THRESHOLD = 0.6  # Manteniamo alto per purezza, come suggerito nel paper BoT-SORT originale
IOU_THRESHOLD = 1.0  # CRITICO: YOLO26 è NMS-Free. Qualsiasi post-processing NMS è ridondante.
MATCH_THRESHOLD = 0.8  # Standard BoT-SORT
RANDOM_SEED = 42
ULTRALYTICS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
TRACKERS_CONFIG_DIR = os.path.join("Configs", "Trackers")
MIN_MODEL_SIZE_BYTES = 1024 * 1024  # 1 MB — soglia minima per considerare un file .pt valido

# Sorgenti per i modelli ReID.
# I file *_ready.pt sono versioni convertite per Ultralytics (via fix_weights.py).
# Se la sorgente diretta non funziona, si scarica il .pth originale da torchreid e si converte inline.
REID_SOURCES = {
    "resnet50_msmt17_ready.pt": {
        # resnet50 trained on MSMT17 (combineall) — da torchreid MODEL_ZOO.md
        # https://drive.google.com/file/d/1yiBteqgIZoOeywE8AhGmEQl7FTVwrQmf/view
        "pth_gdrive_id": "1yiBteqgIZoOeywE8AhGmEQl7FTVwrQmf",
        "pth_name": "resnet50_msmt17.pth",
    },
    "osnet_ain_x1_0_ready.pt": {
        # osnet_ain_x1_0 trained on MSMT17 (cosine, combineall) — da torchreid MODEL_ZOO.md
        # https://drive.google.com/file/d/1hypJvq8G04SOby6jvF337GEkg5K_bmCw/view
        "pth_gdrive_id": "1hypJvq8G04SOby6jvF337GEkg5K_bmCw",
        "pth_name": "osnet_ain_x1_0_msmt17.pth",
    },
}


def _is_valid_model_file(path: str) -> bool:
    """
    Ritorna True solo se il file:
    - esiste sul disco
    - pesa almeno MIN_MODEL_SIZE_BYTES (1 MB)
    - NON è un puntatore Git LFS (header 'version https://git-lfs...')

    I puntatori LFS sono file di testo di ~133 byte che vengono scambiati per modelli
    validi da os.path.exists(), causando l'errore 'invalid load key' di pickle.
    """
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < MIN_MODEL_SIZE_BYTES:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(128).decode("utf-8", errors="ignore")
        if "git-lfs.github.com" in head:
            return False
    except Exception:
        return False
    return True


# Sebbene questi valori siano stati scelti come default basati sulla letteratura (COCO benchmarks),
# il nostro strumento espone esplicitamente questi parametri all'utente tramite GUI,
# permettendo una regolazione fine (fine-tuning) specifica per le condizioni di illuminazione e densità
# della scena analizzata, superando i limiti di un approccio 'one-size-fits-all'.


class ResultsWriter(threading.Thread):
    """
    Gestisce la scrittura su disco in un thread separato per non bloccare
    il loop di inferenza della GPU. Usa una coda per bufferizzare i risultati.
    """

    def __init__(self, path, compress_level=3):
        super().__init__(daemon=True)
        self.path = path
        self.compress_level = compress_level
        self.queue: queue.Queue = queue.Queue(maxsize=256)  # Buffer di ~250 frame
        self.queue: queue.Queue = queue.Queue(maxsize=256)  # type: ignore # Buffer di ~250 frame
        self.stop_event = threading.Event()
        self.error = None

    def put(self, item):
        """Aggiunge un item alla coda. Blocca se la coda è piena (backpressure)."""
        # FIX: Loop con timeout per evitare deadlock se il writer crasha o viene stoppato
        while not self.stop_event.is_set():
            if self.error:
                raise self.error
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def stop(self):
        """Segnala al thread di finire di scrivere e chiudere."""
        self.stop_event.set()
        self.join()
        if self.error:
            raise self.error

    def run(self):
        try:
            # compresslevel=3 è un ottimo compromesso velocità/compressione per dati real-time
            with gzip.open(self.path, "wt", encoding="utf-8", compresslevel=self.compress_level) as f:
                while not self.stop_event.is_set() or not self.queue.empty():
                    try:
                        # Timeout breve per controllare stop_event regolarmente
                        data = self.queue.get(timeout=0.1)
                        f.write(json.dumps(data) + "\n")
                        self.queue.task_done()
                    except queue.Empty:
                        continue
        except Exception as e:
            self.error = e


# ==========================================================================================
# --- LOGIC LAYER ---
# ==========================================================================================
class PoseEstimatorLogic:
    """
    Encapsulates all computational logic for Human Pose Estimation and Tracking (with trackers).
    """

    def __init__(self):
        pass

    def set_determinism(self, seed=42):
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

    def download_model(self, model_name, dest_path, on_progress=None, on_log=None):
        if on_log:
            on_log(f"📥 Downloading model: {model_name}...")

        # I modelli ReID non sono su Ultralytics GitHub Releases.
        # Si scarica il .pth originale da torchreid e si converte inline al formato Ultralytics.
        reid_info = REID_SOURCES.get(model_name)
        if reid_info:
            return self._download_and_convert_reid(reid_info, dest_path, on_log)

        # Modelli YOLO standard: scaricati da Ultralytics GitHub Releases.
        url = ULTRALYTICS_URL + model_name
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with open(dest_path, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and on_progress:
                            on_progress(downloaded, total_size, stage="download")
            if on_log:
                on_log("✅ Download complete.")
            return True
        except Exception as e:
            if on_log:
                on_log(f"❌ Download error: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False

    def _gdrive_download(self, file_id, dest_path, on_log=None):
        """
        Scarica un singolo file da Google Drive tramite gdown o urllib diretto.
        Ritorna True se il file scaricato è valido (≥ 1 MB, non puntatore LFS).
        """
        # Tentativo 1: gdown
        try:
            import gdown  # type: ignore

            url = f"https://drive.google.com/uc?id={file_id}"
            if on_log:
                on_log(f"   gdown: id={file_id}...")
            out = gdown.download(url, dest_path, quiet=False, fuzzy=True)
            if out and _is_valid_model_file(dest_path):
                return True
            if on_log:
                file_sz = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
                on_log(f"   gdown scaricato ma file non valido ({file_sz} byte)")
        except Exception as e:
            if on_log:
                on_log(f"   gdown fallito ({e}), provo urllib diretto...")

        # Tentativo 2: urllib diretto con confirm=t
        try:
            import shutil
            from urllib.request import Request, urlopen

            direct_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            if on_log:
                on_log(f"   urllib diretto: {direct_url[:60]}...")
            req = Request(direct_url, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310
            with urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out_f:  # noqa: S310
                shutil.copyfileobj(resp, out_f)
            if _is_valid_model_file(dest_path):
                return True
            raise RuntimeError(f"File non valido dopo download ({os.path.getsize(dest_path)} byte).")
        except Exception as e:
            if on_log:
                on_log(f"   urllib fallito: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False

    def _download_and_convert_reid(self, reid_info, dest_ready_path, on_log=None):
        """
        Scarica il file .pth originale torchreid da Google Drive e lo converte
        inline al formato Ultralytics (*_ready.pt), applicando la stessa logica
        di Models/fix_weights.py.
        """
        pth_gdrive_id = reid_info["pth_gdrive_id"]
        pth_name = reid_info["pth_name"]
        pth_path = dest_ready_path.replace("_ready.pt", ".pth").replace(".pt", ".pth")
        # Usa come cartella temporanea la stessa cartella di destinazione
        pth_path = os.path.join(os.path.dirname(dest_ready_path), pth_name)

        if on_log:
            on_log(f"   Scarico modello originale torchreid: {pth_name}")

        if not _is_valid_model_file(pth_path):
            ok = self._gdrive_download(pth_gdrive_id, pth_path, on_log)
            if not ok:
                if on_log:
                    on_log(f"❌ Impossibile scaricare {pth_name} da Google Drive.")
                return False

        # Conversione inline (logica identica a Models/fix_weights.py)
        if on_log:
            on_log(f"   Conversione {pth_name} → {os.path.basename(dest_ready_path)}...")
        try:
            raw = torch.load(pth_path, map_location="cpu", weights_only=True)
            state_dict = raw
            if isinstance(raw, dict):
                if "state_dict" in raw:
                    state_dict = raw["state_dict"]
                elif "model" in raw and isinstance(raw["model"], dict):
                    state_dict = raw["model"]
            clean = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            torch.save({"model": clean}, dest_ready_path)
            if on_log:
                on_log(f"✅ Modello ReID pronto: {os.path.basename(dest_ready_path)}")
            return True
        except Exception as e:
            if on_log:
                on_log(f"❌ Conversione fallita: {e}")
            if os.path.exists(dest_ready_path):
                os.remove(dest_ready_path)
            return False

    # Trackers configurabili tramite YAML. Questa funzione genera
    # dinamicamente un file di configurazione per il tracker scelto,
    # basandosi sui parametri forniti dall'utente attraverso
    # l'interfaccia. Supporta i tracker più comuni (BoT-SORT,
    # ByteTrack) e consente di abilitare/disabilitare funzionalità
    # avanzate come ReID, oltre a personalizzare soglie critiche
    # per l'associazione e la gestione dei track.
    def generate_tracker_config(self, params, filename="custom_tracker.yaml", config_dir=TRACKERS_CONFIG_DIR):
        lines = [
            f"tracker_type: {params.get('tracker_type', 'botsort')}",
            f"track_high_thresh: {params.get('conf', 0.5)}",
            f"track_low_thresh: {params.get('low_thresh', 0.1)}",
            f"new_track_thresh: {params.get('new_track_thresh', 0.7)}",
            f"track_buffer: {params.get('buffer', 30)}",
            f"match_thresh: {params.get('match', 0.8)}",
        ]

        tracker_name = params.get("tracker_type", "botsort")
        if tracker_name in ("botsort", "bytetrack"):
            lines.append("fuse_score: True")

        if tracker_name == "botsort":
            lines.append("gmc_method: sparseOptFlow")
            lines.append(f"proximity_thresh: {params.get('prox', 0.5)}")
            lines.append(f"appearance_thresh: {params.get('app', 0.25)}")

            reid_weights = params.get("reid_weights")
            if params.get("with_reid", False) and reid_weights:
                lines.append("with_reid: True")
                lines.append(f"model: '{reid_weights}'")
            elif params.get("with_reid", False):
                lines.append("with_reid: True")
                lines.append("model: resnet50_msmt17_ready.pt")
            else:
                lines.append("with_reid: False")

        tracker_filename = os.path.basename(filename)
        tracker_path = os.path.join(config_dir, tracker_filename)
        os.makedirs(config_dir, exist_ok=True)

        with open(tracker_path, "w") as f:
            f.write("\n".join(lines))
        return tracker_path

    # La funzione export_to_csv_flat converte il file JSON compresso
    # generato dall'analisi YOLO in un formato CSV "piatto"
    # (flattened), dove ogni riga rappresenta una singola rilevazione
    # di persona in un frame, con colonne per frame index, timestamp,
    # track ID, confidenza, coordinate della bounding box e keypoints
    # (x, y, conf) per ciascuno dei 17 punti chiave standard. Questo
    # formato è più facilmente importabile in software di analisi
    # dati come Excel o Python (pandas) per ulteriori elaborazioni
    # o visualizzazioni.
    def export_to_csv_flat(self, json_gz_path, on_log=None):
        try:
            if on_log:
                on_log("Converting output to Flattened CSV...")
            csv_path = json_gz_path.replace(".json.gz", ".csv")

            kp_names = [
                "Nose",
                "L_Eye",
                "R_Eye",
                "L_Ear",
                "R_Ear",
                "L_Shoulder",
                "R_Shoulder",
                "L_Elbow",
                "R_Elbow",
                "L_Wrist",
                "R_Wrist",
                "L_Hip",
                "R_Hip",
                "L_Knee",
                "R_Knee",
                "L_Ankle",
                "R_Ankle",
            ]

            header = ["Frame", "Timestamp", "TrackID", "Conf", "Box_X1", "Box_Y1", "Box_X2", "Box_Y2"]
            for kp in kp_names:
                header.extend([f"{kp}_X", f"{kp}_Y", f"{kp}_C"])

            with open(csv_path, mode="w", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(header)

                with gzip.open(json_gz_path, "rt", encoding="utf-8") as f_json:
                    for line in f_json:
                        frame = json.loads(line)
                        f_idx = frame["f_idx"]
                        ts = frame["ts"]

                        for det in frame.get("det", []):
                            row = [f_idx, ts, det.get("track_id", -1), det.get("conf", 0)]
                            b = det.get("box", {})
                            row.extend([b.get("x1", 0), b.get("y1", 0), b.get("x2", 0), b.get("y2", 0)])
                            kps = det.get("keypoints", [])
                            if not kps:
                                row.extend([0] * (17 * 3))
                            else:
                                for point in kps:
                                    row.extend(point)
                                    if len(point) < 3:
                                        row.append(0)
                            writer.writerow(row)

            if on_log:
                on_log(f"✅ CSV Export complete: {csv_path}")
            return True
        except Exception as e:
            if on_log:
                on_log(f"⚠️ CSV Export error: {e}")
            return False

    # Run_analysis function is the core method that orchestrates the
    # entire process of human pose estimation and tracking.
    # It performs the following steps:
    def run_analysis(self, config, on_progress=None, on_log=None, stop_event=None):
        """
        Main execution method.
        config: dict containing paths, model names, and tracker parameters.
        """
        video_file = config["video_path"]
        out_file = config["output_path"]
        model_name = config["model_name"]
        tracker_params = config["tracker_params"]
        models_dir = config["models_dir"]
        device = config["device"]

        if on_log:
            on_log("--- Analysis Started ---")

        # 1. ReID Mode
        # In Ultralytics 8.4.x, BoT-SORT usa il modello YOLO scelto dall'utente anche come ReID encoder.
        # ReID(model) → load_checkpoint(model) → ckpt["model"].float().eval()
        # Il modello YOLO è già validato al punto 4; passarne il path evita download extra e garantisce
        # coerenza con l'architettura scelta (es. yolo26x-pose.pt → backbone X come feature extractor).
        if tracker_params.get("with_reid") and tracker_params.get("tracker_type") == "botsort":
            reid_model_path = os.path.join(models_dir, model_name)
            if on_log:
                on_log(f"ℹ️ ReID mode: using '{model_name}' as appearance encoder")
            tracker_params["reid_weights"] = reid_model_path

        # 2. Generate Tracker Config
        tracker_config_file = self.generate_tracker_config(
            tracker_params,
            f"custom_{tracker_params['tracker_type']}.yaml",
        )
        if on_log:
            on_log(f"✅ Tracker configuration generated: {tracker_config_file}")

        # 3. Seed
        if on_log:
            on_log(f"Reproducibility seed: {RANDOM_SEED}")
        self.set_determinism(RANDOM_SEED)

        # 4. YOLO Model Check
        # Stessa logica del ReID: _is_valid_model_file rileva anche puntatori LFS
        model_path = os.path.join(models_dir, model_name)
        if not _is_valid_model_file(model_path):
            if on_log:
                on_log(f"Missing or invalid model (LFS pointer?): {model_name}, starting download...")
            if not self.download_model(model_name, model_path, on_progress, on_log):
                raise Exception("Unable to download model.")

        # 5. Load Model
        if on_log:
            on_log("Allocating YOLO weights to VRAM...")
        model = YOLO(model_path)

        # 6. Video Metadata
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise OSError(f"Unable to open video: {video_file}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if on_log:
            on_log(f"Starting tracking (Conf: {tracker_params['conf']}, IoU: {tracker_params['iou']})...")

        # 7. Inference Loop
        is_tracker_enabled = tracker_params["tracker_type"] != "none"

        yolo_args = {
            "source": video_file,
            "stream": True,  # Frame-by-frame streaming per gestione memoria e progress bar
            "verbose": False,  # Deactivate logging from YOLO to keep our custom log clean
            "conf": tracker_params["conf"],  # Confidence threshold for detections
            "iou": tracker_params["iou"],  # IoU threshold for NMS (ignored in YOLO26, but required for older versions)
            "device": 0 if device == "cuda" else "cpu",  # NVIDIA GPU se disponibile, altrimenti CPU
        }

        if is_tracker_enabled:
            yolo_args["persist"] = True
            yolo_args["tracker"] = tracker_config_file
            results = model.track(**yolo_args)
        else:
            if on_log:
                on_log("Starting analysis WITHOUT tracker (detection only)...")
            results = model.predict(**yolo_args)

        # Avvia il thread di scrittura asincrona
        writer = ResultsWriter(out_file)
        writer.start()

        try:
            for i, result in enumerate(results):
                if stop_event and stop_event.is_set():
                    if on_log:
                        on_log("🛑 Analysis interrupted by user.")
                    break

                # Normalize to numpy (handles CPU/GPU and Tensor/ndarray differences)
                result = result.cpu().numpy()

                # Bounding box coordinates in [x1, y1, x2, y2]
                # format (two-points coordinates)
                boxes = result.boxes.xyxy if result.boxes else np.array([])
                # Track IDs assegnati dal tracker (se abilitato),
                # altrimenti -1 per indicare nessun tracking
                ids = result.boxes.id if result.boxes and result.boxes.id is not None else np.array([])
                # Confidence score per ogni rilevazione, utile per
                # filtrare ulteriormente o analizzare la qualità
                # delle rilevazioni
                confs = result.boxes.conf if result.boxes else np.array([])
                # Coordinate dei keypoints in formato [x, y, conf]
                # per ciascuno dei 17 punti chiave standard
                # (se disponibili), altrimenti array vuoto
                keypoints = result.keypoints.data if result.keypoints else np.array([])

                # Lista di rilevazioni per il frame corrente,
                # dove ogni rilevazione è un dizionario contenente
                # track_id, box, conf e keypoints. Questa struttura
                # consente di serializzare facilmente i dati in JSON
                # e mantenere una chiara organizzazione delle
                # informazioni per ogni persona rilevata in ciascun
                # frame.
                det_list = []
                for j in range(len(boxes)):
                    track_id = int(ids[j]) if len(ids) > 0 else -1
                    b = boxes[j].tolist()
                    det_data = {
                        "track_id": track_id,
                        "box": {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]},
                        "conf": float(confs[j]),
                        "keypoints": keypoints[j].tolist() if len(keypoints) > 0 else [],
                    }
                    det_list.append(det_data)

                frame_data = {"f_idx": i, "ts": round(i / fps, 4) if fps > 0 else 0, "det": det_list}

                # Invia al thread di scrittura (non bloccante a meno che la coda sia piena)
                writer.put(frame_data)

                if on_progress and i % 10 == 0:
                    on_progress(i, total_frames, stage="inference")

                if i % 100 == 0 and on_log:
                    on_log(f"Processed Frame: {i}/{total_frames} | Tracked objects: {len(det_list)}")

        except Exception as e:
            # Se c'è un errore nel loop principale, assicuriamoci di fermare il writer
            writer.stop()
            raise e

        # Chiude il writer e attende che finisca di scrivere gli ultimi dati in coda
        writer.stop()

        if on_log:
            on_log("✅ YOLO analysis complete. JSON output saved.")

        # Save Metadata (Parameters)
        try:
            meta_path = out_file.replace(".json.gz", "_meta.json")
            metadata = {
                "timestamp": datetime.datetime.now().isoformat(),
                "video_path": video_file,
                "model_name": model_name,
                "tracker_params": tracker_params,
                "device": device,
            }
            with open(meta_path, "w", encoding="utf-8") as f_meta:
                json.dump(metadata, f_meta, indent=4)
            if on_log:
                on_log(f"📄 Parameters saved to: {os.path.basename(meta_path)}")
        except Exception as e:
            if on_log:
                on_log(f"⚠️ Error saving metadata: {e}")

        # 8. CSV Export
        self.export_to_csv_flat(out_file, on_log)

        return True


# --- REDIRECT PRINT TO GUI ---
class TextRedirector:
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.after(0, self._write_safe, str)

    def _write_safe(self, str):
        try:
            self.widget.configure(state="normal")
            self.widget.insert("end", str, (self.tag,))
            self.widget.see("end")
            self.widget.configure(state="disabled")
        except tk.TclError:
            pass

    def flush(self):
        pass


# --- PRESENTATION LAYER ---
# --- MAIN VIEW CLASS ---
class YoloView:
    def __init__(self, parent, context):
        self.parent = parent
        self.context = context
        self.logic = PoseEstimatorLogic()

        # Variabili Locali
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_name = tk.StringVar(value="yolo26x-pose.pt")
        self.is_running = False
        self.tracker_type = tk.StringVar(value="botsort")
        self.conf_threshold = tk.DoubleVar(value=CONF_THRESHOLD)
        self.iou_threshold = tk.DoubleVar(value=IOU_THRESHOLD)
        self.match_threshold = tk.DoubleVar(value=MATCH_THRESHOLD)
        self.new_track_threshold = tk.DoubleVar(value=0.7)
        # Numero di frame per cui mantenere un ID attivo
        # senza nuove detection (tracking "invisibile")
        self.track_buffer = tk.IntVar(value=30)

        self.stop_event = threading.Event()
        # Bind cleanup quando la vista viene distrutta (es. cambio modulo)
        self.parent.bind("<Destroy>", self._on_destroy, add="+")

        # --- PARAMETRI AVANZATI TRACKER ---
        # Abbiamo adottato un approccio Two-Stage Association
        # (ByteTrack strategy). La soglia bassa di 0.1 permette
        # di mitigare l'occlusione temporanea (occlusion
        # robustness), recuperando rilevazioni a bassa confidenza
        # che sono spazialmente coerenti con le previsioni del
        # filtro di Kalman, riducendo drasticamente i falsi
        # negativi durante incroci complessi.
        self.track_low_thresh = tk.DoubleVar(value=0.1)
        # Il valore di 0.5 per la soglia di prossimità in BoT-SORT
        # è stato scelto per bilanciare efficacemente la capacità
        # del tracker di mantenere l'identità degli individui
        # durante occlusioni parziali, senza essere troppo
        # permissivo da causare errori di associazione (ID switch)
        # in scenari affollati. Questo parametro, combinato con la
        # soglia di apparenza, consente a BoT-SORT di distinguere
        # tra individui vicini ma distinti, migliorando la
        # robustezza complessiva del tracking.
        self.proximity_thresh = tk.DoubleVar(value=0.5)
        # La soglia di apparenza di 0.25 è stata scelta per
        # bilanciare la sensibilità del tracker nel riconoscere
        # caratteristiche distintive degli individui, riducendo
        # i falsi positivi senza compromettere la capacità di
        # mantenere l'identità durante occlusioni parziali.
        self.appearance_thresh = tk.DoubleVar(value=0.25)
        self.with_reid = tk.BooleanVar(value=True)
        self.reid_model_name = tk.StringVar(value="resnet50_msmt17_ready.pt")
        # "I valori sono stati scelti empiricamente basandosi
        # sulle configurazioni di default del paper originale
        # di BoT-SORT [Aharon et al., 2022], che hanno dimostrato
        # robustezza su benchmark standard come MOT17 e MOT20."

        # --- SYNC CONTEXT (LETTURA) ---
        if self.context.video_path:
            self.video_path.set(self.context.video_path)
            self._suggest_output_name(self.context.video_path)

        self._build_ui()
        self._check_hardware_from_context()

    def _on_destroy(self, event):
        if event.widget == self.parent:
            self.stop_event.set()

    def _build_ui(self):
        tk.Label(
            self.parent,
            text="1. Human Pose Estimation & Tracking",
            font=("Segoe UI", 18, "bold"),
            bg="white",
        ).pack(pady=(0, 20), anchor="w")

        # 1. Hardware Info
        self.lbl_hw = tk.Label(self.parent, text="Initializing...", bg="white", font=("Consolas", 10))
        self.lbl_hw.pack(pady=5, anchor="w")

        # 2. Selezione File
        lf_files = tk.LabelFrame(self.parent, text="Input & Output", padx=10, pady=10, bg="white")
        lf_files.pack(fill=tk.X, pady=5)

        self._add_picker(lf_files, "Video Input:", self.video_path, "*.mp4 *.avi *.mov")
        self._add_picker(lf_files, "Output JSON (.gz):", self.output_path, "*.json.gz", save=True)

        # 3. Configurazione Modello
        lf_conf = tk.LabelFrame(self.parent, text="AI Model Configuration", padx=10, pady=10, bg="white")
        lf_conf.pack(fill=tk.X, pady=5)

        tk.Label(lf_conf, text="YOLO Model:", bg="white").pack(side=tk.LEFT)

        # LISTA MODELLI
        models = [
            "yolo26x-pose.pt",
            "yolo26l-pose.pt",
            "yolo26m-pose.pt",
            "yolo26s-pose.pt",
            "yolo26n-pose.pt",
        ]
        self.cb_model = ttk.Combobox(lf_conf, textvariable=self.model_name, values=models, state="readonly", width=25)
        self.cb_model.pack(side=tk.LEFT, padx=10)
        tk.Label(lf_conf, text="(Saved in: Project/Models)", fg="gray", bg="white").pack(side=tk.LEFT)

        # ReID Model Selector
        tk.Label(lf_conf, text="ReID Model:", bg="white").pack(side=tk.LEFT, padx=(15, 0))

        reid_defaults = ["osnet_x0_25_msmt17.pt", "osnet_ain_x1_0_ready.pt", "resnet50_msmt17_ready.pt"]
        found_reid = []
        if self.context and self.context.paths.get("models") and os.path.exists(self.context.paths["models"]):
            found_reid = [
                f for f in os.listdir(self.context.paths["models"]) if f.endswith(".pt") and "yolo" not in f.lower()
            ]

        reid_values = sorted(list(set(reid_defaults + found_reid)))
        self.cb_reid = ttk.Combobox(lf_conf, textvariable=self.reid_model_name, values=reid_values, width=25)
        self.cb_reid.pack(side=tk.LEFT, padx=5)
        tk.Label(lf_conf, text="(Saved in: Project/Models)", fg="gray", bg="white").pack(side=tk.LEFT, padx=5)

        # 3b. Configurazione Tracking
        lf_track = tk.LabelFrame(self.parent, text="Tracking Parameters", padx=10, pady=10, bg="white")
        lf_track.pack(fill=tk.X, pady=5)

        tk.Label(lf_track, text="Tracker:", bg="white").pack(side=tk.LEFT, padx=5)
        trackers = ["none", "botsort", "bytetrack", "deepocsort", "ocsort"]
        self.cb_tracker = ttk.Combobox(
            lf_track,
            textvariable=self.tracker_type,
            values=trackers,
            state="readonly",
            width=12,
        )
        self.cb_tracker.pack(side=tk.LEFT, padx=5)

        tk.Label(lf_track, text="Confidence:", bg="white").pack(side=tk.LEFT, padx=(20, 5))
        tk.Scale(
            lf_track,
            from_=0.1,
            to=0.95,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.conf_threshold,
            bg="white",
            length=100,
        ).pack(side=tk.LEFT)

        tk.Label(lf_track, text="IoU:", bg="white").pack(side=tk.LEFT, padx=(5, 5))
        tk.Scale(
            lf_track,
            from_=0.1,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.iou_threshold,
            bg="white",
            length=100,
        ).pack(side=tk.LEFT)

        tk.Label(lf_track, text="Match Thresh:", bg="white").pack(side=tk.LEFT, padx=(10, 5))
        tk.Scale(
            lf_track,
            from_=0.1,
            to=0.95,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.match_threshold,
            bg="white",
            length=100,
        ).pack(side=tk.LEFT)

        tk.Label(lf_track, text="Buffer:", bg="white").pack(side=tk.LEFT, padx=(10, 5))
        tk.Scale(
            lf_track,
            from_=1,
            to=120,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.track_buffer,
            bg="white",
            length=100,
        ).pack(side=tk.LEFT)

        # Bottone Avanzate
        tk.Button(lf_track, text="⚙ Advanced", command=self.open_tracker_settings).pack(side=tk.LEFT, padx=10)

        # 4. Progress & Log
        self.progress = ttk.Progressbar(self.parent, orient=tk.HORIZONTAL, length=100, mode="determinate")
        self.progress.pack(fill=tk.X, pady=10)

        self.log_text = scrolledtext.ScrolledText(self.parent, height=12, state="disabled", font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        sys.stdout = TextRedirector(self.log_text, "stdout")  # type: ignore
        sys.stderr = TextRedirector(self.log_text, "stderr")  # type: ignore
        self.log_text.tag_config("stderr", foreground="red")

        # 5. Buttons
        self.btn_run = tk.Button(
            self.parent,
            text="START GPU ANALYSIS",
            bg="#007ACC",
            fg="white",
            font=("Bold", 12),
            height=2,
            command=self.start_thread,
        )
        self.btn_run.pack(fill=tk.X, pady=10)

    def _check_hardware_from_context(self):
        if self.context.device == "cuda":
            self.lbl_hw.config(text=f"✅ ACCELERATION ACTIVE: {self.context.gpu_name}", fg="green")
        else:
            self.lbl_hw.config(text=f"⚠️ WARNING: No GPU detected. CPU Mode ({self.context.device}).", fg="orange")

    def _add_picker(self, p, lbl, var, ft, save=False):
        f = tk.Frame(p, bg="white")
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=lbl, width=15, anchor="w", bg="white").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        def cmd():
            if save:
                self.browse_save(var, ft)
            else:
                self.browse_open(var, ft)

        tk.Button(f, text="...", width=3, command=cmd).pack(side=tk.LEFT)

    def _suggest_output_name(self, video_path):
        if not video_path:
            return
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
            # Import the video into the active participant's input folder
            self.context.import_file_for_participant(self.context.current_participant, f)
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
            f = tk.Frame(win)
            f.pack(fill=tk.X, padx=15, pady=5)
            tk.Label(f, text=lbl).pack(anchor="w")
            tk.Scale(f, from_=from_, to=to_, resolution=res, orient=tk.HORIZONTAL, variable=var).pack(fill=tk.X)

        add_scale("Track Low Threshold (Low-confidence track recovery):", self.track_low_thresh, 0.01, 0.6, 0.01)
        add_scale("New Track Threshold (Conservative init):", self.new_track_threshold, 0.1, 0.95, 0.05)
        add_scale("Proximity Threshold (BoT-SORT):", self.proximity_thresh, 0.1, 1.0, 0.05)
        add_scale("Appearance Threshold (BoT-SORT):", self.appearance_thresh, 0.1, 1.0, 0.05)

        f_chk = tk.Frame(win)
        f_chk.pack(fill=tk.X, padx=15, pady=15)
        tk.Checkbutton(f_chk, text="Enable Re-Identification (ReID)", variable=self.with_reid).pack(anchor="w")
        tk.Label(
            f_chk,
            text="(Requires automatic download of extra ReID weights)",
            fg="gray",
            font=("Arial", 8),
        ).pack(anchor="w")

        tk.Button(win, text="Close", command=win.destroy, width=15).pack(pady=10)

    def start_thread(self):
        if self.is_running:
            return
        if not self.video_path.get() or not self.output_path.get():
            messagebox.showwarning("Missing Data", "Select video and output.")
            return

        self.is_running = True
        self.btn_run.config(state="disabled", text="INITIALIZING YOLO...")
        self.stop_event.clear()
        t = threading.Thread(target=self.run_yolo_process, daemon=True)
        t.start()

    def _update_progress(self, current, total, stage="inference"):
        """Thread-safe UI update callback"""
        if stage == "download":
            # Il download occupa il primo 10% della barra (0-10%)
            # Se total è 0 (non si sa la dimensione), non fare nulla per evitare divisione per zero
            if total > 0:
                perc = (current / total) * 10
                self.parent.after(0, lambda: self.progress.config(value=perc, maximum=100))

        elif stage == "inference":
            # L'inferenza occupa il restante 90% (da 10% a 100%)
            if total > 0:
                # Calcoliamo la percentuale relativa all'inferenza (0-100)
                relative_perc = (current / total) * 90
                # Aggiungiamo il 10% del download
                final_val = 10 + relative_perc
                self.parent.after(0, lambda: self.progress.config(value=final_val, maximum=100))

    def _log_message(self, msg):
        """Thread-safe Log callback"""
        print(msg)  # This goes to TextRedirector -> ScrolledText

    def run_yolo_process(self):
        # Collect parameters from UI
        config = {
            "video_path": self.video_path.get(),
            "output_path": self.output_path.get(),
            "model_name": self.model_name.get(),
            "models_dir": self.context.paths["models"],
            "device": self.context.device,
            "tracker_params": {
                "tracker_type": self.tracker_type.get(),
                "conf": self.conf_threshold.get(),
                "iou": self.iou_threshold.get(),
                "match": self.match_threshold.get(),
                "buffer": self.track_buffer.get(),
                "low_thresh": self.track_low_thresh.get(),
                "new_track_thresh": self.new_track_threshold.get(),
                "prox": self.proximity_thresh.get(),
                "app": self.appearance_thresh.get(),
                "with_reid": self.with_reid.get(),
                "reid_model_name": self.reid_model_name.get(),
            },
        }

        try:
            success = self.logic.run_analysis(
                config, on_progress=self._update_progress, on_log=self._log_message, stop_event=self.stop_event
            )

            if success:
                self.context.pose_data_path = config["output_path"]
                self.parent.after(0, lambda: messagebox.showinfo("Finished", "Analysis complete."))

        except Exception as e:
            err_msg = str(e)  # Capture before 'e' goes out of scope
            self._log_message(f"❌ CRITICAL ERROR: {err_msg}\n{traceback.format_exc()}")
            self.parent.after(0, lambda: messagebox.showerror("Error", f"Error during analysis:\n{err_msg}"))

        finally:
            self.is_running = False
            self.parent.after(0, self._reset_btn)

    def _reset_btn(self):
        self.btn_run.config(state="normal", text="START GPU ANALYSIS")
        self.progress.config(value=0)
