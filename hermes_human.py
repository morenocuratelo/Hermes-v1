import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import torch
import cv2
import json
import gzip
import os
import threading
import sys
import requests # Usiamo requests per un download migliore
from ultralytics import YOLO # type: ignore

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
        
        # --- SYNC CONTEXT (LETTURA) ---
        if self.context.video_path:
            self.video_path.set(self.context.video_path)
            self._suggest_output_name(self.context.video_path)

        self._build_ui()
        self._check_hardware_from_context()

    def _build_ui(self):
        tk.Label(self.parent, text="1. Human Pose Extraction (YOLO)", font=("Segoe UI", 18, "bold"), bg="white").pack(pady=(0, 20), anchor="w")

        # 1. Hardware Info
        self.lbl_hw = tk.Label(self.parent, text="Inizializzazione...", bg="white", font=("Consolas", 10))
        self.lbl_hw.pack(pady=5, anchor="w")

        # 2. Selezione File
        lf_files = tk.LabelFrame(self.parent, text="Input & Output", padx=10, pady=10, bg="white")
        lf_files.pack(fill=tk.X, pady=5)
        
        self._add_picker(lf_files, "Video Input:", self.video_path, "*.mp4 *.avi *.mov")
        self._add_picker(lf_files, "Output JSON (.gz):", self.output_path, "*.json.gz", save=True)

        # 3. Configurazione Modello
        lf_conf = tk.LabelFrame(self.parent, text="Configurazione Modello AI", padx=10, pady=10, bg="white")
        lf_conf.pack(fill=tk.X, pady=5)
        
        tk.Label(lf_conf, text="Modello YOLO:", bg="white").pack(side=tk.LEFT)
        
        # LISTA MODELLI
        models = [
            "yolo26x-pose.pt", "yolo26l-pose.pt", "yolo26m-pose.pt", "yolo26s-pose.pt", "yolo26n-pose.pt",
        ]
        self.cb_model = ttk.Combobox(lf_conf, textvariable=self.model_name, values=models, state="readonly", width=25)
        self.cb_model.pack(side=tk.LEFT, padx=10)
        tk.Label(lf_conf, text="(Salvataggio in: Project/Models)", fg="gray", bg="white").pack(side=tk.LEFT)

        # 4. Progress & Log
        self.progress = ttk.Progressbar(self.parent, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(self.parent, height=12, state='disabled', font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        sys.stdout = TextRedirector(self.log_text, "stdout")
        sys.stderr = TextRedirector(self.log_text, "stderr")
        self.log_text.tag_config("stderr", foreground="red")

        # 5. Buttons
        self.btn_run = tk.Button(self.parent, text="AVVIA ANALISI GPU", bg="#007ACC", fg="white", font=("Bold", 12), height=2, command=self.start_thread)
        self.btn_run.pack(fill=tk.X, pady=10)

    def _check_hardware_from_context(self):
        if self.context.device == "cuda":
            self.lbl_hw.config(text=f"✅ ACELERAZIONE ATTIVA: {self.context.gpu_name}", fg="green")
        else:
            self.lbl_hw.config(text=f"⚠️ ATTENZIONE: Nessuna GPU rilevata. Modalità CPU ({self.context.device}).", fg="orange")

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

    def start_thread(self):
        if self.is_running: return
        if not self.video_path.get() or not self.output_path.get():
            messagebox.showwarning("Dati mancanti", "Seleziona video e output.")
            return
            
        self.is_running = True
        self.btn_run.config(state="disabled", text="INIZIALIZZAZIONE YOLO...")
        t = threading.Thread(target=self.run_yolo_process)
        t.start()

    # --- NUOVA FUNZIONE DI DOWNLOAD ---
    def _download_model_manual(self, model_name, dest_path):
        """Scarica manualmente il modello per evitare il flood della console e gestire meglio la rete."""
        print(f"📥 Download modello in corso: {model_name}...")
        
        # URL ufficiali Ultralytics Assets
        base_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
        url = base_url + model_name
        
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Aggiorna progress bar UI (0-50% riservato al download)
                        if total_size > 0:
                            perc = int((downloaded / total_size) * 50)
                            self.parent.after(0, lambda v=perc: self.progress.config(value=v))
            
            print("✅ Download completato.")
            return True
        except Exception as e:
            print(f"❌ Errore download: {e}")
            if os.path.exists(dest_path): os.remove(dest_path) # Rimuovi file corrotto
            return False

    def run_yolo_process(self):
        video_file = self.video_path.get()
        out_file = self.output_path.get()
        model_name = self.model_name.get()
        
        try:
            print(f"--- Inizio Analisi ---")
            print(f"Video: {os.path.basename(video_file)}")
            print(f"Device: {self.context.device.upper()}")
            
            # 1. GESTIONE MODELLO
            model_path = os.path.join(self.context.paths["models"], model_name)
            
            if not os.path.exists(model_path):
                print(f"Il modello {model_name} non è presente nella cartella Models.")
                success = self._download_model_manual(model_name, model_path)
                if not success:
                    raise Exception("Impossibile scaricare il modello. Controlla internet o scaricalo manualmente.")
            else:
                print(f"Modello trovato in: {model_path}")
                self.progress.config(value=50) # Skip download progress

            # 2. CARICAMENTO
            print("Caricamento pesi YOLO in memoria...")
            model = YOLO(model_path) 
            
            # 3. TRACKING
            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Ricalibra progress bar per la fase di tracking (dal 50% al 100%)
            self.parent.after(0, lambda: self.progress.configure(maximum=total_frames + (total_frames))) 
            
            print("Avvio tracking video...")
            
            results = model.track(
                source=video_file,
                stream=True,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False, # <--- DISABILITA SPAM CONSOLE DI ULTRALYTICS
                conf=0.5,
                device=0 if self.context.device == "cuda" else "cpu"
            )
            
            with gzip.open(out_file, 'wt', encoding='utf-8') as f:
                for i, result in enumerate(results):
                    frame_data = {
                        "f_idx": i,
                        "det": json.loads(result.to_json())
                    }
                    f.write(json.dumps(frame_data) + "\n")
                    
                    if i % 10 == 0:
                        # Aggiorna progress bar (offset di partenza per considerare il download fatto)
                        # Usiamo un valore fittizio alto per muovere la barra nella seconda metà
                        current_val = total_frames + i 
                        self.parent.after(0, lambda v=current_val: self.progress.config(value=v))
                        
                        if i % 100 == 0:
                            print(f"Elaborato Frame: {i}/{total_frames}")

            print(f"✅ COMPLETATO! Output salvato in:\n{out_file}")
            
            self.context.pose_data_path = out_file
            messagebox.showinfo("Finito", "Analisi completata.")
            
        except Exception as e:
            print(f"❌ ERRORE: {str(e)}")
            messagebox.showerror("Errore", str(e))
            
        finally:
            self.is_running = False
            self.parent.after(0, self._reset_btn)

    def _reset_btn(self):
        self.btn_run.config(state="normal", text="AVVIA ANALISI GPU")
        self.progress.config(value=0)