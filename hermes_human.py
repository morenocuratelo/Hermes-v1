import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import torch
import cv2
import json
import gzip
import os
import threading
import sys
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
        self.model_name = tk.StringVar(value="yolo11x-pose.pt") # Default V11
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
        # Includiamo sia v8 che v11
        models = [
            "yolo11x-pose.pt", "yolo11l-pose.pt", "yolo11m-pose.pt", 
            "yolo11s-pose.pt", "yolo11n-pose.pt",
            "yolo8x-pose.pt", "yolo8l-pose.pt", "yolo8n-pose.pt"
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
        # Salva nella cartella Output del progetto se possibile
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

    def run_yolo_process(self):
        video_file = self.video_path.get()
        out_file = self.output_path.get()
        model_name = self.model_name.get()
        
        try:
            print(f"--- Inizio Analisi ---")
            print(f"Video: {os.path.basename(video_file)}")
            print(f"Device: {self.context.device.upper()}")
            
            # --- MODIFICA PATH MODELLO ---
            # Cerca o scarica il modello nella cartella del progetto
            model_path = os.path.join(self.context.paths["models"], model_name)
            print(f"Path Modello: {model_path}")
            
            # Se il file non esiste, YOLO lo scaricherà nella Current Working Directory.
            # Workaround: Cambiamo CWD o usiamo il path assoluto se YOLO lo supporta per il download
            # Ultralytics V8+ scarica automaticamente se non trova il file.
            # Se vogliamo forzare la cartella Models, possiamo passare il path assoluto.
            # Se non esiste, lo scarichiamo noi o lasciamo fare a lui e poi lo spostiamo.
            # Per semplicità qui passiamo il path: se YOLO non lo trova lì, potrebbe scaricarlo nella root.
            # Soluzione robusta: passo il nome nudo se non esiste, poi lo sposto.
            
            if os.path.exists(model_path):
                load_arg = model_path
            else:
                print("Modello non trovato nella cartella Models. YOLO lo scaricherà...")
                load_arg = model_name # Scarica nella root
                
            model = YOLO(load_arg) 
            
            # Se è stato appena scaricato nella root, spostiamolo per ordine (Opzionale)
            if load_arg == model_name and os.path.exists(model_name):
                try:
                    shutil.move(model_name, model_path)
                    print(f"Modello spostato in: {model_path}")
                    model = YOLO(model_path) # Ricarica dal path corretto
                except Exception as e:
                    print(f"Impossibile spostare il modello: {e}")

            print("Modello caricato in memoria.")
            
            # Info Video
            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            self.parent.after(0, lambda: self.progress.configure(maximum=total_frames))
            
            print("Avvio tracking (questo processo può richiedere tempo)...")
            
            results = model.track(
                source=video_file,
                stream=True,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
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
                        self.parent.after(0, lambda v=i: self.progress.config(value=v))
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