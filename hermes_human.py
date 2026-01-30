import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import torch
import cv2
import json
import gzip
import os
import threading
import sys
from ultralytics import YOLO

# --- REDIRECT PRINT TO GUI ---
class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")
        self.widget.update_idletasks()

    def flush(self):
        pass

# --- MAIN APP ---
class YoloLauncherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("1. YOLO Extraction Launcher - Lab Modigliani")
        self.root.geometry("800x650")
        
        # Variabili
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_name = tk.StringVar(value="yolo26x-pose.pt") # Default consigliato
        self.is_running = False
        
        self._build_ui()
        self._check_hardware()

    def _build_ui(self):
        main = tk.Frame(self.root, padx=20, pady=20)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Header
        tk.Label(main, text="YOLO Pose Extraction", font=("Segoe UI", 16, "bold")).pack(pady=(0, 20))

        # 1. Hardware Info
        self.lbl_hw = tk.Label(main, text="Rilevamento Hardware...", fg="gray", font=("Consolas", 10))
        self.lbl_hw.pack(pady=5)

        # 2. Selezione File
        lf_files = tk.LabelFrame(main, text="Input & Output", padx=10, pady=10)
        lf_files.pack(fill=tk.X, pady=5)
        
        self._add_picker(lf_files, "Video Input:", self.video_path, "*.mp4 *.avi *.mov")
        self._add_picker(lf_files, "Output JSON (.gz):", self.output_path, "*.json.gz", save=True)

        # 3. Configurazione Modello
        lf_conf = tk.LabelFrame(main, text="Configurazione Modello AI", padx=10, pady=10)
        lf_conf.pack(fill=tk.X, pady=5)
        
        tk.Label(lf_conf, text="Modello YOLO:").pack(side=tk.LEFT)
        models = ["yolo26n-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt", "yolo26l-pose.pt", "yolo26x-pose.pt"]
        self.cb_model = ttk.Combobox(lf_conf, textvariable=self.model_name, values=models, state="readonly", width=25)
        self.cb_model.pack(side=tk.LEFT, padx=10)
        tk.Label(lf_conf, text="(Consigliato: Medium o Large)", fg="gray").pack(side=tk.LEFT)

        # 4. Progress & Log
        self.progress = ttk.Progressbar(main, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(main, height=10, state='disabled', font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Redirezione stdout
        sys.stdout = TextRedirector(self.log_text, "stdout")
        sys.stderr = TextRedirector(self.log_text, "stderr")
        self.log_text.tag_config("stderr", foreground="red")

        # 5. Buttons
        self.btn_run = tk.Button(main, text="AVVIA ANALISI GPU", bg="#007ACC", fg="white", font=("Bold", 12), height=2, command=self.start_thread)
        self.btn_run.pack(fill=tk.X, pady=10)

    def _add_picker(self, p, lbl, var, ft, save=False):
        f = tk.Frame(p); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=lbl, width=15, anchor="w").pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        cmd = lambda: self.browse_save(var, ft) if save else self.browse_open(var, ft)
        tk.Button(f, text="...", width=3, command=cmd).pack(side=tk.LEFT)

    def browse_open(self, var, ft):
        f = filedialog.askopenfilename(filetypes=[("Video", ft)])
        if f: 
            var.set(f)
            # Auto-set output name
            if not self.output_path.get():
                base = os.path.splitext(f)[0]
                self.output_path.set(base + "_yolo.json.gz")

    def browse_save(self, var, ft):
        f = filedialog.asksaveasfilename(filetypes=[("JSON GZ", ft)], defaultextension=".json.gz")
        if f: var.set(f)

    def _check_hardware(self):
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.lbl_hw.config(text=f"✅ GPU Rilevata: {gpu_name} (CUDA Attivo)", fg="green")
            else:
                self.lbl_hw.config(text="⚠️ Nessuna GPU NVIDIA rilevata. L'analisi sarà lenta (CPU).", fg="orange")
        except Exception as e:
            self.lbl_hw.config(text=f"Errore check hardware: {e}", fg="red")

    def start_thread(self):
        if self.is_running: return
        if not self.video_path.get() or not self.output_path.get():
            messagebox.showwarning("Dati mancanti", "Seleziona video e output.")
            return
            
        self.is_running = True
        self.btn_run.config(state="disabled", text="ANALISI IN CORSO...")
        
        # Avvia thread separato per non bloccare la GUI
        t = threading.Thread(target=self.run_yolo_process)
        t.start()

    def run_yolo_process(self):
        video_file = self.video_path.get()
        out_file = self.output_path.get()
        model_name = self.model_name.get()
        
        try:
            print(f"--- Inizio Analisi ---")
            print(f"Video: {os.path.basename(video_file)}")
            print(f"Modello: {model_name}")
            
            # 1. Carica Modello
            model = YOLO(model_name)
            print("Modello caricato in VRAM.")
            
            # 2. Info Video
            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            self.root.after(0, lambda: self.progress.configure(maximum=total_frames))
            
            # 3. Processing Loop
            print("Avvio tracking...")
            
            # IMPORTANTE: persist=True per mantenere gli ID
            # tracker="bytetrack.yaml" per stabilità
            results = model.track(
                source=video_file,
                stream=True,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False, # Sopprime spam in console
                conf=0.5 # Confidenza media
            )
            
            with gzip.open(out_file, 'wt', encoding='utf-8') as f:
                for i, result in enumerate(results):
                    # Struttura dati identica al vecchio script
                    frame_data = {
                        "f_idx": i,
                        "det": json.loads(result.to_json())
                    }
                    f.write(json.dumps(frame_data) + "\n")
                    
                    # Aggiorna GUI ogni 10 frame
                    if i % 10 == 0:
                        self.root.after(0, lambda v=i: self.progress.config(value=v))
                        if i % 100 == 0:
                            print(f"Frame processati: {i}/{total_frames}")

            print(f"✅ COMPLETATO! Output salvato in:\n{out_file}")
            messagebox.showinfo("Finito", "Analisi completata con successo.")
            
        except Exception as e:
            print(f"❌ ERRORE CRITICO: {str(e)}")
            messagebox.showerror("Errore", str(e))
            
        finally:
            self.is_running = False
            self.root.after(0, self._reset_btn)

    def _reset_btn(self):
        self.btn_run.config(state="normal", text="AVVIA ANALISI GPU")
        self.progress.config(value=0)

if __name__ == "__main__":
    root = tk.Tk()
    YoloLauncherGUI(root)
    root.mainloop()