import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import pandas as pd
import bisect
import os
from PIL import Image, ImageTk

class TimelineWidget(tk.Canvas):
    """Barra temporale che mostra i TOI colorati."""
    def __init__(self, parent, command_seek, **kwargs):
        super().__init__(parent, **kwargs)
        self.command_seek = command_seek
        self.duration = 0
        self.tois = []
        self.cursor_x = 0
        self.bind("<Button-1>", self.on_click)

    def set_data(self, duration, df_tois):
        self.duration = duration
        self.tois = []
        if df_tois is not None and not df_tois.empty:
            # Colori ciclici per le condizioni
            colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c']
            cond_map = {}
            for _, row in df_tois.iterrows():
                cond = str(row.get('Condition', 'Base'))
                if cond not in cond_map:
                    cond_map[cond] = colors[len(cond_map) % len(colors)]
                self.tois.append({
                    's': row['Start'], 'e': row['End'], 
                    'c': cond_map[cond], 'n': row['Name']
                })
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if self.duration <= 0:
            return
        
        # Disegna blocchi TOI
        for t in self.tois:
            x1 = (t['s'] / self.duration) * w
            x2 = (t['e'] / self.duration) * w
            self.create_rectangle(x1, 2, x2, h-2, fill=t['c'], outline="gray")
        
        # Cursore
        self.create_line(self.cursor_x, 0, self.cursor_x, h, fill="red", width=2)

    def update_cursor(self, current_sec):
        if self.duration > 0:
            w = self.winfo_width()
            self.cursor_x = (current_sec / self.duration) * w
            self.redraw() # Ridisegna per muovere il cursore (ottimizzabile)

    def on_click(self, event):
        if self.duration > 0:
            perc = event.x / self.winfo_width()
            sec = perc * self.duration
            self.command_seek(sec)

class ReviewerView:
    def __init__(self, parent, context=None):
        self.parent = parent
        self.context = context # Usa il context per caricare i percorsi automaticamente
        
        self.cap = None
        self.df_gaze = None
        self.df_tois = None
        self.fps = 30.0
        self.total_duration = 0
        self.is_playing = False
        
        self._setup_ui()
        
        # Auto-load se lanciato da Hermes
        if self.context:
            self.auto_load()

    def _setup_ui(self):
        tk.Label(self.parent, text="5. Data Reviewer", font=("Arial", 16, "bold")).pack(pady=5)
        
        # Controlli File
        fr_files = tk.LabelFrame(self.parent, text="Load Data")
        fr_files.pack(fill=tk.X, padx=5)
        btn_fr = tk.Frame(fr_files)
        btn_fr.pack(fill=tk.X)
        tk.Button(btn_fr, text="Video", command=self.load_video).pack(side=tk.LEFT)
        tk.Button(btn_fr, text="TOI (.tsv)", command=self.load_tois).pack(side=tk.LEFT)
        tk.Button(btn_fr, text="Gaze (.csv)", command=self.load_gaze).pack(side=tk.LEFT)
        
        # Video
        self.lbl_vid = tk.Label(self.parent, bg="black")
        self.lbl_vid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Timeline
        self.timeline = TimelineWidget(self.parent, command_seek=self.seek, height=40, bg="#eee")
        self.timeline.pack(fill=tk.X, padx=5)
        
        # Controlli Play
        ctrl = tk.Frame(self.parent)
        ctrl.pack(pady=5)
        tk.Button(ctrl, text="Play/Pause", command=self.toggle_play, width=15, bg="#cfc").pack()
        
        self.lbl_info = tk.Label(self.parent, text="Ready", fg="gray")
        self.lbl_info.pack()

    def auto_load(self):
        # Tenta di caricare dai path del context
        if hasattr(self.context, 'video_path') and self.context.video_path:
            self.load_video(self.context.video_path)
        if hasattr(self.context, 'toi_path') and self.context.toi_path:
            self.load_tois(self.context.toi_path)
        # Gaze è più difficile, spesso è in output, proviamo a cercarlo
        if hasattr(self.context, 'paths') and 'output' in self.context.paths:
            # Cerca un CSV che finisce con _mapped.csv o _results.csv
            out_dir = self.context.paths['output']
            for f in os.listdir(out_dir):
                if f.endswith("gazedata_MAPPED.csv") or f.endswith("gazedata_RESULTS.csv"):
                    self.load_gaze(os.path.join(out_dir, f))
                    break

    def load_video(self, path=None):
        if not path:
            path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi")])
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_duration = frames / self.fps
        self.timeline.set_data(self.total_duration, self.df_tois)
        self.show_frame()

    def load_tois(self, path=None):
        if not path:
            path = filedialog.askopenfilename(filetypes=[("TOI", "*.tsv *.csv")])
        if not path:
            return
        try:
            self.df_tois = pd.read_csv(path, sep='\t' if path.endswith('.tsv') else ',')
            if self.cap:
                self.timeline.set_data(self.total_duration, self.df_tois)
            messagebox.showinfo("OK", f"Loaded {len(self.df_tois)} TOIs")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_gaze(self, path=None):
        if not path:
            path = filedialog.askopenfilename(filetypes=[("Gaze", "*.csv")])
        if not path:
            return
        try:
            # Caricamento ottimizzato: assume colonne Timestamp (sec), GazeX, GazeY
            # Adatta i nomi delle colonne in base al tuo CSV!
            df = pd.read_csv(path)
            
            # Logica base per trovare le colonne giuste
            t_col = next((c for c in df.columns if 'time' in c.lower()), None)
            x_col = next((c for c in df.columns if 'x' in c.lower() and 'gaze' in c.lower()), None)
            y_col = next((c for c in df.columns if 'y' in c.lower() and 'gaze' in c.lower()), None)

            if t_col and x_col and y_col:
                self.df_gaze = df[[t_col, x_col, y_col]].sort_values(by=t_col)
                self.gaze_t = self.df_gaze[t_col].values # Per ricerca veloce
                self.gaze_x = self.df_gaze[x_col].values
                self.gaze_y = self.df_gaze[y_col].values
                messagebox.showinfo("OK", "Gaze loaded successfully")
            else:
                messagebox.showwarning("Warning", f"Gaze columns not found in: {df.columns}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.loop()

    def seek(self, sec):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            self.show_frame()

    def loop(self):
        if self.is_playing and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.process_frame(frame)
                self.parent.after(int(1000/self.fps), self.loop)
            else:
                self.is_playing = False

    def show_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.process_frame(frame)

    def process_frame(self, frame):
        # 1. Info Tempo
        curr_sec = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.timeline.update_cursor(curr_sec)
        
        h, w, _ = frame.shape
        
        # 2. Disegna TOI corrente
        toi_txt = ""
        if self.df_tois is not None:
            # Cerca se siamo dentro un TOI
            matches = self.df_tois[(self.df_tois['Start'] <= curr_sec) & (self.df_tois['End'] >= curr_sec)]
            if not matches.empty:
                toi_txt = matches.iloc[0]['Name']
                cv2.putText(frame, toi_txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 3. Disegna Gaze (Interpolazione base o Nearest)
        if self.df_gaze is not None:
            # Trova l'indice del timestamp più vicino
            idx = bisect.bisect_left(self.gaze_t, curr_sec)
            if idx < len(self.gaze_t):
                # Se il dato è entro 100ms dal frame video, disegnalo
                if abs(self.gaze_t[idx] - curr_sec) < 0.1:
                    gx, gy = self.gaze_x[idx], self.gaze_y[idx]
                    # Scala se normalizzato (0-1)
                    if gx <= 1.1 and gy <= 1.1: 
                        gx, gy = int(gx * w), int(gy * h)
                    else:
                        gx, gy = int(gx), int(gy)
                    
                    cv2.circle(frame, (gx, gy), 15, (0, 0, 255), 2) # Cerchio Rosso
                    cv2.line(frame, (gx-10, gy), (gx+10, gy), (0,0,255), 2)
                    cv2.line(frame, (gx, gy-10), (gx, gy+10), (0,0,255), 2)

        # Rendering
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        # Resize per la finestra
        c_w, c_h = self.lbl_vid.winfo_width(), self.lbl_vid.winfo_height()
        if c_w > 10:
            img.thumbnail((c_w, c_h))
        
        imgtk = ImageTk.PhotoImage(image=img)
        self.lbl_vid.imgtk = imgtk # Keep reference!
        self.lbl_vid.configure(image=imgtk)
        self.lbl_info.config(text=f"T: {curr_sec:.2f}s | {toi_txt}")

# Blocco per testarlo da solo
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Reviewer Standalone")
    root.geometry("1000x700")
    app = ReviewerView(root)
    root.mainloop()
