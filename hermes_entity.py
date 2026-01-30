import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog, colorchooser
import cv2
import json
import gzip
import os
import random
import math
from PIL import Image, ImageTk

class IdentityMapperV7:
    def __init__(self, root):
        self.root = root
        self.root.title("Identity Mapper v7.1 (Custom Params) - Lab Modigliani")
        self.root.geometry("1600x950")
        
        # DATI
        self.video_path = None
        self.json_path = None
        self.cap = None
        self.fps = 30.0
        
        self.tracks = {} 
        self.id_lineage = {} 
        
        # --- PARAMETRI CONFIGURABILI (Default da immagine) ---
        # Unsupervised Auto-Stitching
        self.param_lookahead = 15      # Window size (tracks)
        self.param_time_gap = 2.0      # Seconds (occlusion tolerance)
        self.param_stitch_dist = 150   # Pixels (centroid distance)
        
        # Supervised Noise Absorption
        self.param_noise_dist = 100    # Pixels (tighter precision)
        # -----------------------------------------------------

        # CAST
        self.cast = {
            "Target": {"color": (0, 255, 0)},       
            "Confederate_1": {"color": (0, 255, 255)}, 
            "Confederate_2": {"color": (0, 165, 255)}  
        }
        
        self.hide_short_var = tk.BooleanVar(value=True)
        
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False

        self._setup_ui()

    def _setup_ui(self):
        main = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # 1. VIDEO (SX)
        left = tk.Frame(main, bg="black")
        main.add(left, minsize=900)
        self.lbl_video = tk.Label(left, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)
        
        ctrl = tk.Frame(left)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        self.slider = ttk.Scale(ctrl, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_seek)
        self.slider.pack(fill=tk.X, padx=5)
        
        btns = tk.Frame(ctrl)
        btns.pack(pady=5)
        tk.Button(btns, text="📂 Carica Dati", command=self.load_data).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="⏯ Play/Pausa", command=self.toggle_play).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="💾 SALVA MAPPATURA", bg="#4CAF50", fg="white", font=("bold"), command=self.save_mapping).pack(side=tk.LEFT, padx=20)

        # 2. GESTIONE (DX)
        right = tk.Frame(main, padx=5, pady=5)
        main.add(right, minsize=450)

        # A. CAST
        lbl_cast = tk.LabelFrame(right, text="1. Cast (Persone)", padx=5, pady=5)
        lbl_cast.pack(fill=tk.X, pady=5)
        self.list_cast = tk.Listbox(lbl_cast, height=6)
        self.list_cast.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        btn_cast = tk.Frame(lbl_cast)
        btn_cast.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Button(btn_cast, text="➕ Aggiungi", command=self.add_person).pack(fill=tk.X)
        tk.Button(btn_cast, text="🎨 Colore", command=self.change_person_color).pack(fill=tk.X)
        tk.Button(btn_cast, text="➖ Rimuovi", command=self.remove_person).pack(fill=tk.X)

        # B. TRACCE
        lbl_tracks = tk.LabelFrame(right, text="2. Tracce YOLO & Strumenti", padx=5, pady=5)
        lbl_tracks.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # TOOLBAR
        tools = tk.Frame(lbl_tracks)
        tools.pack(fill=tk.X, pady=5)
        
        # Checkbox nascondi brevi
        chk = tk.Checkbutton(tools, text="Nascondi brevi (<1s)", variable=self.hide_short_var, command=self.refresh_tree)
        chk.pack(side=tk.LEFT, padx=5)

        # --- NUOVO: BOTTONE SETTINGS ---
        tk.Button(tools, text="⚙ Parametri", command=self.open_settings_dialog).pack(side=tk.RIGHT, padx=5)
        
        # PRIMA RIGA BOTTONI (Automazioni)
        row1 = tk.Frame(lbl_tracks)
        row1.pack(fill=tk.X, pady=2)
        tk.Button(row1, text="⚡ Auto-Stitch", command=self.auto_stitch).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(row1, text="🧹 Assorbi Noise (Gap Fill)", command=self.absorb_noise_logic).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # SECONDA RIGA BOTTONI (Merging)
        row2 = tk.Frame(lbl_tracks)
        row2.pack(fill=tk.X, pady=2)
        tk.Button(row2, text="🔗 Unisci Selezionati", command=self.manual_merge).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(row2, text="🔗 Unisci TUTTI per Ruolo", bg="#d1e7dd", command=self.merge_all_by_role).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # TREEVIEW
        cols = ("ID", "Origine", "Durata", "Assegnato A")
        self.tree = ttk.Treeview(lbl_tracks, columns=cols, show="headings", selectmode="extended")
        self.tree.heading("ID", text="ID")
        self.tree.heading("Origine", text="Storia")
        self.tree.heading("Durata", text="Sec")
        self.tree.heading("Assegnato A", text="Persona")
        
        self.tree.column("ID", width=40)
        self.tree.column("Origine", width=60)
        self.tree.column("Durata", width=50)
        self.tree.column("Assegnato A", width=100)

        sb = ttk.Scrollbar(lbl_tracks, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Button-3>", self.show_context_menu)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.refresh_cast_list()

    # --- NUOVA FEATURE: DIALOG PARAMETRI ---
    def open_settings_dialog(self):
        """Apre una finestra popup per modificare i parametri hardcoded."""
        win = tk.Toplevel(self.root)
        win.title("Impostazioni Algoritmi")
        win.geometry("350x250")
        
        # Variabili temporanee
        v_lookahead = tk.IntVar(value=self.param_lookahead)
        v_time = tk.DoubleVar(value=self.param_time_gap)
        v_s_dist = tk.IntVar(value=self.param_stitch_dist)
        v_n_dist = tk.IntVar(value=self.param_noise_dist)
        
        # Layout
        tk.Label(win, text="1. Auto-Stitching", font=("bold")).pack(pady=(10,5))
        
        f1 = tk.Frame(win); f1.pack(fill=tk.X, padx=20)
        tk.Label(f1, text="Look-ahead (tracks):").pack(side=tk.LEFT)
        tk.Entry(f1, textvariable=v_lookahead, width=8).pack(side=tk.RIGHT)
        
        f2 = tk.Frame(win); f2.pack(fill=tk.X, padx=20)
        tk.Label(f2, text="Max Time Gap (sec):").pack(side=tk.LEFT)
        tk.Entry(f2, textvariable=v_time, width=8).pack(side=tk.RIGHT)

        f3 = tk.Frame(win); f3.pack(fill=tk.X, padx=20)
        tk.Label(f3, text="Max Distance (px):").pack(side=tk.LEFT)
        tk.Entry(f3, textvariable=v_s_dist, width=8).pack(side=tk.RIGHT)

        tk.Label(win, text="2. Noise Absorption", font=("bold")).pack(pady=(10,5))
        
        f4 = tk.Frame(win); f4.pack(fill=tk.X, padx=20)
        tk.Label(f4, text="Precision Dist (px):").pack(side=tk.LEFT)
        tk.Entry(f4, textvariable=v_n_dist, width=8).pack(side=tk.RIGHT)
        
        def save():
            self.param_lookahead = v_lookahead.get()
            self.param_time_gap = v_time.get()
            self.param_stitch_dist = v_s_dist.get()
            self.param_noise_dist = v_n_dist.get()
            win.destroy()
            messagebox.showinfo("Salvataggio", "Parametri aggiornati con successo.")

        tk.Button(win, text="Salva", command=save, bg="#4CAF50", fg="white").pack(pady=15)

    # --- LOGICHE AGGIORNATE CON PARAMETRI ---

    def absorb_noise_logic(self):
        """Supervised Noise Absorption usando i parametri configurabili."""
        if not messagebox.askyesno("Conferma", f"Assorbire il noise (Dist < {self.param_noise_dist}px)?"): return
        
        main_tracks = [tid for tid, d in self.tracks.items() if d['role'] in self.cast]
        candidates = [tid for tid, d in self.tracks.items() if d['role'] not in self.cast]
        
        absorbed = 0
        changed = True
        
        # Parametri dinamici
        MAX_DIST = self.param_noise_dist
        MAX_TIME_GAP = self.param_time_gap * self.fps
        
        while changed:
            changed = False
            curr_main = list(main_tracks)
            
            for main_id in curr_main:
                if main_id not in self.tracks: continue
                main_data = self.tracks[main_id]
                main_frames = sorted(main_data['frames'])
                
                to_remove = []
                for cand_id in candidates:
                    if cand_id not in self.tracks: continue
                    cand_data = self.tracks[cand_id]
                    c_frames = cand_data['frames']
                    
                    if not set(main_frames).isdisjoint(c_frames): continue # Overlap
                    
                    # Distanza temporale
                    gap_after = c_frames[0] - main_frames[-1]
                    dist_after = float('inf')
                    
                    if 0 < gap_after < MAX_TIME_GAP:
                        end_m = main_data['boxes'][main_data['frames'].index(main_frames[-1])]
                        start_c = cand_data['boxes'][0]
                        cm = ((end_m[0]+end_m[2])/2, (end_m[1]+end_m[3])/2)
                        cc = ((start_c[0]+start_c[2])/2, (start_c[1]+start_c[3])/2)
                        dist_after = math.sqrt((cm[0]-cc[0])**2 + (cm[1]-cc[1])**2)
                        
                    gap_before = main_frames[0] - c_frames[-1]
                    dist_before = float('inf')
                    
                    if 0 < gap_before < MAX_TIME_GAP:
                        end_c = cand_data['boxes'][-1]
                        start_m = main_data['boxes'][main_data['frames'].index(main_frames[0])]
                        cc = ((end_c[0]+end_c[2])/2, (end_c[1]+end_c[3])/2)
                        cm = ((start_m[0]+start_m[2])/2, (start_m[1]+start_m[3])/2)
                        dist_before = math.sqrt((cm[0]-cc[0])**2 + (cm[1]-cc[1])**2)
                        
                    if dist_after < MAX_DIST or dist_before < MAX_DIST:
                        self._merge_logic(main_id, cand_id)
                        absorbed += 1
                        changed = True
                        to_remove.append(cand_id)
                        main_frames = sorted(self.tracks[main_id]['frames'])
                        
                for c in to_remove: 
                    if c in candidates: candidates.remove(c)
                    
        self.refresh_tree()
        messagebox.showinfo("Info", f"Assorbiti {absorbed} frammenti.")

    def auto_stitch(self):
        """Unsupervised Auto-Stitching usando i parametri configurabili."""
        # Recupera parametri
        p_win = self.param_lookahead
        p_time = self.param_time_gap
        p_dist = self.param_stitch_dist
        
        sorted_ids = sorted(self.tracks.keys(), key=lambda x: self.tracks[x]['frames'][0])
        merged = 0
        changed = True
        
        while changed:
            changed = False
            curr_ids = sorted(self.tracks.keys(), key=lambda x: min(self.tracks[x]['frames']))
            i = 0
            while i < len(curr_ids) - 1:
                id_a = curr_ids[i]
                best_match = None
                min_dist = float('inf')
                
                # Usa param_lookahead invece di 15 fisso
                search_limit = min(i + p_win, len(curr_ids))
                
                for j in range(i+1, search_limit):
                    id_b = curr_ids[j]
                    t_a = self.tracks[id_a]; t_b = self.tracks[id_b]
                    
                    gap = t_b['frames'][0] - t_a['frames'][-1]
                    
                    # Usa param_time_gap (converted to frames)
                    if gap < 0 or gap > (p_time * self.fps): continue
                    
                    ba = t_a['boxes'][-1]; bb = t_b['boxes'][0]
                    ca = ((ba[0]+ba[2])/2, (ba[1]+ba[3])/2)
                    cb = ((bb[0]+bb[2])/2, (bb[1]+bb[3])/2)
                    d = math.sqrt((ca[0]-cb[0])**2 + (ca[1]-cb[1])**2)
                    
                    # Usa param_stitch_dist
                    if d < p_dist and d < min_dist:
                        min_dist = d; best_match = id_b
                
                if best_match:
                    self._merge_logic(id_a, best_match)
                    merged += 1; changed = True; break
                i += 1
        self.refresh_tree()
        messagebox.showinfo("Info", f"Uniti {merged} frammenti (Lookahead:{p_win}, Time:{p_time}s, Dist:{p_dist}px).")

    # --- STANDARD METHODS (Unchanged) ---
    def merge_all_by_role(self):
        if not messagebox.askyesno("Conferma", "Vuoi unire tutte le tracce assegnate allo stesso ruolo?"): return
        merge_count = 0
        roles_processed = []
        for person_name in self.cast:
            ids_with_role = [tid for tid, data in self.tracks.items() if data['role'] == person_name]
            ids_with_role.sort()
            if len(ids_with_role) > 1:
                master_id = ids_with_role[0]
                for slave_id in ids_with_role[1:]:
                    self._merge_logic(master_id, slave_id)
                    merge_count += 1
                roles_processed.append(person_name)
        self.refresh_tree()
        if merge_count > 0: msg = f"Fusi {merge_count} frammenti per: {', '.join(roles_processed)}."
        else: msg = "Nessuna fusione necessaria."
        messagebox.showinfo("Risultato Merge", msg)

    def manual_merge(self):
        sel = self.tree.selection()
        if len(sel) < 2: return
        ids = sorted([int(x) for x in sel])
        master = ids[0]
        role = self.tracks[master]['role']
        for s in ids[1:]:
            sr = self.tracks[s]['role']
            if role not in self.cast and sr in self.cast: role = sr
            self._merge_logic(master, s)
        self.tracks[master]['role'] = role
        self.refresh_tree()
        self.tree.selection_set(str(master))

    def _merge_logic(self, master, slave):
        if slave not in self.tracks: return
        self.tracks[master]['frames'].extend(self.tracks[slave]['frames'])
        self.tracks[master]['boxes'].extend(self.tracks[slave]['boxes'])
        self.tracks[master]['merged_from'].extend(self.tracks[slave]['merged_from'])
        for oid, curr in self.id_lineage.items():
            if curr == slave: self.id_lineage[oid] = master
        del self.tracks[slave]
        z = sorted(zip(self.tracks[master]['frames'], self.tracks[master]['boxes']), key=lambda x:x[0])
        self.tracks[master]['frames'] = [x[0] for x in z]
        self.tracks[master]['boxes'] = [x[1] for x in z]

    def refresh_tree(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        hide = self.hide_short_var.get()
        for tid in sorted(self.tracks.keys()):
            d = self.tracks[tid]
            role = d['role']
            dur = len(d['frames']) / self.fps
            if hide and dur < 1.0 and role not in self.cast: continue
            
            merged = str(d['merged_from']) if len(d['merged_from']) > 1 else str(tid)
            tag = "Ignore"
            if role in self.cast: tag = role
            
            self.tree.insert("", "end", iid=str(tid), values=(tid, merged, f"{dur:.2f}", role), tags=(tag,))
            
        self.tree.tag_configure("Ignore", background="white")
        for n in self.cast:
            b,g,r = self.cast[n]['color']
            self.tree.tag_configure(n, background='#{:02x}{:02x}{:02x}'.format(min(r+180,255), min(g+180,255), min(b+180,255)))

    def load_data(self):
        v = filedialog.askopenfilename(); 
        if not v: return
        j = filedialog.askopenfilename()
        if not j: return
        self.video_path = v; self.json_path = j
        self.cap = cv2.VideoCapture(v)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.config(to=self.total_frames-1)
        self.tracks = {}; self.id_lineage = {}
        try:
            with gzip.open(j, 'rt', encoding='utf-8') as f:
                for line in f:
                    d = json.loads(line)
                    idx = d['f_idx']
                    for det in d['det']:
                        tid = det.get('track_id')
                        if tid is None: continue
                        tid = int(tid)
                        if tid not in self.tracks:
                            self.tracks[tid] = {'frames':[], 'boxes':[], 'role':'Ignore', 'merged_from':[tid]}
                            self.id_lineage[tid] = tid
                        self.tracks[tid]['frames'].append(idx)
                        b=det['box']
                        self.tracks[tid]['boxes'].append([b['x1'],b['y1'],b['x2'],b['y2']])
        except Exception as e: messagebox.showerror("Err", str(e))
        self.refresh_tree(); self.show_frame()

    def save_mapping(self):
        if not self.json_path: return
        out_map = {}
        count = 0
        for oid, master in self.id_lineage.items():
            if master in self.tracks:
                role = self.tracks[master]['role']
                if role in self.cast:
                    out_map[str(oid)] = role
                    count += 1
        out = self.json_path.replace(".json.gz", "_identity.json")
        with open(out, 'w') as f: json.dump(out_map, f, indent=4)
        messagebox.showinfo("Fatto", f"Mappati {count} ID originali.")

    def refresh_cast_list(self):
        self.list_cast.delete(0, tk.END)
        for n in self.cast:
            self.list_cast.insert(tk.END, n)
            b,g,r = self.cast[n]['color']
            self.list_cast.itemconfig(self.list_cast.size()-1, bg='#{:02x}{:02x}{:02x}'.format(r,g,b))
            
    def add_person(self):
        n = simpledialog.askstring("Nuovo", "Nome:")
        if n: self.cast[n] = {"color":(random.randint(50,200),random.randint(50,200),random.randint(50,200))}; self.refresh_cast_list()
    
    def remove_person(self):
        s = self.list_cast.curselection()
        if s: 
            n = self.list_cast.get(s[0])
            del self.cast[n]
            for t in self.tracks.values(): 
                if t['role']==n: t['role']='Ignore'
            self.refresh_cast_list(); self.refresh_tree()
            
    def change_person_color(self):
        s = self.list_cast.curselection()
        if s:
            n = self.list_cast.get(s[0])
            c = colorchooser.askcolor()
            if c[0]: 
                r,g,b = map(int, c[0])
                self.cast[n]['color'] = (b,g,r)
                self.refresh_cast_list(); self.show_frame()

    def show_context_menu(self, e):
        i = self.tree.identify_row(e.y)
        if i:
            if i not in self.tree.selection(): self.tree.selection_set(i)
            self.context_menu.delete(0, tk.END)
            for p in self.cast: self.context_menu.add_command(label=f"Assegna a {p}", command=lambda n=p: self.assign_sel(n))
            self.context_menu.add_separator()
            self.context_menu.add_command(label="Rimuovi", command=lambda: self.assign_sel("Ignore"))
            self.context_menu.add_separator()
            self.context_menu.add_command(label="Unisci", command=self.manual_merge)
            self.context_menu.post(e.x_root, e.y_root)

    def assign_sel(self, r):
        for i in self.tree.selection(): self.tracks[int(i)]['role'] = r
        self.refresh_tree(); self.show_frame()

    def on_tree_select(self, e):
        s = self.tree.selection()
        if s: self.current_frame = self.tracks[int(s[0])]['frames'][0]; self.slider.set(self.current_frame); self.show_frame()

    def on_seek(self, v): self.current_frame = int(float(v)); self.show_frame()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_loop()

    def play_loop(self):
        if self.is_playing and self.cap:
            self.current_frame += 1
            if self.current_frame >= self.total_frames: self.is_playing=False
            self.slider.set(self.current_frame); self.show_frame(); self.root.after(30, self.play_loop)
            
    def show_frame(self):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret: return
        hide = self.hide_short_var.get()
        for tid, d in self.tracks.items():
            if self.current_frame in d['frames']:
                role = d['role']
                # if hide and len(d['frames'])/self.fps < 1.0 and role not in self.cast: continue 
                idx = d['frames'].index(self.current_frame)
                box = d['boxes'][idx]
                col = (100,100,100)
                if role in self.cast: col = self.cast[role]['color']
                x1,y1,x2,y2 = map(int, box)
                cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
                cv2.putText(frame, f"{tid} {role if role!='Ignore' else ''}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.thumbnail((self.lbl_video.winfo_width(), self.lbl_video.winfo_height()))
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.lbl_video.config(image=self.tk_img)

if __name__ == "__main__":
    root = tk.Tk()
    IdentityMapperV7(root)
    root.mainloop()