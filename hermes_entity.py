import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog, colorchooser
import cv2
import json
import gzip
import os
import random
import math
from PIL import Image, ImageTk
import pickle
import tempfile
import shutil
import time
import threading
import queue

class HistoryManager:
    def __init__(self, max_history=20):
        self.max_history = max_history
        self.temp_dir = tempfile.mkdtemp(prefix="hermes_history_")
        self.undo_stack = []
        self.redo_stack = []
        self.current_state_file = None

    def push_state(self, data):
        # Create unique filename
        filename = os.path.join(self.temp_dir, f"state_{time.time_ns()}.pkl")
        
        # Serializza data
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            
        # Manage stacks
        if self.current_state_file:
            self.undo_stack.append(self.current_state_file)
            
        self.current_state_file = filename
        
        # Clear redo stack and delete associated files
        for f in self.redo_stack:
            self._delete_file(f)
        self.redo_stack = []
        
        # Enforce max history
        if len(self.undo_stack) > self.max_history:
            oldest = self.undo_stack.pop(0)
            self._delete_file(oldest)

    def undo(self):
        if not self.undo_stack:
            return None
            
        # Move current to redo
        if self.current_state_file:
            self.redo_stack.append(self.current_state_file)
            
        # Pop from undo
        prev_file = self.undo_stack.pop()
        self.current_state_file = prev_file
        
        return self._load_file(prev_file)

    def redo(self):
        if not self.redo_stack:
            return None
            
        # Move current to undo
        if self.current_state_file:
            self.undo_stack.append(self.current_state_file)
            
        # Pop from redo
        next_file = self.redo_stack.pop()
        self.current_state_file = next_file
        
        return self._load_file(next_file)

    def clear(self):
        self.undo_stack = []
        self.redo_stack = []
        self.current_state_file = None
        # Clean directory contents
        if os.path.exists(self.temp_dir):
            for f in os.listdir(self.temp_dir):
                self._delete_file(os.path.join(self.temp_dir, f))

    def _load_file(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"History Load Error: {e}")
            return None
            
    def _delete_file(self, filepath):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass
            
    def __del__(self):
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass

class IdentityView:  # <--- CAMBIATO NOME
    def __init__(self, parent, context): # <--- NUOVI ARGOMENTI
        self.parent = parent
        self.context = context
        
        # Rimuovi self.root.title e geometry, non servono più
        
        # DATI
        self.video_path = None
        self.json_path = None
        self.cap = None
        self.fps = 30.0
        
        self.tracks = {} 
        self.id_lineage = {} 
        self.history = HistoryManager()
        self.load_queue = queue.Queue()
        
        # --- PARAMETRI CONFIGURABILI (Invariati) ---
        self.param_lookahead = 15      
        self.param_time_gap = 2.0      
        self.param_stitch_dist = 150   
        self.param_noise_dist = 100    
        # -------------------------------------------

        # CAST (Recupera dal CONTEXT se esiste, altrimenti usa default)
        if self.context.cast:
            self.cast = self.context.cast
        else:
            self.cast = {
                "Target": {"color": (0, 255, 0)},       
                "Confederate_1": {"color": (0, 0, 255)}, 
                "Confederate_2": {"color": (255, 0, 0)}  
            }
            self.context.cast = self.cast # Salva nel context per il futuro

        self.hide_short_var = tk.BooleanVar(value=True)
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False

        self._setup_ui()
        self._setup_hotkeys()
        
        # AUTO-LOAD: If Human has finished, load data automatically
        if self.context.video_path and self.context.pose_data_path:
            self.load_data_direct(self.context.video_path, self.context.pose_data_path)

        # --- AUTOSAVE INIT ---
        self.AUTOSAVE_INTERVAL_MS = 300000 # 5 minutes
        self.AUTOSAVE_FILENAME = "hermes_autosave_identity.json"
        self._check_for_autosave()
        self._start_autosave_loop()
        
        # Cleanup on exit
        self.parent.winfo_toplevel().protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self):
        # Header visivo
        tk.Label(self.parent, text="2. Identity Assignment", font=("Segoe UI", 18, "bold"), bg="white").pack(pady=(0, 10), anchor="w")

        # Usa self.parent invece di self.root
        main = tk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
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
        tk.Button(btns, text="📂 Load Video", command=self.browse_video).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="📂 Load Yolo", command=self.browse_pose).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="📂 Load Mapping", command=self.load_mapping).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="⏯ Play/Pause", command=self.toggle_play).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="💾 SAVE MAPPING", bg="#4CAF50", fg="white", font=("bold"), command=self.save_mapping).pack(side=tk.LEFT, padx=20)
        # 2. GESTIONE (DX)
        right = tk.Frame(main, padx=5, pady=5)
        main.add(right, minsize=450)
        self.right_panel = right

        # Progress Bar (Hidden)
        self.progress = ttk.Progressbar(right, mode='indeterminate')

        # A. CAST
        lbl_cast = tk.LabelFrame(right, text="1. Cast (Persone)", padx=5, pady=5)
        lbl_cast.pack(fill=tk.X, pady=5)
        self.list_cast = tk.Listbox(lbl_cast, height=6)
        self.list_cast.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        btn_cast = tk.Frame(lbl_cast)
        btn_cast.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Button(btn_cast, text="➕ Add", command=self.add_person).pack(fill=tk.X)
        tk.Button(btn_cast, text="🎨 Color", command=self.change_person_color).pack(fill=tk.X)
        tk.Button(btn_cast, text="➖ Remove", command=self.remove_person).pack(fill=tk.X)

        # B. TRACKS
        lbl_tracks = tk.LabelFrame(right, text="2. YOLO & Tools Tracks", padx=5, pady=5)
        lbl_tracks.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # TOOLBAR
        tools = tk.Frame(lbl_tracks)
        tools.pack(fill=tk.X, pady=5)
        
        # Checkbox hide short
        chk = tk.Checkbutton(tools, text="Hide short (<1s)", variable=self.hide_short_var, command=self.refresh_tree)
        chk.pack(side=tk.LEFT, padx=5)

        # --- NEW: SETTINGS BUTTON ---
        tk.Button(tools, text="⚙ Parameters", command=self.open_settings_dialog).pack(side=tk.RIGHT, padx=5)
        
        # FIRST ROW BUTTONS (Automations)
        row1 = tk.Frame(lbl_tracks)
        row1.pack(fill=tk.X, pady=2)
        tk.Button(row1, text="⚡ Auto-Stitch", command=self.auto_stitch).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(row1, text="🧹 Absorb Noise (Gap Fill)", command=self.absorb_noise_logic).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # SECOND ROW BUTTONS (Merging)
        row2 = tk.Frame(lbl_tracks)
        row2.pack(fill=tk.X, pady=2)
        tk.Button(row2, text="🔗 Merge Selected", command=self.manual_merge).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(row2, text="🔗 Merge ALL by Role", bg="#d1e7dd", command=self.merge_all_by_role).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # TREEVIEW
        cols = ("ID", "Origin", "Duration", "Assigned To")
        self.tree = ttk.Treeview(lbl_tracks, columns=cols, show="headings", selectmode="extended")
        self.tree.heading("ID", text="ID")
        self.tree.heading("Origin", text="Story")
        self.tree.heading("Duration", text="Sec")
        self.tree.heading("Assigned To", text="Person")
        
        self.tree.column("ID", width=40)
        self.tree.column("Origin", width=60)
        self.tree.column("Duration", width=50)
        self.tree.column("Assigned To", width=100)

        sb = ttk.Scrollbar(lbl_tracks, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Button-3>", self.show_context_menu)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        
        self.context_menu = tk.Menu(self.parent, tearoff=0)
        self.refresh_cast_list()

    def _setup_hotkeys(self):
        root = self.parent.winfo_toplevel()
        # Playback & Navigation
        root.bind("<Space>", self._on_space)
        root.bind("<Left>", self._on_left)
        root.bind("<Right>", self._on_right)
        root.bind("<Shift-Left>", self._on_shift_left)
        root.bind("<Shift-Right>", self._on_shift_right)
        # Assignment 1-9
        for i in range(1, 10):
            root.bind(str(i), self._on_number)
        root.bind("<Control-z>", self.perform_undo)

    def _is_hotkey_safe(self):
        focused = self.parent.focus_get()
        # Ignore hotkeys if user is typing in an Entry or Text field
        if focused and focused.winfo_class() in ['Entry', 'Text']:
            return False
        return True

    def _on_space(self, event):
        if self._is_hotkey_safe(): self.toggle_play()

    def _on_left(self, event):
        if self._is_hotkey_safe(): self.seek_relative(-1)

    def _on_right(self, event):
        if self._is_hotkey_safe(): self.seek_relative(1)

    def _on_shift_left(self, event):
        if self._is_hotkey_safe(): self.seek_relative(-10)

    def _on_shift_right(self, event):
        if self._is_hotkey_safe(): self.seek_relative(10)

    def seek_relative(self, delta):
        if not self.cap: return
        self.current_frame = max(0, min(self.total_frames - 1, self.current_frame + delta))
        self.slider.set(self.current_frame)
        self.show_frame()

    def _on_number(self, event):
        if not self._is_hotkey_safe(): return
        try:
            idx = int(event.char) - 1
            if 0 <= idx < self.list_cast.size() and self.tree.selection():
                self.assign_sel(self.list_cast.get(idx))
        except ValueError: pass

    def _snapshot(self):
        self.history.push_state(self.tracks)

    def perform_undo(self, event=None):
        prev_state = self.history.undo()
        if prev_state is not None:
            self.tracks = prev_state
            # Rebuild lineage from tracks to maintain consistency
            self.id_lineage = {}
            for tid, data in self.tracks.items():
                self.id_lineage[tid] = tid
                for merged in data.get('merged_from', []):
                    self.id_lineage[merged] = tid
            self.refresh_tree()
            self.show_frame()
            print("Undo performed.")

    def _start_autosave_loop(self):
        try:
            self.parent.after(self.AUTOSAVE_INTERVAL_MS, self._perform_autosave)
        except Exception:
            pass # Widget destroyed

    def _perform_autosave(self):
        if not self.tracks:
            self._start_autosave_loop()
            return

        # Determine path
        if self.context and self.context.project_path:
            save_path = os.path.join(self.context.project_path, self.AUTOSAVE_FILENAME)
        else:
            save_path = self.AUTOSAVE_FILENAME

        self.AUTOSAVE_FILENAME
        save_path = self._get_autosave_path()

        def save_task(data, filepath):
            try:
                tmp_path = filepath + ".tmp"
                with open(tmp_path, 'w') as f:
                    json.dump(data, f)
                if os.path.exists(filepath): os.remove(filepath)
                os.rename(tmp_path, filepath)
                print(f"[Autosave] Saved to {filepath}")
            except Exception as e:
                print(f"[Autosave] Error: {e}")

        threading.Thread(target=save_task, args=(self.tracks.copy(), save_path), daemon=True).start()
        self._start_autosave_loop()

    def _get_autosave_path(self):
        if self.context and self.context.project_path:
            return os.path.join(self.context.project_path, self.AUTOSAVE_FILENAME)
        return self.AUTOSAVE_FILENAME

    def _check_for_autosave(self):
        path = self._get_autosave_path()
        if os.path.exists(path):
            if messagebox.askyesno("Recovery", "Found an autosave file from a previous session.\nDo you want to restore it?"):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    # JSON keys are strings, convert back to int
                    self.tracks = {int(k): v for k, v in data.items()}
                    
                    # Rebuild lineage
                    self.id_lineage = {}
                    for tid, d in self.tracks.items():
                        self.id_lineage[tid] = tid
                        for merged in d.get('merged_from', []):
                            self.id_lineage[merged] = tid
                            
                    self.refresh_tree()
                    self.show_frame()
                    print(f"Restored autosave: {path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Corrupt autosave file: {e}")
            else:
                try: os.remove(path)
                except: pass

    def _on_close(self):
        path = self._get_autosave_path()
        if os.path.exists(path):
            try: os.remove(path)
            except: pass
        self.parent.winfo_toplevel().destroy()

    # --- NEW FEATURE: SETTINGS DIALOG ---
    def open_settings_dialog(self):
        """Opens a popup window to modify hardcoded parameters."""
        win = tk.Toplevel(self.parent)
        win.title("Algorithm Settings")
        win.geometry("350x250")
        
        # Temporary variables
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
            messagebox.showinfo("Save", "Parameters updated successfully.")

        tk.Button(win, text="Save", command=save, bg="#4CAF50", fg="white").pack(pady=15)

    # --- UPDATED LOGIC WITH PARAMETERS ---

    def absorb_noise_logic(self):
        """Supervised Noise Absorption using configurable parameters."""
        if not messagebox.askyesno("Confirm", f"Absorb noise (Dist < {self.param_noise_dist}px)?"): return
        self._snapshot()
        
        main_tracks = [tid for tid, d in self.tracks.items() if d['role'] in self.cast]
        candidates = [tid for tid, d in self.tracks.items() if d['role'] not in self.cast]
        
        absorbed = 0
        changed = True
        
        # Dynamic parameters
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
        """Unsupervised Auto-Stitching using configurable parameters."""
        self._snapshot()
        # Retrieve parameters
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
        messagebox.showinfo("Info", f"Stitched {merged} fragments (Lookahead:{p_win}, Time:{p_time}s, Dist:{p_dist}px).")

    # --- STANDARD METHODS (Unchanged) ---
    def merge_all_by_role(self):
        if not messagebox.askyesno("Confirm", "Do you want to merge all tracks assigned to the same role?"): return
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
        if merge_count > 0: msg = f"Merged {merge_count} fragments for: {', '.join(roles_processed)}."
        else: msg = "No merges necessary."
        messagebox.showinfo("Merge Result", msg)

    def manual_merge(self):
        sel = self.tree.selection()
        if len(sel) < 2: return
        self._snapshot()
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

    def split_track_at_current_frame(self):
        """
        Splits a selected track into two at the current video frame.
        The second part of the track gets a new ID.
        """
        # 1. Validation
        selection = self.tree.selection()
        first = next(iter(selection), None)
        if first is None:
            messagebox.showwarning("Warning", "No track selected.")
            return
        track_id_to_split = int(first)

        
        if len(selection) > 1:
            messagebox.showwarning("Warning", "Select only one track to split.")
            return

        split_frame = self.current_frame
        
        track_data = self.tracks.get(track_id_to_split)
        if not track_data:
            return

        # Check if split_frame is a valid point
        if split_frame <= track_data['frames'][0]:
            messagebox.showinfo("Info", "You cannot split a track at its first frame or before.")
            return
        
        if split_frame > track_data['frames'][-1]:
            messagebox.showinfo("Info", "The split frame is beyond the end of the track.")
            return

        # 2. Data Slicing
        try:
            # Find the index of the first frame >= split_frame
            split_index = next(i for i, f in enumerate(track_data['frames']) if f >= split_frame)
        except StopIteration:
            return # Should be caught by the check above, but for safety

        # Part A (original track) and Part B (new track)
        original_frames, new_frames = track_data['frames'][:split_index], track_data['frames'][split_index:]
        original_boxes, new_boxes = track_data['boxes'][:split_index], track_data['boxes'][split_index:]

        if not new_frames:
            messagebox.showinfo("Info", "You cannot split a track at its last frame.")
            return

        # --- ASK USER PREFERENCE ---
        msg = (f"Splitting track {track_id_to_split} at frame {split_frame}.\n\n"
               "Which part should KEEP the original ID and Role?\n"
               "YES = The PREVIOUS part (up to the cursor)\n"
               "NO = The NEXT part (from the cursor onwards)")
        
        keep_head = messagebox.askyesno("Confirm Split", msg)

        self._snapshot()

        # 3. ID Generation & State Update
        new_track_id = max(self.tracks.keys()) + 1 if self.tracks else 1
        
        if keep_head:
            # Standard: Original ID stays with the PREVIOUS part
            self.tracks[track_id_to_split]['frames'], self.tracks[track_id_to_split]['boxes'] = original_frames, original_boxes
            self.tracks[new_track_id] = {'frames': new_frames, 'boxes': new_boxes, 'role': 'Ignore', 'merged_from': [new_track_id]}
            created_len = len(new_frames)
        else:
            # Swap: ID Originale passa alla parte SUCCESSIVA
            self.tracks[track_id_to_split]['frames'], self.tracks[track_id_to_split]['boxes'] = new_frames, new_boxes
            self.tracks[new_track_id] = {'frames': original_frames, 'boxes': original_boxes, 'role': 'Ignore', 'merged_from': [new_track_id]}
            created_len = len(original_frames)

        # FIX: Se la nuova traccia è breve (<1s) e il filtro è attivo, verrebbe nascosta.
        # Disattiviamo il filtro automaticamente per mostrare il risultato dell'operazione.
        if self.hide_short_var.get() and (created_len / self.fps) < 1.0:
            self.hide_short_var.set(False)

        self.refresh_tree()
        # messagebox.showinfo("Successo", f"Traccia {track_id_to_split} divisa.\nNuova traccia creata: ID {new_track_id}.")
        self.tree.selection_set(str(new_track_id)); self.tree.focus(str(new_track_id))

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

    def browse_video(self):
        v = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")]) 
        if v: self.load_data_direct(v, self.json_path)

    def browse_pose(self):
        j = filedialog.askopenfilename(filetypes=[("Pose JSON", "*.json.gz")])
        if j: self.load_data_direct(self.video_path, j)

    def load_data_direct(self, video_path, json_path):
        if video_path:
            self.video_path = video_path
            self.context.video_path = video_path
        
        if json_path:
            self.json_path = json_path
            self.context.pose_data_path = json_path

        if self.video_path and os.path.exists(self.video_path):
            self.cap = cv2.VideoCapture(self.video_path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.config(to=self.total_frames-1)
        
        if self.json_path and os.path.exists(self.json_path):
            # Async Load
            siblings = [c for c in self.right_panel.winfo_children() if c != self.progress]
            if siblings:
                self.progress.pack(fill=tk.X, pady=5, side=tk.TOP, before=siblings[0])
            else:
                self.progress.pack(fill=tk.X, pady=5, side=tk.TOP)
            self.progress.start(10)
            
            threading.Thread(target=self._load_json_thread, args=(self.json_path,), daemon=True).start()
            self.parent.after(100, self._check_load_queue)
        else:
            self.refresh_tree()
            self.show_frame()

    def _load_json_thread(self, path):
        try:
            tracks = {}
            id_lineage = {}
            has_untracked = False

            with gzip.open(path, 'rt', encoding='utf-8') as f:
                for line in f:
                    d = json.loads(line)
                    idx = d['f_idx']
                    for i, det in enumerate(d['det']):
                        tid = det.get('track_id')
                        
                        if tid is None: tid = -1
                        tid = int(tid)
                        
                        if tid == -1:
                            tid = 9000000 + (idx * 1000) + i
                            has_untracked = True
                        
                        if tid not in tracks:
                            tracks[tid] = {'frames':[], 'boxes':[], 'role':'Ignore', 'merged_from':[tid]}
                            id_lineage[tid] = tid
                        tracks[tid]['frames'].append(idx)
                        b=det['box']
                        tracks[tid]['boxes'].append([b['x1'],b['y1'],b['x2'],b['y2']])
            
            self.load_queue.put(("success", tracks, id_lineage, has_untracked))
        except Exception as e:
            self.load_queue.put(("error", str(e)))

    def _check_load_queue(self):
        try:
            msg = self.load_queue.get_nowait()
            status = msg[0]
            
            self.progress.stop()
            self.progress.pack_forget()
            
            if status == "success":
                _, tracks, lineage, has_untracked = msg
                self.tracks = tracks
                self.id_lineage = lineage
                
                if has_untracked:
                    self.hide_short_var.set(False)
                    print("Info: Untracked detections detected (ID -1). 'Hide short' disabled.")
                
                self.refresh_tree()
                self.show_frame()
            elif status == "error":
                messagebox.showerror("Error", f"Failed to load JSON: {msg[1]}")
                
        except queue.Empty:
            self.parent.after(100, self._check_load_queue)

    def load_mapping(self):
        if not self.tracks:
            messagebox.showwarning("Warning", "Load video and pose data before loading an identity mapping.")
            return

        f = filedialog.askopenfilename(filetypes=[("Identity JSON", "*.json")])
        if not f: return
        
        try:
            with open(f, 'r') as file:
                mapping = json.load(file)
            
            loaded_count = 0
            new_roles = set()
            
            for tid_str, role in mapping.items():
                tid = int(tid_str)
                if tid in self.tracks:
                    self.tracks[tid]['role'] = role
                    loaded_count += 1
                    if role not in self.cast and role != "Ignore":
                        new_roles.add(role)
            
            # Add new roles to the cast if they don't exist
            for role in new_roles:
                self.cast[role] = {"color": (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))}
            
            self.refresh_cast_list()
            self.refresh_tree()
            self.show_frame()
            
            # Update context
            if self.context:
                self.context.identity_map_path = f
                
            messagebox.showinfo("Loaded", f"Restored {loaded_count} assignments.\nNew roles added: {len(new_roles)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Unable to load the file:\n{e}")

    def save_mapping(self):
        if not self.json_path: return

        base_name = os.path.basename(self.json_path).replace(".json.gz", "_identity.json")
        if self.context and self.context.paths.get("output"):
            out = os.path.join(self.context.paths["output"], base_name)
        else:
            out = self.json_path.replace(".json.gz", "_identity.json")

        # --- CRITICAL FIX: Use id_lineage instead of tracks ---
        # This ensures that if ID 10 was merged into ID 5, 
        # the JSON will also include "10": "RoleOf5", not just "5": "RoleOf5".
        mapping = {}
        for original_id, current_master in self.id_lineage.items():
            if current_master in self.tracks:
                role = self.tracks[current_master]['role']
                if role != 'None':
                    mapping[original_id] = role

        with open(out, 'w') as f:
            json.dump(mapping, f, indent=4)

        if self.context:
            self.context.identity_map_path = out
            print(f"CONTEXT: Identity Map updated -> {out}")

        count = len(mapping)
        messagebox.showinfo("Done", f"Mapped {count} IDs (including merged historical IDs).\nSaved to: {out}")

    def refresh_cast_list(self):
        self.list_cast.delete(0, tk.END)
        for n in self.cast:
            self.list_cast.insert(tk.END, n)
            b,g,r = self.cast[n]['color']
            self.list_cast.itemconfig(self.list_cast.size()-1, bg='#{:02x}{:02x}{:02x}'.format(r,g,b))
            
    def add_person(self):
        n = simpledialog.askstring("New", "Name:")
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
        if not s: return

        n = self.list_cast.get(s[0])
        c = colorchooser.askcolor()
        
        if c and isinstance(c, tuple) and len(c) >= 2:
            rgb = c[0]
            if rgb:
                r,g,b = map(int, rgb)
                self.cast[n]['color'] = (b,g,r)
                self.refresh_cast_list(); self.show_frame()

    def show_context_menu(self, e):
        i = self.tree.identify_row(e.y)
        if i:
            if i not in self.tree.selection(): self.tree.selection_set(i)
            self.context_menu.delete(0, tk.END)
            for p in self.cast: self.context_menu.add_command(label=f"Assign to {p}", command=lambda n=p: self.assign_sel(n))
            self.context_menu.add_separator()
            self.context_menu.add_command(label="Remove Assignment", command=lambda: self.assign_sel("Ignore"))
            self.context_menu.add_separator()
            self.context_menu.add_command(label="🔗 Merge Selected", command=self.manual_merge)
            self.context_menu.add_command(label="✂️ Split at Current Frame", command=self.split_track_at_current_frame)
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
            self.slider.set(self.current_frame); self.show_frame(); self.parent.after(30, self.play_loop)
            
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
