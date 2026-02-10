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
    def __init__(self, max_history=20, ram_buffer=5):
        self.max_history = max_history
        self.ram_buffer = ram_buffer
        self.temp_dir = tempfile.mkdtemp(prefix="hermes_history_")
        self.undo_stack = []
        self.redo_stack = []
        self.current_state = None  # Stores the current snapshot entry

    def push_state(self, data):
        # 1. Create RAM Entry (Fast, no disk I/O)
        # Use pickle.dumps to create a deep copy in RAM (bytes)
        blob = pickle.dumps(data)
        timestamp = time.time_ns()
        new_entry = {'type': 'ram', 'blob': blob, 'timestamp': timestamp}

        # 2. Push current to undo stack
        if self.current_state:
            self.undo_stack.append(self.current_state)

        self.current_state = new_entry

        # 3. Clear Redo
        self._clear_stack(self.redo_stack)
        self.redo_stack = []

        # 4. Manage RAM Buffer (Spill to disk if needed)
        # We keep 'ram_buffer' amount of states in RAM (including current)
        # The undo_stack has older states.
        
        # Total items in RAM we want = self.ram_buffer.
        # self.current_state is in RAM.
        # So we can keep (self.ram_buffer - 1) items in undo_stack as RAM.
        
        ram_slots_for_stack = self.ram_buffer - 1
        if len(self.undo_stack) > ram_slots_for_stack:
            # The item at this index is pushing out of the RAM window
            idx_to_spill = len(self.undo_stack) - 1 - ram_slots_for_stack
            if idx_to_spill >= 0:
                self._spill_to_disk(self.undo_stack[idx_to_spill])

        # 5. Enforce Max History (Delete oldest)
        if len(self.undo_stack) > self.max_history:
            oldest = self.undo_stack.pop(0)
            self._delete_entry(oldest)

    def undo(self):
        if not self.undo_stack:
            return None

        # Current becomes Redo
        if self.current_state:
            self.redo_stack.append(self.current_state)

        # Pop Undo becomes Current
        self.current_state = self.undo_stack.pop()
        return self._load_entry(self.current_state)

    def redo(self):
        if not self.redo_stack:
            return None

        # Current becomes Undo
        if self.current_state:
            self.undo_stack.append(self.current_state)

        # Pop Redo becomes Current
        self.current_state = self.redo_stack.pop()
        return self._load_entry(self.current_state)

    def _spill_to_disk(self, entry):
        if entry['type'] == 'ram':
            filename = os.path.join(self.temp_dir, f"state_{entry['timestamp']}.pkl")
            try:
                with open(filename, 'wb') as f:
                    f.write(entry['blob'])
                entry['type'] = 'disk'
                entry['path'] = filename
                entry['blob'] = None # Free RAM
            except Exception as e:
                print(f"History Spill Error: {e}")

    def _load_entry(self, entry):
        try:
            if entry['type'] == 'ram':
                return pickle.loads(entry['blob'])
            else:
                with open(entry['path'], 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"History Load Error: {e}")
            return None

    def _delete_entry(self, entry):
        if entry['type'] == 'disk' and os.path.exists(entry['path']):
            try: os.remove(entry['path'])
            except: pass
        # If RAM, just let GC handle it

    def _clear_stack(self, stack):
        for entry in stack:
            self._delete_entry(entry)
        # Clear list in place
        del stack[:]
            
    def __del__(self):
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass

class IdentityLogic:
    """
    Encapsulates all business logic for identity management, including
    data loading, track manipulation (merge, split), and algorithmic processing.
    This class is completely decoupled from the UI (Tkinter).
    """
    def __init__(self, fps=30.0):
        self.tracks = {}
        self.id_lineage = {}
        self.audit_log = []
        self.fps = fps
        self.lock = threading.RLock()

    def set_fps(self, fps):
        self.fps = fps

    def get_data(self):
        with self.lock:
            return self.tracks.copy(), self.id_lineage.copy()

    def get_data_snapshot(self):
        with self.lock:
            return self.tracks.copy(), self.id_lineage.copy()

    def set_data(self, tracks, id_lineage):
        with self.lock:
            self.tracks = tracks
            self.id_lineage = id_lineage

    def _log_operation(self, action, details):
        entry = {
            "timestamp": time.time(),
            "action": action,
            "details": details
        }
        self.audit_log.append(entry)

    def get_audit_log(self):
        with self.lock:
            return list(self.audit_log)

    def set_audit_log(self, log):
        with self.lock:
            self.audit_log = log

    def load_from_json_gz(self, path):
        """Loads and parses track data from a YOLO .json.gz file."""
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
                    b = det['box']
                    tracks[tid]['boxes'].append([b['x1'], b['y1'], b['x2'], b['y2']])
        
        with self.lock:
            self.tracks = tracks
            self.id_lineage = id_lineage
            self.audit_log = []
            self._log_operation("Load Data", {"path": path, "track_count": len(tracks)})
        return has_untracked

    def assign_role_to_ids(self, ids, role):
        with self.lock:
            for i in ids:
                self.tracks[int(i)]['role'] = role
            self._log_operation("Assign Role", {"ids": ids, "role": role})

    def merge_logic(self, master, slave):
        with self.lock:
            if slave not in self.tracks or master not in self.tracks: return
            self.tracks[master]['frames'].extend(self.tracks[slave]['frames'])
            self.tracks[master]['boxes'].extend(self.tracks[slave]['boxes'])
            self.tracks[master]['merged_from'].extend(self.tracks[slave]['merged_from'])
            for oid, curr in self.id_lineage.items():
                if curr == slave: self.id_lineage[oid] = master
            del self.tracks[slave]
            z = sorted(zip(self.tracks[master]['frames'], self.tracks[master]['boxes']), key=lambda x: x[0])
            self.tracks[master]['frames'] = [x[0] for x in z]
            self.tracks[master]['boxes'] = [x[1] for x in z]

    def manual_merge(self, ids, valid_roles=None):
        with self.lock:
            ids = sorted([int(x) for x in ids])
            master = ids[0]
            
            # Resolve role priority
            final_role = self.tracks[master]['role']
            if valid_roles:
                for s in ids[1:]:
                    if s in self.tracks:
                        s_role = self.tracks[s]['role']
                        if final_role not in valid_roles and s_role in valid_roles:
                            final_role = s_role

            for s in ids[1:]:
                self.merge_logic(master, s)
            self.tracks[master]['role'] = final_role
            self._log_operation("Manual Merge", {"master": master, "merged_ids": ids, "final_role": final_role})
            return master

    def merge_all_by_role(self, cast):
        with self.lock:
            merge_count = 0
            roles_processed = []
            for person_name in cast:
                ids_with_role = [tid for tid, data in self.tracks.items() if data['role'] == person_name]
                ids_with_role.sort()
                if len(ids_with_role) > 1:
                    master_id = ids_with_role[0]
                    for slave_id in ids_with_role[1:]:
                        self.merge_logic(master_id, slave_id)
                        merge_count += 1
                    roles_processed.append(person_name)
            self._log_operation("Merge All By Role", {"merged_count": merge_count, "roles": roles_processed})
            return merge_count, roles_processed

    def split_track(self, track_id_to_split, split_frame, keep_head):
        with self.lock:
            track_data = self.tracks.get(track_id_to_split)
            if not track_data: return None, "Track not found"

            if split_frame <= track_data['frames'][0]: return None, "Cannot split at or before the first frame."
            if split_frame > track_data['frames'][-1]: return None, "Split frame is beyond the end of the track."

            try:
                split_index = next(i for i, f in enumerate(track_data['frames']) if f >= split_frame)
            except StopIteration:
                return None, "Split frame not found in track."

            original_frames, new_frames = track_data['frames'][:split_index], track_data['frames'][split_index:]
            original_boxes, new_boxes = track_data['boxes'][:split_index], track_data['boxes'][split_index:]

            if not new_frames: return None, "Cannot split at the last frame."

            new_track_id = max(self.tracks.keys()) + 1 if self.tracks else 1
            
            if keep_head:
                self.tracks[track_id_to_split]['frames'], self.tracks[track_id_to_split]['boxes'] = original_frames, original_boxes
                self.tracks[new_track_id] = {'frames': new_frames, 'boxes': new_boxes, 'role': 'Ignore', 'merged_from': [new_track_id]}
                self.id_lineage[new_track_id] = new_track_id
                created_len = len(new_frames)
            else:
                self.tracks[track_id_to_split]['frames'], self.tracks[track_id_to_split]['boxes'] = new_frames, new_boxes
                self.tracks[new_track_id] = {'frames': original_frames, 'boxes': original_boxes, 'role': 'Ignore', 'merged_from': [new_track_id]}
                self.id_lineage[new_track_id] = new_track_id
                created_len = len(original_frames)
            
            self._log_operation("Split Track", {"original_id": track_id_to_split, "new_id": new_track_id, "frame": split_frame, "keep_head": keep_head})
            return new_track_id, created_len

    def auto_stitch(self, lookahead, time_gap, stitch_dist):
        with self.lock:
            merged = 0
            changed = True
            while changed:
                changed = False
                curr_ids = sorted(self.tracks.keys(), key=lambda x: min(self.tracks[x]['frames']))
                i = 0
                while i < len(curr_ids) - 1:
                    id_a = curr_ids[i]
                    best_match, min_dist = None, float('inf')
                    search_limit = min(i + lookahead, len(curr_ids))
                    for j in range(i + 1, search_limit):
                        id_b = curr_ids[j]
                        t_a, t_b = self.tracks[id_a], self.tracks[id_b]
                        gap = t_b['frames'][0] - t_a['frames'][-1]
                        if not (0 < gap <= (time_gap * self.fps)): continue
                        ba, bb = t_a['boxes'][-1], t_b['boxes'][0]
                        ca, cb = ((ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2), ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
                        d = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
                        if d < stitch_dist and d < min_dist:
                            min_dist, best_match = d, id_b
                    if best_match:
                        self.merge_logic(id_a, best_match)
                        merged += 1; changed = True; break
                    i += 1
            self._log_operation("Auto Stitch", {"merged_count": merged, "params": {"lookahead": lookahead, "time_gap": time_gap, "stitch_dist": stitch_dist}})
            return merged

    def absorb_noise(self, cast, noise_dist, time_gap):
        with self.lock:
            main_tracks = [tid for tid, d in self.tracks.items() if d['role'] in cast]
            candidates = [tid for tid, d in self.tracks.items() if d['role'] not in cast]
            absorbed, changed = 0, True
            MAX_DIST, MAX_TIME_GAP = noise_dist, time_gap * self.fps
            while changed:
                changed = False
                for main_id in list(main_tracks):
                    if main_id not in self.tracks: continue
                    main_data = self.tracks[main_id]
                    main_frames = sorted(main_data['frames'])
                    to_remove = []
                    for cand_id in candidates:
                        if cand_id not in self.tracks or not set(main_frames).isdisjoint(self.tracks[cand_id]['frames']): continue
                        cand_data = self.tracks[cand_id]
                        c_frames = cand_data['frames']
                        gap_after, dist_after = c_frames[0] - main_frames[-1], float('inf')
                        if 0 < gap_after < MAX_TIME_GAP:
                            end_m, start_c = main_data['boxes'][main_data['frames'].index(main_frames[-1])], cand_data['boxes'][0]
                            cm, cc = ((end_m[0] + end_m[2]) / 2, (end_m[1] + end_m[3]) / 2), ((start_c[0] + start_c[2]) / 2, (start_c[1] + start_c[3]) / 2)
                            dist_after = math.hypot(cm[0] - cc[0], cm[1] - cc[1])
                        gap_before, dist_before = main_frames[0] - c_frames[-1], float('inf')
                        if 0 < gap_before < MAX_TIME_GAP:
                            end_c, start_m = cand_data['boxes'][-1], main_data['boxes'][main_data['frames'].index(main_frames[0])]
                            cc, cm = ((end_c[0] + end_c[2]) / 2, (end_c[1] + end_c[3]) / 2), ((start_m[0] + start_m[2]) / 2, (start_m[1] + start_m[3]) / 2)
                            dist_before = math.hypot(cm[0] - cc[0], cm[1] - cc[1])
                        if dist_after < MAX_DIST or dist_before < MAX_DIST:
                            self.merge_logic(main_id, cand_id)
                            absorbed += 1; changed = True; to_remove.append(cand_id)
                            main_frames = sorted(self.tracks[main_id]['frames'])
                    for c in to_remove:
                        if c in candidates: candidates.remove(c)
            self._log_operation("Absorb Noise", {"absorbed_count": absorbed, "params": {"noise_dist": noise_dist, "time_gap": time_gap}})
            return absorbed

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
        self.logic = IdentityLogic(self.fps)
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
        tk.Button(tools, text="📜 Log", command=self.show_audit_log_window).pack(side=tk.RIGHT, padx=5)
        
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
        # Ensure view is visible and user is not typing
        if not self.parent.winfo_viewable(): return False
        focused = self.parent.focus_get()
        if focused and focused.winfo_class() in ['Entry', 'Text', 'Spinbox', 'TEntry']:
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
                self.assign_role_to_selection(self.list_cast.get(idx))
        except ValueError: pass

    def _snapshot(self):
        tracks, id_lineage = self.logic.get_data()
        self.history.push_state((tracks, id_lineage))

    def perform_undo(self, event=None):
        prev_state = self.history.undo()
        if prev_state is not None:
            tracks, id_lineage = prev_state
            self.logic.set_data(tracks, id_lineage)
            self.refresh_tree()
            self.show_frame()
            print("Undo performed.")
    
    def _start_autosave_loop(self):
        try:
            self.parent.after(self.AUTOSAVE_INTERVAL_MS, self._perform_autosave)
        except Exception:
            pass # Widget destroyed

    def _perform_autosave(self):
        tracks, lineage = self.logic.get_data_snapshot()
        audit_log = self.logic.get_audit_log()
        if not tracks:
            self._start_autosave_loop()
            return

        save_path = self._get_autosave_path()

        def save_task(data_to_save, filepath):
            try:
                tmp_path = filepath + ".tmp"
                with open(tmp_path, 'w') as f:
                    json.dump(data_to_save, f)
                if os.path.exists(filepath): os.remove(filepath)
                os.rename(tmp_path, filepath)
                print(f"[Autosave] Saved to {filepath}")
            except Exception as e:
                print(f"[Autosave] Error: {e}")

        data_to_save = {
            'tracks': tracks,
            'id_lineage': lineage,
            'audit_log': audit_log
        }
        threading.Thread(target=save_task, args=(data_to_save, save_path), daemon=True).start()
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
                        saved_data = json.load(f)

                    # Handle old (dict) and new ({'tracks':...}) format
                    if 'tracks' in saved_data and 'id_lineage' in saved_data:
                        tracks_data, lineage_data = saved_data['tracks'], saved_data['id_lineage']
                        audit_data = saved_data.get('audit_log', [])
                    else:
                        tracks_data, lineage_data = saved_data, {}
                        audit_data = []

                    tracks = {int(k): v for k, v in tracks_data.items()}
                    id_lineage = {int(k): v for k, v in lineage_data.items()}

                    # For backward compatibility, rebuild lineage if not present
                    if not id_lineage:
                        for tid, d in tracks.items():
                            id_lineage[tid] = tid
                            for merged in d.get('merged_from', []):
                                id_lineage[merged] = tid
                    
                    self.logic.set_data(tracks, id_lineage)
                    if audit_data:
                        self.logic.set_audit_log(audit_data)

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
        if not messagebox.askyesno("Confirm", f"Absorb noise (Dist < {self.param_noise_dist}px)? This cannot be undone in one step."): return
        self._snapshot()
        
        absorbed = self.logic.absorb_noise(self.cast, self.param_noise_dist, self.param_time_gap)
        
        self.refresh_tree()
        messagebox.showinfo("Info", f"Assorbiti {absorbed} frammenti.")

    def auto_stitch(self):
        """Unsupervised Auto-Stitching using configurable parameters."""
        if not messagebox.askyesno("Confirm", "Run auto-stitching? This may merge unrelated tracks and cannot be undone in one step."): return
        self._snapshot()
        p_win = self.param_lookahead
        p_time = self.param_time_gap
        p_dist = self.param_stitch_dist
        
        merged = self.logic.auto_stitch(p_win, p_time, p_dist)
        
        self.refresh_tree()
        messagebox.showinfo("Info", f"Stitched {merged} fragments (Lookahead:{p_win}, Time:{p_time}s, Dist:{p_dist}px).")

    # --- STANDARD METHODS (Unchanged) ---
    def merge_all_by_role(self):
        if not messagebox.askyesno("Confirm", "Do you want to merge all tracks assigned to the same role?"): return
        self._snapshot()
        merge_count, roles_processed = self.logic.merge_all_by_role(self.cast)
        self.refresh_tree()
        if merge_count > 0: msg = f"Merged {merge_count} fragments for: {', '.join(roles_processed)}."
        else: msg = "No merges necessary."
        messagebox.showinfo("Merge Result", msg)

    def manual_merge(self):
        sel = self.tree.selection()
        if len(sel) < 2: return
        self._snapshot()
        
        master = self.logic.manual_merge(sel, self.cast)

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
        
        track_data = self.logic.tracks.get(track_id_to_split)
        if not track_data: return

        # Early exit checks are now inside the logic, but we can keep them here for UI feedback
        if split_frame <= track_data['frames'][0] or split_frame > track_data['frames'][-1]:
            messagebox.showinfo("Info", "Cannot split at the very start or after the end of a track.")
            return

        # 2. Data Slicing
        try:
            # Find the index of the first frame >= split_frame
            split_index = next(i for i, f in enumerate(track_data['frames']) if f >= split_frame)
        except StopIteration:
            return # Should be caught by the check above, but for safety

        # --- ASK USER PREFERENCE ---
        msg = (f"Splitting track {track_id_to_split} at frame {split_frame}.\n\n"
               "Which part should KEEP the original ID and Role?\n"
               "YES = The PREVIOUS part (up to the cursor)\n"
               "NO = The NEXT part (from the cursor onwards)")
        keep_head = messagebox.askyesno("Confirm Split", msg)

        self._snapshot()

        # 3. Delegate to logic
        new_track_id, created_len_or_msg = self.logic.split_track(track_id_to_split, split_frame, keep_head)

        if new_track_id is None:
            messagebox.showerror("Split Error", str(created_len_or_msg))
            return

        created_len = int(created_len_or_msg)
        if self.hide_short_var.get() and (created_len / self.fps) < 1.0:
            self.hide_short_var.set(False)

        self.refresh_tree()
        self.tree.selection_set(str(new_track_id)); self.tree.focus(str(new_track_id))

    def show_audit_log_window(self):
        logs = self.logic.get_audit_log()
        win = tk.Toplevel(self.parent)
        win.title("Audit Log Explorer")
        win.geometry("600x400")

        sb = ttk.Scrollbar(win)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        text = tk.Text(win, wrap="word", yscrollcommand=sb.set, font=("Consolas", 9))
        text.pack(side=tk.LEFT, fill="both", expand=True)
        sb.config(command=text.yview)

        for entry in reversed(logs): # Dal più recente
            ts = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
            text.insert("end", f"[{ts}] {entry['action']}\n")
            text.insert("end", f"   Details: {entry['details']}\n\n")
        
        text.config(state="disabled")

    def refresh_tree(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        hide = self.hide_short_var.get()
        for tid in sorted(self.logic.tracks.keys()):
            d = self.logic.tracks[tid]
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
            self.logic.set_fps(self.fps)
            self.slider.config(to=self.total_frames-1)
        
        if self.json_path and os.path.exists(self.json_path):
            # Async Load
            siblings = [c for c in self.right_panel.winfo_children() if c != self.progress]
            if siblings:
                self.progress.pack(fill=tk.X, pady=5, side=tk.TOP, before=siblings[0])
            else:
                self.progress.pack(fill=tk.X, pady=5, side=tk.TOP)
            self.progress.start(10)
            threading.Thread(target=self._load_json_thread_refactored, args=(self.json_path,), daemon=True).start()
            self.parent.after(100, self._check_load_queue)
        else:
            self.refresh_tree()
            self.show_frame()

    def _load_json_thread_refactored(self, path):
        try:
            has_untracked = self.logic.load_from_json_gz(path)
            self.load_queue.put(("success", has_untracked))
        except Exception as e:
            self.load_queue.put(("error", str(e)))

    def _check_load_queue(self):
        try:
            msg = self.load_queue.get_nowait()
            status = msg[0]
            
            self.progress.stop()
            self.progress.pack_forget()
            
            if status == "success":
                _, has_untracked = msg
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
        if not self.logic.tracks:
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
                if tid in self.logic.tracks:
                    self.logic.tracks[tid]['role'] = role
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
        mapping = {} # Using logic's data now
        for original_id, current_master in self.logic.id_lineage.items():
            if current_master in self.logic.tracks:
                role = self.logic.tracks[current_master]['role']
                if role != 'Ignore':
                    mapping[original_id] = role

        with open(out, 'w') as f:
            json.dump(mapping, f, indent=4)

        if self.context:
            self.context.identity_map_path = out
            print(f"CONTEXT: Identity Map updated -> {out}")

        # --- AUDIT LOG SAVE ---
        audit_out = out.replace("_identity.json", "_audit.json")
        try:
            with open(audit_out, 'w') as f:
                json.dump(self.logic.get_audit_log(), f, indent=4)
            print(f"Audit Log saved -> {audit_out}")
        except Exception as e:
            print(f"Error saving audit log: {e}")

        count = len(mapping)
        messagebox.showinfo("Done", f"Mapped {count} IDs (including merged historical IDs).\nSaved to: {out}\nAudit Log: {os.path.basename(audit_out)}")

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
            for t in self.logic.tracks.values():
                if t['role'] == n: t['role'] = 'Ignore'
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
            for p in self.cast: self.context_menu.add_command(label=f"Assign to {p}", command=lambda n=p: self.assign_role_to_selection(n))
            self.context_menu.add_separator()
            self.context_menu.add_command(label="Remove Assignment", command=lambda: self.assign_role_to_selection("Ignore"))
            self.context_menu.add_separator()
            self.context_menu.add_command(label="🔗 Merge Selected", command=self.manual_merge)
            self.context_menu.add_command(label="✂️ Split at Current Frame", command=self.split_track_at_current_frame)
            self.context_menu.post(e.x_root, e.y_root)

    def assign_role_to_selection(self, role):
        selected_ids = [int(i) for i in self.tree.selection()]
        self.logic.assign_role_to_ids(selected_ids, role)
        self.refresh_tree(); self.show_frame()

    def on_tree_select(self, e):
        s = self.tree.selection()
        if s: self.current_frame = self.logic.tracks[int(s[0])]['frames'][0]; self.slider.set(self.current_frame); self.show_frame()

    def on_seek(self, v): self.current_frame = int(float(v)); self.show_frame()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_loop()

    def play_loop(self):
        if self.is_playing and self.cap:
            start_t = time.time()
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.is_playing = False
                return
            self.slider.set(self.current_frame); self.show_frame()
            
            dt = (time.time() - start_t) * 1000
            wait = max(1, int((1000/self.fps) - dt))
            self.parent.after(wait, self.play_loop)
            
    def show_frame(self):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret: return
        hide = self.hide_short_var.get()
        for tid, d in self.logic.tracks.items():
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
