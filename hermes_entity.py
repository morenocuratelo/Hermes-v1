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
from typing import Optional, Dict, List, Tuple, Any, Set


class HistoryManager:
    def __init__(self, max_history: int = 20, ram_buffer: int = 5):
        self.max_history: int = max_history
        self.ram_buffer: int = ram_buffer
        self.temp_dir: str = tempfile.mkdtemp(prefix="hermes_history_")
        self.undo_stack: List[dict] = []
        self.redo_stack: List[dict] = []
        self.current_state: Optional[dict] = None

    def push_state(self, data: Any) -> None:
        blob = pickle.dumps(data)
        timestamp = time.time_ns()
        new_entry = {'type': 'ram', 'blob': blob, 'timestamp': timestamp}

        if self.current_state:
            self.undo_stack.append(self.current_state)

        self.current_state = new_entry

        self._clear_stack(self.redo_stack)
        self.redo_stack = []

        ram_slots_for_stack = self.ram_buffer - 1
        if len(self.undo_stack) > ram_slots_for_stack:
            idx_to_spill = len(self.undo_stack) - 1 - ram_slots_for_stack
            if idx_to_spill >= 0:
                self._spill_to_disk(self.undo_stack[idx_to_spill])

        if len(self.undo_stack) > self.max_history:
            oldest = self.undo_stack.pop(0)
            self._delete_entry(oldest)

    def undo(self) -> Optional[Any]:
        if not self.undo_stack:
            return None

        if self.current_state:
            self.redo_stack.append(self.current_state)

        self.current_state = self.undo_stack.pop()
        return self._load_entry(self.current_state)

    def redo(self) -> Optional[Any]:
        if not self.redo_stack:
            return None

        if self.current_state:
            self.undo_stack.append(self.current_state)

        self.current_state = self.redo_stack.pop()
        return self._load_entry(self.current_state)

    def _spill_to_disk(self, entry: dict) -> None:
        if entry['type'] == 'ram':
            filename = os.path.join(self.temp_dir, f"state_{entry['timestamp']}.pkl")
            try:
                with open(filename, 'wb') as f:
                    f.write(entry['blob'])
                entry['type'] = 'disk'
                entry['path'] = filename
                entry['blob'] = None
            except Exception as e:
                print(f"History Spill Error: {e}")

    def _load_entry(self, entry: dict) -> Optional[Any]:
        try:
            if entry['type'] == 'ram':
                return pickle.loads(entry['blob'])
            else:
                with open(entry['path'], 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"History Load Error: {e}")
            return None

    def _delete_entry(self, entry: dict) -> None:
        if entry['type'] == 'disk' and os.path.exists(entry['path']):
            try:
                os.remove(entry['path'])
            except OSError:
                pass

    def _clear_stack(self, stack: List[dict]) -> None:
        for entry in stack:
            self._delete_entry(entry)
        del stack[:]
            
    def __del__(self) -> None:
        try:
            temp_dir = getattr(self, "temp_dir", None)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except OSError:
            pass


class IdentityLogic:
    """
    Encapsulates all business logic for identity management, including
    data loading, track manipulation (merge, split), and algorithmic processing.
    This class is completely decoupled from the UI (Tkinter).
    """
    def __init__(self, fps: float = 30.0):
        self.tracks: Dict[int, dict] = {}
        self.id_lineage: Dict[int, int] = {}
        self.audit_log: List[dict] = []
        self.fps: float = fps
        self.lock: threading.RLock = threading.RLock()

    def set_fps(self, fps: float) -> None:
        self.fps = fps

    def get_data(self) -> Tuple[Dict[int, dict], Dict[int, int]]:
        with self.lock:
            return self.tracks.copy(), self.id_lineage.copy()

    def get_data_snapshot(self) -> Tuple[Dict[int, dict], Dict[int, int]]:
        with self.lock:
            return self.tracks.copy(), self.id_lineage.copy()

    def set_data(self, tracks: Dict[int, dict], id_lineage: Dict[int, int]) -> None:
        with self.lock:
            self.tracks = tracks
            self.id_lineage = id_lineage

    def _log_operation(self, action: str, details: dict) -> None:
        entry = {
            "timestamp": time.time(),
            "action": action,
            "details": details
        }
        self.audit_log.append(entry)

    def get_audit_log(self) -> List[dict]:
        with self.lock:
            return list(self.audit_log)

    def set_audit_log(self, log: List[dict]) -> None:
        with self.lock:
            self.audit_log = log

    def load_from_json_gz(self, path: str) -> bool:
        """Loads and parses track data from a YOLO .json.gz file.
        Uses a temporary dict so partial/corrupt files don't pollute state."""
        tmp_tracks: Dict[int, dict] = {}
        tmp_lineage: Dict[int, int] = {}
        has_untracked = False

        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                idx = d['f_idx']
                for i, det in enumerate(d['det']):
                    tid = det.get('track_id')
                    if tid is None:
                        tid = -1
                    tid = int(tid)
                    
                    if tid == -1:
                        tid = 9000000 + (idx * 1000) + i
                        has_untracked = True
                    
                    if tid not in tmp_tracks:
                        tmp_tracks[tid] = {'frames':[], 'boxes':[], 'role':'Ignore', 'merged_from':[tid]}
                        tmp_lineage[tid] = tid
                    tmp_tracks[tid]['frames'].append(idx)
                    b = det['box']
                    tmp_tracks[tid]['boxes'].append([b['x1'], b['y1'], b['x2'], b['y2']])
        
        # Only commit if full parse succeeded (no exception above)
        with self.lock:
            self.tracks = tmp_tracks
            self.id_lineage = tmp_lineage
            self.audit_log = []
            self._log_operation("Load Data", {"path": path, "track_count": len(tmp_tracks)})
        return has_untracked

    def assign_role_to_ids(self, ids: List[int], role: str) -> None:
        with self.lock:
            for i in ids:
                self.tracks[int(i)]['role'] = role
            self._log_operation("Assign Role", {"ids": ids, "role": role})

    def merge_logic(self, master: int, slave: int) -> None:
        with self.lock:
            if slave not in self.tracks or master not in self.tracks:
                return
            self.tracks[master]['frames'].extend(self.tracks[slave]['frames'])
            self.tracks[master]['boxes'].extend(self.tracks[slave]['boxes'])
            self.tracks[master]['merged_from'].extend(self.tracks[slave]['merged_from'])
            for oid, curr in self.id_lineage.items():
                if curr == slave:
                    self.id_lineage[oid] = master
            del self.tracks[slave]
            z = sorted(zip(self.tracks[master]['frames'], self.tracks[master]['boxes']), key=lambda x: x[0])
            self.tracks[master]['frames'] = [x[0] for x in z]
            self.tracks[master]['boxes'] = [x[1] for x in z]

    def manual_merge(self, ids: List[Any], valid_roles: Optional[Dict] = None) -> int:
        with self.lock:
            ids = sorted([int(x) for x in ids])
            master = ids[0]
            
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

    def merge_all_by_role(self, cast: Dict[str, dict]) -> Tuple[int, List[str]]:
        with self.lock:
            merge_count = 0
            roles_processed: List[str] = []
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

    def split_track(self, track_id: int, split_frame: int, keep_head: bool) -> Tuple[Optional[int], Any]:
        with self.lock:
            if track_id not in self.tracks:
                return None, "Track not found"
            
            data = self.tracks[track_id]
            frames = data['frames']
            boxes = data['boxes']
            
            # Find split index
            try:
                split_idx = next(i for i, f in enumerate(frames) if f >= split_frame)
            except StopIteration:
                return None, "Split frame out of bounds"
            
            if split_idx == 0:
                return None, "Cannot split at start"
            
            # Generate new ID
            max_id = max(self.tracks.keys()) if self.tracks else 0
            new_id = max_id + 1
            
            head_frames = frames[:split_idx]
            head_boxes = boxes[:split_idx]
            tail_frames = frames[split_idx:]
            tail_boxes = boxes[split_idx:]
            
            if keep_head:
                # Original keeps head
                self.tracks[track_id]['frames'] = head_frames
                self.tracks[track_id]['boxes'] = head_boxes
                
                # New gets tail
                self.tracks[new_id] = {'frames': tail_frames, 'boxes': tail_boxes, 'role': 'Ignore', 'merged_from': []}
                self.id_lineage[new_id] = new_id
                self._log_operation("Split Track", {"original": track_id, "new": new_id, "split_frame": split_frame, "kept": "head"})
                return new_id, len(tail_frames)
            else:
                # Original keeps tail
                self.tracks[track_id]['frames'] = tail_frames
                self.tracks[track_id]['boxes'] = tail_boxes
                
                # New gets head
                self.tracks[new_id] = {'frames': head_frames, 'boxes': head_boxes, 'role': 'Ignore', 'merged_from': []}
                self.id_lineage[new_id] = new_id
                self._log_operation("Split Track", {"original": track_id, "new": new_id, "split_frame": split_frame, "kept": "tail"})
                return new_id, len(head_frames)

    def auto_stitch(self, lookahead: int, time_gap: float, stitch_dist: float) -> int:
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
                        if not (0 < gap <= (time_gap * self.fps)):
                            continue
                        ba, bb = t_a['boxes'][-1], t_b['boxes'][0]
                        ca, cb = ((ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2), ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
                        d = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
                        if d < stitch_dist and d < min_dist:
                            min_dist, best_match = d, id_b
                    if best_match:
                        self.merge_logic(id_a, best_match)
                        merged += 1
                        changed = True
                        break
                    i += 1
            self._log_operation("Auto Stitch", {"merged_count": merged, "params": {"lookahead": lookahead, "time_gap": time_gap, "stitch_dist": stitch_dist}})
            return merged

    def absorb_noise(self, cast: Dict[str, dict], noise_dist: float, time_gap: float) -> int:
        with self.lock:
            main_tracks = [tid for tid, d in self.tracks.items() if d['role'] in cast]
            candidates = [tid for tid, d in self.tracks.items() if d['role'] not in cast]
            absorbed, changed = 0, True
            MAX_DIST, MAX_TIME_GAP = noise_dist, time_gap * self.fps
            while changed:
                changed = False
                for main_id in list(main_tracks):
                    if main_id not in self.tracks:
                        continue
                    main_data = self.tracks[main_id]
                    main_frames = sorted(main_data['frames'])
                    to_remove: List[int] = []
                    for cand_id in candidates:
                        if cand_id not in self.tracks or not set(main_frames).isdisjoint(self.tracks[cand_id]['frames']):
                            continue
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
                            absorbed += 1
                            changed = True
                            to_remove.append(cand_id)
                            main_frames = sorted(self.tracks[main_id]['frames'])
                    for c in to_remove:
                        if c in candidates:
                            candidates.remove(c)
            self._log_operation("Absorb Noise", {"absorbed_count": absorbed, "params": {"noise_dist": noise_dist, "time_gap": time_gap}})
            return absorbed

    def get_track_at_point(self, frame: int, x: int, y: int) -> Optional[int]:
        """Returns the track ID whose bounding box contains (x, y) at the given frame.
        If multiple overlap, returns the smallest box (most specific)."""
        best_id: Optional[int] = None
        best_area = float('inf')
        for tid, d in self.tracks.items():
            if frame in d['frames']:
                idx = d['frames'].index(frame)
                bx1, by1, bx2, by2 = d['boxes'][idx]
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    area = (bx2 - bx1) * (by2 - by1)
                    if area < best_area:
                        best_area = area
                        best_id = tid
        return best_id


class IdentityView:
    def __init__(self, parent: tk.Widget, context: Any):
        self.parent: tk.Widget = parent
        self.context = context
        
        # DATI
        self.video_path: Optional[str] = None
        self.json_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.logic: IdentityLogic = IdentityLogic(self.fps)
        self.history: HistoryManager = HistoryManager()
        self.load_queue: queue.Queue = queue.Queue()
        
        # --- PARAMETRI CONFIGURABILI ---
        self.param_lookahead: int = 15      
        self.param_time_gap: float = 2.0      
        self.param_stitch_dist: int = 150   
        self.param_noise_dist: int = 100    

        # CAST
        if self.context.cast:
            self.cast: Dict[str, dict] = self.context.cast
        else:
            self.cast = {
                "Target": {"color": (0, 255, 0)},       
                "Confederate_1": {"color": (0, 0, 255)}, 
                "Confederate_2": {"color": (255, 0, 0)}  
            }
            self.context.cast = self.cast

        self.hide_short_var: tk.BooleanVar = tk.BooleanVar(value=True)
        self.current_frame: int = 0
        self.total_frames: int = 0
        self.is_playing: bool = False

        # Video display scaling state (for click-to-select)
        self._video_offset_x: int = 0
        self._video_offset_y: int = 0
        self._video_scale: float = 1.0
        self._video_orig_w: int = 0
        self._video_orig_h: int = 0

        self._setup_ui()
        self._setup_hotkeys()
        
        # AUTO-LOAD
        if self.context.video_path and self.context.pose_data_path:
            self.load_data_direct(self.context.video_path, self.context.pose_data_path)

        # --- AUTOSAVE INIT ---
        self.AUTOSAVE_INTERVAL_MS: int = 300000
        self.AUTOSAVE_FILENAME: str = "hermes_autosave_identity.json"
        self._check_for_autosave()
        self._start_autosave_loop()
        
        self._toplevel = self.parent.winfo_toplevel()
        self._toplevel.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self) -> None:
        tk.Label(self.parent, text="2. Identity Assignment", font=("Segoe UI", 18, "bold"), bg="white").pack(pady=(0, 10), anchor="w")

        main = tk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # 1. VIDEO (SX)
        left = tk.Frame(main, bg="black")
        main.add(left, minsize=900)
        self.lbl_video = tk.Label(left, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)
        # Bind click on video for synchronized selection
        self.lbl_video.bind("<Button-1>", self._on_video_click)
        
        # --- TIMELINE CANVAS ---
        self.timeline_canvas = tk.Canvas(left, height=60, bg="#1e1e1e", highlightthickness=0)
        self.timeline_canvas.pack(fill=tk.X, side=tk.BOTTOM)
        self.timeline_canvas.bind("<Button-1>", self._on_timeline_click)
        self.timeline_canvas.bind("<Configure>", lambda e: self._draw_timeline())

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
        
        tools = tk.Frame(lbl_tracks)
        tools.pack(fill=tk.X, pady=5)
        
        chk = tk.Checkbutton(tools, text="Hide short (<1s)", variable=self.hide_short_var, command=self.refresh_tree)
        chk.pack(side=tk.LEFT, padx=5)

        tk.Button(tools, text="⚙ Parameters", command=self.open_settings_dialog).pack(side=tk.RIGHT, padx=5)
        tk.Button(tools, text="📜 Log", command=self.show_audit_log_window).pack(side=tk.RIGHT, padx=5)
        
        row1 = tk.Frame(lbl_tracks)
        row1.pack(fill=tk.X, pady=2)
        tk.Button(row1, text="⚡ Auto-Stitch", command=self.auto_stitch).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(row1, text="🧹 Absorb Noise (Gap Fill)", command=self.absorb_noise_logic).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        row2 = tk.Frame(lbl_tracks)
        row2.pack(fill=tk.X, pady=2)
        tk.Button(row2, text="🔗 Merge Selected", command=self.manual_merge).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(row2, text="🔗 Merge ALL by Role", bg="#d1e7dd", command=self.merge_all_by_role).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

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
        self.tree.bind("<Double-1>", self.on_tree_double_click)
        
        self.context_menu = tk.Menu(self.parent, tearoff=0)
        self.refresh_cast_list()

    # ------------------------------------------------------------------ #
    #  TIMELINE VISIVA                                                     #
    # ------------------------------------------------------------------ #
    def _draw_timeline(self) -> None:
        """Draws horizontal colored bars on the timeline canvas showing
        where each assigned track is active.  Unassigned ('Ignore') tracks
        are drawn as thin grey lines; cast members use their assigned colour."""
        c = self.timeline_canvas
        c.delete("all")
        if self.total_frames == 0 or not self.logic.tracks:
            return

        cw: int = c.winfo_width()
        ch: int = c.winfo_height()
        if cw < 10:
            return  # not yet laid out

        scale: float = cw / self.total_frames

        # Collect tracks that belong to the cast first, then the rest
        cast_tracks: List[Tuple[int, dict]] = []
        other_tracks: List[Tuple[int, dict]] = []
        for tid in sorted(self.logic.tracks.keys()):
            d = self.logic.tracks[tid]
            if d['role'] in self.cast:
                cast_tracks.append((tid, d))
            else:
                other_tracks.append((tid, d))

        # Assign a row to each cast role
        role_order: List[str] = list(self.cast.keys())
        n_roles = max(len(role_order), 1)
        row_h = max(4, (ch - 14) // n_roles)  # leave room for playhead

        # Draw "Ignore" tracks as thin grey background bars (all in row 0 area, semi-transparent)
        for tid, d in other_tracks:
            if not d['frames']:
                continue
            x0 = int(d['frames'][0] * scale)
            x1 = max(x0 + 1, int(d['frames'][-1] * scale))
            c.create_rectangle(x0, 0, x1, ch - 14, fill="#3a3a3a", outline="", stipple="gray25")

        # Draw cast tracks as solid colored bars
        for tid, d in cast_tracks:
            if not d['frames']:
                continue
            role = d['role']
            if role not in role_order:
                continue
            row_idx = role_order.index(role)
            y0 = row_idx * row_h
            y1 = y0 + row_h - 1

            b, g, r = self.cast[role]['color']
            hex_col = '#{:02x}{:02x}{:02x}'.format(r, g, b)

            # Find contiguous segments to draw fewer rectangles
            frames = d['frames']
            seg_start = frames[0]
            prev = frames[0]
            for fi in range(1, len(frames)):
                if frames[fi] > prev + 1:
                    # end of segment
                    x0 = int(seg_start * scale)
                    x1 = max(x0 + 1, int(prev * scale))
                    c.create_rectangle(x0, y0, x1, y1, fill=hex_col, outline="")
                    seg_start = frames[fi]
                prev = frames[fi]
            # last segment
            x0 = int(seg_start * scale)
            x1 = max(x0 + 1, int(prev * scale))
            c.create_rectangle(x0, y0, x1, y1, fill=hex_col, outline="")

        # Draw playhead
        px = int(self.current_frame * scale)
        c.create_line(px, 0, px, ch, fill="white", width=2)

        # Draw role labels
        for i, role in enumerate(role_order):
            y = i * row_h + row_h // 2
            c.create_text(4, y, text=role, anchor="w", fill="white", font=("Segoe UI", 7))

    def _on_timeline_click(self, event: tk.Event) -> None:
        """Seek to the frame the user clicked on in the timeline."""
        if self.total_frames == 0:
            return
        cw = self.timeline_canvas.winfo_width()
        if cw <= 0:
            return
        frame = int(event.x / cw * self.total_frames)
        frame = max(0, min(self.total_frames - 1, frame))
        self.current_frame = frame
        self.slider.set(self.current_frame)
        self.show_frame()

    # ------------------------------------------------------------------ #
    #  SELEZIONE SINCRONIZZATA (click on video → select treeview)         #
    # ------------------------------------------------------------------ #
    def _on_video_click(self, event: tk.Event) -> None:
        """When the user clicks on a bounding box in the video, select the
        corresponding track in the Treeview."""
        if not self.cap or self._video_scale == 0:
            return

        # Convert widget coords → original frame coords
        orig_x = int((event.x - self._video_offset_x) / self._video_scale)
        orig_y = int((event.y - self._video_offset_y) / self._video_scale)

        # Bounds check
        if orig_x < 0 or orig_y < 0 or orig_x > self._video_orig_w or orig_y > self._video_orig_h:
            return

        tid = self.logic.get_track_at_point(self.current_frame, orig_x, orig_y)
        if tid is not None:
            tid_str = str(tid)
            # Check the item exists in tree (might be hidden)
            if self.tree.exists(tid_str):
                self.tree.selection_set(tid_str)
                self.tree.focus(tid_str)
                self.tree.see(tid_str)

    # ------------------------------------------------------------------ #
    #  HOTKEYS                                                             #
    # ------------------------------------------------------------------ #
    def _setup_hotkeys(self) -> None:
        root = self.parent.winfo_toplevel()
        root.bind("<space>", self._on_space)
        root.bind("<Left>", self._on_left)
        root.bind("<Right>", self._on_right)
        root.bind("<Shift-Left>", self._on_shift_left)
        root.bind("<Shift-Right>", self._on_shift_right)
        for i in range(1, 10):
            root.bind(str(i), self._on_number)
        root.bind("<Control-z>", self.perform_undo)

    def _is_hotkey_safe(self) -> bool:
        if not self.parent.winfo_viewable():
            return False
        focused = self.parent.focus_get()
        if focused and focused.winfo_class() in ['Entry', 'Text', 'Spinbox', 'TEntry']:
            return False
        return True

    def _on_space(self, event: tk.Event) -> None:
        if self._is_hotkey_safe():
            self.toggle_play()

    def _on_left(self, event: tk.Event) -> None:
        if self._is_hotkey_safe():
            self.seek_relative(-1)
    
    def _on_right(self, event: tk.Event) -> None:
        if self._is_hotkey_safe():
            self.seek_relative(1)

    def _on_shift_left(self, event: tk.Event) -> None:
        if self._is_hotkey_safe():
            self.seek_relative(-10)

    def _on_shift_right(self, event: tk.Event) -> None:
        if self._is_hotkey_safe():
            self.seek_relative(10)

    def seek_relative(self, delta: int) -> None:
        if not self.cap:
            return
        self.current_frame = max(0, min(self.total_frames - 1, self.current_frame + delta))
        self.slider.set(self.current_frame)
        self.show_frame()

    def _on_number(self, event: tk.Event) -> None:
        if not self._is_hotkey_safe():
            return
        try:
            idx = int(event.char) - 1
            if 0 <= idx < self.list_cast.size() and self.tree.selection():
                self.assign_role_to_selection(self.list_cast.get(idx))
        except ValueError:
            pass

    def _snapshot(self) -> None:
        tracks, id_lineage = self.logic.get_data()
        self.history.push_state((tracks, id_lineage))

    def perform_undo(self, event: Optional[tk.Event] = None) -> None:
        prev_state = self.history.undo()
        if prev_state is not None:
            tracks, id_lineage = prev_state
            self.logic.set_data(tracks, id_lineage)
            self.refresh_tree()
            self.show_frame()
            print("Undo performed.")
    
    def _start_autosave_loop(self) -> None:
        try:
            self.parent.after(self.AUTOSAVE_INTERVAL_MS, self._perform_autosave)
        except Exception:
            pass # Widget destroyed

    def _perform_autosave(self) -> None:
        tracks, lineage = self.logic.get_data_snapshot()
        audit_log = self.logic.get_audit_log()
        if not tracks:
            self._start_autosave_loop()
            return

        save_path = self._get_autosave_path()

        def save_task(data_to_save: dict, filepath: str) -> None:
            try:
                tmp_path = filepath + ".tmp"
                with open(tmp_path, 'w') as f:
                    json.dump(data_to_save, f)
                if os.path.exists(filepath):
                    os.remove(filepath)
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

    def _get_autosave_path(self) -> str:
        if self.context and self.context.project_root:
            return os.path.join(self.context.project_root, self.AUTOSAVE_FILENAME)
        return self.AUTOSAVE_FILENAME

    def _check_for_autosave(self) -> None:
        path = self._get_autosave_path()
        if os.path.exists(path):
            if messagebox.askyesno("Recovery", "Found an autosave file from a previous session.\nDo you want to restore it?"):
                try:
                    with open(path, 'r') as f:
                        saved_data = json.load(f)

                    if 'tracks' in saved_data and 'id_lineage' in saved_data:
                        tracks_data, lineage_data = saved_data['tracks'], saved_data['id_lineage']
                        audit_data = saved_data.get('audit_log', [])
                    else:
                        tracks_data, lineage_data = saved_data, {}
                        audit_data = []

                    tracks = {int(k): v for k, v in tracks_data.items()}
                    id_lineage = {int(k): v for k, v in lineage_data.items()}

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
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _on_close(self) -> None:
        path = self._get_autosave_path()
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            self._toplevel.destroy()
        except (tk.TclError, AttributeError):
            pass

    def open_settings_dialog(self) -> None:
        """Opens a popup window to modify hardcoded parameters."""
        win = tk.Toplevel(self.parent)
        win.title("Algorithm Settings")
        win.geometry("350x250")
        
        v_lookahead = tk.IntVar(value=self.param_lookahead)
        v_time = tk.DoubleVar(value=self.param_time_gap)
        v_s_dist = tk.IntVar(value=self.param_stitch_dist)
        v_n_dist = tk.IntVar(value=self.param_noise_dist)
        
        tk.Label(win, text="1. Auto-Stitching", font=("bold")).pack(pady=(10,5))
        
        f1 = tk.Frame(win)
        f1.pack(fill=tk.X, padx=20)
        tk.Label(f1, text="Look-ahead (tracks):").pack(side=tk.LEFT)
        tk.Entry(f1, textvariable=v_lookahead, width=8).pack(side=tk.RIGHT)
        
        f2 = tk.Frame(win)
        f2.pack(fill=tk.X, padx=20)
        tk.Label(f2, text="Max Time Gap (sec):").pack(side=tk.LEFT)
        tk.Entry(f2, textvariable=v_time, width=8).pack(side=tk.RIGHT)

        f3 = tk.Frame(win)
        f3.pack(fill=tk.X, padx=20)
        tk.Label(f3, text="Max Distance (px):").pack(side=tk.LEFT)
        tk.Entry(f3, textvariable=v_s_dist, width=8).pack(side=tk.RIGHT)

        tk.Label(win, text="2. Noise Absorption", font=("bold")).pack(pady=(10,5))
        
        f4 = tk.Frame(win)
        f4.pack(fill=tk.X, padx=20)
        tk.Label(f4, text="Precision Dist (px):").pack(side=tk.LEFT)
        tk.Entry(f4, textvariable=v_n_dist, width=8).pack(side=tk.RIGHT)
        
        def save() -> None:
            self.param_lookahead = v_lookahead.get()
            self.param_time_gap = v_time.get()
            self.param_stitch_dist = v_s_dist.get()
            self.param_noise_dist = v_n_dist.get()
            win.destroy()
            messagebox.showinfo("Save", "Parameters updated successfully.")

        tk.Button(win, text="Save", command=save, bg="#4CAF50", fg="white").pack(pady=15)

    def absorb_noise_logic(self) -> None:
        """Supervised Noise Absorption using configurable parameters."""
        if not messagebox.askyesno("Confirm", f"Absorb noise (Dist < {self.param_noise_dist}px)? This cannot be undone in one step."):
            return
        self._snapshot()
        
        absorbed = self.logic.absorb_noise(self.cast, self.param_noise_dist, self.param_time_gap)
        
        self.refresh_tree()
        messagebox.showinfo("Info", f"Assorbiti {absorbed} frammenti.")

    def auto_stitch(self) -> None:
        """Unsupervised Auto-Stitching using configurable parameters."""
        if not messagebox.askyesno("Confirm", "Run auto-stitching? This may merge unrelated tracks and cannot be undone in one step."):
            return
        self._snapshot()
        p_win = self.param_lookahead
        p_time = self.param_time_gap
        p_dist = self.param_stitch_dist
        
        merged = self.logic.auto_stitch(p_win, p_time, p_dist)
        
        self.refresh_tree()
        messagebox.showinfo("Info", f"Stitched {merged} fragments (Lookahead:{p_win}, Time:{p_time}s, Dist:{p_dist}px).")

    def merge_all_by_role(self) -> None:
        if not messagebox.askyesno("Confirm", "Do you want to merge all tracks assigned to the same role?"):
            return
        self._snapshot()
        merge_count, roles_processed = self.logic.merge_all_by_role(self.cast)
        self.refresh_tree()
        if merge_count > 0:
            msg = f"Merged {merge_count} fragments for: {', '.join(roles_processed)}."
        else:
            msg = "No merges necessary."
        messagebox.showinfo("Merge Result", msg)

    def manual_merge(self) -> None:
        sel = self.tree.selection()
        if len(sel) < 2:
            return
        self._snapshot()
        
        master = self.logic.manual_merge(list(sel), self.cast)

        self.refresh_tree()
        self.tree.selection_set(str(master))

    def split_track_at_current_frame(self, track_id: Optional[int] = None, override_frame: Optional[int] = None) -> None:
        """Splits a specific track into two at the specified frame."""
        if track_id is not None:
            track_id_to_split = track_id
        else:
            selection = self.tree.selection()
            first = next(iter(selection), None)
            if first is None:
                messagebox.showwarning("Warning", "No track selected.")
                return
            track_id_to_split = int(first)

            if len(selection) > 1:
                messagebox.showwarning("Warning", "Select only one track to split.")
                return

        split_frame = override_frame if override_frame is not None else self.current_frame
        
        track_data = self.logic.tracks.get(track_id_to_split)
        if not track_data:
            return

        if split_frame <= track_data['frames'][0] or split_frame > track_data['frames'][-1]:
            messagebox.showinfo("Split Error", 
                                f"Cannot split at frame {split_frame}.\n"
                                f"Track range: {track_data['frames'][0]} - {track_data['frames'][-1]}.\n"
                                "You must be strictly inside the track (after the first frame).")
            return

        try:
            next(i for i, f in enumerate(track_data['frames']) if f >= split_frame)
        except StopIteration:
            return

        msg = (f"Splitting track {track_id_to_split} at frame {split_frame}.\n\n"
               "Which part should KEEP the original ID and Role?\n"
               "YES = The PREVIOUS part (up to the cursor)\n"
               "NO = The NEXT part (from the cursor onwards)")
        keep_head = messagebox.askyesno("Confirm Split", msg)

        self._snapshot()

        new_track_id, created_len_or_msg = self.logic.split_track(track_id_to_split, split_frame, keep_head)

        if new_track_id is None:
            messagebox.showerror("Split Error", str(created_len_or_msg))
            return

        created_len = int(created_len_or_msg)
        if self.hide_short_var.get() and (created_len / self.fps) < 1.0:
            self.hide_short_var.set(False)

        self.refresh_tree()
        self.tree.selection_set(str(new_track_id))
        self.tree.focus(str(new_track_id))

    def show_audit_log_window(self) -> None:
        logs = self.logic.get_audit_log()
        win = tk.Toplevel(self.parent)
        win.title("Audit Log Explorer")
        win.geometry("600x400")

        sb = ttk.Scrollbar(win)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        text = tk.Text(win, wrap="word", yscrollcommand=sb.set, font=("Consolas", 9))
        text.pack(side=tk.LEFT, fill="both", expand=True)
        sb.config(command=text.yview)

        for entry in reversed(logs):
            ts = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
            text.insert("end", f"[{ts}] {entry['action']}\n")
            text.insert("end", f"   Details: {entry['details']}\n\n")
        
        text.config(state="disabled")

    def refresh_tree(self) -> None:
        for i in self.tree.get_children():
            self.tree.delete(i)
        hide = self.hide_short_var.get()
        for tid in sorted(self.logic.tracks.keys()):
            d = self.logic.tracks[tid]
            role = d['role']
            dur = len(d['frames']) / self.fps
            if hide and dur < 1.0 and role not in self.cast:
                continue
            
            merged = str(d['merged_from']) if len(d['merged_from']) > 1 else str(tid)
            tag = "Ignore"
            if role in self.cast:
                tag = role
            
            self.tree.insert("", "end", iid=str(tid), values=(tid, merged, f"{dur:.2f}", role), tags=(tag,))
            
        self.tree.tag_configure("Ignore", background="white")
        for n in self.cast:
            b,g,r = self.cast[n]['color']
            self.tree.tag_configure(n, background='#{:02x}{:02x}{:02x}'.format(min(r+180,255), min(g+180,255), min(b+180,255)))
        
        # Redraw timeline whenever tree data changes
        self._draw_timeline()

    def browse_video(self) -> None:
        v = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")]) 
        if v:
            self.load_data_direct(v, self.json_path)

    def browse_pose(self) -> None:
        j = filedialog.askopenfilename(filetypes=[("Pose JSON", "*.json.gz")])
        if j:
            self.load_data_direct(self.video_path, j)

    def load_data_direct(self, video_path: Optional[str], json_path: Optional[str]) -> None:
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

    def _load_json_thread_refactored(self, path: str) -> None:
        try:
            has_untracked = self.logic.load_from_json_gz(path)
            self.load_queue.put(("success", has_untracked))
        except Exception as e:
            self.load_queue.put(("error", str(e)))

    def _check_load_queue(self) -> None:
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

    def load_mapping(self) -> None:
        if not self.logic.tracks:
            messagebox.showwarning("Warning", "Load video and pose data before loading an identity mapping.")
            return

        f = filedialog.askopenfilename(filetypes=[("Identity JSON", "*.json")])
        if not f:
            return
        
        try:
            with open(f, 'r') as file:
                mapping = json.load(file)
            
            loaded_count = 0
            new_roles: Set[str] = set()
            
            for tid_str, role in mapping.items():
                tid = int(tid_str)
                if tid in self.logic.tracks:
                    self.logic.tracks[tid]['role'] = role
                    loaded_count += 1
                    if role not in self.cast and role != "Ignore":
                        new_roles.add(role)
            
            for role in new_roles:
                self.cast[role] = {"color": (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))}
            
            self.refresh_cast_list()
            self.refresh_tree()
            self.show_frame()
            
            if self.context:
                self.context.identity_map_path = f
                
            messagebox.showinfo("Loaded", f"Restored {loaded_count} assignments.\nNew roles added: {len(new_roles)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Unable to load the file:\n{e}")

    def save_mapping(self) -> None:
        if not self.json_path:
            return

        base_name = os.path.basename(self.json_path).replace(".json.gz", "_identity.json")
        if self.context and self.context.paths.get("output"):
            out = os.path.join(self.context.paths["output"], base_name)
        else:
            out = self.json_path.replace(".json.gz", "_identity.json")

        mapping: Dict[int, str] = {}
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

        audit_out = out.replace("_identity.json", "_audit.json")
        try:
            with open(audit_out, 'w') as f:
                json.dump(self.logic.get_audit_log(), f, indent=4)
            print(f"Audit Log saved -> {audit_out}")
        except Exception as e:
            print(f"Error saving audit log: {e}")

        count = len(mapping)
        messagebox.showinfo("Done", f"Mapped {count} IDs (including merged historical IDs).\nSaved to: {out}\nAudit Log: {os.path.basename(audit_out)}")

    def refresh_cast_list(self) -> None:
        self.list_cast.delete(0, tk.END)
        for n in self.cast:
            self.list_cast.insert(tk.END, n)
            b,g,r = self.cast[n]['color']
            self.list_cast.itemconfig(self.list_cast.size()-1, bg='#{:02x}{:02x}{:02x}'.format(r,g,b))
            
    def add_person(self) -> None:
        n = simpledialog.askstring("New", "Name:")
        if n:
            self.cast[n] = {"color":(random.randint(50,200),random.randint(50,200),random.randint(50,200))}
            self.refresh_cast_list()
    
    def remove_person(self) -> None:
        s = self.list_cast.curselection()
        if s: 
            n = self.list_cast.get(s[0])
            del self.cast[n]
            for t in self.logic.tracks.values():
                if t['role'] == n:
                    t['role'] = 'Ignore'
            self.refresh_cast_list()
            self.refresh_tree()
            
    def change_person_color(self) -> None:
        s = self.list_cast.curselection()
        if not s:
            return

        n = self.list_cast.get(s[0])
        c = colorchooser.askcolor()
        
        if c and isinstance(c, tuple) and len(c) >= 2:
            rgb = c[0]
            if rgb:
                r,g,b = map(int, rgb)
                self.cast[n]['color'] = (b,g,r)
                self.refresh_cast_list()
                self.show_frame()

    def show_context_menu(self, e: tk.Event) -> None:
        i = self.tree.identify_row(e.y)
        if i:
            # 1. Seleziona la riga se non lo è (questo potrebbe causare un salto video, che è corretto per feedback)
            if i not in self.tree.selection(): 
                self.tree.selection_set(i)
                # Forziamo l'aggiornamento eventi per assicurarci che on_tree_select sia scattato
                self.parent.update_idletasks()

            # 2. Cattura il frame e l'ID DOPO l'eventuale selezione/salto
            frozen_frame_for_split = self.current_frame
            tid = int(i)

            # 3. Costruisci il menu
            self.context_menu.delete(0, tk.END)
            for p in self.cast: 
                self.context_menu.add_command(label=f"Assign to {p}", command=lambda n=p: self.assign_role_to_selection(n))
            
            self.context_menu.add_separator()
            self.context_menu.add_command(label="Remove Assignment", command=lambda: self.assign_role_to_selection("Ignore"))
            self.context_menu.add_separator()
            self.context_menu.add_command(label="🔗 Merge Selected", command=self.manual_merge)
            
            # 5. PUNTO CRITICO: Passiamo il frame congelato direttamente alla funzione
            # Check bounds to prevent errors
            can_split = False
            try:
                if tid in self.logic.tracks:
                    t_frames = self.logic.tracks[tid]['frames']
                    # Logic requires: split_frame > first_frame AND split_frame <= last_frame
                    if t_frames and (frozen_frame_for_split > t_frames[0]) and (frozen_frame_for_split <= t_frames[-1]):
                        can_split = True
            except Exception:
                pass

            if can_split:
                self.context_menu.add_command(
                    label=f"✂️ Split at Frame {frozen_frame_for_split}", 
                    command=lambda: self.split_track_at_current_frame(track_id=tid, override_frame=frozen_frame_for_split)
                )
            else:
                self.context_menu.add_command(
                    label=f"✂️ Split (Frame {frozen_frame_for_split} out of bounds)", 
                    state="disabled"
                )
            
            self.context_menu.post(e.x_root, e.y_root)

    def assign_role_to_selection(self, role: str) -> None:
        selected_ids = [str(i) for i in self.tree.selection()]
        self.logic.assign_role_to_ids([int(i) for i in selected_ids], role)
        self.refresh_tree()
        # Restore selection and focus after tree rebuild
        existing = [iid for iid in selected_ids if self.tree.exists(iid)]
        if existing:
            self.tree.selection_set(existing)
            self.tree.focus(existing[0])
            self.tree.see(existing[0])
        self.show_frame()

    def on_tree_select(self, e: tk.Event) -> None:
        # Comportamento normale (Click sinistro / Frecce):
        # Salta all'inizio della traccia per rapida identificazione
        s = self.tree.selection()
        if s: 
            try:
                track_id = int(s[0])
                track_data = self.logic.tracks[track_id]
                frames = track_data['frames']

                # FIX: Se siamo già dentro la traccia, non saltare all'inizio.
                if frames and (frames[0] <= self.current_frame <= frames[-1]):
                    return

                self.current_frame = frames[0]
                self.slider.set(self.current_frame)
                self.show_frame()
            except (ValueError, IndexError, KeyError):
                pass

    def on_tree_double_click(self, e: tk.Event) -> None:
        # Questa funzione ripristina il comportamento di "Salto al frame" 
        # ma solo quando lo richiedi esplicitamente col doppio click.
        s = self.tree.selection()
        if s: 
            start_frame = self.logic.tracks[int(s[0])]['frames'][0]
            self.current_frame = start_frame
            self.slider.set(self.current_frame)
            self.show_frame()

    def on_seek(self, v: str) -> None:
        self.current_frame = int(float(v))
        self.show_frame()

    def toggle_play(self) -> None:
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_loop()

    def play_loop(self) -> None:
        if self.is_playing and self.cap:
            start_t = time.time()
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.is_playing = False
                return
            self.slider.set(self.current_frame)
            self.show_frame()
            
            dt = (time.time() - start_t) * 1000
            wait = max(1, int((1000/self.fps) - dt))
            self.parent.after(wait, self.play_loop)
            
    def show_frame(self) -> None:
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return

        self._video_orig_h, self._video_orig_w = frame.shape[:2]

        for tid, d in self.logic.tracks.items():
            if self.current_frame in d['frames']:
                role = d['role']
                idx = d['frames'].index(self.current_frame)
                box = d['boxes'][idx]
                col = (100,100,100)
                if role in self.cast:
                    col = self.cast[role]['color']
                x1,y1,x2,y2 = map(int, box)
                cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
                cv2.putText(frame, f"{tid} {role if role!='Ignore' else ''}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        # Compute thumbnail scale + offsets for click mapping
        disp_w = self.lbl_video.winfo_width()
        disp_h = self.lbl_video.winfo_height()
        if disp_w > 1 and disp_h > 1:
            scale_x = disp_w / self._video_orig_w
            scale_y = disp_h / self._video_orig_h
            self._video_scale = min(scale_x, scale_y)
            thumb_w = int(self._video_orig_w * self._video_scale)
            thumb_h = int(self._video_orig_h * self._video_scale)
            self._video_offset_x = (disp_w - thumb_w) // 2
            self._video_offset_y = (disp_h - thumb_h) // 2
            img = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        else:
            img.thumbnail((disp_w, disp_h))
            self._video_scale = 1.0
            self._video_offset_x = 0
            self._video_offset_y = 0

        self.tk_img = ImageTk.PhotoImage(image=img)
        self.lbl_video.config(image=self.tk_img)

        # Update timeline playhead
        self._draw_timeline()
