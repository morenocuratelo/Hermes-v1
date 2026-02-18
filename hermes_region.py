import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import json
import gzip
import os
import math
import copy
import threading
import pandas as pd
from PIL import Image, ImageTk
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

KEYPOINTS_MAP = {
    0: "Nose", 1: "L_Eye", 2: "R_Eye", 3: "L_Ear", 4: "R_Ear",
    5: "L_Shoulder", 6: "R_Shoulder", 7: "L_Elbow", 8: "R_Elbow",
    9: "L_Wrist", 10: "R_Wrist", 11: "L_Hip", 12: "R_Hip",
    13: "L_Knee", 14: "R_Knee", 15: "L_Ankle", 16: "R_Ankle"
}


# ═══════════════════════════════════════════════════════════════════
# PROFILE MANAGER — I/O for AOI profile JSON files
# ═══════════════════════════════════════════════════════════════════

class AOIProfileManager:
    def __init__(self, folder="profiles_aoi"):
        self.folder = folder

    def create_default_profile(self):
        profile = {
            "name": "BW Invasion Granular",
            "roles": {
                "Target": [
                    {"name": "Face",   "shape": "box", "kps": [0,1,2,3,4],     "margin_px": 30, "expand_factor": 1.0},
                    {"name": "Torso",  "shape": "box", "kps": [5,6,11,12],     "margin_px": 20, "expand_factor": 1.0},
                    {"name": "Arms",   "shape": "box", "kps": [7,8,9,10],      "margin_px": 20, "expand_factor": 1.0},
                    {"name": "Legs",   "shape": "box", "kps": [13,14,15,16],   "margin_px": 20, "expand_factor": 1.0},
                    {"name": "Peripersonal", "shape": "box", "kps": list(range(17)), "margin_px": 0, "expand_factor": 3.0},
                ],
                "DEFAULT": [
                    {"name": "FullBody", "shape": "box", "kps": list(range(17)), "margin_px": 20, "expand_factor": 1.0}
                ]
            }
        }
        self.save_profile("default_invasion.json", profile)
        return profile

    def load_profile(self, name):
        try:
            with open(os.path.join(self.folder, name), 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_profile(self, name, data):
        with open(os.path.join(self.folder, name), 'w') as f:
            json.dump(data, f, indent=4)

    def list_profiles(self):
        if not os.path.exists(self.folder):
            return []
        return [f for f in os.listdir(self.folder) if f.endswith(".json")]

# ═══════════════════════════════════════════════════════════════════
# MODEL — Pure logic, no tkinter
# ═══════════════════════════════════════════════════════════════════

class RegionLogic:
    """
    Pure computational logic for AOI region calculation.
    No tkinter imports or UI references.
    """

    # Roles that are always skipped during rendering / export
    IGNORED_ROLES = {"Ignore", "Noise", "Unknown"}

    def __init__(self):
        self.pose_data = {}      # { frame_idx: { track_id: [[x,y,conf], ...] } }
        self.identity_map = {}   # { "track_id_str": "RoleName" }
        # {(frame_idx, track_id, role, aoi_name): (x1, y1, x2, y2)}
        self.manual_overrides = {}
        self.total_frames = 0
        self.fps = 30.0
        self._cancel_flag = False
        self.lock = threading.RLock()

    def cancel(self):
        self._cancel_flag = True

    # ── Pose I/O ────────────────────────────────────────────────

    def load_pose_data(self, path, progress_callback=None):
        """
        Load a .json.gz pose file and populate self.pose_data.

        Parameters
        ----------
        path : str
            Path to the <name>_yolo.json.gz file.
        progress_callback : callable(str) | None
            Optional feedback hook.
        """
        self._cancel_flag = False
        temp_data = {}

        if progress_callback:
            progress_callback(f"Loading poses: {os.path.basename(path)}")

        with gzip.open(path, 'rt', encoding='utf-8') as file:
            for line_num, line in enumerate(file):
                if self._cancel_flag:
                    raise InterruptedError("Pose loading cancelled by user.")

                d = json.loads(line)
                f_idx = d['f_idx']
                frame_dict = {}

                for i, det in enumerate(d['det']):
                    if 'keypoints' not in det:
                        continue

                    # Track-ID handling (replicate Entity synthetic-ID logic)
                    raw_tid = det.get('track_id', -1)
                    if raw_tid is None:
                        raw_tid = -1
                    tid = int(raw_tid)
                    if tid == -1:
                        tid = 9000000 + (f_idx * 1000) + i

                    # Parse keypoints (support both dict and list formats)
                    raw_kps = det['keypoints']
                    final_kps = []
                    if isinstance(raw_kps, dict) and 'x' in raw_kps:
                        xs, ys = raw_kps['x'], raw_kps['y']
                        confs = raw_kps.get('visible', raw_kps.get('confidence', [1.0] * len(xs)))
                        for k in range(len(xs)):
                            final_kps.append([xs[k], ys[k], confs[k] if k < len(confs) else 0])
                    elif isinstance(raw_kps, list):
                        final_kps = raw_kps

                    frame_dict[tid] = final_kps

                temp_data[f_idx] = frame_dict

                if progress_callback and line_num % 1000 == 0:
                    progress_callback(f"Loading frame {f_idx}...")

        # Atomic swap — single lock acquisition
        with self.lock:
            self.pose_data = temp_data

        count = len(self.pose_data)
        if progress_callback:
            progress_callback(f"Poses loaded: {count} frames.")
        return count

    # ── Identity I/O ────────────────────────────────────────────

    def load_identity_map(self, path):
        """Load the identity JSON and return the number of mapped IDs."""
        with open(path, 'r') as f:
            self.identity_map = json.load(f)
        return len(self.identity_map)

    # ── Geometry Engine ─────────────────────────────────────────

    def calculate_box(self, kps_data, rule, kp_conf_thresh=0.3):
        """
        Compute a bounding box from a subset of keypoints.

        Parameters
        ----------
        kps_data : list[list]
            List of [x, y, conf] triples for one tracked person.
        rule : dict
            AOI rule with keys: kps, margin_px, expand_factor,
            and optional scale_w, scale_h, offset_y_bottom.
        kp_conf_thresh : float
            Minimum confidence to consider a keypoint valid.

        Returns
        -------
        tuple(int, int, int, int) | None
            (x1, y1, x2, y2) or None if insufficient keypoints.
        """
        indices = rule.get('kps', [])
        xs, ys = [], []

        for i in indices:
            if i >= len(kps_data):
                continue
            pt = kps_data[i]
            x, y, conf = 0, 0, 0
            if isinstance(pt, list):
                if len(pt) >= 2:
                    x, y = pt[0], pt[1]
                if len(pt) >= 3:
                    conf = pt[2]
                else:
                    conf = 1.0

            if conf > kp_conf_thresh and x > 1 and y > 1:
                xs.append(x)
                ys.append(y)

        if not xs:
            return None

        # Base bounding box
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 1. Uniform margin padding
        m = int(rule.get('margin_px', 0))
        min_x -= m
        max_x += m
        min_y -= m
        max_y += m

        # 2. Independent width/height scaling from centre
        base_exp = float(rule.get('expand_factor', 1.0))
        scale_w = float(rule.get('scale_w', base_exp))
        scale_h = float(rule.get('scale_h', base_exp))

        w = max_x - min_x
        h = max_y - min_y
        cx = min_x + w / 2
        cy = min_y + h / 2

        new_w = w * scale_w
        new_h = h * scale_h

        min_x = cx - new_w / 2
        max_x = cx + new_w / 2
        min_y = cy - new_h / 2
        max_y = cy + new_h / 2

        # 3. Bottom offset AFTER expansion (so N px always means N real px)
        if 'offset_y_bottom' in rule:
            max_y += int(rule['offset_y_bottom'])

        # 4. Clamp to non-negative coordinates
        min_x = max(0, min_x)
        min_y = max(0, min_y)

        return (int(min_x), int(min_y), int(max_x), int(max_y))

    @staticmethod
    def _extract_valid_points(kps_data, indices, kp_conf_thresh):
        pts = []
        for i in indices:
            if i >= len(kps_data):
                continue
            pt = kps_data[i]
            x, y, conf = 0, 0, 0
            if isinstance(pt, list):
                if len(pt) >= 2:
                    x, y = pt[0], pt[1]
                if len(pt) >= 3:
                    conf = pt[2]
                else:
                    conf = 1.0
            if conf > kp_conf_thresh and x > 1 and y > 1:
                pts.append((float(x), float(y)))
        return pts

    @staticmethod
    def _bbox_from_points(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _map_points_between_boxes(points, src_box, dst_box):
        sx1, sy1, sx2, sy2 = src_box
        dx1, dy1, dx2, dy2 = dst_box
        sw = max(1e-6, sx2 - sx1)
        sh = max(1e-6, sy2 - sy1)
        dw = max(1e-6, dx2 - dx1)
        dh = max(1e-6, dy2 - dy1)

        mapped = []
        for x, y in points:
            nx = (x - sx1) / sw
            ny = (y - sy1) / sh
            mx = dx1 + nx * dw
            my = dy1 + ny * dh
            mapped.append((int(mx), int(my)))
        return mapped

    @staticmethod
    def _order_polygon(points):
        if len(points) <= 2:
            return [(int(x), int(y)) for x, y in points]
        cx = sum(x for x, _ in points) / len(points)
        cy = sum(y for _, y in points) / len(points)
        ordered = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        return [(int(x), int(y)) for x, y in ordered]

    def _shape_from_box(self, shape_type, box, base_shape=None):
        shape_type = str(shape_type).lower()
        b = self._sanitize_box(box)
        x1, y1, x2, y2 = b

        if shape_type == "circle":
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            r = max(1, int(min(x2 - x1, y2 - y1) / 2))
            sb = self._sanitize_box((cx - r, cy - r, cx + r, cy + r))
            return {
                "shape_type": "circle",
                "box": sb,
                "cx": int((sb[0] + sb[2]) / 2),
                "cy": int((sb[1] + sb[3]) / 2),
                "radius": max(1, int(min(sb[2] - sb[0], sb[3] - sb[1]) / 2)),
            }

        if shape_type in ("oval", "ellipse"):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            rx = max(1, int((x2 - x1) / 2))
            ry = max(1, int((y2 - y1) / 2))
            return {
                "shape_type": "oval",
                "box": b,
                "cx": cx,
                "cy": cy,
                "rx": rx,
                "ry": ry,
                "angle": 0,
            }

        if shape_type == "polygon":
            if base_shape and base_shape.get("points"):
                pts = self._map_points_between_boxes(
                    [(float(px), float(py)) for px, py in base_shape["points"]],
                    base_shape["box"],
                    b,
                )
            else:
                pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            return {"shape_type": "polygon", "box": b, "points": pts}

        return {"shape_type": "box", "box": b}

    def calculate_shape(self, kps_data, rule, kp_conf_thresh=0.3):
        """
        Compute shape geometry from selected keypoints.

        Supported rule['shape']: box, polygon, circle, oval.
        """
        indices = rule.get('kps', [])
        if not indices:
            return None

        base_box = self.calculate_box(kps_data, rule, kp_conf_thresh)
        if not base_box:
            return None

        shape_type = str(rule.get("shape", "box")).lower()
        valid_pts = self._extract_valid_points(kps_data, indices, kp_conf_thresh)

        if shape_type == "polygon":
            if len(valid_pts) >= 3:
                src_box = self._bbox_from_points(valid_pts)
                mapped = self._map_points_between_boxes(valid_pts, src_box, base_box)
                pts = self._order_polygon(mapped)
            else:
                x1, y1, x2, y2 = base_box
                pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            return {"shape_type": "polygon", "box": base_box, "points": pts}

        if shape_type == "circle":
            return self._shape_from_box("circle", base_box)

        if shape_type in ("oval", "ellipse"):
            return self._shape_from_box("oval", base_box)

        return {"shape_type": "box", "box": base_box}

    # ── Helper: safe profile role lookup ────────────────────────

    @staticmethod
    def _get_rules(profile, role):
        """Safely retrieve AOI rules for a role, with DEFAULT fallback."""
        if not profile or 'roles' not in profile:
            return []
        return profile['roles'].get(role, profile['roles'].get("DEFAULT", []))

    @staticmethod
    def _override_key(frame_idx, track_id, role, aoi_name):
        return (int(frame_idx), int(track_id), str(role), str(aoi_name))

    @staticmethod
    def _sanitize_box(box):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)
        return (x1, y1, x2, y2)

    def set_manual_override(self, frame_idx, track_id, role, aoi_name, box):
        key = self._override_key(frame_idx, track_id, role, aoi_name)
        with self.lock:
            self.manual_overrides[key] = self._sanitize_box(box)

    def clear_manual_override(self, frame_idx, track_id, role, aoi_name):
        key = self._override_key(frame_idx, track_id, role, aoi_name)
        with self.lock:
            if key in self.manual_overrides:
                del self.manual_overrides[key]

    def clear_overrides_for_frame(self, frame_idx):
        frame_idx = int(frame_idx)
        with self.lock:
            keys_to_remove = [k for k in self.manual_overrides if k[0] == frame_idx]
            for k in keys_to_remove:
                del self.manual_overrides[k]

    def get_frame_aoi_data(self, frame_idx, profile, kp_conf_thresh=0.3):
        """
        Return all AOIs for one frame with manual overrides applied.

        Returns
        -------
        list[dict]
            Each item has: frame, track_id, role, aoi, box, corrected.
        """
        items = []
        if not profile or 'roles' not in profile:
            return items

        with self.lock:
            frame_poses = dict(self.pose_data.get(frame_idx, {}))
            frame_overrides = {
                k: v for k, v in self.manual_overrides.items() if k[0] == int(frame_idx)
            }

        if not frame_poses:
            return items

        for tid, kps in frame_poses.items():
            role = self.identity_map.get(str(tid), "Unknown")
            if role in self.IGNORED_ROLES:
                continue

            rules = self._get_rules(profile, role)
            for rule in rules:
                base_shape = self.calculate_shape(kps, rule, kp_conf_thresh)
                if not base_shape:
                    continue

                key = self._override_key(frame_idx, tid, role, rule['name'])
                if key in frame_overrides:
                    shape = self._shape_from_box(
                        base_shape.get("shape_type", "box"),
                        frame_overrides[key],
                        base_shape=base_shape,
                    )
                    corrected = True
                else:
                    shape = base_shape
                    corrected = False

                row = {
                    "frame": int(frame_idx),
                    "track_id": int(tid),
                    "role": role,
                    "aoi": rule['name'],
                    "shape_type": shape.get("shape_type", "box"),
                    "box": shape["box"],
                    "corrected": corrected,
                }
                for extra_key in ("points", "cx", "cy", "radius", "rx", "ry", "angle"):
                    if extra_key in shape:
                        row[extra_key] = shape[extra_key]
                items.append(row)
        return items

    # ── Render Data (for View) ──────────────────────────────────

    def get_render_data(self, frame_idx, profile, kp_conf_thresh=0.3):
        """
        Return a list of drawable items for a single frame.

        Returns
        -------
        list[dict]
            Each dict has keys: box, color, label, track_id, role, aoi, corrected, shape_type.
        """
        items = []
        frame_aois = self.get_frame_aoi_data(frame_idx, profile, kp_conf_thresh)
        for aoi in frame_aois:
            if aoi["corrected"]:
                color = (0, 220, 0)
            else:
                color = (255, 0, 255) if aoi["aoi"] == "Peripersonal" else (0, 255, 255)
            items.append({
                "box": aoi["box"],
                "color": color,
                "label": f"{aoi['role']}:{aoi['aoi']}",
                "track_id": aoi["track_id"],
                "role": aoi["role"],
                "aoi": aoi["aoi"],
                "corrected": aoi["corrected"],
                "shape_type": aoi.get("shape_type", "box"),
            })
            for extra_key in ("points", "cx", "cy", "radius", "rx", "ry", "angle"):
                if extra_key in aoi:
                    items[-1][extra_key] = aoi[extra_key]
        return items

    # ── Diagnostics ─────────────────────────────────────────────

    def get_diagnostics_report(self, frame_idx, profile, kp_conf_thresh=0.3):
        """
        Build a human-readable diagnostics string for a single frame.

        Returns
        -------
        str
            Multi-line report.
        """
        lines = []
        lines.append("=" * 40)
        lines.append(f"FRAME DIAGNOSTICS {frame_idx}")
        lines.append("=" * 40)

        if not profile or 'roles' not in profile:
            lines.append("❌ NO PROFILE loaded (profile is empty or missing 'roles').")
            return "\n".join(lines)

        with self.lock:
            frame_poses = self.pose_data.get(frame_idx)

        if not frame_poses:
            lines.append(f"❌ NO POSE found for frame {frame_idx}.")
            lines.append("Verify video and JSON alignment.")
            return "\n".join(lines)

        lines.append(f"✅ Found {len(frame_poses)} Tracked IDs: {list(frame_poses.keys())}")

        for tid, kps in frame_poses.items():
            lines.append(f"\n--- ID Analysis {tid} ---")

            role = self.identity_map.get(str(tid), "Unknown")
            lines.append(f"   Mapped Role (Identity): '{role}'")

            if role in ("Ignore", "Noise"):
                lines.append("   ⛔ SKIPPED: Role is Ignore or Noise.")
                continue
            if role == "Unknown":
                lines.append("   ⚠️ SKIPPED: Role is Unknown (unmapped).")
                continue

            rules = self._get_rules(profile, role)
            if role not in profile['roles']:
                lines.append(f"   ℹ️ INFO: Role '{role}' not in profile. Using 'DEFAULT'.")

            for rule in rules:
                shape = self.calculate_shape(kps, rule, kp_conf_thresh)
                if shape:
                    shp = shape.get("shape_type", "box")
                    lines.append(f"   ✅ AOI '{rule['name']}' ({shp}): Box {shape['box']}")
                else:
                    lines.append(f"   ❌ AOI '{rule['name']}': Failed (Insufficient keypoints or low conf)")
                    indices = rule.get('kps', [])
                    valid_pts = 0
                    for i in indices:
                        if i < len(kps):
                            pt = kps[i]
                            conf = pt[2] if len(pt) > 2 else 0
                            if conf > kp_conf_thresh:
                                valid_pts += 1
                    lines.append(f"      (Threshold: {kp_conf_thresh})")
                    lines.append(f"      Valid points: {valid_pts}/{len(indices)}")

        lines.append("=" * 40)
        return "\n".join(lines)

    # ── CSV Export ──────────────────────────────────────────────

    def export_csv(self, output_path, profile, kp_conf_thresh=0.3,
                   progress_callback=None, profile_for_frame_fn=None):
        """
        Iterate over all frames and export AOI bounding boxes to CSV.

        Parameters
        ----------
        output_path : str
        profile : dict
            The currently-active AOI profile.
        kp_conf_thresh : float
        progress_callback : callable(str) | None

        Returns
        -------
        int
            Number of rows written.
        """
        self._cancel_flag = False

        if not profile or 'roles' not in profile:
            raise ValueError("Cannot export: no valid AOI profile loaded.")

        rows = []

        with self.lock:
            all_frames = sorted(self.pose_data.keys())
            total = len(all_frames)

        for count, f_idx in enumerate(all_frames):
            if self._cancel_flag:
                raise InterruptedError("Export cancelled by user.")
            if profile_for_frame_fn:
                prof_for_frame = profile_for_frame_fn(f_idx, profile)
            else:
                prof_for_frame = profile
            frame_aois = self.get_frame_aoi_data(f_idx, prof_for_frame, kp_conf_thresh)
            for aoi in frame_aois:
                b = aoi["box"]
                shape_type = aoi.get("shape_type", "box")
                shape_points = json.dumps(aoi.get("points", [])) if shape_type == "polygon" else ""
                rows.append({
                    "Frame": f_idx,
                    "Timestamp": round(f_idx / self.fps, 4),
                    "TrackID": aoi["track_id"],
                    "Role": aoi["role"],
                    "AOI": aoi["aoi"],
                    "ShapeType": shape_type,
                    "ShapePoints": shape_points,
                    "CenterX": aoi.get("cx", -1),
                    "CenterY": aoi.get("cy", -1),
                    "Radius": aoi.get("radius", -1),
                    "RadiusX": aoi.get("rx", -1),
                    "RadiusY": aoi.get("ry", -1),
                    "Angle": aoi.get("angle", 0),
                    "x1": b[0], "y1": b[1],
                    "x2": b[2], "y2": b[3],
                    "Corrected": int(aoi["corrected"]),
                })

            if progress_callback and count % 500 == 0:
                progress_callback(f"Exporting… {count}/{total} frames")

        pd.DataFrame(rows).to_csv(output_path, index=False)

        if progress_callback:
            progress_callback(f"Export complete: {len(rows)} rows.")

        return len(rows)

class TOITimelineWidget(tk.Canvas):
    """Timeline with TOI epochs and playhead."""

    def __init__(self, parent, command_seek, **kwargs):
        super().__init__(parent, **kwargs)
        self.command_seek = command_seek
        self.duration = 0.0
        self.tois = []
        self.cursor_x = 0
        self.bind("<Button-1>", self.on_click)
        self.bind("<Configure>", lambda _e: self.redraw())

    def set_data(self, duration, df_tois):
        self.duration = max(0.0, float(duration))
        self.tois = []
        if df_tois is not None and not df_tois.empty:
            colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f']
            cond_map = {}
            for _, row in df_tois.iterrows():
                cond = str(row.get('Condition', row.get('Phase', row.get('Name', 'Base'))))
                if cond not in cond_map:
                    cond_map[cond] = colors[len(cond_map) % len(colors)]
                self.tois.append({
                    's': float(row.get('Start', 0.0)),
                    'e': float(row.get('End', 0.0)),
                    'c': cond_map[cond],
                    'n': str(row.get('Name', 'TOI')),
                    'p': str(row.get('Phase', '')),
                    'cond': cond,
                })
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if self.duration <= 0 or w <= 2 or h <= 2:
            return

        for t in self.tois:
            x1 = int(max(0, min(w, (t['s'] / self.duration) * w)))
            x2 = int(max(0, min(w, (t['e'] / self.duration) * w)))
            if x2 <= x1:
                x2 = min(w, x1 + 1)
            self.create_rectangle(x1, 2, x2, h - 2, fill=t['c'], outline="gray")

        self.create_line(self.cursor_x, 0, self.cursor_x, h, fill="#d32f2f", width=2)

    def update_cursor(self, current_sec):
        if self.duration > 0:
            w = self.winfo_width()
            self.cursor_x = (max(0.0, min(self.duration, current_sec)) / self.duration) * w
            self.redraw()

    def on_click(self, event):
        if self.duration <= 0:
            return
        w = max(1, self.winfo_width())
        perc = max(0.0, min(1.0, event.x / w))
        sec = perc * self.duration
        self.command_seek(sec)

# ═══════════════════════════════════════════════════════════════════
# VIEW / CONTROLLER — UI only
# ═══════════════════════════════════════════════════════════════════

class RegionView:
    def __init__(self, parent, context):
        self.parent = parent
        self.context = context
        self.logic = RegionLogic()

        # --- Profile manager from project path ---
        if self.context.paths["profiles_aoi"]:
            profile_dir = self.context.paths["profiles_aoi"]
        else:
            profile_dir = "profiles_aoi_fallback"
            if not os.path.exists(profile_dir):
                os.makedirs(profile_dir)

        self.pm = AOIProfileManager(folder=profile_dir)

        # Create default profile if empty
        if not self.pm.list_profiles():
            self.pm.create_default_profile()

        self.video_path = None
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0
        self.is_playing = False
        self.manual_mode = False
        self.manual_mode_snapshot = None
        self.manual_mode_undo_snapshot = []
        self.manual_mode_redo_snapshot = []
        self._manual_prev_scope = None
        self.toi_df = None
        self.toi_records = []
        self.toi_rule_overrides = {}
        self.param_scope_var = tk.StringVar(value="Current TOI")
        self.edit_scope_var = tk.StringVar(value="Current TOI")
        self.kp_conf_thresh = tk.DoubleVar(value=0.3)
        self.frame_aoi_items = []
        self.frame_aoi_lookup = {}
        self.selected_frame_aoi = tk.StringVar()
        self.edit_x1 = tk.IntVar(value=0)
        self.edit_y1 = tk.IntVar(value=0)
        self.edit_x2 = tk.IntVar(value=1)
        self.edit_y2 = tk.IntVar(value=1)
        self.drag_state = None
        self._orig_w = 0
        self._orig_h = 0
        self._disp_w = 0
        self._disp_h = 0
        self._disp_off_x = 0
        self._disp_off_y = 0
        self.undo_stack = []
        self.redo_stack = []
        self.last_manual_save_ts = None
        self.session_state_path = None
        self.current_toi_idx = None

        # Load first available profile
        profs = self.pm.list_profiles()
        self.current_profile = self.pm.load_profile(profs[0]) if profs else {}

        self._setup_ui()
        self._setup_hotkeys()

        # --- AUTO-LOAD FROM CONTEXT ---
        if self.context.video_path:
            self.load_video_direct(self.context.video_path)

        if self.context.pose_data_path:
            self.load_pose_direct(self.context.pose_data_path)

        if self.context.identity_map_path:
            self.load_identity_direct(self.context.identity_map_path)
        if self.context.toi_path and os.path.exists(self.context.toi_path):
            self.load_toi_direct(self.context.toi_path)
        self._init_session_state_path()
        self._try_restore_newer_autosave()

    # ── UI Construction ─────────────────────────────────────────

    def _setup_ui(self):
        tk.Label(self.parent, text="3. Spatial Area of Interest (AOI) Definition",
                 font=("Segoe UI", 18, "bold"), bg="white").pack(pady=(0, 10), anchor="w")

        main = tk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # LEFT: Video
        left = tk.Frame(main, bg="black")
        main.add(left, minsize=900)
        self.lbl_video = tk.Label(left, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)
        self.lbl_video.bind("<Button-1>", self._on_video_mouse_down)
        self.lbl_video.bind("<B1-Motion>", self._on_video_mouse_drag)
        self.lbl_video.bind("<ButtonRelease-1>", self._on_video_mouse_up)

        ctrl = tk.Frame(left)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM, pady=5)

        self.lbl_player = tk.Label(ctrl, text="Ready", anchor="w")
        self.lbl_player.pack(fill=tk.X, padx=6)

        self.timeline = TOITimelineWidget(
            ctrl, command_seek=self.seek_to_seconds, height=44, bg="#eeeeee"
        )
        self.timeline.pack(fill=tk.X, padx=6, pady=(2, 4))

        self.lbl_toi = tk.Label(ctrl, text="TOI: n/a", fg="gray", anchor="w")
        self.lbl_toi.pack(fill=tk.X, padx=6)

        self.slider = ttk.Scale(ctrl, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_seek)
        self.slider.pack(fill=tk.X, padx=6, pady=(2, 4))

        btns = tk.Frame(ctrl)
        btns.pack(pady=5, fill=tk.X)
        tk.Button(btns, text="1. Video Source",       command=self.browse_video).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="2. Pose Data (.gz)",    command=self.browse_pose).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="3. Identity Map (.json)", command=self.browse_identity).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="4. TOI (.tsv/.csv)", command=self.browse_toi).pack(side=tk.LEFT, padx=5)

        tk.Button(btns, text="🔍 FRAME DIAGNOSTICS", bg="red", fg="white",
                  font=("Bold", 10), command=self.run_diagnostics).pack(side=tk.RIGHT, padx=20)

        transport = tk.Frame(ctrl)
        transport.pack(fill=tk.X, pady=(2, 4))
        tk.Button(transport, text="⏮ -10", width=8, command=lambda: self.seek_relative(-10)).pack(side=tk.LEFT, padx=2)
        tk.Button(transport, text="⏴ -1", width=8, command=lambda: self.seek_relative(-1)).pack(side=tk.LEFT, padx=2)
        self.btn_play = tk.Button(transport, text="▶ Play", width=10, command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=8)
        tk.Button(transport, text="+1 ⏵", width=8, command=lambda: self.seek_relative(1)).pack(side=tk.LEFT, padx=2)
        tk.Button(transport, text="+10 ⏭", width=8, command=lambda: self.seek_relative(10)).pack(side=tk.LEFT, padx=2)
        tk.Button(transport, text="Prev TOI", width=10, command=lambda: self.seek_toi(-1)).pack(side=tk.LEFT, padx=8)
        tk.Button(transport, text="Next TOI", width=10, command=lambda: self.seek_toi(1)).pack(side=tk.LEFT, padx=2)

        # RIGHT: Config
        right = tk.Frame(main, padx=10, pady=10)
        main.add(right, minsize=400)

        tk.Label(right, text="AOI Configuration", font=("Bold", 14)).pack(pady=10)

        f_prof = tk.Frame(right)
        f_prof.pack(fill=tk.X)
        tk.Label(f_prof, text="Profile:").pack(side=tk.LEFT)

        self.cb_profile = ttk.Combobox(f_prof, values=self.pm.list_profiles(), state="readonly")
        self.cb_profile.pack(side=tk.LEFT, padx=5)
        if self.pm.list_profiles():
            self.cb_profile.current(0)
        self.cb_profile.bind("<<ComboboxSelected>>", self.on_profile_change)

        tk.Button(f_prof, text="✨ New (Wizard)", command=self.open_profile_wizard, bg="#e1f5fe").pack(side=tk.LEFT, padx=5)
        tk.Button(f_prof, text="💾 Save Session", command=self.save_session_now).pack(side=tk.LEFT, padx=5)

        f_mode = tk.Frame(right)
        f_mode.pack(fill=tk.X, pady=(6, 2))
        tk.Label(f_mode, text="Parameter Scope:").pack(side=tk.LEFT)
        self.cb_param_scope = ttk.Combobox(
            f_mode, textvariable=self.param_scope_var, state="readonly",
            values=["Current TOI", "Whole Video"], width=14
        )
        self.cb_param_scope.pack(side=tk.LEFT, padx=5)

        tk.Label(f_mode, text="Manual Scope:").pack(side=tk.LEFT, padx=(10, 0))
        self.cb_edit_scope = ttk.Combobox(
            f_mode, textvariable=self.edit_scope_var, state="readonly",
            values=["Current TOI", "Frame", "Whole Video"], width=12
        )
        self.cb_edit_scope.pack(side=tk.LEFT, padx=5)

        f_manual = tk.Frame(right)
        f_manual.pack(fill=tk.X, pady=(2, 6))
        self.btn_manual_mode = tk.Button(
            f_manual, text="Manual Correction: OFF", bg="#d32f2f", fg="white",
            command=self.toggle_manual_mode
        )
        self.btn_manual_mode.pack(side=tk.LEFT, padx=2)
        self.btn_undo = tk.Button(f_manual, text="Undo", width=8, command=self.undo_last_action)
        self.btn_undo.pack(side=tk.LEFT, padx=4)
        self.btn_redo = tk.Button(f_manual, text="Redo", width=8, command=self.redo_last_action)
        self.btn_redo.pack(side=tk.LEFT, padx=2)

        # Confidence slider
        lf_conf = tk.LabelFrame(right, text="Detection Sensitivity / Confidence Threshold", padx=5, pady=5)
        lf_conf.pack(fill=tk.X, pady=10)

        tk.Label(lf_conf, text="Keypoint Confidence Threshold (0.0 - 1.0):").pack(anchor="w")
        s_conf = tk.Scale(lf_conf, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                          variable=self.kp_conf_thresh, command=lambda v: self.show_frame())
        s_conf.pack(fill=tk.X)
        tk.Label(lf_conf, text="(Lower to recover missing limbs, Raise to reduce noise)",
                 fg="gray", font=("Arial", 8)).pack(anchor="w")

        # Notebook for rule editors
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        self.frame_target = tk.Frame(self.notebook)
        self.notebook.add(self.frame_target, text="Target Rules")
        self.frame_others = tk.Frame(self.notebook)
        self.notebook.add(self.frame_others, text="Non-Target Rules")

        self.refresh_editors()
        self._build_frame_correction_editor(right)

        tk.Button(right, text="GENERATE & EXPORT AOI CSV", bg="#4CAF50", fg="white",
                  font=("Bold", 12), height=2, command=self.export_data
                  ).pack(side=tk.BOTTOM, fill=tk.X, pady=20)

        self._update_history_buttons()

    # ── Hotkeys ─────────────────────────────────────────────────

    def _setup_hotkeys(self):
        root = self.parent.winfo_toplevel()
        root.bind("<space>",       self._on_space)
        root.bind("<Left>",        self._on_left)
        root.bind("<Right>",       self._on_right)
        root.bind("<Shift-Left>",  self._on_shift_left)
        root.bind("<Shift-Right>", self._on_shift_right)
        root.bind("<Control-z>",   self._on_undo)
        root.bind("<Control-y>",   self._on_redo)

    def _is_hotkey_safe(self):
        if not self.parent.winfo_viewable():
            return False
        focused = self.parent.focus_get()
        if focused and focused.winfo_class() in ['Entry', 'Text', 'Spinbox', 'TEntry']:
            return False
        return True

    def _on_space(self, event):
        if self._is_hotkey_safe():
            self.toggle_play()

    def _on_left(self, event):
        if self._is_hotkey_safe():
            self.seek_relative(-1)

    def _on_right(self, event):
        if self._is_hotkey_safe():
            self.seek_relative(1)

    def _on_shift_left(self, event):
        if self._is_hotkey_safe():
            self.seek_relative(-10)

    def _on_shift_right(self, event):
        if self._is_hotkey_safe():
            self.seek_relative(10)

    def _on_undo(self, event):
        if self._is_hotkey_safe():
            self.undo_last_action()

    def _on_redo(self, event):
        if self._is_hotkey_safe():
            self.redo_last_action()

    def seek_relative(self, delta):
        if not self.cap:
            return
        self.drag_state = None
        self.current_frame = max(0, min(self.total_frames - 1, self.current_frame + delta))
        self.slider.set(self.current_frame)
        self.show_frame()

    # ── TOI / Scope / Session Helpers ─────────────────────────

    @staticmethod
    def _parse_timestamp(ts):
        if ts is None:
            return None
        try:
            return datetime.fromisoformat(str(ts))
        except Exception:
            return None

    def _init_session_state_path(self):
        out_dir = self.context.paths.get("output", "")
        if out_dir and os.path.exists(out_dir):
            self.session_state_path = os.path.join(out_dir, "_aoi_edit_session.json")
        else:
            self.session_state_path = os.path.join(os.getcwd(), "_aoi_edit_session.json")

    def _snapshot_edit_state(self):
        return {
            "manual_overrides": copy.deepcopy(self.logic.manual_overrides),
            "toi_rule_overrides": copy.deepcopy(self.toi_rule_overrides),
            "current_profile": copy.deepcopy(self.current_profile),
        }

    def _restore_edit_state(self, state):
        self.logic.manual_overrides = copy.deepcopy(state.get("manual_overrides", {}))
        self.toi_rule_overrides = copy.deepcopy(state.get("toi_rule_overrides", {}))
        self.current_profile = copy.deepcopy(state.get("current_profile", {}))
        self.refresh_editors()
        self.show_frame()

    def _update_history_buttons(self):
        if hasattr(self, "btn_undo"):
            self.btn_undo.config(state=tk.NORMAL if self.undo_stack else tk.DISABLED)
        if hasattr(self, "btn_redo"):
            self.btn_redo.config(state=tk.NORMAL if self.redo_stack else tk.DISABLED)

    def _commit_action(self, label, before_state):
        after_state = self._snapshot_edit_state()
        if before_state == after_state:
            return
        self.undo_stack.append({
            "label": label,
            "before": before_state,
            "after": after_state,
        })
        self.redo_stack.clear()
        self._update_history_buttons()
        self._autosave_session_state()

    def undo_last_action(self):
        if not self.undo_stack:
            return
        cmd = self.undo_stack.pop()
        self._restore_edit_state(cmd["before"])
        self.redo_stack.append(cmd)
        self._update_history_buttons()

    def redo_last_action(self):
        if not self.redo_stack:
            return
        cmd = self.redo_stack.pop()
        self._restore_edit_state(cmd["after"])
        self.undo_stack.append(cmd)
        self._update_history_buttons()

    def _serialize_session_state(self):
        payload = {
            "updated_at": datetime.now().isoformat(),
            "manual_save_ts": self.last_manual_save_ts,
            "manual_overrides": [],
            "toi_rule_overrides": self.toi_rule_overrides,
            "current_profile": self.current_profile,
        }
        for (f_idx, tid, role, aoi), box in self.logic.manual_overrides.items():
            payload["manual_overrides"].append({
                "frame": int(f_idx),
                "track_id": int(tid),
                "role": str(role),
                "aoi": str(aoi),
                "box": [int(v) for v in box],
            })
        return payload

    def _deserialize_session_state(self, payload):
        self.last_manual_save_ts = payload.get("manual_save_ts")
        self.toi_rule_overrides = payload.get("toi_rule_overrides", {}) or {}
        self.current_profile = payload.get("current_profile", self.current_profile)
        self.logic.manual_overrides = {}
        for row in payload.get("manual_overrides", []):
            key = (
                int(row.get("frame", 0)),
                int(row.get("track_id", -1)),
                str(row.get("role", "")),
                str(row.get("aoi", "")),
            )
            box = row.get("box", [0, 0, 1, 1])
            self.logic.manual_overrides[key] = self.logic._sanitize_box(tuple(box))
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._update_history_buttons()
        self.refresh_editors()

    def _autosave_session_state(self):
        if not self.session_state_path:
            return
        try:
            with open(self.session_state_path, "w", encoding="utf-8") as f:
                json.dump(self._serialize_session_state(), f, indent=2)
        except Exception as exc:
            print(f"Autosave session failed: {exc}")

    def save_session_now(self):
        self.last_manual_save_ts = datetime.now().isoformat()
        self._autosave_session_state()
        messagebox.showinfo("Saved", "AOI session state saved.")

    def _try_restore_newer_autosave(self):
        if not self.session_state_path or not os.path.exists(self.session_state_path):
            return
        try:
            with open(self.session_state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return
        updated = self._parse_timestamp(payload.get("updated_at"))
        manual = self._parse_timestamp(payload.get("manual_save_ts"))
        newer = (updated is not None) and ((manual is None) or (updated > manual))
        if not newer:
            self.last_manual_save_ts = payload.get("manual_save_ts")
            return
        if messagebox.askyesno("Restore Session", "A newer autosaved AOI session was found. Restore it?"):
            self._deserialize_session_state(payload)
            self.show_frame()

    def toggle_manual_mode(self):
        if not self.manual_mode:
            self.manual_mode = True
            self.manual_mode_snapshot = self._snapshot_edit_state()
            self.manual_mode_undo_snapshot = copy.deepcopy(self.undo_stack)
            self.manual_mode_redo_snapshot = copy.deepcopy(self.redo_stack)
            self._manual_prev_scope = self.edit_scope_var.get()
            self.edit_scope_var.set("Frame")
            self.is_playing = False
            self.btn_play.config(text="▶ Play")
            self.btn_manual_mode.config(text="Manual Correction: ON", bg="#2e7d32")
            self.lbl_player.config(text="Manual correction mode enabled: use mouse drag and arrow keys frame-by-frame.")
            self._update_manual_editor_visibility()
            return

        if self.manual_mode_snapshot is None:
            self.manual_mode = False
            self.btn_manual_mode.config(text="Manual Correction: OFF", bg="#d32f2f")
            self._update_manual_editor_visibility()
            return

        keep = messagebox.askyesno(
            "Manual Correction",
            "Keep the manual corrections made in this session?\n\nYes = Keep\nNo = Discard",
        )
        if not keep:
            self._restore_edit_state(self.manual_mode_snapshot)
            self.undo_stack = copy.deepcopy(self.manual_mode_undo_snapshot)
            self.redo_stack = copy.deepcopy(self.manual_mode_redo_snapshot)
            self._update_history_buttons()

        self.manual_mode = False
        if self._manual_prev_scope:
            self.edit_scope_var.set(self._manual_prev_scope)
        self.manual_mode_snapshot = None
        self.manual_mode_undo_snapshot = []
        self.manual_mode_redo_snapshot = []
        self.btn_manual_mode.config(text="Manual Correction: OFF", bg="#d32f2f")
        self._update_manual_editor_visibility()
        self.show_frame()

    def browse_toi(self):
        f = filedialog.askopenfilename(filetypes=[("TOI", "*.tsv *.csv *.txt")])
        if f:
            self.load_toi_direct(f)

    def load_toi_direct(self, path):
        if not os.path.exists(path):
            return
        try:
            if path.lower().endswith(".tsv") or path.lower().endswith(".txt"):
                df = pd.read_csv(path, sep="\t")
            else:
                df = pd.read_csv(path)
            if "Start" not in df.columns or "End" not in df.columns:
                raise ValueError("TOI file must contain Start and End columns.")
            self.toi_df = df.sort_values(by=["Start", "End"]).reset_index(drop=True)
            self.toi_records = self.toi_df.to_dict("records")
            self.timeline.set_data(self.total_frames / max(1e-6, self.fps), self.toi_df)
            self.show_frame()
        except Exception as exc:
            messagebox.showerror("TOI Error", str(exc))

    def seek_to_seconds(self, sec):
        if self.total_frames <= 0:
            return
        frame = int(max(0.0, sec) * max(1e-6, self.fps))
        self.current_frame = max(0, min(self.total_frames - 1, frame))
        self.slider.set(self.current_frame)
        self.show_frame()

    def _active_toi_for_frame(self, frame_idx):
        if not self.toi_records:
            return None, None
        sec = frame_idx / max(1e-6, self.fps)
        match = None
        match_idx = None
        for idx, row in enumerate(self.toi_records):
            try:
                s = float(row.get("Start", -1))
                e = float(row.get("End", -1))
            except Exception:
                continue
            if s <= sec <= e:
                match = row
                match_idx = idx  # last match wins
        return match, match_idx

    def _toi_frame_range(self, toi_row):
        if not toi_row:
            return None
        try:
            s = float(toi_row.get("Start", 0.0))
            e = float(toi_row.get("End", 0.0))
        except Exception:
            return None
        f0 = int(max(0.0, s) * max(1e-6, self.fps))
        f1 = int(max(0.0, e) * max(1e-6, self.fps))
        if self.total_frames > 0:
            f0 = max(0, min(self.total_frames - 1, f0))
            f1 = max(0, min(self.total_frames - 1, f1))
        return f0, max(f0, f1)

    def seek_toi(self, direction):
        if not self.toi_records:
            return
        _row, idx = self._active_toi_for_frame(self.current_frame)
        if idx is None:
            idx = -1 if direction > 0 else 0
        target = idx + direction
        target = max(0, min(len(self.toi_records) - 1, target))
        rng = self._toi_frame_range(self.toi_records[target])
        if not rng:
            return
        self.current_frame = rng[0]
        self.slider.set(self.current_frame)
        self.show_frame()

    def _effective_profile_for_frame(self, frame_idx):
        prof = copy.deepcopy(self.current_profile)
        _row, toi_idx = self._active_toi_for_frame(frame_idx)
        if toi_idx is None:
            return prof
        toi_map = self.toi_rule_overrides.get(str(toi_idx), {})
        if not toi_map:
            return prof
        roles = prof.get("roles", {})
        for role_name, idx_map in toi_map.items():
            role_rules = roles.get(role_name)
            if not isinstance(role_rules, list):
                continue
            for idx_key, changes in idx_map.items():
                try:
                    ridx = int(idx_key)
                except Exception:
                    continue
                if 0 <= ridx < len(role_rules):
                    role_rules[ridx].update(changes)
        return prof

    def _update_toi_labels(self):
        sec = self.current_frame / max(1e-6, self.fps) if self.fps > 0 else 0.0
        self.timeline.update_cursor(sec)
        row, idx = self._active_toi_for_frame(self.current_frame)
        self.current_toi_idx = idx
        if row:
            name = row.get("Name", "TOI")
            phase = row.get("Phase", "")
            cond = row.get("Condition", "")
            self.lbl_toi.config(text=f"TOI: {name} | Phase: {phase} | Condition: {cond}")
        else:
            self.lbl_toi.config(text="TOI: n/a")

    def _apply_manual_box_with_scope(self, item, box):
        scope = self.edit_scope_var.get()
        box = self.logic._sanitize_box(box)
        targets = []

        if scope == "Whole Video":
            targets = list(range(self.total_frames))
        elif scope == "Current TOI":
            row, _idx = self._active_toi_for_frame(self.current_frame)
            rng = self._toi_frame_range(row)
            if rng:
                targets = list(range(rng[0], rng[1] + 1))
        if not targets:
            targets = [self.current_frame]

        for f_idx in targets:
            self.logic.set_manual_override(
                f_idx, item["track_id"], item["role"], item["aoi"], box
            )

    # ── Profile Wizard ──────────────────────────────────────────

    def open_profile_wizard(self):
        win = tk.Toplevel(self.parent)
        win.title("Advanced Profile Wizard")
        win.geometry("760x900")

        v_name = tk.StringVar(value="New_Strategy_Profile")

        self.strat_vars = {}
        rule_rows = []

        # --- Section 1: Name ---
        tk.Label(win, text="1. Profile Name", font=("Bold", 12)).pack(pady=(10, 5))
        f_name = tk.Frame(win)
        f_name.pack(fill=tk.X, padx=20)
        tk.Label(f_name, text="Filename:").pack(side=tk.LEFT)
        tk.Entry(f_name, textvariable=v_name).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # --- Section 2: Roles ---
        tk.Label(win, text="2. Role Configuration", font=("Bold", 12)).pack(pady=(15, 5))

        def load_identity_wiz():
            path = filedialog.askopenfilename(filetypes=[("Identity JSON", "*.json")])
            if path:
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        roles = set(data.values())
                        refresh_roles_ui(roles)
                        messagebox.showinfo("Info", f"Loaded {len(roles)} roles from file.")
                except Exception as e:
                    messagebox.showerror("Error", str(e))

        tk.Button(win, text="📂 Load Identity JSON (Refresh List)", command=load_identity_wiz).pack(fill=tk.X, padx=20, pady=5)

        lf_strat = tk.LabelFrame(win, text="Visualization Mode", padx=10, pady=10)
        lf_strat.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        canvas = tk.Canvas(lf_strat, height=200)
        sb = ttk.Scrollbar(lf_strat, orient="vertical", command=canvas.yview)
        frame_roles = tk.Frame(canvas)
        frame_roles.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame_roles, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        def refresh_roles_ui(roles_set):
            for w in frame_roles.winfo_children():
                w.destroy()
            self.strat_vars.clear()

            roles_set.discard("Ignore")
            roles_set.discard("Noise")
            roles_set.discard("Unknown")
            if not roles_set:
                roles_set = {"Target"}
            roles_set.add("DEFAULT")

            tk.Label(frame_roles, text="Role",     font=("Bold", 9)).grid(row=0, column=0, sticky="w", padx=5)
            tk.Label(frame_roles, text="Strategy", font=("Bold", 9)).grid(row=0, column=1, sticky="w", padx=5)

            r_idx = 1
            for role in sorted(list(roles_set)):
                tk.Label(frame_roles, text=f"{role}:").grid(row=r_idx, column=0, sticky="w", padx=5, pady=2)
                val = 1 if role == "Target" else 2
                v = tk.IntVar(value=val)
                self.strat_vars[role] = v
                fr = tk.Frame(frame_roles)
                fr.grid(row=r_idx, column=1, sticky="w", padx=5)
                tk.Radiobutton(fr, text="AOI",  variable=v, value=1).pack(side=tk.LEFT)
                tk.Radiobutton(fr, text="Box",  variable=v, value=2).pack(side=tk.LEFT)
                tk.Radiobutton(fr, text="Hide", variable=v, value=0).pack(side=tk.LEFT)
                r_idx += 1

        current_roles = set(self.logic.identity_map.values()) if self.logic.identity_map else set()
        refresh_roles_ui(current_roles)

        # --- Section 3: AOI Rule Builder ---
        tk.Label(win, text="3. AOI Rule Builder (from keypoints)", font=("Bold", 12)).pack(pady=(15, 5))
        lf_rules = tk.LabelFrame(win, text="Custom AOI Rules", padx=10, pady=10)
        lf_rules.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        tk.Label(
            lf_rules,
            text="Create reusable AOI rules by selecting YOLO keypoints and shape (box / polygon / circle / oval).",
            fg="gray",
            wraplength=680,
            justify="left",
        ).pack(anchor="w", pady=(0, 6))

        rules_canvas = tk.Canvas(lf_rules, height=260)
        rules_scroll = ttk.Scrollbar(lf_rules, orient="vertical", command=rules_canvas.yview)
        rules_frame = tk.Frame(rules_canvas)
        rules_frame.bind("<Configure>", lambda e: rules_canvas.configure(scrollregion=rules_canvas.bbox("all")))
        rules_canvas.create_window((0, 0), window=rules_frame, anchor="nw")
        rules_canvas.configure(yscrollcommand=rules_scroll.set)
        rules_canvas.pack(side="left", fill="both", expand=True)
        rules_scroll.pack(side="right", fill="y")

        def remove_rule(row):
            if row in rule_rows:
                rule_rows.remove(row)
            row["frame"].destroy()

        def add_rule(seed=None):
            seed = seed or {}
            row = {}

            lf = tk.LabelFrame(rules_frame, text=f"AOI Rule #{len(rule_rows) + 1}", padx=8, pady=8)
            lf.pack(fill=tk.X, pady=5)
            row["frame"] = lf

            v_rname = tk.StringVar(value=seed.get("name", f"AOI_{len(rule_rows) + 1}"))
            v_shape = tk.StringVar(value=str(seed.get("shape", "box")).lower())
            v_margin = tk.IntVar(value=int(seed.get("margin_px", 20)))
            v_scale_w = tk.DoubleVar(value=float(seed.get("scale_w", seed.get("expand_factor", 1.0))))
            v_scale_h = tk.DoubleVar(value=float(seed.get("scale_h", seed.get("expand_factor", 1.0))))
            v_offset = tk.IntVar(value=int(seed.get("offset_y_bottom", 0)))

            top = tk.Frame(lf)
            top.pack(fill=tk.X, pady=2)
            tk.Label(top, text="Name:").pack(side=tk.LEFT)
            tk.Entry(top, textvariable=v_rname, width=22).pack(side=tk.LEFT, padx=5)
            tk.Label(top, text="Shape:").pack(side=tk.LEFT, padx=(10, 0))
            cb_shape = ttk.Combobox(
                top, values=["box", "polygon", "circle", "oval"],
                textvariable=v_shape, state="readonly", width=10
            )
            cb_shape.pack(side=tk.LEFT, padx=5)
            cb_shape.set(v_shape.get())

            params = tk.Frame(lf)
            params.pack(fill=tk.X, pady=2)
            tk.Label(params, text="Margin(px):").pack(side=tk.LEFT)
            tk.Spinbox(params, from_=-200, to=1000, width=8, textvariable=v_margin).pack(side=tk.LEFT, padx=4)
            tk.Label(params, text="Scale W:").pack(side=tk.LEFT)
            tk.Spinbox(params, from_=0.1, to=8.0, increment=0.1, width=6, textvariable=v_scale_w).pack(side=tk.LEFT, padx=4)
            tk.Label(params, text="Scale H:").pack(side=tk.LEFT)
            tk.Spinbox(params, from_=0.1, to=8.0, increment=0.1, width=6, textvariable=v_scale_h).pack(side=tk.LEFT, padx=4)
            tk.Label(params, text="Bottom Off:").pack(side=tk.LEFT)
            tk.Spinbox(params, from_=-500, to=1000, width=8, textvariable=v_offset).pack(side=tk.LEFT, padx=4)

            kp_frame = tk.Frame(lf)
            kp_frame.pack(fill=tk.X, pady=(4, 2))
            tk.Label(kp_frame, text="Keypoints (Ctrl/Shift multi-select):").pack(anchor="w")
            lb = tk.Listbox(kp_frame, height=6, selectmode=tk.EXTENDED, exportselection=False)
            for kp_idx in sorted(KEYPOINTS_MAP.keys()):
                lb.insert(tk.END, f"{kp_idx}: {KEYPOINTS_MAP[kp_idx]}")
            lb.pack(fill=tk.X, padx=2)

            selected_kps = seed.get("kps", [0, 1, 2, 3, 4])
            for kp_idx in selected_kps:
                if 0 <= int(kp_idx) < lb.size():
                    lb.selection_set(int(kp_idx))

            tk.Button(lf, text="Remove Rule", fg="#c62828",
                      command=lambda r=row: remove_rule(r)).pack(anchor="e", pady=(4, 0))

            row["name"] = v_rname
            row["shape"] = v_shape
            row["margin"] = v_margin
            row["scale_w"] = v_scale_w
            row["scale_h"] = v_scale_h
            row["offset"] = v_offset
            row["kps_listbox"] = lb
            rule_rows.append(row)

        def collect_custom_rules():
            rules = []
            for idx, row in enumerate(rule_rows):
                lb = row["kps_listbox"]
                selected = lb.curselection()
                if not selected:
                    continue

                kps = []
                for sidx in selected:
                    token = lb.get(sidx).split(":")[0].strip()
                    try:
                        kps.append(int(token))
                    except ValueError:
                        continue
                if not kps:
                    continue

                shape = str(row["shape"].get()).strip().lower()
                if shape not in ("box", "polygon", "circle", "oval"):
                    shape = "box"

                name = row["name"].get().strip() or f"AOI_{idx + 1}"
                sw = float(row["scale_w"].get())
                sh = float(row["scale_h"].get())

                rule = {
                    "name": name,
                    "shape": shape,
                    "kps": kps,
                    "margin_px": int(row["margin"].get()),
                    "expand_factor": 1.0,
                    "scale_w": sw,
                    "scale_h": sh,
                }
                off = int(row["offset"].get())
                if off != 0:
                    rule["offset_y_bottom"] = off
                rules.append(rule)
            return rules

        buttons_rules = tk.Frame(win)
        buttons_rules.pack(fill=tk.X, padx=20, pady=(2, 6))
        tk.Button(
            buttons_rules,
            text="+ Add AOI Rule",
            bg="#e3f2fd",
            command=lambda: add_rule()
        ).pack(side=tk.LEFT)

        add_rule({
            "name": "FullBody",
            "shape": "box",
            "kps": list(range(17)),
            "margin_px": 20,
            "scale_w": 1.0,
            "scale_h": 1.0,
        })

        # --- Save ---
        def save_wiz():
            name = v_name.get().strip()
            if not name.endswith(".json"):
                name += ".json"

            custom_rules = collect_custom_rules()

            def clone_rules(rules):
                return json.loads(json.dumps(rules))

            def build_rules(strategy_code):
                if strategy_code == 1:
                    return clone_rules(custom_rules)
                elif strategy_code == 2:
                    return [{
                        "name": "FullBody",
                        "shape": "box",
                        "kps": list(range(17)),
                        "margin_px": 20,
                        "expand_factor": 1.0,
                    }]
                else:
                    return []

            has_aoi_roles = any(v.get() == 1 for v in self.strat_vars.values())
            if has_aoi_roles and not custom_rules:
                messagebox.showwarning(
                    "Missing Rules",
                    "At least one custom AOI rule is required when a role is set to AOI mode."
                )
                return

            roles_config = {}
            for role_name, var in self.strat_vars.items():
                roles_config[role_name] = build_rules(var.get())

            new_profile = {"name": name.replace(".json", ""), "roles": roles_config}
            self.pm.save_profile(name, new_profile)
            messagebox.showinfo("Success", f"Profile '{name}' saved!")
            win.destroy()

            self.cb_profile['values'] = self.pm.list_profiles()
            self.cb_profile.set(name)
            self.on_profile_change(None)

        tk.Button(win, text="💾 GENERATE PROFILE", bg="#4CAF50", fg="white",
                  font=("Bold", 12), command=save_wiz).pack(side=tk.BOTTOM, fill=tk.X, pady=20)

    # ── Rule Editors ────────────────────────────────────────────

    def refresh_editors(self):
        for widget in self.frame_target.winfo_children():
            widget.destroy()
        for widget in self.frame_others.winfo_children():
            widget.destroy()
        self._build_role_editor(self.frame_target, "Target")
        self._build_role_editor(self.frame_others, "DEFAULT")

    def _build_frame_correction_editor(self, parent):
        self.frame_correction_container = tk.Frame(parent)
        self.frame_correction_container.pack(fill=tk.X, pady=(0, 10))

        self.lbl_manual_collapsed = tk.Label(
            self.frame_correction_container,
            text="Manual correction tools are hidden. Turn ON 'Manual Correction' to edit AOIs frame-by-frame.",
            fg="gray",
            justify="left",
            wraplength=340,
        )

        lf = tk.LabelFrame(self.frame_correction_container, text="Frame-by-Frame AOI Correction", padx=5, pady=5)
        self.frame_correction_lf = lf
        tk.Label(
            lf,
            text="Tip: turn Manual Correction ON, then click an AOI and drag a green corner (or drag inside to move).",
            fg="gray",
            font=("Arial", 8),
            wraplength=340,
            justify="left",
        ).grid(row=0, column=0, columnspan=4, sticky="w")

        tk.Label(lf, text="AOI in current frame:").grid(row=1, column=0, sticky="w", columnspan=4)
        self.cb_frame_aoi = ttk.Combobox(
            lf, textvariable=self.selected_frame_aoi, state="readonly"
        )
        self.cb_frame_aoi.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(2, 6))
        self.cb_frame_aoi.bind("<<ComboboxSelected>>", self.on_frame_aoi_select)

        tk.Label(lf, text="x1").grid(row=3, column=0, sticky="w")
        tk.Entry(lf, textvariable=self.edit_x1, width=8).grid(row=3, column=1, sticky="w")
        tk.Label(lf, text="y1").grid(row=3, column=2, sticky="w")
        tk.Entry(lf, textvariable=self.edit_y1, width=8).grid(row=3, column=3, sticky="w")

        tk.Label(lf, text="x2").grid(row=4, column=0, sticky="w")
        tk.Entry(lf, textvariable=self.edit_x2, width=8).grid(row=4, column=1, sticky="w")
        tk.Label(lf, text="y2").grid(row=4, column=2, sticky="w")
        tk.Entry(lf, textvariable=self.edit_y2, width=8).grid(row=4, column=3, sticky="w")

        btn_row = tk.Frame(lf)
        btn_row.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        tk.Button(btn_row, text="Apply to Frame", bg="#1976d2", fg="white",
                  command=self.apply_selected_override).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Clear Selected", command=self.clear_selected_override).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Clear Frame", command=self.clear_frame_overrides).pack(side=tk.LEFT, padx=2)

        lf.grid_columnconfigure(0, weight=1)
        lf.grid_columnconfigure(1, weight=1)
        lf.grid_columnconfigure(2, weight=1)
        lf.grid_columnconfigure(3, weight=1)
        self._update_manual_editor_visibility()

    def _update_manual_editor_visibility(self):
        if not hasattr(self, "frame_correction_lf") or not hasattr(self, "lbl_manual_collapsed"):
            return

        if self.manual_mode:
            if self.lbl_manual_collapsed.winfo_manager() == "pack":
                self.lbl_manual_collapsed.pack_forget()
            if self.frame_correction_lf.winfo_manager() != "pack":
                self.frame_correction_lf.pack(fill=tk.X)
        else:
            if self.frame_correction_lf.winfo_manager() == "pack":
                self.frame_correction_lf.pack_forget()
            if self.lbl_manual_collapsed.winfo_manager() != "pack":
                self.lbl_manual_collapsed.pack(fill=tk.X)

    def _set_box_vars(self, box):
        x1, y1, x2, y2 = box
        self.edit_x1.set(int(x1))
        self.edit_y1.set(int(y1))
        self.edit_x2.set(int(x2))
        self.edit_y2.set(int(y2))

    @staticmethod
    def _aoi_key(item):
        shp = item.get("shape_type", "box")
        return f"ID {item['track_id']} | {item['role']} | {item['aoi']} [{shp}]"

    def _current_selected_aoi_item(self):
        key = self.selected_frame_aoi.get()
        return self.frame_aoi_lookup.get(key)

    def _select_frame_aoi_item(self, item):
        key = self._aoi_key(item)
        if key not in self.frame_aoi_lookup:
            return False
        self.selected_frame_aoi.set(key)
        self._set_box_vars(self.frame_aoi_lookup[key]["box"])
        return True

    @staticmethod
    def _point_in_box(px, py, box):
        x1, y1, x2, y2 = box
        return x1 <= px <= x2 and y1 <= py <= y2

    @staticmethod
    def _point_in_polygon(px, py, points):
        if len(points) < 3:
            return False
        inside = False
        j = len(points) - 1
        for i in range(len(points)):
            xi, yi = points[i]
            xj, yj = points[j]
            intersects = ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / max(1e-6, (yj - yi)) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def _point_in_item(self, px, py, item):
        shape_type = item.get("shape_type", "box")
        if shape_type == "circle":
            cx = item.get("cx", int((item["box"][0] + item["box"][2]) / 2))
            cy = item.get("cy", int((item["box"][1] + item["box"][3]) / 2))
            r = max(1, item.get("radius", int(min(item["box"][2] - item["box"][0],
                                                   item["box"][3] - item["box"][1]) / 2)))
            dx = px - cx
            dy = py - cy
            return (dx * dx + dy * dy) <= (r * r)

        if shape_type in ("oval", "ellipse"):
            cx = item.get("cx", int((item["box"][0] + item["box"][2]) / 2))
            cy = item.get("cy", int((item["box"][1] + item["box"][3]) / 2))
            rx = max(1, item.get("rx", int((item["box"][2] - item["box"][0]) / 2)))
            ry = max(1, item.get("ry", int((item["box"][3] - item["box"][1]) / 2)))
            nx = (px - cx) / rx
            ny = (py - cy) / ry
            return (nx * nx + ny * ny) <= 1.0

        if shape_type == "polygon":
            return self._point_in_polygon(px, py, item.get("points", []))

        return self._point_in_box(px, py, item["box"])

    @staticmethod
    def _corner_points(box):
        x1, y1, x2, y2 = box
        return {
            "tl": (x1, y1),
            "tr": (x2, y1),
            "bl": (x1, y2),
            "br": (x2, y2),
        }

    def _corner_hit(self, px, py, box):
        if self._disp_w <= 0 or self._orig_w <= 0:
            return None

        scale_x = self._orig_w / max(1, self._disp_w)
        scale_y = self._orig_h / max(1, self._disp_h)
        radius = max(6, int(10 * max(scale_x, scale_y)))
        r2 = radius * radius

        for name, (cx, cy) in self._corner_points(box).items():
            dx = px - cx
            dy = py - cy
            if (dx * dx + dy * dy) <= r2:
                return name
        return None

    def _find_aoi_from_point(self, px, py):
        selected = self._current_selected_aoi_item()
        if selected:
            corner = self._corner_hit(px, py, selected["box"])
            if corner:
                return selected, "corner", corner

        best_corner = None
        for item in self.frame_aoi_items:
            corner = self._corner_hit(px, py, item["box"])
            if corner:
                best_corner = (item, corner)
                break
        if best_corner:
            return best_corner[0], "corner", best_corner[1]

        inside = []
        for item in self.frame_aoi_items:
            if self._point_in_item(px, py, item):
                x1, y1, x2, y2 = item["box"]
                area = max(1, (x2 - x1) * (y2 - y1))
                inside.append((area, item))
        if inside:
            inside.sort(key=lambda it: it[0])
            return inside[0][1], "move", None

        return None, None, None

    def _widget_to_frame_xy(self, wx, wy):
        if self._disp_w <= 0 or self._disp_h <= 0:
            return None

        if wx < self._disp_off_x or wy < self._disp_off_y:
            return None
        if wx >= self._disp_off_x + self._disp_w or wy >= self._disp_off_y + self._disp_h:
            return None

        rel_x = wx - self._disp_off_x
        rel_y = wy - self._disp_off_y
        px = int(rel_x * self._orig_w / max(1, self._disp_w))
        py = int(rel_y * self._orig_h / max(1, self._disp_h))
        px = max(0, min(self._orig_w - 1, px))
        py = max(0, min(self._orig_h - 1, py))
        return px, py

    def on_frame_aoi_select(self, _event=None):
        item = self._current_selected_aoi_item()
        if not item:
            return
        self._set_box_vars(item["box"])
        self.show_frame()

    def apply_selected_override(self):
        if not self.manual_mode:
            messagebox.showinfo("Manual Mode", "Enable Manual Correction mode to apply frame-level edits.")
            return
        item = self._current_selected_aoi_item()
        if not item:
            messagebox.showwarning("No AOI", "Select an AOI in the current frame first.")
            return

        before = self._snapshot_edit_state()
        box = (self.edit_x1.get(), self.edit_y1.get(), self.edit_x2.get(), self.edit_y2.get())
        sanitized = self.logic._sanitize_box(box)
        self._apply_manual_box_with_scope(item, sanitized)
        self.show_frame()
        self._commit_action("Apply manual override", before)

    def clear_selected_override(self):
        if not self.manual_mode:
            messagebox.showinfo("Manual Mode", "Enable Manual Correction mode to clear frame-level edits.")
            return
        item = self._current_selected_aoi_item()
        if not item:
            messagebox.showwarning("No AOI", "Select an AOI in the current frame first.")
            return

        before = self._snapshot_edit_state()
        scope = self.edit_scope_var.get()
        if scope == "Whole Video":
            for f_idx in range(self.total_frames):
                self.logic.clear_manual_override(f_idx, item["track_id"], item["role"], item["aoi"])
        elif scope == "Current TOI":
            row, _idx = self._active_toi_for_frame(self.current_frame)
            rng = self._toi_frame_range(row)
            if rng:
                for f_idx in range(rng[0], rng[1] + 1):
                    self.logic.clear_manual_override(f_idx, item["track_id"], item["role"], item["aoi"])
            else:
                self.logic.clear_manual_override(self.current_frame, item["track_id"], item["role"], item["aoi"])
        else:
            self.logic.clear_manual_override(self.current_frame, item["track_id"], item["role"], item["aoi"])
        self.show_frame()
        self._commit_action("Clear selected override", before)

    def clear_frame_overrides(self):
        if not self.manual_mode:
            messagebox.showinfo("Manual Mode", "Enable Manual Correction mode to clear overrides.")
            return
        before = self._snapshot_edit_state()
        scope = self.edit_scope_var.get()
        if scope == "Whole Video":
            self.logic.manual_overrides = {}
        elif scope == "Current TOI":
            row, _idx = self._active_toi_for_frame(self.current_frame)
            rng = self._toi_frame_range(row)
            if rng:
                for f_idx in range(rng[0], rng[1] + 1):
                    self.logic.clear_overrides_for_frame(f_idx)
            else:
                self.logic.clear_overrides_for_frame(self.current_frame)
        else:
            self.logic.clear_overrides_for_frame(self.current_frame)
        self.show_frame()
        self._commit_action("Clear overrides by scope", before)

    def _refresh_frame_correction_ui(self, items):
        previous = self.selected_frame_aoi.get()
        self.frame_aoi_items = items
        self.frame_aoi_lookup = {}
        values = []

        for item in items:
            key = self._aoi_key(item)
            self.frame_aoi_lookup[key] = item
            values.append(key)

        self.cb_frame_aoi["values"] = values

        if previous in self.frame_aoi_lookup:
            self.selected_frame_aoi.set(previous)
        elif values:
            self.selected_frame_aoi.set(values[0])
        else:
            self.selected_frame_aoi.set("")
            self._set_box_vars((0, 0, 1, 1))
            return

        item = self._current_selected_aoi_item()
        if item:
            self._set_box_vars(item["box"])

    def _build_role_editor(self, parent, role_key):
        rules = self.current_profile.get("roles", {}).get(role_key, [])
        canvas = tk.Canvas(parent)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        for idx, rule in enumerate(rules):
            shp = rule.get("shape", "box")
            lf = tk.LabelFrame(frame, text=f"AOI: {rule['name']} [{shp}]", pady=5, padx=5)
            lf.pack(fill=tk.X, pady=5)

            # Margin
            tk.Label(lf, text="Margin (px):").grid(row=0, column=0, sticky="e")
            s_margin = tk.Scale(lf, from_=0, to=100, orient=tk.HORIZONTAL)
            s_margin.set(rule.get("margin_px", 0))
            s_margin.grid(row=0, column=1, sticky="ew")
            s_margin.bind("<ButtonRelease-1>",
                          lambda e, r=role_key, i=idx, s=s_margin: self.update_rule_val(r, i, "margin_px", s.get()))

            # Width Scale
            cur_w = rule.get("scale_w", rule.get("expand_factor", 1.0))
            tk.Label(lf, text="Width Scale:", fg="#d32f2f").grid(row=1, column=0, sticky="e")
            s_w = tk.Scale(lf, from_=0.5, to=4.0, resolution=0.1, orient=tk.HORIZONTAL, fg="#d32f2f")
            s_w.set(cur_w)
            s_w.grid(row=1, column=1, sticky="ew")
            s_w.bind("<ButtonRelease-1>",
                     lambda e, r=role_key, i=idx, s=s_w: self.update_rule_val(r, i, "scale_w", s.get()))

            # Height Scale
            cur_h = rule.get("scale_h", rule.get("expand_factor", 1.0))
            tk.Label(lf, text="Height Scale:", fg="#1976d2").grid(row=2, column=0, sticky="e")
            s_h = tk.Scale(lf, from_=0.5, to=4.0, resolution=0.1, orient=tk.HORIZONTAL, fg="#1976d2")
            s_h.set(cur_h)
            s_h.grid(row=2, column=1, sticky="ew")
            s_h.bind("<ButtonRelease-1>",
                     lambda e, r=role_key, i=idx, s=s_h: self.update_rule_val(r, i, "scale_h", s.get()))

            # Bottom offset (only if present)
            if "offset_y_bottom" in rule:
                tk.Label(lf, text="Bottom Ext:", fg="gray").grid(row=3, column=0, sticky="e")
                s_off = tk.Scale(lf, from_=0, to=100, orient=tk.HORIZONTAL, fg="gray")
                s_off.set(rule.get("offset_y_bottom", 0))
                s_off.grid(row=3, column=1, sticky="ew")
                s_off.bind("<ButtonRelease-1>",
                           lambda e, r=role_key, i=idx, s=s_off: self.update_rule_val(r, i, "offset_y_bottom", s.get()))

    def update_rule_val(self, role, idx, key, val):
        before = self._snapshot_edit_state()
        scope = self.param_scope_var.get()
        if scope == "Current TOI":
            _row, toi_idx = self._active_toi_for_frame(self.current_frame)
            if toi_idx is None:
                messagebox.showwarning("No TOI", "No active TOI at current frame. Switch scope to Whole Video.")
                return
            toi_map = self.toi_rule_overrides.setdefault(str(toi_idx), {})
            role_map = toi_map.setdefault(role, {})
            idx_map = role_map.setdefault(str(idx), {})
            idx_map[key] = val
        else:
            self.current_profile["roles"][role][idx][key] = val
            # Reset any TOI-specific override for this exact key so whole-video stays authoritative.
            for toi_map in self.toi_rule_overrides.values():
                role_map = toi_map.get(role, {})
                idx_map = role_map.get(str(idx), {})
                if key in idx_map:
                    del idx_map[key]
        self.show_frame()
        self._commit_action(f"Update rule {role}[{idx}] {key}", before)

    def on_profile_change(self, e):
        before = self._snapshot_edit_state()
        self.current_profile = self.pm.load_profile(self.cb_profile.get())
        self.toi_rule_overrides = {}
        self.logic.manual_overrides = {}
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._update_history_buttons()
        self.refresh_editors()
        self.show_frame()
        self._commit_action("Change profile", before)

    # ── Diagnostics (delegates to Logic) ────────────────────────

    def run_diagnostics(self):
        prof = self._effective_profile_for_frame(self.current_frame)
        report = self.logic.get_diagnostics_report(
            self.current_frame, prof, self.kp_conf_thresh.get()
        )
        print("\n" + report + "\n")

    # ── Data Loading (thin wrappers) ────────────────────────────

    def browse_video(self):
        f = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi")])
        if f:
            self.load_video_direct(f)

    def load_video_direct(self, path):
        if not os.path.exists(path):
            return
        self.video_path = path
        self.context.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.logic.fps = self.fps
        self.logic.total_frames = self.total_frames
        self.slider.config(to=self.total_frames - 1)
        self.timeline.set_data(self.total_frames / max(1e-6, self.fps), self.toi_df)
        self.show_frame()

    def browse_pose(self):
        f = filedialog.askopenfilename(filetypes=[("Pose JSON", "*.json.gz")])
        if f:
            self.load_pose_direct(f)

    def load_pose_direct(self, path):
        if not os.path.exists(path):
            return
        self.context.pose_data_path = path

        def _worker():
            try:
                count = self.logic.load_pose_data(
                    path, progress_callback=lambda m: print(m)
                )
                self.parent.after(0, lambda: self._on_pose_loaded(count))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                err_msg = f"Error loading poses: {exc}"
                self.parent.after(0, lambda: messagebox.showerror(
                    "Error", err_msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_pose_loaded(self, count):
        print(f"Poses loaded: {count} frames.")
        self.show_frame()

    def browse_identity(self):
        f = filedialog.askopenfilename(filetypes=[("Identity", "*.json")])
        if f:
            self.load_identity_direct(f)

    def load_identity_direct(self, path):
        if not os.path.exists(path):
            return
        self.context.identity_map_path = path
        count = self.logic.load_identity_map(path)
        print(f"Identities loaded: {count} IDs.")
        self.show_frame()

    def _on_video_mouse_down(self, event):
        if not self.manual_mode:
            return
        if not self.cap or not self.frame_aoi_items:
            return

        pt = self._widget_to_frame_xy(event.x, event.y)
        if not pt:
            return

        self.is_playing = False
        px, py = pt
        item, mode, corner = self._find_aoi_from_point(px, py)
        if not item:
            return

        self._select_frame_aoi_item(item)
        self.drag_state = {
            "mode": mode,
            "corner": corner,
            "start_pt": (px, py),
            "start_box": tuple(item["box"]),
            "track_id": item["track_id"],
            "role": item["role"],
            "aoi": item["aoi"],
            "before_state": self._snapshot_edit_state(),
        }
        self.show_frame()

    def _on_video_mouse_drag(self, event):
        if not self.drag_state:
            return

        pt = self._widget_to_frame_xy(event.x, event.y)
        if not pt:
            return

        px, py = pt
        x1, y1, x2, y2 = self.drag_state["start_box"]

        if self.drag_state["mode"] == "corner":
            corner = self.drag_state["corner"]
            if corner == "tl":
                x1, y1 = px, py
            elif corner == "tr":
                x2, y1 = px, py
            elif corner == "bl":
                x1, y2 = px, py
            elif corner == "br":
                x2, y2 = px, py
        else:
            sx, sy = self.drag_state["start_pt"]
            dx = px - sx
            dy = py - sy
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            x1 = max(0, x1 + dx)
            y1 = max(0, y1 + dy)
            x2 = x1 + w
            y2 = y1 + h

        new_box = self.logic._sanitize_box((x1, y1, x2, y2))
        item = {
            "track_id": self.drag_state["track_id"],
            "role": self.drag_state["role"],
            "aoi": self.drag_state["aoi"],
        }
        self._apply_manual_box_with_scope(item, new_box)
        self.show_frame()

    def _on_video_mouse_up(self, _event):
        if self.drag_state and "before_state" in self.drag_state:
            self._commit_action("Manual drag AOI", self.drag_state["before_state"])
        self.drag_state = None

    @staticmethod
    def _draw_aoi_shape(frame, item, color, thickness=2):
        shape_type = item.get("shape_type", "box")
        x1, y1, x2, y2 = item["box"]

        if shape_type == "circle":
            cx = int(item.get("cx", (x1 + x2) / 2))
            cy = int(item.get("cy", (y1 + y2) / 2))
            r = max(1, int(item.get("radius", min(x2 - x1, y2 - y1) / 2)))
            cv2.circle(frame, (cx, cy), r, color, thickness)
            return

        if shape_type in ("oval", "ellipse"):
            cx = int(item.get("cx", (x1 + x2) / 2))
            cy = int(item.get("cy", (y1 + y2) / 2))
            rx = max(1, int(item.get("rx", (x2 - x1) / 2)))
            ry = max(1, int(item.get("ry", (y2 - y1) / 2)))
            angle = float(item.get("angle", 0))
            cv2.ellipse(frame, (cx, cy), (rx, ry), angle, 0, 360, color, thickness)
            return

        if shape_type == "polygon":
            pts = item.get("points", [])
            if len(pts) >= 2:
                pts_cv = [(int(p[0]), int(p[1])) for p in pts]
                for i in range(len(pts_cv)):
                    p1 = pts_cv[i]
                    p2 = pts_cv[(i + 1) % len(pts_cv)]
                    cv2.line(frame, p1, p2, color, thickness)
                return

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # ── Rendering ───────────────────────────────────────────────

    def show_frame(self):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return
        self._orig_h, self._orig_w = frame.shape[:2]

        # Delegate all geometry to Logic
        prof = self._effective_profile_for_frame(self.current_frame)
        items = self.logic.get_render_data(
            self.current_frame, prof, self.kp_conf_thresh.get()
        )
        for item in items:
            x1, y1, x2, y2 = item["box"]
            c = item["color"]
            self._draw_aoi_shape(frame, item, c, 2)
            suffix = "*" if item.get("corrected") else ""
            shape_suffix = f"[{item.get('shape_type', 'box')}]"
            label = f"{item['label']}#{item['track_id']}{shape_suffix}{suffix}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

        selected_key = self.selected_frame_aoi.get()
        selected_item = None
        if selected_key:
            for item in items:
                if self._aoi_key(item) == selected_key:
                    selected_item = item
                    break
        if selected_item:
            x1, y1, x2, y2 = selected_item["box"]
            self._draw_aoi_shape(frame, selected_item, (0, 255, 0), 2)
            for cx, cy in self._corner_points(selected_item["box"]).values():
                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), -1)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        w, h = self.lbl_video.winfo_width(), self.lbl_video.winfo_height()
        if w < 10:
            w = 800
            h = 600
        img.thumbnail((w, h), Image.Resampling.BILINEAR)
        self._disp_w, self._disp_h = img.size
        self._disp_off_x = max(0, (w - self._disp_w) // 2)
        self._disp_off_y = max(0, (h - self._disp_h) // 2)
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.lbl_video.config(image=self.tk_img)
        self._refresh_frame_correction_ui(items)
        self._update_toi_labels()

        sec = self.current_frame / max(1e-6, self.fps) if self.fps > 0 else 0.0
        self.lbl_player.config(
            text=f"Frame {self.current_frame}/{max(0, self.total_frames - 1)} | {sec:.3f}s | FPS {self.fps:.2f} | Manual {'ON' if self.manual_mode else 'OFF'}"
        )

    # ── Playback ────────────────────────────────────────────────

    def on_seek(self, v):
        self.drag_state = None
        self.current_frame = int(float(v))
        self.show_frame()

    def toggle_play(self):
        if self.manual_mode:
            return
        self.is_playing = not self.is_playing
        self.btn_play.config(text="⏸ Pause" if self.is_playing else "▶ Play")
        if self.is_playing:
            self.play_loop()

    def play_loop(self):
        if not self.is_playing or not self.cap:
            return
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.is_playing = False
            self.btn_play.config(text="▶ Play")
            return
        self.slider.set(self.current_frame)
        self.show_frame()
        delay = int(1000 / max(1e-6, self.fps))
        self.parent.after(max(1, delay), self.play_loop)

    # ── Export (threaded) ───────────────────────────────────────

    def export_data(self):
        if not self.logic.pose_data:
            messagebox.showwarning("No Data", "Load pose data first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".csv")
        if not out:
            return
        self.context.export_path = out

        def _worker():
            try:
                count = self.logic.export_csv(
                    out, self.current_profile, self.kp_conf_thresh.get(),
                    progress_callback=lambda m: print(m),
                    profile_for_frame_fn=lambda f_idx, _p: self._effective_profile_for_frame(f_idx),
                )
                self.parent.after(0, lambda: self._on_export_done(out, count))
            except Exception as exc:
                err_msg = str(exc)
                self.parent.after(0, lambda: messagebox.showerror("Export Error", err_msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_export_done(self, path, count):
        self.context.aoi_csv_path = path
        messagebox.showinfo("OK", f"Export complete: {count} rows.")
