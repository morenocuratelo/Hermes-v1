# HERMES - Region Module Developer Guide

This document details the internal logic, parameters, and workflows of the **Region** module (Spatial AOI Definition). It is intended for developers or researchers wishing to modify the underlying scripts for dynamic Area of Interest generation.

## 1. Global Constants & Keypoints

The module relies on the standard COCO Keypoint format used by YOLO-Pose. These indices are mapped to anatomical names in `KEYPOINTS_MAP`.

```python
KEYPOINTS_MAP = {
    0: "Nose", 1: "L_Eye", 2: "R_Eye", 3: "L_Ear", 4: "R_Ear",
    5: "L_Shoulder", 6: "R_Shoulder", 7: "L_Elbow", 8: "R_Elbow",
    9: "L_Wrist", 10: "R_Wrist", 11: "L_Hip", 12: "R_Hip",
    13: "L_Knee", 14: "R_Knee", 15: "L_Ankle", 16: "R_Ankle"
}
```

## 2. Profile Management (`AOIProfileManager`)

Profiles define how raw keypoints are converted into semantic Areas of Interest (e.g., "Face", "Hands"). They are stored as JSON files in `assets/profiles_aoi`.

### 2.1. Profile Structure

A profile consists of **Roles** (mapped from the Entity module) and **Rules** for each role.

```json
{
    "name": "Invasion Profile",
    "roles": {
        "Target": [
            {
                "name": "Face",
                "shape": "box",
                "kps": [0, 1, 2, 3, 4],
                "margin_px": 30,
                "scale_w": 1.0,
                "scale_h": 1.0
            }
        ],
        "DEFAULT": [...]
    }
}
```

### 2.2. Rule Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **name** | String | Name of the AOI (e.g., "Face"). |
| **shape** | String | Geometry type: `box`, `circle`, `oval`, `polygon`. |
| **kps** | List[int] | Indices of keypoints used to calculate the base bounding box. |
| **margin_px** | Int | Padding added to the raw bounding box of keypoints. |
| **scale_w/h** | Float | Multiplier to expand/shrink width or height from the center. |
| **offset_y_bottom** | Int | Extra pixels added to the bottom edge (useful for torsos/legs). |

---

## 3. Logic Layer: `RegionLogic`

The `RegionLogic` class handles the geometric computations and state management, decoupled from the UI.

### 3.1. Geometry Engine (`calculate_shape`)

This is the core function that transforms keypoints into shapes.

1.  **Base Box Calculation:**
    *   Extracts valid keypoints (confidence > threshold).
    *   Computes min/max X and Y.
    *   Applies `margin_px`.
    *   Applies `scale_w` and `scale_h` relative to the center.
    *   Applies `offset_y_bottom`.

2.  **Shape Morphing:**
    *   **Box:** Returns the calculated rectangle.
    *   **Circle:** Calculates center and radius based on the box dimensions.
    *   **Oval:** Calculates center and semi-axes (rx, ry).
    *   **Polygon:** Maps the relative positions of keypoints from the source box to the expanded box, ordering points to form a convex hull-like shape.

### 3.2. Data Hierarchy & Overrides

The system uses a strict hierarchy to determine what is shown on screen for a given frame/track/AOI:

1.  **Level 0 (Base):** Automatic calculation from YOLO keypoints using Profile Rules.
2.  **Level 1 (Manual Override):** Explicit shapes stored in `self.manual_overrides`. These take precedence over Level 0.

**Storage Key:** `(frame_idx, track_id, role, aoi_name)`

### 3.3. Ghost Tracks (`find_ghost_tracks`)

*   **Function:** Identifies tracks that are missing in the current frame but present in neighbors.
*   **Logic:**
    *   Scans a window of +/- `ghost_window_var` frames.
    *   If a track exists in a neighbor but not in the current frame (and hasn't been manually overridden), it is returned as a "Ghost".
    *   **Purpose:** Allows the user to "Force Add" an AOI by copying the position from a nearby frame, useful for occlusions or detection failures.

### 3.4. Export (`export_csv`)

Generates the final dataset.

*   **Columns:** Frame, Timestamp, TrackID, Role, AOI, ShapeType, Coordinates (x1, y1, x2, y2), Geometric details (Radius, Angle, etc.), Corrected (Bool).
*   **Logic:** Iterates through all frames, applying the Profile Rules and Manual Overrides to generate the final geometry state.

---

## 4. Presentation Layer: `RegionView`

The UI orchestrates the workflow and visualization.

### 4.1. Manual Correction Mode

A stateful mode that enables editing tools.

*   **State Snapshot:** When entering, the current session state is saved to allow cancellation.
*   **Force Add:**
    *   **Auto/Ghost:** Uses `find_ghost_tracks` to seed the new box position.
    *   **Center:** Seeds the box at the screen center if no ghost is found.
    *   **Draw:** Allows drawing the box/circle directly on the video canvas.
*   **Interpolation:**
    *   **Linear:** Linearly interpolates box coordinates between two existing anchor frames (manual or automatic) over a selected scope.

### 4.2. Scopes

Operations (like applying a manual edit or interpolation) act on a specific scope:
*   **Frame:** Only the current frame.
*   **Current TOI:** All frames within the active Time Interval of Interest.
*   **Whole Video:** Every frame in the dataset.

### 4.3. Session Management

*   **Autosave:** State is saved to `_aoi_edit_session.json` on every commit action.
*   **Undo/Redo:** Implements a command pattern stack storing before/after states of overrides.

### 4.4. Profile Wizard

A GUI tool to generate JSON profiles without editing text files.
*   **Visual Feedback:** Lists available keypoints.
*   **Strategies:** Allows defining different logic for Targets vs. Non-Targets.