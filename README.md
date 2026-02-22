# H.E.R.M.E.S. - Human-centric Eye-tracking & Robust Motion Estimation Suite

**HERMES** is a modular research framework designed to synchronize, analyze, and visualize eye-tracking data in conjunction with computer-vision-based kinematic extraction. The suite addresses the methodological challenge of defining dynamic Areas of Interest (AOIs) on moving human targets without manual annotation.

By integrating **YOLO**-based pose estimation with rigorous temporal synchronization protocols, HERMES enables researchers to map gaze data onto semantic body regions (e.g., Face, Hands, Peripersonal Space) within complex experimental settings.

---

## 📥 Installation and Setup

### Prerequisites
*   **Operating System:** Windows 10/11 (64-bit).
*   **Hardware:** An NVIDIA GPU is recommended (CUDA support) for the Human module; CPU inference is supported but significantly slower.
*   **Software:** Python 3.10+ (the setup script manages automatic dependency installation).

### Installation Procedure

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/morenocuratelo/Hermes-v1
    ```
2.  **Run the Setup:**
    *   Navigate to the project directory.
    *   Execute the `SETUP_LAB.bat` file.
    *   *Note:* This script will install `uv` (package manager), create an isolated virtual environment, install all dependencies, and download the required AI model weights.

3.  **Launch:**
    *   Once the setup is complete, launch the application via `AVVIA_HERMES.bat`.

---

## 🚀 Workflow

The software enforces a sequential workflow to ensure data integrity. Follow the numbered modules in the interface:

1.  **Human (Pose Estimation):** Extracts the skeleton and tracks individuals across video frames.
2.  **Entity (Identity Assignment):** Assigns roles (e.g., "Target") to anonymous tracks and corrects tracking errors.
3.  **Region (AOI Definition):** Defines the geometric rules for AOIs (e.g., Face = Nose + Eyes).
4.  **Master TOI (Synchronization):** Synchronizes eye-tracker logs with the video and defines temporal phases.
5.  **Eye Mapping:** Performs geometric intersection between gaze coordinates and dynamic AOIs.
6.  **Stats:** Generates statistical reports and comprehensive Excel files.

---

## ⚙️ Module Tuning and Configuration Guide

This section describes how to configure and optimize each module to suit specific experimental requirements.

### 1. Human (Kinematic Extraction)
Employs YOLO to extract 17 skeletal keypoints per detected individual.

*   **Key Parameters:**
    *   **Confidence Threshold (`CONF_THRESHOLD`):** Default `0.6`. A conservative threshold that balances precision and recall. Lower this value if the participant is not detected under low-illumination conditions.
    *   **Tracker:** Supports `BoT-SORT` (default) and `ByteTrack`.
    *   **Re-Identification (ReID):** Enable this option to reduce identity switches (ID switches) when participants cross paths. Requires automatic download of supplementary models (e.g., `resnet50`).
*   **Output:** Generates a `.json.gz` file (raw data) and a flattened `.csv` file containing coordinates and confidence values for each keypoint.

### 2. Entity (Identity Assignment)
A post-processing interface for correcting tracking errors.

*   **Features:**
    *   **Merge:** Combines two fragmented tracks (e.g., ID 5 becomes part of ID 2).
    *   **Split:** Divides a single track into two segments when the ID has erroneously jumped from one individual to another.
    *   **Auto-Stitch:** Attempts to automatically join fragmented tracks based on spatial and temporal proximity.
*   **Auto-Stitch Tuning Parameters:**
    *   `Lookahead`: Number of future frames to search for a match.
    *   `Time Gap`: Maximum time interval (in seconds) permitted between two tracks to be joined.
    *   `Stitch Dist`: Maximum distance in pixels between the endpoint of Track A and the starting point of Track B.

### 3. Region (Dynamic AOIs)
Defines Areas of Interest based on detected keypoints.

*   **JSON Profiles:** Rules are stored in `assets/profiles_aoi`. Example rule for "Face":
    *   `kps`: [0, 1, 2, 3, 4] (Nose, Eyes, Ears).
    *   `margin_px`: Padding added to the keypoint bounding box.
    *   `shape`: `box`, `circle`, `oval`, or `polygon`.
*   **Ghost Tracks:** The system automatically detects frames in which tracking data is absent but present in adjacent frames, enabling interpolation or positional copying of the AOI ("Force Add").

### 4. Master TOI (Synchronization)
Aligns asynchronous data streams (Tobii vs. Video).

*   **Synchronization Logic:** Computes a linear offset based on a common event marker (e.g., "VideoStart" in both the Tobii log and the experimental log).
*   **Data Cropping:** Once the TOIs (Times of Interest) are defined, the module can generate `_CROPPED.csv` files containing only data pertaining to the phases of interest, substantially reducing file sizes for subsequent statistical analysis.

### 5. Eye Mapping
Performs geometric hit-testing.

*   **Logic:**
    1.  Loads AOIs generated by the Region module.
    2.  Converts gaze timestamps to video frames: `Frame = (Timestamp - Offset) * FPS`.
    3.  Determines whether the gaze point falls within one or more AOIs.
    4.  **Overlap Resolution:** When the gaze intersects multiple AOIs simultaneously (e.g., Face within Body), the AOI with the **smallest area** (i.e., the most specific region) takes precedence.

### 6. Statistics
Generates the final analytical report.

*   **Computed Metrics:**
    *   **Duration:** Total dwell time within the AOI.
    *   **Percentage:** Proportion of phase duration spent fixating on the AOI.
    *   **Latency:** Time to first entry into the AOI from the onset of the phase.
    *   **Glances:** Number of gaze transitions into the AOI.
*   **Master Report:** Capable of generating a multi-sheet Excel file containing Stats, Raw Data, Mapping, and configurations, ready for archival and reproducibility purposes.

---

## 🛠 System Architecture

The software is built on **Python 3.12** and leverages **Tkinter** for the graphical user interface, ensuring native Windows compatibility without reliance on heavyweight frameworks. It employs a "Hub & Spoke" architecture governed by a centralized `AppContext` that guarantees state persistence across modules.

### Technical Stack
*   **GUI:** Tkinter / Tcl
*   **Computer Vision:** OpenCV, Ultralytics (YOLOv8/v11)
*   **Data Manipulation:** Pandas, NumPy, SciPy
*   **Packaging:** uv (environment management), PyInstaller (distribution)

---

## 🐞 Debugging Tools

The suite includes dedicated scripts for diagnostics and code validation:

1.  **Hermes Diagnostics (`hermes_diagnostics.py`):** Analyzes input files (CSV, MAT, JSON) to detect format errors.
    *   *Usage:* `python hermes_diagnostics.py`
2.  **Architecture Validator (`hermes_architecture_validator.py`):** Verifies the separation between Logic and GUI.
    *   *Usage:* `python hermes_architecture_validator.py`
3.  **Logic Smoke Test (`hermes_logic_test.py`):** Unit tests for logic components.
    *   *Usage:* `python hermes_logic_test.py`

For full documentation, see `docs/hermes_debug_tools.md`.

---

## � Citation and Disclaimer

If you use HERMES in your research, please refer to the internal laboratory documentation for the appropriate citation format.

**Disclaimer:** This software is provided "as is" for research purposes only. Users must ensure compliance with GDPR regulations and institutional ethical guidelines when processing video data containing identifiable human participants.


# HERMES - Entity Module Developer Guide

This document describes the internal logic, data flows, and transformations implemented in the **Entity** module (`hermes_entity.py`). The module is responsible for assigning identities (Roles) to the tracks generated by YOLO, enabling manual and semi-automatic correction of tracking errors (fragmentation, ID switches).

## 1. Memory Management: `HistoryManager`

To support destructive operations such as Merge and Split with Undo/Redo functionality, the module implements a hybrid RAM/Disk state manager.

*   **Logic:**
    *   Maintains a stack of states (`undo_stack`).
    *   Each state is a deep copy (pickle-serialized) of the track data.
    *   **RAM Buffer:** The first N states (default: 5) are retained in RAM for rapid access.
    *   **Disk Spilling:** Older states are serialized to temporary files on disk to prevent memory saturation, particularly with long-duration videos containing numerous tracks.
    *   **Cleanup:** Upon termination, all temporary files are deleted.

## 2. Track Manipulation Logic: `IdentityLogic`

This class manages the primary data structures and algorithmic operations.

### 2.1. Data Structure (`self.tracks`)

Data are not maintained as a frame-indexed list (as in YOLO output) but are aggregated by ID ("Track-Oriented").

*   **Structure:** Dictionary `{ TrackID : TrackData }`
*   **TrackData:**
    *   `frames`: Ordered list of frames in which the ID appears.
    *   `boxes`: Corresponding list of bounding boxes `[x1, y1, x2, y2]`.
    *   `role`: Assigned role (e.g., "Target", "Ignore"). Default: "Ignore".
    *   `merged_from`: List of original IDs that have been merged into this track.
*   **Lineage (`self.id_lineage`):** Map `{ Original_ID : Current_Master_ID }`. This is essential for final export: it records, for instance, that YOLO's ID 5 is now part of ID 2.

### 2.2. Data Loading (`load_from_json_gz`)

*   **Input:** `.json.gz` file generated by the Human module (YOLO).
*   **Transformation:**
    1.  Reads line by line (streaming).
    2.  **Handling ID -1 (Untracked):** If YOLO has not assigned an ID (isolated detection), a **Synthetic ID** is generated: `9000000 + (frame_idx * 1000) + detection_idx`. This renders each untracked detection an independently manipulable track.
    3.  Aggregates detections into the `self.tracks` dictionary.

### 2.3. Merge Operations (`merge_logic`)

Combines two tracks ("Master" and "Slave") into a single entity.

*   **Logic:**
    1.  Transfers all frames and bounding boxes from the Slave to the Master.
    2.  Updates the Master's `merged_from` list.
    3.  Updates `id_lineage`: all IDs previously pointing to the Slave now point to the Master.
    4.  Removes the Slave's key from `self.tracks`.
    5.  **Reordering:** Sorts the Master's `frames` and `boxes` lists by temporal index.

### 2.4. Split Operations (`split_track`)

Divides a track into two segments at a specified frame boundary.

*   **Input:** `track_id`, `split_frame`, `keep_head` (boolean).
*   **Logic:**
    1.  Locates the split index within the `frames` lists.
    2.  Generates a `new_id` (Max existing ID + 1).
    3.  Partitions the `frames` and `boxes` lists into `head` (before the split) and `tail` (after the split).
    4.  If `keep_head` is True:
        *   The original ID retains the `head` segment.
        *   The `new_id` receives the `tail` segment.
    5.  If `keep_head` is False (default for ID switch correction):
        *   The original ID retains the `tail` segment.
        *   The `new_id` receives the `head` segment.

### 2.5. Automatic Correction Algorithms

*   **Auto-Stitch (`auto_stitch`):**
    *   Attempts to join temporally consecutive unassigned fragments.
    *   **Criteria:**
        1.  Temporal gap < `time_gap` (e.g., 2 seconds).
        2.  Spatial distance (Euclidean distance between the center of the final bounding box of Track A and the initial bounding box of Track B) < `stitch_dist`.
*   **Absorb Noise (`absorb_noise`):**
    *   Attempts to merge "Ignore" fragments (noise) into primary tracks ("Target", etc.).
    *   Useful for recovering lost limb detections or transient detections that belong to the principal participant.
    *   Employs strict spatial proximity criteria.

## 3. User Interface: `IdentityView`

Manages visual interaction and synchronization between the Video, Timeline, and Track List components.

### 3.1. Visual Timeline (`_draw_timeline`)

Renders a temporal representation of all tracks.

*   **Rendering:**
    *   "Ignore" tracks are drawn as a grey background layer.
    *   Tracks assigned to a Role are rendered in the role's designated color.
    *   Each Role occupies a dedicated "lane" (row) to prevent visual overlap.

### 3.2. Synchronized Selection (`_on_video_click`)

Enables track selection by clicking directly on the video feed.

*   **Logic:**
    1.  Receives click coordinates (x, y) on the video widget.
    2.  Converts them to original video coordinates (accounting for widget rescaling and letterboxing).
    3.  Queries `logic.get_track_at_point` to identify which ID possesses a bounding box containing the selected point in the current frame.
    4.  Highlights the corresponding ID in the lateral Treeview.

## 4. Data Output (`save_mapping`)

Generates the final identity mapping file.

*   **File:** `_identity.json`
*   **Content:** A flat dictionary `{ Original_YOLO_ID : "RoleName" }`.
*   **Transformation:**
    1.  Iterates over `id_lineage`.
    2.  For each original ID, resolves the current "Master" ID.
    3.  If the Master has a role other than "Ignore", the mapping is written.
*   **Purpose:** This file is consumed by downstream modules (Region, Eye Mapping) to determine that, for example, IDs 45, 46, and 98 all correspond to "Target".

---

### Note on Autosave Files
The module periodically saves the current state to `hermes_autosave_identity.json` in the project directory to prevent data loss in the event of a crash. Upon restart, the user is prompted to restore the saved session.

# HERMES - Region Module Developer Guide

This document details the internal logic, parameters, and workflows of the **Region** module (Spatial AOI Definition). It is intended for developers or researchers wishing to modify the underlying scripts for dynamic Area of Interest generation.

## 1. Global Constants and Keypoints

The module relies on the standard COCO Keypoint format employed by YOLO-Pose. These indices are mapped to anatomical labels in `KEYPOINTS_MAP`.

```python
KEYPOINTS_MAP = {
    0: "Nose", 1: "L_Eye", 2: "R_Eye", 3: "L_Ear", 4: "R_Ear",
    5: "L_Shoulder", 6: "R_Shoulder", 7: "L_Elbow", 8: "R_Elbow",
    9: "L_Wrist", 10: "R_Wrist", 11: "L_Hip", 12: "R_Hip",
    13: "L_Knee", 14: "R_Knee", 15: "L_Ankle", 16: "R_Ankle"
}
```

## 2. Profile Management (`AOIProfileManager`)

Profiles define how raw keypoints are transformed into semantic Areas of Interest (e.g., "Face", "Hands"). They are stored as JSON files in `assets/profiles_aoi`.

### 2.1. Profile Structure

A profile comprises **Roles** (mapped from the Entity module) and **Rules** for each role.

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
| **name** | String | Label for the AOI (e.g., "Face"). |
| **shape** | String | Geometry type: `box`, `circle`, `oval`, `polygon`. |
| **kps** | List[int] | Indices of keypoints used to compute the base bounding box. |
| **margin_px** | Int | Padding applied to the raw keypoint bounding box. |
| **scale_w/h** | Float | Multiplicative factor to expand or contract width or height relative to the center. |
| **offset_y_bottom** | Int | Additional pixels appended to the bottom edge (useful for torso/leg regions). |

---

## 3. Logic Layer: `RegionLogic`

The `RegionLogic` class handles the geometric computations and state management, decoupled from the UI.

### 3.1. Geometry Engine (`calculate_shape`)

This is the core function that transforms keypoints into geometric shapes.

1.  **Base Bounding Box Calculation:**
    *   Extracts valid keypoints (confidence > threshold).
    *   Computes the minimum and maximum X and Y coordinates.
    *   Applies `margin_px`.
    *   Applies `scale_w` and `scale_h` relative to the centroid.
    *   Applies `offset_y_bottom`.

2.  **Shape Morphing:**
    *   **Box:** Returns the computed rectangle.
    *   **Circle:** Derives center and radius from the bounding box dimensions.
    *   **Oval:** Derives center and semi-axes (rx, ry).
    *   **Polygon:** Maps the relative positions of keypoints from the source bounding box to the expanded bounding box, ordering vertices to form a convex hull-like shape.

### 3.2. Data Hierarchy and Overrides

The system employs a strict hierarchy to determine the rendered output for a given frame/track/AOI combination:

1.  **Level 0 (Base):** Automatic computation from YOLO keypoints using Profile Rules.
2.  **Level 1 (Manual Override):** Explicit shapes stored in `self.manual_overrides`. These take precedence over Level 0.

**Storage Key:** `(frame_idx, track_id, role, aoi_name)`

### 3.3. Ghost Tracks (`find_ghost_tracks`)

*   **Function:** Identifies tracks that are absent in the current frame but present in neighboring frames.
*   **Logic:**
    *   Scans a window of ± `ghost_window_var` frames.
    *   If a track exists in a neighboring frame but not in the current frame (and has not been manually overridden), it is flagged as a "Ghost."
    *   **Purpose:** Enables the user to "Force Add" an AOI by copying positional data from an adjacent frame—particularly useful for handling temporary occlusions or detection failures.

### 3.4. Export (`export_csv`)

Generates the final AOI dataset.

*   **Columns:** Frame, Timestamp, TrackID, Role, AOI, ShapeType, Coordinates (x1, y1, x2, y2), Geometric details (Radius, Angle, etc.), Corrected (Boolean).
*   **Logic:** Iterates through all frames, applying the Profile Rules and Manual Overrides to produce the definitive geometric state for each AOI.

---

## 4. Presentation Layer: `RegionView`

The UI orchestrates the workflow and visualization.

### 4.1. Manual Correction Mode

A stateful mode that activates editing tools.

*   **State Snapshot:** Upon entry, the current session state is saved to permit cancellation.
*   **Force Add:**
    *   **Auto/Ghost:** Uses `find_ghost_tracks` to seed the initial bounding box position.
    *   **Center:** Seeds the bounding box at the screen center if no ghost data is available.
    *   **Draw:** Allows the user to draw the bounding box or circle directly on the video canvas.
*   **Interpolation:**
    *   **Linear:** Linearly interpolates bounding box coordinates between two existing anchor frames (either manual or automatic) over a user-selected scope.

### 4.2. Scopes

Operations (such as applying a manual edit or performing interpolation) act upon a defined scope:
*   **Frame:** Current frame only.
*   **Current TOI:** All frames within the active Time Interval of Interest.
*   **Whole Video:** Every frame in the dataset.

### 4.3. Session Management

*   **Autosave:** State is persisted to `_aoi_edit_session.json` upon every commit action.
*   **Undo/Redo:** Implements a command-pattern stack storing before and after states of overrides.

### 4.4. Profile Wizard

A GUI tool for generating JSON profiles without manual text file editing.
*   **Visual Feedback:** Lists all available keypoints for selection.
*   **Strategies:** Supports defining distinct AOI logic for Targets versus Non-Targets.

# HERMES - Master TOI Module Developer Guide

This document describes the internal logic, data flows, and transformations implemented in the **Master TOI** module (`hermes_master_toi.py`). The module is responsible for temporal synchronization between heterogeneous data sources (eye-tracker, behavioral logs, video) and for defining the temporal intervals of interest (TOIs) that structure the analysis.

## 1. Management Logic: `MasterToiLogic`

The `MasterToiLogic` class handles log importation, synchronization offset computation, and TOI table construction.

### 1.1. Log Importation and Parsing

*   **Input:**
    *   Eye-Tracker Log (e.g., Tobii TSV/Excel) containing timestamps and event triggers.
    *   External Log (e.g., E-Prime, MATLAB, manual CSV) containing the sequence of experimental events.
*   **Logic:**
    *   Normalizes column names (e.g., searches for columns labeled "Event", "Timestamp").
    *   Identifies common synchronization markers (e.g., "VideoStart", "TrialStart").

### 1.2. Temporal Synchronization

Aligns the eye-tracker timeline (absolute or machine-relative time) with the video or experimental timeline.

*   **Method:** Linear Offset Computation.
    *   `Offset = Timestamp_Event_EyeTracker - Timestamp_Event_ExternalLog`
    *   This offset is applied to all timestamps to project them into a common reference frame (typically relative to video onset).

### 1.3. TOI (Time of Interest) Definition

Constructs the principal table that governs the statistical analysis.

*   **Data Structure:** DataFrame with the following columns:
    *   `Phase`: Phase label (e.g., "Fixation", "Stimulus").
    *   `Condition`: Experimental condition.
    *   `Start`: Phase onset time (in seconds, synchronized).
    *   `End`: Phase offset time (in seconds, synchronized).
    *   `Trial`: Sequential trial number (optional).
*   **Output:** A `_TOI.tsv` file consumed by `hermes_stats.py`.

## 2. Data Pruning and Export: `DataCropper`

This class specializes in cropping voluminous datasets to retain only data pertaining to the defined TOIs, thereby reducing noise and file sizes for downstream analyses.

### 2.1. Cropping Logic

*   **Input:**
    *   Defined TOI table.
    *   Complete datasets (YOLO Raw, Gaze Mapped).
*   **Process:**
    1.  Iterates over each TOI.
    2.  Extracts rows from the original datasets where `Start_TOI <= Timestamp <= End_TOI`.
    3.  Appends TOI metadata (Phase, Condition) to the extracted rows.
    4.  Concatenates the results into a new "Cropped" dataset.

### 2.2. Flattened Output Generation

The module exports cropped data in a "Long" (tidy) CSV format to facilitate analysis in R or Pandas, replicating the export structure of `hermes_human.py`.

*   **YOLO Transformation:**
    *   Input: JSON/CSV with nested structures or multi-column keypoint representations.
    *   Output: `_video_yolo_CROPPED.csv`.
    *   Structure: `Frame`, `Timestamp`, `TrackID`, `Box`, `Keypoints` (flattened), `Phase`, `Condition`.

## 3. User Interface: `MasterToiView`

Manages user interaction for manual or assisted TOI definition.

### 3.1. Tabular Editor

*   Allows the user to inspect the imported or generated TOI table.
*   Supports manual editing of `Start`, `End`, and label fields (`Phase`, `Condition`).

### 3.2. Synchronization Visualization

*   Displays the timestamps of detected events from both logs to enable visual verification of temporal alignment.
*   Permits manual adjustment of the offset if automatic synchronization fails.

---

### Data Flow Summary

| Input | Process | Output |
| :--- | :--- | :--- |
| **Tobii Log + Exp. Log** | **Synchronization**<br>Timestamp alignment via common events | **Temporal Offset** |
| **Offset + Exp. Log** | **Phase Definition**<br>Event-to-interval mapping | **TOI Table (.tsv)** |
| **TOI Table + Raw Data** | **Data Cropper**<br>Temporal filtering and enrichment | **Cropped Datasets (.csv)**<br>(Reduced YOLO and Gaze) |

# HERMES - Eye Mapping Module Developer Guide

This document describes the internal logic, data flows, and transformations implemented in the **Eye Mapping** module (`hermes_eye.py`). The module is responsible for the geometric intersection between gaze data and the dynamically defined Areas of Interest (AOIs) on a frame-by-frame basis.

## 1. Computation Logic: `GazeLogic`

The `GazeLogic` class manages the spatial mapping mathematics and temporal synchronization.

### 1.1. AOI Loading and Indexing (`load_aoi_data`)

*   **Input:** CSV file generated by the Region module (containing bounding boxes for each frame).
*   **Transformation:**
    1.  Loads the CSV into a Pandas DataFrame.
    2.  Dynamically identifies the ID column (`ID` or `TrackID`).
    3.  **Grouping:** Groups rows by `Frame`.
    4.  **Indexing:** Constructs a dictionary (hash map) `{ frame_index : [list_of_aoi_dicts] }`.
*   **Purpose:** Enables constant-time (O(1)) access to all active AOIs for a given frame during gaze data streaming.

### 1.2. Geometric Hit-Testing (`calculate_hit`)

This is the core function that determines "what the participant is looking at."

*   **Input:** Gaze coordinates in pixels (x, y), list of AOIs in the current frame.
*   **Logic:**
    1.  Iterates over all AOIs present in the frame.
    2.  For each AOI, evaluates geometric intersection according to its shape (`_shape_hit_and_area`):
        *   **Box:** Simple boundary check: `x1 <= x <= x2` and `y1 <= y <= y2`.
        *   **Circle:** Euclidean distance from center ≤ radius.
        *   **Polygon:** Ray Casting algorithm (`_point_in_polygon`).
    3.  **Overlap Resolution:** If the gaze point falls within multiple AOIs simultaneously (e.g., "Face" inside "Body"), the AOI with the **smallest area** is selected. This ensures maximum specificity (e.g., a hit on "Eye" takes precedence over "Face").
*   **Output:** The winning AOI object, or `None` (gaze directed at the background).

### 1.3. Temporal Synchronization (`timestamp_to_frame`)

Converts the absolute eye-tracker timestamp to the corresponding video-relative frame index.

*   **Formula:** `Frame = int((Timestamp_Gaze - Sync_Offset) * FPS)`
*   **Parameters:**
    *   `Timestamp_Gaze`: Time in seconds from the Tobii recording file.
    *   `Sync_Offset`: Temporal delta aligning video onset with the start of eye-tracking recording.
    *   `FPS`: Video frame rate.

### 1.4. Mapping Pipeline (`run_mapping`)

Orchestrates the entire process in streaming mode to handle large files without exhausting system memory.

*   **Input:** AOI file (CSV), Gaze file (JSON.GZ), Video resolution, FPS, Offset.
*   **Data Flow:**
    1.  **AOI Loading:** Executes `load_aoi_data` to construct the spatial map in memory.
    2.  **Gaze Streaming:** Opens the `.gz` file and reads line by line.
    3.  **Parsing:**
        *   Discards invalid packets or those lacking `gaze2d` coordinates.
        *   Extracts `timestamp` and normalized coordinates `(gx, gy)` [0.0–1.0].
    4.  **Spatial Conversion:**
        *   `Pixel_X = gx * Video_Width`
        *   `Pixel_Y = gy * Video_Height`
    5.  **Temporal Conversion:** Computes the `frame_idx` using the synchronization formula.
    6.  **Hit-Test:** Retrieves AOIs for `frame_idx` and executes `calculate_hit`.
    7.  **Accumulation:** Stores the result (Hit/Miss, Role, AOI, TrackID) in a buffer list.
    8.  **Export:** Writes the final `_MAPPED.csv` file.

## 2. Data Output (`_MAPPED.csv`)

The generated file contains one row per eye-tracker sample mapped onto the video.

| Column | Description |
| :--- | :--- |
| **Timestamp** | Original eye-tracker timestamp. |
| **Frame_Est** | Estimated corresponding video frame. |
| **Gaze_X, Gaze_Y** | Gaze coordinates in pixels on the video plane. |
| **Hit_Role** | Role of the intersected target (e.g., "Target", "Confederate"). "None" if miss. |
| **Hit_AOI** | Name of the intersected AOI (e.g., "Face", "Hands"). "None" if miss. |
| **Hit_TrackID** | Numeric ID of the intersected participant. -1 if miss. |
| **Hit_Shape** | Geometric shape type intersected (box, circle, polygon). |

## 3. User Interface

### 3.1. `GazeView`

Configuration panel for initiating the mapping process.

*   **Input:** File selectors for AOI data and Gaze data.
*   **Parameters:** Video resolution (default: 1920×1080), FPS, Synchronization offset.
*   **Threading:** Executes `run_mapping` in a dedicated thread (`_thread_worker`) to maintain UI responsiveness, displaying an indeterminate progress bar (since the streaming reader cannot determine total file length a priori).

### 3.2. `GazeResultPlayer`

A dedicated video player for qualitative verification of the mapping output.

*   **Functionality:** Loads the video alongside the newly generated mapped CSV file.
*   **Visualization:**
    *   Renders the gaze point (yellow/red circle).
    *   When a hit is registered, displays the name of the intersected AOI as an overlay.
    *   Supports frame-by-frame navigation to verify the precision of both synchronization and geometric tracking.
    *   **Optimization:** Indexes the CSV in memory (`data_map`) for rapid access during video playback.

# HERMES - Stats Module Developer Guide

This document describes the internal logic, data flows, and transformations implemented in the **Stats** module (`hermes_stats.py`). The module is responsible for aggregating gaze data mapped onto AOIs and computing statistical metrics within defined temporal windows (TOIs).

## 1. Computation Logic: `StatsLogic`

The `StatsLogic` class encapsulates the mathematical engine. It operates independently of the graphical interface.

### 1.1. Sampling Rate Estimation (`calculate_actual_sampling_rate`)

*   **Input:** Gaze data DataFrame (`df_gaze`) containing the `Timestamp` column.
*   **Logic:**
    1.  Computes inter-sample time differences ($\Delta t$) between consecutive rows: `df['Timestamp'].diff()`.
    2.  **Gap Filtering:** Excludes differences exceeding 0.1 s (100 ms) to prevent data gaps (blinks, tracking loss) from biasing the mean.
    3.  Computes the mean of the valid $\Delta t$ values.
    4.  Sampling frequency ($Hz$) = $1.0 / \overline{\Delta t}$.
*   **Fallback:** If insufficient data are available, defaults to 50.0 Hz.
*   **Purpose:** Essential for converting the *number of samples* into *time (seconds)*.

### 1.2. Raw Dataset Generation (`generate_raw_dataset`)

This function enriches the "Mapped" file (sample-level) with contextual information from the TOIs (phase-level).

*   **Input:**
    *   `mapped_path`: CSV generated by the Eye Mapping module.
    *   `toi_path`: TSV generated by the Master TOI module.
*   **Transformation:**
    1.  **Loading:** Reads both files into Pandas DataFrames.
    2.  **Sorting:** Orders gaze data by `Timestamp` (critical for computational efficiency).
    3.  **Initialization:** Appends empty columns `Phase`, `Condition`, `Trial` to the Gaze DataFrame.
    4.  **TOI Iteration:** For each row in the TOI file (representing a temporal phase):
        *   Extracts `Start` and `End`.
        *   **Binary Search:** Uses `np.searchsorted` on gaze timestamps to instantaneously locate the start (`idx_start`) and end (`idx_end`) indices in the Gaze DataFrame corresponding to the temporal window.
        *   **Vectorized Assignment:** Assigns the `Phase`, `Condition`, and `Trial` values to all rows in the range `[idx_start:idx_end]` in a single operation.
*   **Output:** A DataFrame in which each individual gaze sample is annotated with its corresponding experimental phase.

### 1.3. Statistical Analysis Engine (`run_analysis`)

This is the core of the module. It crosses spatial data (AOI hits) with temporal data (TOIs).

*   **Input:** Mapped file, TOI file, Frequency (optional), Format flag (Wide/Long).
*   **Data Flow:**
    1.  **Validation:** Verifies the presence of essential columns (`Hit_Role`, `Hit_AOI`, `Timestamp` in Gaze; `Start`, `End` in TOI).
    2.  **Frequency Setup:** If the user does not specify a frequency, it is computed via `calculate_actual_sampling_rate`.
        *   `sample_dur` = $1.0 / f$.
    3.  **Combination Discovery:** Scans the entire Gaze file to identify all unique `(Hit_Role, Hit_AOI)` pairs (e.g., "Target_Face", "Confederate_Hand"). This ensures the final report includes columns for all AOIs, even those that are never fixated in a particular phase (value: 0).
    4.  **Phase Loop (TOI):** Iterates over each temporal interval defined in the TOI file.
        *   **Slicing:** Extracts the subset of Gaze samples falling within the phase boundaries (`t_start` → `t_end`).
        *   **General Phase Metrics:**
            *   `Gaze_Samples_Total`: Number of samples in the subset.
            *   `Gaze_Valid_Time`: `Samples × sample_dur`.
            *   `Tracking_Ratio`: `Valid_Time / (t_end - t_start)`.
        *   **Per-AOI Metric Computation:** Groups the subset by `Hit_Role` and `Hit_AOI`.
            *   **Duration:** `Count × sample_dur`.
            *   **Percentage:** `Duration / Phase_Duration`.
            *   **Latency:** `Timestamp_First_Hit - t_start`. Empty if no hits are registered.
            *   **Glances:** Counts the number of gaze *entries* into the AOI.
                *   Logic: `(Current == AOI) AND (Previous ≠ AOI)`.
    5.  **Output Formatting:**
        *   **Wide Format (Classical):** One row per Phase. AOI metrics are appended as horizontal columns (e.g., `Target_Face_Dur`, `Target_Face_Perc`, `Target_Face_Lat`).
        *   **Long Format (Tidy):** One row per Phase–AOI combination. Fixed columns: `Phase`, `Condition`, `Hit_Role`, `Hit_AOI`, `Duration`, `Percentage`, etc.

## 2. Master Report Generation (`export_master_report`)

This function produces a comprehensive Excel file consolidating all experimental data.

*   **Input:** A dictionary of DataFrames (`data_frames_dict`) containing Stats, Raw Data, Mapping, AOI, Identity, YOLO, and TOI data.
*   **Logic:**
    1.  Uses `xlsxwriter` as the engine.
    2.  For each DataFrame in the dictionary:
        *   Creates a **Legend** sheet (`L - SheetName`) using a static dictionary of descriptions (`legends_dict`).
        *   Creates the **Data** sheet (`SheetName`) by writing the DataFrame.
        *   Applies auto-fit column widths for readability.
*   **Data Integration:**
    *   The controller (`GazeStatsView`) attempts to automatically load related files based on naming conventions (e.g., if the mapped file is `P01_MAPPED.csv`, it searches for `P01_AOI.csv`, `P01_video_yolo.csv`, etc.) to populate the dictionary.

## 3. User Interface: `GazeStatsView`

Manages thread orchestration to prevent UI blocking during computationally intensive operations.

*   **Thread Worker:**
    1.  Executes `run_analysis` to compute the statistics.
    2.  If requested, executes `generate_raw_dataset`.
    3.  If "Master Report" is requested, collects all auxiliary CSV/JSON files from the project directory and invokes `export_master_report`.
    4.  Otherwise, saves the individual CSV files (`_STATS.csv` and optionally `_RAW.csv`).

---

### Transformation Summary

| Input | Process | Output |
| :--- | :--- | :--- |
| **Gaze Mapped (.csv)**<br>(Timestamp, X, Y, Hit_AOI) | **Temporal Slicing**<br>Segmentation based on TOI Start/End | **Gaze Subset**<br>(Phase-specific samples) |
| **Gaze Subset** | **Aggregation**<br>Count, Sum(Duration), Min(Timestamp) | **AOI Metrics**<br>(Duration, Latency, Glances) |
| **AOI Metrics** | **Pivoting (Wide)**<br>Flattening AOI keys into columns | **Stats Row**<br>(Phase, Cond, Face_Dur, Hand_Dur…) |
| **All DataFrames** | **Excel Writer**<br>Merge into multiple sheets + Legends | **Master Report (.xlsx)** |
