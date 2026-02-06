# HERMES UX Improvement Roadmap

This document outlines specific tasks to improve the User Experience (UX) of the HERMES platform, focusing on efficiency, feedback, and modernization.

## 1. Interaction & Efficiency (Workflow Speed)

### 1.1 Keyboard Shortcuts (Hotkeys)
**Goal:** Reduce reliance on mouse clicks for repetitive video navigation tasks.

**Reflections:**
*   **Pros:** Essential for frame-by-frame analysis. Drastically reduces Time on Task for labeling.
*   **Risks:** Focus conflicts. If the user is typing in an Entry field, pressing "Space" or "1" must not trigger playback or assignment.
*   **Implementation:** Use `focus_get()` or conditional bindings to check active widget before executing action.

- [x] **IdentityView (`hermes_entity.py`)**
    - [x] Bind `<Space>` key to `toggle_play`.
    - [x] Bind `<Left>` arrow to step backward 1 frame.
    - [x] Bind `<Right>` arrow to step forward 1 frame.
    - [x] Bind `<Shift>+<Left/Right>` to step 10 frames.
    - [x] Bind Number keys (`1`-`9`) to instantly assign the selected track to a specific Cast member (e.g., `1` -> Target, `2` -> Confederate).
    - [x] **Task:** Implement `RequiresVideoFocus` decorator/mixin to ignore hotkeys if `root.focus_get()` is `tk.Entry` or `tk.Text`.

- [x] **GazeResultPlayer (`hermes_eye.py`)**
    - [x] Implement the same playback controls (Space, Arrows) as IdentityView for consistency.

- [x] **RegionView (`hermes_region.py`)**
    - [x] Add shortcuts for frame navigation to quickly verify AOI stability across the video.

### 1.2 Undo/Redo System
**Goal:** Mitigate user anxiety regarding destructive operations like merging tracks.

**Reflections:**
*   **Refinement:** Instead of deep-copying the entire state (RAM heavy), consider a **Delta Approach** (record only changed track IDs) or **Disk-based History** (serialize undo stack to temp pickle file).

- [ ] **IdentityView (`hermes_entity.py`)**
    - [ ] Create a `HistoryManager` class to manage a stack of states.
    - [ ] Implement `snapshot()` (or delta recording) before: `manual_merge`, `auto_stitch`, `absorb_noise_logic`, `split_track_at_current_frame`.
    - [ ] Hook `snapshot()` into: `manual_merge`, `auto_stitch`, `absorb_noise_logic`, `split_track_at_current_frame`.
    - [ ] Add `Ctrl+Z` (Undo) binding to restore the previous state.

## 2. Visual Feedback & Responsiveness

### 2.1 Asynchronous Loading (Threading)
**Goal:** Prevent the application window from freezing ("Not Responding") during heavy file I/O.

**Reflections:**
*   **Pros:** Prevents OS from marking app as "Not Responding".
*   **Risks:** Tkinter is not thread-safe. Cannot update GUI directly from background thread.
*   **Implementation:** Use `queue.Queue` to pass messages from loader thread to main thread; main thread polls queue via `.after()`.

- [ ] **IdentityView (`hermes_entity.py`)**
    - [ ] Refactor `load_data_direct` to run JSON parsing in a background thread (`threading.Thread`).
    - [ ] Implement a `queue` or `after()` loop to update the UI once data is ready.
    - [ ] Display an indeterminate `ttk.Progressbar` (Spinner) overlay while loading.

- [ ] **GazeView (`hermes_eye.py`)**
    - [ ] Ensure the Gaze Mapping loop in `run_process` yields to the UI thread more frequently or runs entirely in a background thread.

### 2.2 Interactive Canvas (Direct Manipulation)
**Goal:** Allow intuitive definition of Areas of Interest (AOI) instead of guessing pixel values.

**Reflections:**
*   **Pros:** User thinks in "spaces", not "pixels".
*   **Risks:** Coordinate Mapping bugs. Video resolution (e.g., 1920x1080) vs Display size (e.g., 800x600).
*   **Implementation:** Robust scaling logic: `real_x = canvas_x * (video_w / canvas_w)`. Must handle window resizing events.

- [ ] **RegionView (`hermes_region.py`)**
    - [ ] Overlay a transparent `tk.Canvas` on top of the video label.
    - [ ] Implement `<Button-1>`, `<B1-Motion>`, `<ButtonRelease-1>` events to draw rectangles.
    - [ ] Implement scaling logic to convert Canvas coords <-> Video coords.
    - [ ] Add hover effects to existing AOI boxes to show their name/role.

### 2.3 Visualization Enhancements
**Goal:** Make complex spatial data easier to understand.

- [ ] **IdentityView (`hermes_entity.py`)**
    - [ ] **Trajectory Tails:** Draw a line connecting the center points of the last 20-50 frames behind the bounding box to visualize movement history.
    - [ ] **Selection Highlight:** Render the currently selected track (in the TreeView) with a thicker, distinct border (e.g., Cyan or Dashed) in the video overlay.

- [ ] **TOIGeneratorView (`hermes_master_toi.py`)**
    - [ ] Add a visual plot (using `matplotlib` backend for Tkinter) to show the alignment between Matlab signals and Tobii events before generating files.

## 3. Visual Polish & Modernization

### 3.1 Theming
**Goal:** Move away from the "Windows 95" look to a professional scientific tool appearance.

**Reflections:**
*   **Pros:** Reduces visual fatigue (Dark Mode).
*   **Risks:** Libraries like `ttkbootstrap` might interfere with matplotlib plots or OpenCV canvas (DPI scaling issues).

- [ ] **Global (`hermes_unified.py`)**
    - [ ] Integrate a modern theme library like `ttkbootstrap` or `sv_ttk`.
    - [ ] Apply a consistent theme (Dark/Light) across all Toplevel windows.

- [ ] **Icons**
    - [ ] Replace text-based buttons (e.g., "⏯", "📂", "🚀") with consistent PNG/SVG icons (Play, Pause, Folder, Rocket).

## 4. Persistence & Onboarding

### 4.1 Configuration Management
**Goal:** Allow users to resume work quickly without re-selecting paths every time.

- [x] **AppContext (`hermes_context.py`)**
    - [x] Implement `save_config()` and `load_config()` using a local `config.json` file.
    - [x] Persist "Last Opened Project" path.
    - [x] Persist "Recent Files" list for Video and Data inputs.
    - [x] Auto-load the last project on startup (with a confirmation prompt).

### 4.2 User Guidance
**Goal:** Make the tool accessible to non-developers.

- [ ] **Tooltips**
    - [ ] Add hover tooltips to scientific parameters in `YoloView` (e.g., "IoU Threshold") and `IdentityView` (e.g., "Lookahead", "Stitch Distance").

- [ ] **Workflow Status Indicators**
    - [ ] **Sidebar (`hermes_unified.py`)**: Add visual checkmarks (✅) next to navigation buttons when specific data is present in `AppContext` (e.g., check "1. HUMAN" if `pose_data_path` is set).

- [ ] **Contextual Help / Cheatsheet**
    - [ ] Add toggleable overlay (`?` key) showing hotkeys and color codes.

## 5. Critical Missing Features (Must-Have)

### 5.1 Auto-Save & Crash Recovery
**Context:** Research software is prone to crashes (OOM, CUDA).
**Goal:** Prevent data loss.

- [ ] **IdentityView (`hermes_entity.py`)**
    - [ ] Implement background Auto-Save timer (2-5 mins) to temp file (`_autosave_identity.json`).
    - [ ] On startup, check for temp file and prompt to restore.

### 5.2 Input Validation & "Pre-Flight Checks"
**Context:** Mismatched files (FPS, Duration, Coordinates) cause late errors.
**Goal:** Early detection of issues.

- [ ] **HermesUnifiedApp (`hermes_unified.py`)**
    - [ ] Create a Dashboard Widget/Status Panel.
    - [ ] Implement checks: Duration mismatch (Video vs Pose vs Gaze), Frequency consistency, Coordinate bounds.

## 6. Advanced Efficiency Features

### 6.1 "Smart Seek" / Jump to Event
**Goal:** Drastically reduce navigation time.

- [ ] **IdentityView (`hermes_entity.py`)**
    - [ ] Add "Jump to Next Untracked" (Gap in ID).
    - [ ] Add "Jump to Next Interaction" (Proximity < X).

### 6.2 Bulk Editing
**Goal:** Handle crowd scenes or noise efficiently.

- [ ] **IdentityView (`hermes_entity.py`)**
    - [ ] **Box Selection:** Allow drawing rectangle to select all tracks inside area.
    - [ ] **Batch Action:** "Delete all short tracks (< 1s) in selection".

### 6.3 Dynamic Resolution Scaling
**Goal:** Performance on high-res videos (4K).

- [ ] **Video Modules**
    - [ ] Implement "Preview Proxy": Process visualization at 720p while keeping data logic at native resolution.

## 7. Specific Module Improvements

- [ ] **GazeStatsView (`hermes_stats.py`)**
    - [ ] Add an "Export Summary" button that generates a PDF or HTML report with charts, not just a CSV.