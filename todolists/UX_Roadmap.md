# HERMES UX Improvement Roadmap

This document outlines specific tasks to improve the User Experience (UX) of the HERMES platform, focusing on efficiency, feedback, and modernization.

## 1. Interaction & Efficiency (Workflow Speed)

### 1.1 Keyboard Shortcuts (Hotkeys)
**Goal:** Reduce reliance on mouse clicks for repetitive video navigation tasks.

- [ ] **IdentityView (`hermes_entity.py`)**
    - [ ] Bind `<Space>` key to `toggle_play`.
    - [ ] Bind `<Left>` arrow to step backward 1 frame.
    - [ ] Bind `<Right>` arrow to step forward 1 frame.
    - [ ] Bind `<Shift>+<Left/Right>` to step 10 frames.
    - [ ] Bind Number keys (`1`-`9`) to instantly assign the selected track to a specific Cast member (e.g., `1` -> Target, `2` -> Confederate).

- [ ] **GazeResultPlayer (`hermes_eye.py`)**
    - [ ] Implement the same playback controls (Space, Arrows) as IdentityView for consistency.

- [ ] **RegionView (`hermes_region.py`)**
    - [ ] Add shortcuts for frame navigation to quickly verify AOI stability across the video.

### 1.2 Undo/Redo System
**Goal:** Mitigate user anxiety regarding destructive operations like merging tracks.

- [ ] **IdentityView (`hermes_entity.py`)**
    - [ ] Create a `HistoryManager` class to manage a stack of states.
    - [ ] Implement `snapshot()` method to deepcopy `self.tracks` and `self.id_lineage` before operations.
    - [ ] Hook `snapshot()` into: `manual_merge`, `auto_stitch`, `absorb_noise_logic`, `split_track_at_current_frame`.
    - [ ] Add `Ctrl+Z` (Undo) binding to restore the previous state.

## 2. Visual Feedback & Responsiveness

### 2.1 Asynchronous Loading (Threading)
**Goal:** Prevent the application window from freezing ("Not Responding") during heavy file I/O.

- [ ] **IdentityView (`hermes_entity.py`)**
    - [ ] Refactor `load_data_direct` to run JSON parsing in a background thread (`threading.Thread`).
    - [ ] Implement a `queue` or `after()` loop to update the UI once data is ready.
    - [ ] Display an indeterminate `ttk.Progressbar` (Spinner) overlay while loading.

- [ ] **GazeView (`hermes_eye.py`)**
    - [ ] Ensure the Gaze Mapping loop in `run_process` yields to the UI thread more frequently or runs entirely in a background thread.

### 2.2 Interactive Canvas (Direct Manipulation)
**Goal:** Allow intuitive definition of Areas of Interest (AOI) instead of guessing pixel values.

- [ ] **RegionView (`hermes_region.py`)**
    - [ ] Overlay a transparent `tk.Canvas` on top of the video label.
    - [ ] Implement `<Button-1>`, `<B1-Motion>`, `<ButtonRelease-1>` events to draw rectangles.
    - [ ] Convert drawn rectangle coordinates to the margin/offset values required by the profile logic.
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

- [ ] **Global (`hermes_unified.py`)**
    - [ ] Integrate a modern theme library like `ttkbootstrap` or `sv_ttk`.
    - [ ] Apply a consistent theme (Dark/Light) across all Toplevel windows.

- [ ] **Icons**
    - [ ] Replace text-based buttons (e.g., "⏯", "📂", "🚀") with consistent PNG/SVG icons (Play, Pause, Folder, Rocket).

## 4. Persistence & Onboarding

### 4.1 Configuration Management
**Goal:** Allow users to resume work quickly without re-selecting paths every time.

- [ ] **AppContext (`hermes_context.py`)**
    - [ ] Implement `save_config()` and `load_config()` using a local `config.json` file.
    - [ ] Persist "Last Opened Project" path.
    - [ ] Persist "Recent Files" list for Video and Data inputs.
    - [ ] Auto-load the last project on startup (with a confirmation prompt).

### 4.2 User Guidance
**Goal:** Make the tool accessible to non-developers.

- [ ] **Tooltips**
    - [ ] Add hover tooltips to scientific parameters in `YoloView` (e.g., "IoU Threshold") and `IdentityView` (e.g., "Lookahead", "Stitch Distance").

- [ ] **Workflow Status Indicators**
    - [ ] **Sidebar (`hermes_unified.py`)**: Add visual checkmarks (✅) next to navigation buttons when specific data is present in `AppContext` (e.g., check "1. HUMAN" if `pose_data_path` is set).

## 5. Specific Module Improvements

- [ ] **IdentityView (`hermes_entity.py`)**
    - [ ] **"Find Gaps" Feature:** Add a button to jump the video timeline to the next frame where a specific Target is *missing* (lost tracking).
    - [ ] **Filter by Duration:** Improve the "Hide short tracks" filter to be a slider (e.g., hide < 0.5s, < 1.0s, < 2.0s).

- [ ] **GazeStatsView (`hermes_stats.py`)**
    - [ ] Add an "Export Summary" button that generates a PDF or HTML report with charts, not just a CSV.