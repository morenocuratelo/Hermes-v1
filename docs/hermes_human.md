# HERMES - Human Module Developer Guide

This document details the internal logic, parameters, and workflows of the **Human** module (Kinematic Extraction). It is intended for developers or researchers wishing to modify the underlying scripts.

## 1. Global Parameters & Heuristics

These constants are defined at the module level. While initialized here, many are exposed to the user via the GUI sliders for fine-tuning.

```python
# --- RESEARCH PARAMETERS & HEURISTICS (CONSTANTS) ---

# CONF_THRESHOLD: Conservative threshold to balance Precision and Recall.
# Kept high (0.6) for purity, as suggested in the original BoT-SORT paper.
CONF_THRESHOLD = 0.6 

# IOU_THRESHOLD: Threshold for Non-Maximum Suppression (NMS).
# CRITICAL: YOLO26 (v8/v11) is NMS-Free. Setting this to 1.0 effectively disables redundant NMS post-processing.
IOU_THRESHOLD = 1.0 

# MATCH_THRESHOLD: Specific for tracking association (e.g., BoT-SORT).
# Determines how strictly detections are matched to existing tracks.
MATCH_THRESHOLD = 0.8 

# RANDOM_SEED: Ensures deterministic results for reproducibility.
RANDOM_SEED = 42

# ULTRALYTICS_URL: Base URL for downloading model assets.
ULTRALYTICS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"

# TRACKERS_CONFIG_DIR: Directory where dynamic tracker configurations are stored.
TRACKERS_CONFIG_DIR = os.path.join("Configs", "Trackers")
```

**Note:** Although defaults are based on literature (COCO benchmarks), HERMES allows users to fine-tune these via the GUI to adapt to specific lighting or scene density, overcoming "one-size-fits-all" limitations. Selected parameters are saved in the `_meta.json` file.

---

## 2. Logic Layer: `PoseEstimatorLogic`

The `PoseEstimatorLogic` class encapsulates the backend logic for video analysis, decoupling it from the UI.

### 2.1. Reproducibility (`set_determinism`)

*   **Function:** Sets seeds for Python's `random`, `numpy`, and `torch` generators.
*   **Purpose:** Ensures that running the analysis twice on the same video with identical parameters yields **exactly** the same results.
*   **Technical Detail:** Forces PyTorch and CUDNN to use deterministic algorithms, disabling specific hardware optimizations that might introduce stochastic variance. Currently, the seed is forced to `42`.

### 2.2. Model Management (`download_model`)

*   **Function:** Automatically downloads AI models (e.g., `yolo26x-pose.pt`, ReID models) if missing.
*   **Purpose:** Simplifies setup by fetching weights from official repositories (Ultralytics) upon first use.

**YOLO Pose Model Variants:**
| Model | Size (Pixels) | Layers | Parameters | Gradients | GFLOPs | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **n** | 640 | 363 | 3.7M | 3.7M | 10.7 | Nano - Fastest, lowest accuracy |
| **s** | 640 | 363 | 11.9M | 11.9M | 29.6 | Small |
| **m** | 640 | 383 | 24.3M | 24.3M | 85.9 | Medium |
| **l** | 640 | 495 | 28.7M | 28.7M | 104.3 | Large |
| **x** | 640 | 495 | 62.9M | 62.9M | 226.3 | XLarge - Slowest, highest accuracy |

### 2.3. Tracker Configuration (`generate_tracker_config`)

*   **Function:** Dynamically creates a `.yaml` configuration file for the selected tracker (BoT-SORT or ByteTrack).
*   **Purpose:** Translates GUI parameters (confidence thresholds, buffers, ReID flags) into a format consumable by the YOLO tracking engine.

**Configuration Parameters:**

| Parameter | Valid Values | Description |
| :--- | :--- | :--- |
| **tracker_type** | `botsort`, `bytetrack` | Specifies the tracking algorithm. |
| **track_high_thresh** | 0.0 - 1.0 | Threshold for the first association pass. High confidence detections. |
| **track_low_thresh** | 0.0 - 1.0 | Threshold for the second association pass (lower confidence). |
| **new_track_thresh** | 0.0 - 1.0 | Minimum confidence to initialize a new track ID. |
| **track_buffer** | >= 0 | Frames to keep a lost track alive (tolerance for occlusion). |
| **match_thresh** | 0.0 - 1.0 | Lenience of track matching. Higher = more lenient. |
| **fuse_score** | `True`, `False` | Fuses confidence scores with IoU distances during matching. |
| **gmc_method** | `orb`, `sift`, `ecc`, `sparseOptFlow`, `None` | Global Motion Compensation method to account for camera movement. |
| **proximity_thresh** | 0.0 - 1.0 | Min IoU required before attempting ReID (spatial gate). |
| **appearance_thresh** | 0.0 - 1.0 | Min visual similarity for ReID matching. |
| **with_reid** | `True`, `False` | Enables Re-Identification (BoT-SORT only). |

**Re-Identification (ReID) Notes:**
*   **Native features (`model: auto`):** Uses YOLO detector features. Low overhead.
*   **Classification models:** Explicitly set a model (e.g., `yolo26n-cls.pt`) for higher accuracy but higher latency.
*   **Optimization:** For production, export ReID models to **TensorRT** for speed:
    ```python
    from torch import nn
    from ultralytics import YOLO

    # Load and modify model for export
    model = YOLO("yolo26n-cls.pt")
    head = model.model.model[-1]
    pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1))
    pool.f, pool.i = head.f, head.i
    model.model.model[-1] = pool

    # Export
    model.export(format="engine", half=True, dynamic=True, batch=32)
    ```

### 2.4. Data Export (`export_to_csv_flat`)

*   **Function:** Converts the raw output (`.json.gz`) into a flat CSV file.
*   **Purpose:** Makes data accessible for Excel, MATLAB, or Pandas.
*   **Structure:**
    *   **Metadata:** Frame Index, Timestamp, Track ID.
    *   **Bounding Box:** x1, y1, x2, y2.
    *   **Keypoints (17):** Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles (X, Y, Conf for each).

### 2.5. Analysis Engine (`run_analysis`)

This is the orchestration method. It executes the following pipeline:

1.  **ReID Model Check:**
    *   If ReID is enabled (BoT-SORT), checks for the secondary classification model (e.g., `resnet50_msmt17_ready.pt`).
    *   Downloads if missing. If download fails, ReID is disabled to prevent crashes.

2.  **Generate Tracker Config:**
    *   Calls `generate_tracker_config` to write the temporary `.yaml` file required by YOLO.

3.  **Seeding:**
    *   Calls `set_determinism(RANDOM_SEED)` to lock random states for scientific validity.

4.  **Load YOLO Model:**
    *   Checks for the main pose model (e.g., `yolo26x-pose.pt`) and loads it into VRAM (or RAM).

5.  **Video Metadata:**
    *   Opens the video via OpenCV to retrieve `total_frames` (for progress bars) and `fps` (for timestamp calculation).

6.  **Inference Configuration:**
    *   Prepares arguments for the Ultralytics engine:
        ```python
        yolo_args = {
            "source": video_file,
            "stream": True,       # Generator mode to manage memory
            "verbose": False,     # Suppress internal logs
            "conf": tracker_params['conf'],
            "iou": tracker_params['iou'],
            "device": 0 if device == "cuda" else "cpu"
        }
        ```

7.  **Inference Loop (Streaming):**
    *   Iterates frame-by-frame:
        *   **Extract:** Convert GPU tensors to NumPy arrays (Box, ID, Keypoints).
        *   **Normalize:** Handle empty detections safely.
        *   **Structure:** Create a JSON object for the frame.
        *   **Write:** Append immediately to a `.json.gz` file (streaming write) to handle long videos without RAM saturation.
        *   **Feedback:** Update GUI progress bar.

8.  **Finalization:**
    *   **Save Metadata:** Writes `_meta.json` with all experiment parameters.
    *   **Export:** Calls `export_to_csv_flat` to generate the final CSV.
