 # H.E.R.M.E.S. Human-centric Eye-tracking & Robust Motion Estimation Suite
 
 ## Abstract
 
 HERMES is a modular research framework developed to synchronise, analyse, and visualise eye-tracking data alongside computer vision-based kinematic extraction. The suite addresses the methodological challenge of defining dynamic Areas of Interest (AOIs) on moving human targets without manual annotation. By integrating YOLO26-based pose estimation with rigorous time-synchronisation protocols, HERMES allows researchers to map gaze data onto semantic body regions (e.g., Face, Hands, Peripersonal Space) in complex experimental settings.
 
 ## System Architecture
 
 The software is built on Python 3.12 and utilises a Tkinter graphical interface for broad compatibility on Windows systems. It employs a "Hub & Spoke" architecture managed by a central AppContext, ensuring state persistence across seven distinct processing modules.
 
 ## Core Modules
 
 *   **Human (Kinematic Extraction):** Utilises ultralytics (YOLO26) to extract 17-point skeletal keypoints. Supports BoT-SORT/ByteTrack algorithms for consistent multi-object tracking.
 *   **Master TOI (Synchronisation):** Aligns asynchronous data streams (e.g., Tobii Pro logs vs. external triggers/MATLAB logs) to define Times of Interest (TOIs).
 *   **Entity (Identity Assignment):** A post-processing interface to correct tracking errors (ID switches) and assign semantic roles (e.g., "Target", "Confederate") to anonymised track IDs.
 *   **Region (AOI Definition):** Defines geometric rules for dynamic AOIs (e.g., "Face" = nose + eyes + 30px margin). Supports bounding box expansion for peripersonal space analysis.
 *   **Eye Mapping:** Performs the geometric hit-testing between gaze coordinates (normalised) and the calculated dynamic AOIs.
 *   **Analytics:** Generates statistical reports (dwell time, latency, hit ratios) in both wide and long (tidy) formats suitable for R/Python analysis.
 *   **Reviewer:** A visual validation tool to inspect the coherence between video, gaze, and defined time windows.
 
 ## Installation
 
 ### Prerequisites
 
 *   **OS:** Windows 10/11 (64-bit).
 *   **Hardware:** NVIDIA GPU recommended (CUDA support) for the Human module; CPU inference is supported but significantly slower.
 *   **Dependencies:** Python 3.10+ (The setup script handles Python 3.12 installation automatically via uv).
 
 ### Setup Procedure
 
 HERMES uses uv, a modern Python package manager, for rapid and isolated environment creation.
 
 1.  Clone or download this repository.
 2.  Navigate to the root directory.
 3.  Execute `SETUP_LAB.bat`.
 
 **Note:** This script will automatically install uv, create a virtual environment, install dependencies from requirements.txt, and download necessary model weights.
 
 Once completed, launch the application using `AVVIA_HERMES.bat`.
 
 ## Usage Workflow
 
 The software enforces a sequential workflow to ensure data integrity:
 
 *   **Project Initialisation:** Create a project folder structure.
 *   **Input:** Import video files (.mp4) and eye-tracking logs.
 *   **Processing:** Run the modules in numerical order (1 through 6).
 *   **Output:** The final output is a CSV file containing row-wise gaze data enriched with semantic labels (Phase, Condition, Hit_AOI).
 
 ## Critical Analysis
 
 This section provides an objective evaluation of the software's current state, highlighting both technical advantages and limitations.
 
 ### Strengths (Pros)
 
 *   **Reproducibility:** The Human module enforces deterministic seeding for YOLO and CUDNN, ensuring that re-running analysis on the same video yields identical results—a critical requirement for scientific publication.
 *   **Dependency Management:** The implementation of uv in SETUP_LAB.bat provides a robust, isolated environment, resolving the "dependency hell" often found in academic Python projects.
 *   **Dynamic AOIs:** Automates a historically manual process (frame-by-frame AOI drawing), significantly reducing coding time for researchers.
 *   **Data Integrity:** The AppContext hub prevents user error by enforcing path consistencies and auto-saving project states (json persistence).
 
 ### Limitations (Cons)
 
 *   **Platform Dependency:** The current build system (.bat scripts) and path handling are strictly tied to the Windows ecosystem. Porting to Linux/macOS would require refactoring the startup scripts and path separators.
 *   **GUI Technology:** The use of tkinter ensures stability but results in a dated user interface that lacks modern responsiveness or hardware-accelerated rendering for video playback (relying on CPU-bound PIL conversions).
 *   **Tracking Drift:** While BoT-SORT is robust, extreme occlusion or rapid motion can still cause ID switches. The Entity module allows manual correction, but it remains a time-consuming bottleneck for complex footage.
 *   **Sync Logic:** The synchronisation relies on linear offsets. It assumes a constant frame rate and no clock drift between the eye-tracker and the video recording device, which may not hold true for long-duration recordings.
 
 ## Technical Stack
 
 *   **Language:** Python 3.12
 *   **GUI:** Tkinter / Tcl
 *   **Computer Vision:** OpenCV, Ultralytics (YOLO26)
 *   **Data Manipulation:** Pandas, NumPy
 *   **Packaging:** PyInstaller (via .spec file), uv
 
 ## Citation
 
 If you use HERMES in your research, please refer to the internal lab documentation for the appropriate citation format.
 
 ## Disclaimer
 
 This software is provided "as is" for research purposes. Ensure compliance with GDPR and ethical guidelines when processing video data containing identifiable human subjects.