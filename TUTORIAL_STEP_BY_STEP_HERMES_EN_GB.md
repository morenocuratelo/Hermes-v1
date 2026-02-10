# Step-by-Step Tutorial - HERMES (Windows 11)

This document provides a clear, procedural guide for operating HERMES on a 64-bit Windows 11 system. It is written for users with limited technical experience.

## 1. Requirements

- A **64-bit Windows 11** computer
- Internet access
- At least **10-15 GB** of available storage
- The project folder **Hermes v1**

## 2. Install Python (one-time operation)

1. Visit: https://www.python.org/downloads/windows/
2. Download Python 3.10 or later (64-bit).
3. Run the installer.
4. **Important:** tick `Add python.exe to PATH`.
5. Select `Install Now`.

## 3. Open the HERMES project folder

1. Open File Explorer.
2. Navigate to the `Hermes v1` directory.
3. Confirm that the following files are present:
   - `SETUP_LAB.bat`
   - `AVVIA_HERMES.bat`

## 4. Initial automated setup (once per computer)

1. Double-click `SETUP_LAB.bat`.
2. A black terminal window will open; this is expected behaviour.
3. Wait until the procedure is complete. The script will:
   - create a virtual environment;
   - install required libraries;
   - download AI model files.
4. When `Installazione completata` appears, press any key to close the window.

## 5. Daily application launch

1. Double-click `AVVIA_HERMES.bat`.
2. HERMES will launch automatically.

For routine use, always start the software via this file.

## 6. First use inside HERMES

At first launch:
1. Select `Create New Project` or `Open Existing Project`.
2. If creating a new project, choose a project name, folder, and participant.
3. Proceed through the workflow modules in sequence.

## 7. Troubleshooting

### Error: "Python non trovato"
- Python was not installed correctly or is not available in PATH.
- Reinstall Python and ensure `Add python.exe to PATH` is selected.

### Error during `SETUP_LAB.bat`
- Verify that internet access is active.
- Run `SETUP_LAB.bat` again.
- If SmartScreen blocks the file, select `More info` and then `Run anyway`.

### Model download failure
- Ensure each Google Drive link is shared as:
  `Anyone with the link / Viewer`.
- Run `SETUP_LAB.bat` again.

### Application launches but is slow
- In a virtual machine, or on systems without NVIDIA GPU acceleration, HERMES runs on CPU. Reduced performance is therefore expected.

## 8. Project updates

When a new HERMES version is provided:
1. Update or replace the project files.
2. Run `SETUP_LAB.bat` again (recommended).
3. Launch with `AVVIA_HERMES.bat`.

## 9. Laboratory deployment (multiple computers)

For each additional computer:
1. Copy the `Hermes v1` folder.
2. Install Python.
3. Execute `SETUP_LAB.bat`.
4. Launch via `AVVIA_HERMES.bat`.

No manual terminal workflow is required for regular operation.
