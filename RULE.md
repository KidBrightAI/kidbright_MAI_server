# KidBright MAI Server - AI Developer Context Rules

This file is intended to provide AI coding assistants (like Cursor, GitHub Copilot, Gemini etc.) with the domain context for generating, refactoring, and exploring the KidBright MAI Server codebase.

## Repository Context

- **Purpose**: Backend server for the KidBright IDE, specifically utilized for training AI models (Classification, Object Detection, Voice) via block programming.
- **Framework**: Python 3, Flask, PyTorch, NCNN.
- **Hardware Targets**: The models trained here are converted to INT8 NCNN formats for deployment on edge devices like KidBright hardware (V831 / Maix equivalents). Supports execution on Windows (EDGE backend), Linux, Raspberry Pi, Jetson, and Google Colab.

## System Architecture Highlights

1. **Flask API Structure (main.py)**: 
   - `upload()` extracts the ZIP payload.
   - `start_training()` kicks off `training_task()` in a background thread.
   - `listen()` uses Server-Sent Events (SSE) via a `MessageAnnouncer` to emit live training metrics to the KidBright block IDE.
   - `convert_model()` handles the pipeline from `.pth` -> `.onnx` -> `.param/.bin` -> `int8`. `tools/spnntools` is heavily used.
2. **Training Modules**:
   - `train_object_detection.py` is for YOLOv2 (`SlimYOLOv2`).
   - `train_image_classification.py` covers Image Classification (`ResNet`, `MobileNet`).
   - `train_voice_classification.py` targets voice pattern data.
3. **Data Locations**:
   - `projects/<project_id>/` contains user-uploaded datasets and the `project.json` configuration config.
   - Output models write to `projects/<project_id>/output/`.

## Coding Conventions & AI Workflow Guidelines

- **Do Not Break the SSE Pipe**: The `/listen` system relies on `MessageAnnouncer` (`q.announce()`). Any new training log or error output MUST be sent through this pipeline so the frontend KidBright IDE can read it.
- **Thread Safety**: Training runs in a background thread. Avoid mutating global variables unsafely.
- **Hardcoded Paths**: The current structure uses `projects/` and `tools/` with relative system calls (e.g., `os.system("tools/spnntools ...")`). Be cautious when refactoring file locations.
- **PyTorch to ONNX Strictness**: Any architectural changes to the PyTorch models (`models/` dir) MUST REMAIN strictly exportable through ONNX and NCNN without unsupported ops. 
- **Dependencies**: Uses `OpenCV` (cv2), `torch`, `torchvision`, `numpy`. Maintain numpy backwards compatibility where required (e.g. `np.float`, `np.bool` mapping to `np.float32`, `np.bool_`).

When proposing updates, prefer writing helper functions in `utils/helper.py` over bloating `main.py`. For training script modifications, make changes uniformly across all three `train_*.py` files if they pertain to generalized training loops (such as progress reporting).
