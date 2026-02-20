# KidBright MAI Server

KidBright MAI Server is the backend training and inference server for KidBright IDE. It enables users to train AI models using block-based programming interfaces. The server receives datasets, trains PyTorch models, and then converts them to a deployable format (NCNN int8) suited for edge devices like the KidBright board (or similar hardware like Sipeed Maix).

## Project Overview

This repository provides a Flask-based API that coordinates:
1. **Dataset Uploading**: Receives project data (images, labels, configurations) in a `.zip` file from the KidBright IDE.
2. **Model Training**: Spawns independent threads to train different types of models:
   - **Image Classification** (ResNet18, MobileNet varieties)
   - **Object Detection** (Slim YOLOv2)
   - **Voice Classification** (Custom CNN for Audio)
3. **Model Conversion**: Automatically converts the best PyTorch checkpoint to ONNX, then to NCNN (`.param`, `.bin`), and finally optimizes and quantizes to `INT8` using the provided `spnntools` in the `tools/` directory.
4. **Real-time Inference**: Provides an endpoint (`/inference_image`) to test the model dynamically once it has been trained (or loaded) in the server environment.

## API Endpoints

- `GET /ping`: Check server status, device environment, backend, and current Stage.
- `POST /upload`: Uploads `project.zip` which contains `project.json` and datasets to start a new project.
- `POST /train`: Initiates the model training based on the `target project_id`. Spawns a background thread.
- `GET /listen`: Server-Sent Events (SSE) endpoint to transmit real-time training progress and logs back to the IDE.
- `GET /convert`: Converts the trained PyTorch `.pth` model to an `INT8` quantized NCNN model.
- `GET /download_model`: Downloads the final exported model package (`model.zip`).
- `POST /terminate_training`: Stops an ongoing training thread.
- `POST /inference_image`: Performs inference using the currently trained model for rapid testing.

## Model Training & Structure

- **`projects/<project_id>`**: Each user project has a folder containing dataset files, `project.json` configurations, and an `output` folder for the saved models (`.pth`, `.onnx`, `.param`, `.bin`, etc.).
- **`train_image_classification.py`**: Handles ResNet and MobileNet based classification.
- **`train_object_detection.py`**: Handles YOLO-based bounding box detection training.
- **`train_voice_classification.py`**: Handles audio/voice classification using a unified 1D/2D CNN layout.
- **`convert.py`**: Manages the PyTorch -> ONNX -> NCNN pipeline.

## Conversion Pipeline Details

For edge deployment, the server runs a multi-step sequence:
1. Load PyTorch base model with learned weights.
2. Export to `ONNX`.
3. Utilize `onnx2ncnn` to convert to `NCNN`.
4. Run `spnntools optimize` on the NCNN model.
5. Run `spnntools calibrate` on a sample dataset to generate an INT8 calibration table.
6. Run `spnntools quantize` to finalize the `model_int8.param` and `model_int8.bin` output for embedded board utilization.