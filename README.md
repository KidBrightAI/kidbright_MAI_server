# KidBright mAI Server

Backend training, conversion, and inference server สำหรับ KidBright mAI IDE รับ dataset จาก IDE, train PyTorch model, แปลงเป็น format ที่เหมาะกับแต่ละบอร์ด แล้วส่งกลับ

## Supported Boards

| Board | Board ID | Chip | Protocol | Model Format |
|-------|----------|------|----------|-------------|
| KidBright uAI | `kidbright-mai` | MaixII | web-adb | `.bin` + `.param` (NCNN INT8) |
| KidBright uAI Plus | `kidbright-mai-plus` | CV181x (RISC-V) | websocket-shell | `.cvimodel` + `.mud` |

Device detection (main.py):
- Windows → `DEVICE="WINDOWS"`, `BACKEND="EDGE"`
- `COLAB_GPU` env → `DEVICE="COLAB"`, `BACKEND="COLAB"`
- `/proc/device-tree/model` containing "Jetson Nano" → `DEVICE="JETSON"`, `BACKEND="EDGE"`
- `/proc/device-tree/model` containing "Raspberry Pi" → `DEVICE="RPI"`, `BACKEND="EDGE"`

---

## Verified Hardware (SSH Inspection 2026-04-09)

ข้อมูลจากบอร์ดจริงที่ SSH ตรวจสอบ: `root@10.155.55.1`

### Board Identity

| Field | Value |
|-------|-------|
| device-tree/model | `LicheeRv Nano` |
| device-tree/compatible | `cvitek,cv181x` |
| Kernel | `5.10.4-tag-` RISC-V 64-bit (`rv64imafdvcsu`) |
| OS | Buildroot 2023.11.2 |
| Board Type | **MaixCAM** (ไม่ใช่ MaixCAM2) |

**สรุป: บอร์ดนี้คือ MaixCAM (CV181x / SG2002) ไม่ใช่ MaixCAM2 (AX630C)**

หลักฐาน:
- `compatible = cvitek,cv181x` → chip CV181x (SG2002) ซึ่งเป็น MaixCAM
- MaixCAM2 จะใช้ chip AX630C (Axera) ซึ่งจะมี `/dev/ax*` devices แทน `/dev/cvi*`
- NPU devices ทั้งหมดเป็น `cvi-tpu0`, `cvi-vpss` ฯลฯ → CVITEK TPU

### Hardware Specs

| Field | Value |
|-------|-------|
| SoC | CV181x (SG2002) — CVITEK RISC-V |
| RAM | 128 MB (total 127.8M, available ~55M) |
| Storage | 32 GB SD card (28.8G usable, 1.5G used) |
| NPU | CVITEK TPU (`/dev/cvi-tpu0`) — 0.5 TOPS INT8 |
| CVI Runtime | `libcviruntime.so`, `libcvikernel.so`, `libcvimath.so` |
| Model Format | `.cvimodel` (CVITEK quantized INT8) |
| Python | 3.11.6 |
| MaixCAM Lib | 1.24.0 |

### MaixPy NN Modules Available

```
Classifier, YOLO11, YOLOv8, YOLOv5, FaceDetector, FaceLandmarks,
FaceRecognizer, HandLandmarks, DepthAnything, NanoTrack, PP_OCR,
SelfLearnClassifier, YOLOWorld, MixFormerV2, Retinaface,
Qwen, InternVL, SmolVLM, Whisper, MeloTTS, Speech
```

`nn.YOLO11` ใช้ได้ ← ยืนยันว่า runtime รองรับ YOLO11 detection

### Models Currently Deployed on Board

| Hash | model_type | type | labels | Date |
|------|-----------|------|--------|------|
| `2d41b22f...` | yolo11 | (ไม่ระบุ) | coke, oleang | 2026-02-20 |
| `435969207...` | yolo11 | detector | coke, oleang | 2026-02-20 |
| `52bfceeb5...` | classifier | (ไม่ระบุ) | wall, water | 2026-02-20 |
| `9f4bbfddb...` | yolo11 | obb | coke, oleang | 2026-02-21 |

**MUD file variants พบบนบอร์ด:**

```ini
# Model 1 — ไม่มี type field (เวอร์ชันเก่า)
[extra]
model_type = yolo11
input_type = rgb
mean = 0, 0, 0
scale = 0.00392156862745098, ...
labels = coke, oleang

# Model 2 — type = detector
[extra]
model_type = yolo11
type = detector
...

# Model 3 — classifier (mobilenet)
[extra]
model_type = classifier
input_type = rgb
mean = 123.675, 116.28, 103.53
scale = 0.017124..., 0.017507..., 0.017429...
labels = wall, water

# Model 4 — type = obb (ล่าสุดจาก main.py)
[extra]
model_type = yolo11
type = obb
...
```

### Implications for Detection Model

เนื่องจากบอร์ดคือ **MaixCAM (CV181x)** ไม่ใช่ MaixCAM2:
- ต้องใช้ `.cvimodel` format (CVITEK TPU) ← ถูกแล้ว
- ตาม MaixPy doc ควรใช้ **Method B** (2 output nodes) สำหรับ YOLO11 detect:
  - `/model.23/dfl/conv/Conv_output_0`
  - `/model.23/Sigmoid_output_0`
- MUD file `model_type = yolo11` ← ถูกแล้ว
- MUD file ไม่ควรมี `type = obb` สำหรับ object detection ปกติ
- Hardware reference: https://wiki.sipeed.com/hardware/en/maixcam/maixcam.html

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/ping` | Server status, device, backend, current STAGE |
| POST | `/upload` | Upload `project.zip` (FormData: `project` file + `project_id`) |
| POST | `/train` | Start training (JSON body: `project` = project_id). Training config อ่านจาก `project.json` ภายใน ZIP |
| GET | `/listen` | SSE endpoint สำหรับ real-time training/conversion progress |
| GET | `/convert?project_id=xxx` | Convert trained model เป็น format เฉพาะบอร์ด |
| GET | `/download_model?project_id=xxx` | Download model.zip |
| POST | `/terminate_training` | หยุด training thread ที่กำลังทำงาน |
| POST | `/inference_image` | Inference ทดสอบด้วยรูปภาพ (FormData: `image`, `project_id`, `type`) |
| GET | `/projects/<path>` | Static file serving สำหรับ output files |

## STAGE Values

| STAGE | สถานะ |
|-------|-------|
| 0 | None (idle) |
| 1 | Preparing dataset |
| 2 | Training |
| 3 | Trained (พร้อม convert) |
| 4 | Converting |
| 5 | Converted (พร้อม download) |

---

## Data Flow: trainConfig

```
Frontend                                        Backend
────────                                        ───────
ModelDesigner.computeGraph()
  → trainConfig object
  → เก็บใน Pinia store (workspace.js)
  → serialize ลง project.json ภายใน ZIP

uploadProject()
  POST /upload → project.zip ถูกเก็บที่ projects/{project_id}/project.zip

trainColab()
  POST /train { project: id }
                                               training_task(project_id):
                                                 1. unzip project.zip
                                                 2. อ่าน project.json
                                                 3. ใช้ project["trainConfig"] ทั้งหมด
```

**หมายเหตุ:** Frontend ส่ง `train_config` ใน POST body ของ `/train` ด้วย แต่ backend ไม่ได้ใช้ค่านี้ — อ่านจาก `project.json` ภายใน ZIP เท่านั้น

### trainConfig Structure

```json
{
  "modelType": "slim_yolo_v2 | yolo11n | yolo5s | mobilenet-75 | resnet18 | ...",
  "objectThreshold": 0.5,
  "iouThreshold": 0.5,
  "train_split": 80,
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "validateMatrix": "mAP | val_accuracy | val_loss",
  "saveMethod": "best | last | best_one_of_third | best_one_of_half"
}
```

สร้างจาก Frontend Model Graph nodes:
- `default-input.js` → train_split, epochs, batch_size, learning_rate
- `yolo.js` → modelType, objectThreshold, iouThreshold (Object Detection)
- `image_classification.js` → modelType (Classification)
- `object-detection-output.js` → validateMatrix, saveMethod

### Frontend Model Type Options

**YOLO Node (yolo.js):**

| Display | Value | Default |
|---------|-------|---------|
| YOLO v2 slim | `slim_yolo_v2` | |
| YOLO v5s | `yolo5s` | |
| YOLO v11n | `yolo11n` | |

Default value ใน code: `"yolo_v11s"` (ไม่ตรงกับ value ใน dropdown ทั้ง 3 ตัว)

**Image Classification Node (image_classification.js):**

| Display | Value |
|---------|-------|
| MobileNet-100 | `mobilenet-100` |
| MobileNet-75 | `mobilenet-75` |
| MobileNet-50 | `mobilenet-50` |
| MobileNet-25 | `mobilenet-25` |
| MobileNet-10 | `mobilenet-10` |
| Resnet-18 | `resnet18` |

---

## Training Pipeline

### Routing Logic (main.py → training_task)

```
project["trainConfig"]["modelType"] → เลือก training function:

projectType == "VOICE_CLASSIFICATION"
  → train_voice_classification()

modelType == "slim_yolo_v2"
  → train_object_detection()

modelType == "yolo11n"
  → train_object_detection_yolo11n()

modelType.startswith("resnet")
  → train_image_classification()

modelType.startswith("mobilenet")
  → train_image_classification()

modelType == "yolo5s"
  → ไม่มี branch รองรับ (ไม่มี training script)
```

### Training Scripts

#### train_object_detection.py — Slim YOLOv2

| Parameter | Value |
|-----------|-------|
| Model | SlimYOLOv2 (DarkNet-19 backbone) |
| Input Size | 416x416 (multi-scale 640→416) |
| Anchors | `[[1.19,1.98],[2.79,4.59],[4.53,8.92],[8.06,5.29],[10.32,10.65]]` |
| Dataset Format | VOC (JPEGImages/, Annotations/, ImageSets/) |
| Augmentation | SSDAugmentation (mean: 0.5, std: 128/255) |
| Optimizer | Adam, weight_decay=5e-4 |
| Loss | conf_loss + cls_loss + txtytwth_loss |
| Stride | 32 |
| Output | `output/best_map.pth` |

#### train_object_detection_yolo11n.py — YOLO11 Nano

| Parameter | Value |
|-----------|-------|
| Model | Ultralytics YOLO("yolo11n.pt") |
| Training imgsz | `[224, 320]` (hardcoded ใน code, ไม่ใช้ high_resolution param) |
| Dataset Format | VOC → แปลงเป็น YOLO format อัตโนมัติ (convert_voc_to_yolo) |
| Output Dir | `output/yolo11n_run/weights/best.pt` |
| Copy | best.pt → `output/best_map.pth` (เพิ่มเติม) |
| Callbacks | on_train_epoch_start, on_train_batch_start/end, on_fit_epoch_end, on_train_end |

#### train_image_classification.py — MobileNet / ResNet

| Parameter | Value |
|-----------|-------|
| Models | mobilenet_v2 (width_mult variants), resnet18/34/50/101/152 |
| Input Size | 224x224 |
| Normalization | ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| Augmentation | Resize(255) → RandomRotation(30) → RandomResizedCrop(224) |
| Optimizer | Adam, weight_decay=5e-4 |
| Loss | CrossEntropyLoss |
| Output | `output/best_acc.pth` |

#### train_voice_classification.py — Voice CNN

| Parameter | Value |
|-----------|-------|
| Models | ResNet variants, VoiceCnn (custom from code string) |
| Input Size | (3, 147, 13) — MFCC features |
| Dataset | `dataset/mfcc/{label}/` |
| Output | `output/best_acc.pth` |

---

## Conversion Pipeline

### Routing Logic (main.py → convert_model)

```
1. หา best_file ตาม modelType:
   - mobilenet/resnet18/code → output/best_acc.pth
   - slim_yolo_v2            → output/best_map.pth
   - yolo11n                 → output/yolo11n_run/weights/best.pt
                               fallback: runs/detect/{path}/output/yolo11n_run/weights/best.pt

2. โหลด model (ยกเว้น yolo11n ที่ set net = None)

3. Export ONNX:
   - modelType != yolo11n: torch_to_onnx(net, input_shape) → model.onnx
   - modelType == yolo11n: YOLO(best_file).export(format="onnx", imgsz=[224,320]) → model.onnx

4. Board-specific conversion (ดูตาราง Conversion Matrix ด้านล่าง)
```

### Conversion Matrix

| Board ID | modelType | Conversion Path |
|----------|-----------|----------------|
| `kidbright-mai-plus` | `mobilenet*` | ONNX → MLIR → calibrate → INT8 cvimodel + MUD |
| `kidbright-mai-plus` | `yolo11n` | Ultralytics ONNX export → MLIR → calibrate → INT8 cvimodel + MUD |
| `kidbright-mai-plus` | `slim_yolo_v2` | ไม่มี cvimodel path เฉพาะ → ตกไป NCNN path (else branch) |
| `kidbright-mai-plus` | `resnet18` | ไม่มี cvimodel path เฉพาะ → ตกไป NCNN path (else branch) |
| `kidbright-mai` | `slim_yolo_v2` | ONNX → NCNN → optimize → calibrate → quantize INT8 |
| `kidbright-mai` | `yolo11n` | ตกไป NCNN path แต่ ncnn_out_param/ncnn_out_bin ไม่ถูก define (variable ไม่มี) |
| อื่นๆ | ทุก model | ONNX → NCNN → spnntools optimize → calibrate → quantize INT8 |

### CVI Conversion: kidbright-mai-plus + mobilenet

```
CMD1: model_transform.py
  --model_name mobilenet
  --input_shapes [[1,3,224,224]]
  --mean 123.675,116.28,103.53
  --scale 0.0171,0.0175,0.0174
  --keep_aspect_ratio
  --pixel_format rgb
  --channel_format nchw
  --test_input data/test_images2/cat.jpg
  → mobilenet.mlir + mobilenet_top_outputs.npz

CMD2: run_calibration.py
  --dataset {images_path}
  --input_num 24
  --processor cv181x
  → mobilenet_cali_table

CMD3: model_deploy.py
  --quantize INT8
  --quant_input
  --chip cv181x
  --processor cv181x
  → model.cvimodel

MUD file:
  [basic]
  type = cvimodel
  model = model.cvimodel
  [extra]
  model_type = classifier
  input_type = rgb
  mean = 123.675, 116.28, 103.53
  scale = 0.017124753831663668, 0.01750700280112045, 0.017429193899782137
  labels = {sorted labels}
```

### CVI Conversion: kidbright-mai-plus + yolo11n

```
CMD1: model_transform.py
  --model_name yolo11n
  --input_shapes [[1,3,224,320]]
  --mean "0,0,0"
  --output_names "/model.23/Sigmoid_output_0"     ← ใช้ 1 output node
  --scale "0.00392156862745098,..."
  --keep_aspect_ratio
  --pixel_format rgb
  --channel_format nchw
  --test_input data/test_images2/cat.jpg
  → model.mlir + yolo11s_top_outputs.npz

CMD2: run_calibration.py
  --dataset data/test_images
  --input_num 24
  (ไม่มี --processor)
  → yolo11n_cali_table

CMD3: model_deploy.py
  --quantize INT8
  --processor cv181x
  (ไม่มี --chip, ไม่มี --quant_input)
  → model.cvimodel

MUD file:
  [basic]
  type = cvimodel
  model = model.cvimodel
  [extra]
  model_type = yolo11
  type = obb
  input_type = rgb
  mean = 0, 0, 0
  scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
  labels = {sorted labels}
```

### NCNN Conversion (else branch — kidbright-mai default)

```
1. onnx_to_ncnn(input_shape, onnx, ncnn_param, ncnn_bin) → model.param + model.bin
2. spnntools optimize → model_opt.param + model_opt.bin
3. spnntools calibrate → model_opt.table
   mean="127.5,127.5,127.5", norm="0.0078125,0.0078125,0.0078125"
4. spnntools quantize → model_int8.param + model_int8.bin
```

---

## Colab Notebook Conversion Tests

ไฟล์: `Copy_of_KidBright_μAI_Server_ver_1_0.ipynb`

มี 2 version ทดลอง yolo11n conversion:

### Version A (Cell 18-20): Method B — 2 output nodes

```
model_transform.py
  --output_names "/model.23/dfl/conv/Conv_output_0,/model.23/Sigmoid_output_0"
  --input_shapes [[1,3,224,320]]

run_calibration.py → yolo11n_cali_table

model_deploy.py
  --processor cv181x
  → model.cvimodel   ← สำเร็จ (ดูจาก output log)
```

### Version B (Cell 22-24): Method A — 3 output nodes

```
model_transform.py
  --output_names "/model.23/Concat_output_0,/model.23/Concat_1_output_0,/model.23/Concat_2_output_0"
  --input_shapes [[1,3,224,224]]   ← ใช้ 224x224 ไม่ใช่ 224x320

run_calibration.py → yolo11n_cali_table

model_deploy.py
  --processor cv181x
  → model.cvimodel
```

---

## MaixPy Documentation Reference

ที่มา: `Offline Training for YOLO11_YOLOv8 Models on MaixCAM MaixPy` (Sipeed Wiki)
- Doc version: v3.0 (2025-07-01) — เพิ่ม MaixCAM2 support
- v2.0 (2024-10-10) — เพิ่ม YOLO11 support
- v1.0 (2024-06-21) — Document creation

### YOLOv8 vs YOLO11 — ความเหมือน/ต่าง

- preprocessing / postprocessing **เหมือนกัน**
- training + conversion steps **identical**
- ต่างกันแค่ **output node names**: YOLOv8 ใช้ `model.22`, YOLO11 ใช้ `model.23`

### Supported Tasks (MaixPy/MaixCDK)

| Task | YOLOv8 | YOLO11 |
|------|--------|--------|
| Object Detection | `yolov8n` | `yolo11n` |
| Keypoint (Pose) | `yolov8n-pose` | `yolo11n-pose` |
| Segmentation | `yolov8n-seg` | `yolo11n-seg` |
| OBB (Oriented BBox) | `yolov8n-obb` | `yolo11n-obb` |

### ONNX Export (MaixPy doc)

```python
# export_onnx.py — สร้างไว้ใน ultralytics directory
from ultralytics import YOLO
import sys

net_name = sys.argv[1]       # yolov8n.pt / yolo11n.pt / yolov8n-pose.pt
input_width = int(sys.argv[2])
input_height = int(sys.argv[3])

model = YOLO(net_name)
results = model("https://ultralytics.com/images/bus.jpg")
path = model.export(format="onnx", imgsz=[input_height, input_width])
```

Resolution ตาม doc:
- **MaixCAM**: `python export_onnx.py yolov8n.pt 320 224` → imgsz=[224, 320]
  - เหตุผล: ใกล้เคียง aspect ratio หน้าจอ MaixCAM → แสดงผลง่าย
- **MaixCAM2**: `640x480` หรือ `320x240`

### Output Node Selection — ตารางครบทุก task

#### Method A vs Method B

| Feature | Method A | Method B |
|---------|----------|----------|
| Devices | **MaixCAM2** (แนะนำ), MaixCAM (ช้ากว่า B) | **MaixCAM** (แนะนำ) |
| Computation | CPU มาก (quantization ปลอดภัยกว่า, ช้ากว่า B เล็กน้อย) | NPU มาก (computation อยู่ใน quantization) |
| ข้อจำกัด | ไม่มี | Quantization **fail** บน MaixCAM2 ในการทดสอบจริง |

#### Detect Nodes

| Model | Method A (3 nodes) | Method B (2 nodes) |
|-------|-------------------|-------------------|
| YOLOv8 | `/model.22/Concat_1_output_0`, `/model.22/Concat_2_output_0`, `/model.22/Concat_3_output_0` | `/model.22/dfl/conv/Conv_output_0`, `/model.22/Sigmoid_output_0` |
| YOLO11 | `/model.23/Concat_output_0`, `/model.23/Concat_1_output_0`, `/model.23/Concat_2_output_0` | `/model.23/dfl/conv/Conv_output_0`, `/model.23/Sigmoid_output_0` |

#### Keypoint (Pose) Nodes — Detect nodes + เพิ่ม:

| Model | Method A เพิ่ม | Method B เพิ่ม |
|-------|---------------|---------------|
| YOLOv8-pose | + `/model.22/Concat_output_0` | + `/model.22/Concat_output_0` |
| YOLO11-pose | + `/model.23/Concat_output_0` | + `/model.23/Concat_output_0` |

#### Segmentation Nodes — Detect nodes + เพิ่ม:

| Model | Method A เพิ่ม | Method B เพิ่ม |
|-------|---------------|---------------|
| YOLOv8-seg | + `/model.22/Concat_output_0` + `output1` | + `/model.22/Concat_output_0` + `output1` |
| YOLO11-seg | + `/model.23/Concat_output_0` + `output1` | + `/model.23/Concat_output_0` + `output1` |

#### OBB Nodes

| Model | Method A (4 nodes) | Method B (3 nodes) |
|-------|-------------------|-------------------|
| YOLOv8-obb | Concat_1,2,3 + Concat_output_0 | dfl/conv/Conv + **Sigmoid_1** + Sigmoid |
| YOLO11-obb | Concat_output_0,1,2 + Concat_output_0 | dfl/conv/Conv + **Sigmoid_1** + Sigmoid |

### MUD File Format — MaixCAM/MaixCAM-Pro (Detection)

```ini
[basic]
type = cvimodel
model = yolov8n.cvimodel

[extra]
model_type = yolov8          ; หรือ yolo11 สำหรับ YOLO11
input_type = rgb
mean = 0, 0, 0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
labels = label1, label2, ...
```

### MUD File `type` field (section [extra]) — ตาม Task

| Task | type field |
|------|-----------|
| Detection | **ไม่ต้องมี** (หรือ ละไว้) |
| Pose | `type=pose` |
| Segmentation | `type=seg` |
| OBB | `type=obb` |

### MUD File Format — MaixCAM2 (Detection)

```ini
[basic]
type = axmodel
model_npu = yolo11n_640x480_npu.axmodel
model_vnpu = yolo11n_640x480_vnpu.axmodel

[extra]
model_type = yolo11
type = detector
input_type = rgb
labels = label1, label2, ...
input_cache = true
output_cache = true
input_cache_flush = false
output_cache_inval = true
mean = 0,0,0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
```

### Cross-Reference: Doc vs Board vs Code

| Topic | MaixPy Doc | Board จริง (SSH) | main.py Code |
|-------|-----------|-----------------|-------------|
| MaixCAM output method | **Method B** (2 nodes) แนะนำ | Model ที่โหลดได้มี 1 output node (Concat_1 only), 3 nodes fail | ใช้ 1 node (Sigmoid only) |
| MUD type field (detect) | **ไม่ต้องมี** | `type=obb` ก็โหลดได้, ไม่มี type ก็ fail → type ไม่ใช่สาเหตุ fail | สร้าง `type=obb` |
| model_type | `yolov8` หรือ `yolo11` | `yolo11` ทุกตัว | สร้าง `yolo11` |
| mean/scale | `0,0,0` / `1/255` | ตรงกัน | ตรงกัน |
| imgsz | `[224, 320]` สำหรับ MaixCAM | Working model: `[1,3,224,320]` | training + export ใช้ `[224,320]` |
| Model size | ไม่ระบุ | 2.4-2.9 MB | - |
| Inference speed | ไม่ระบุ | 14ms per frame | - |

---

## Board Runtime (Generated Python Code)

### kidbright-mai-plus — Object Detection

```python
# generators_ai.js สร้างโค้ดนี้:
from maix import nn

class Yolo:
  labels = ["label1", "label2", ...]
  def __init__(self):
    self.detector = nn.YOLO11(model="/root/model/{hash}.mud", dual_buff=True)

_yolo = Yolo()
_boxes = _yolo.detector.detect(image, conf_th=threshold, iou_th=nms)
# box.x, box.y, box.w, box.h, box.class_id, box.score
```

### kidbright-mai-plus — Image Classification

```python
from maix import nn

class _Classifier:
  def __init__(self):
    self.classifier = nn.Classifier(model="/root/model/{hash}.mud", dual_buff=True)

_model = _Classifier()
_model_result = _model.classify(image)
# result[0][0] = class_id, result[0][1] = probability
```

### kidbright-mai-plus — Voice Classification

```python
# ใช้ .bin/.param format (AWNN) ไม่ใช่ .mud
from maix import nn
import voice_mfcc

class _Resnet:
  m = {"bin": "/root/model/{hash}.bin", "param": "/root/model/{hash}.param"}
  options = {
    "model_type": "awnn",
    "inputs": {"input0": (147, 13, 3)},
    "outputs": {"output0": (1, 1, num_classes)},
    "mean": [127.5, 127.5, 127.5],
    "norm": [0.00784313725490196, ...],
  }
  def __init__(self):
    self.model = nn.load(self.m, opt=self.options)

_model = _Resnet()
_model_result = _model.model.forward(mfcc_image, quantize=True)
```

### kidbright-mai — Object Detection (legacy)

```python
# ใช้ .bin/.param + manual decoder
from maix import nn
from maix.nn import decoder

class Yolo:
  labels = [...]
  anchors = [1.19,1.98, 2.79,4.59, 4.53,8.92, 8.06,5.29, 10.32,10.65]
  m = {"bin": "/root/model/{hash}.bin", "param": "/root/model/{hash}.param"}
  options = {
    "model_type": "awnn",
    "inputs": {"input0": (224, 224, 3)},
    "outputs": {"output0": (7, 7, (1+4+len(labels))*5)},
    "mean": [127.5, 127.5, 127.5],
    "norm": [0.0078125, 0.0078125, 0.0078125],
  }
  def __init__(self):
    self.model = nn.load(self.m, opt=self.options)
    self.decoder = decoder.Yolo2(len(self.labels), self.anchors,
                                 net_in_size=(224,224), net_out_size=(7,7))

_yolo = Yolo()
_out = _yolo.model.forward(image.tobytes(), quantize=True, layout="hwc")
_boxes, _probs = _yolo.decoder.run(_out, nms=nms, threshold=threshold, img_size=(224,224))
```

### kidbright-mai — Image Classification (legacy)

```python
from maix import nn

class _Resnet:
  m = {"bin": "/root/model/{hash}.bin", "param": "/root/model/{hash}.param"}
  options = {
    "model_type": "awnn",
    "inputs": {"input0": (224, 224, 3)},
    "outputs": {"output0": (1, 1, num_classes)},
    "mean": [127.5, 127.5, 127.5],
    "norm": [0.00784313725490196, ...],
  }
  def __init__(self):
    self.model = nn.load(self.m, opt=self.options)

_model = _Resnet()
_model_result = _model.model.forward(image, quantize=True)
```

---

## Real Board Benchmark (MaixCAM CV181x — 2026-04-09)

### Official Pre-built Models บนบอร์ด (/root/models/)

| Model | File | cvimodel Size | model_type | Output Nodes |
|-------|------|-------------|-----------|-------------|
| **YOLO11n** | `yolo11n_320x224_int8.cvimodel` | **2.8 MB** | yolo11 | 2 nodes: `dfl/conv/Conv` + `Sigmoid` **(Method B)** |
| **YOLOv8n** | `yolov8n.cvimodel` | **3.2 MB** | yolov8 | 2 nodes: `dfl/conv/Conv` + `Sigmoid` **(Method B)** |
| **YOLOv5s** | `yolov5s_320x224_int8.cvimodel` | **7.2 MB** | yolov5 | 3 nodes: `m.0/Conv`, `m.1/Conv`, `m.2/Conv` (multi-scale) |
| YOLO11n-obb | `yolo11n-obb_320x224_int8.cvimodel` | - | yolo11, type=obb | - |
| YOLO11n-pose | `yolo11n-pose_320x224_int8.cvimodel` | - | yolo11, type=pose | - |
| YOLO11n-seg | `yolo11n-seg_320x224_int8.cvimodel` | - | yolo11, type=seg | - |
| YOLOv8n-pose | `yolov8n_pose.cvimodel` | 3.3 MB | yolov8, type=pose | - |
| YOLOv8n-seg | `yolov8n_seg.cvimodel` | 3.4 MB | yolov8, type=seg | - |

### Official Model Output Node Structure

```
YOLO11n (Method B — MaixCAM recommended):
  Input:  images, INT8, [1, 3, 224, 320]
  Output[0]: /model.23/dfl/conv/Conv_output_0, INT8, [1, 1, 4, 1470]     ← bounding box
  Output[1]: /model.23/Sigmoid_output_0,       INT8, [1, 80, 1470, 1]    ← class scores

YOLOv8n (Method B — MaixCAM recommended):
  Input:  images, INT8, [1, 3, 224, 320]
  Output[0]: /model.22/dfl/conv/Conv_output_0, INT8, [1, 1, 4, 1470]
  Output[1]: /model.22/Sigmoid_output_0,       INT8, [1, 80, 1470, 1]

YOLOv5s (multi-scale output):
  Input:  images, INT8, [1, 3, 224, 320]
  Output[0]: /model.24/m.0/Conv_output_0, INT8, [1, 255, 28, 40]         ← stride 8
  Output[1]: /model.24/m.1/Conv_output_0, INT8, [1, 255, 14, 20]         ← stride 16
  Output[2]: /model.24/m.2/Conv_output_0, INT8, [1, 255, 7, 10]          ← stride 32
```

### Inference Benchmark (blank 320x224 image, 20 frames avg)

| Model | Avg | Min | Max | FPS | Status |
|-------|-----|-----|-----|-----|--------|
| **YOLO11n** | **16.5ms** | 13.5ms | 19.2ms | **60.4** | OK |
| **YOLOv8n** | **16.5ms** | 13.2ms | 19.3ms | **60.4** | OK |
| **YOLOv5s** | - | - | - | - | **SIGSEGV** (crash) |

YOLOv5s SegFault เมื่อ detect — shared memory 860KB อาจเกินขีดจำกัดบอร์ด

### MUD File ของ YOLOv5s มี field พิเศษ

```ini
# YOLOv5s ต้องมี anchors field (YOLO11/v8 ไม่ต้อง)
anchors = 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
```

### สรุป: Official models ทั้งหมดใช้ Method B (2 output nodes) สำหรับ YOLO11/v8

นี่คือหลักฐานยืนยันว่า **Method B คือ standard ที่ถูกต้องสำหรับ MaixCAM** — ตรงกับ MaixPy doc

Model ที่เรา convert เอง (main.py) ใช้ output_names ผิด:
- main.py ใช้: `/model.23/Sigmoid_output_0` (1 node เท่านั้น)
- Official model ใช้: `/model.23/dfl/conv/Conv_output_0` + `/model.23/Sigmoid_output_0` (2 nodes)

---

## MaixPy Usage Doc Reference

ที่มา: `MaixPy MaixCAM Using YOLOv5 / YOLOv8 / YOLO11 for Object Detection` (Sipeed Wiki)

### Official Model Comparison (จาก Sipeed)

| Model | Size Variant | Pre-built | Accuracy Ranking |
|-------|-------------|-----------|-----------------|
| YOLOv5 | **s** (small) | yolov5s_320x224 | ต่ำสุด |
| YOLOv8 | **n** (nano) | yolov8n_320x224 | กลาง |
| YOLO11 | **n** (nano) | yolo11n_320x224 | **สูงสุด** |

**Accuracy ranking (Sipeed official):** `YOLO11n > YOLOv8n > YOLOv5s`

### s-variant (Small) — Higher Accuracy Option

จาก doc:
> "Additionally, you may try YOLOv8s or YOLO11s, which will have a lower frame rate
> (e.g., yolov8s_320x224 is **10ms slower** than yolov8n_320x224),
> but offer **higher accuracy**."

| Variant | Speed (320x224 est.) | Accuracy | cvimodel Size (est.) |
|---------|---------------------|----------|---------------------|
| YOLO11n | 16.5ms / 60 FPS | baseline | 2.8 MB |
| YOLO11s | **~26.5ms / ~38 FPS** | **สูงกว่า n** | ~6-8 MB (est.) |
| YOLOv8n | 16.5ms / 60 FPS | ต่ำกว่า YOLO11n | 3.2 MB |
| YOLOv8s | **~26.5ms / ~38 FPS** | สูงกว่า v8n | ~8-10 MB (est.) |
| YOLOv5s | crash (SIGSEGV) | ต่ำสุด | 7.2 MB |

**หมายเหตุ:** ไม่มี pre-built YOLO11s/YOLOv8s cvimodel บนบอร์ด — ต้อง export เองจาก official YOLO repo

### Resolution Variants (จาก MaixHub)

Doc แนะนำว่า model มีหลาย resolution ให้เลือก:
- Default: **320x224** (ใกล้เคียงหน้าจอ MaixCAM)
- Higher resolution → **accuracy สูงขึ้น** แต่ **ช้าลง**
- Download จาก MaixHub:
  - YOLOv5: maixhub.com/model/zoo/365
  - YOLOv8: maixhub.com/model/zoo/400
  - YOLO11: maixhub.com/model/zoo/453

### Ultralytics Official Benchmark (COCO val2017)

| Model | Params | FLOPs | mAP50-95 |
|-------|--------|-------|----------|
| YOLOv5s | 7.2M | 16.5G | 37.4 |
| YOLOv8n | 3.2M | 8.7G | 37.3 |
| **YOLOv8s** | **11.2M** | **28.6G** | **44.9** |
| YOLO11n | 2.6M | 6.5G | 39.5 |
| **YOLO11s** | **9.4M** | **21.5G** | **47.0** |

### Model Selection Analysis สำหรับ KidBright mAI Plus (MaixCAM)

**ข้อจำกัดของบอร์ด:**
- RAM: 128 MB (available ~55 MB)
- TPU: 0.5 TOPS INT8
- Working model shared memory: yolo11n=412KB, yolov5s=860KB (crash)
- cvimodel max practical size: ประมาณ 8-10 MB (ต้องทดสอบ)

**ถ้าเน้น accuracy (ยอม FPS ลดลง):**

| Choice | mAP50-95 | Est. Speed | Risk |
|--------|----------|-----------|------|
| **YOLO11s** | **47.0** (+19% vs 11n) | ~26.5ms / ~38 FPS | ต้องทดสอบว่า cvimodel ขนาด ~8MB จะ run ได้บน 128MB RAM หรือไม่ |
| YOLO11n | 39.5 | 16.5ms / 60 FPS | ใช้งานได้แน่นอน (ทดสอบแล้ว) |
| YOLOv8s | 44.9 (+20% vs v8n) | ~26.5ms / ~38 FPS | เหมือน YOLO11s แต่ accuracy ต่ำกว่า |

**สรุป: YOLO11s เป็นตัวเลือกที่ดีที่สุดถ้าเน้น accuracy**
- mAP สูงสุด (47.0) ในทุก variant
- ช้ากว่า YOLO11n ประมาณ 10ms (ยังได้ ~38 FPS ซึ่งใช้งานได้)
- ต้องทดสอบ convert + deploy บนบอร์ดจริงก่อน
- ถ้า YOLO11s ไม่ผ่าน (RAM/TPU ไม่พอ) → fallback เป็น YOLO11n

**แผน: ควรรองรับทั้ง YOLO11n (default) และ YOLO11s (accuracy mode) ใน frontend dropdown**

---

## Model Deployment to Board

### kidbright-mai-plus (websocket-shell.js)

1. Model files ถูก upload ไปที่ `/root/model/{md5_hash}.{ext}` บนบอร์ด
2. ก่อน upload `.mud` file จะถูก patch: `model = model.cvimodel` → `model = {hash}.cvimodel`
3. ตรวจสอบ file size ก่อน upload ถ้าซ้ำจะข้ามไป
4. Upload เป็น chunks ขนาด 256KB ผ่าน WebSocket
5. Generated Python code ถูก inject ที่ `##{main}##` ใน main.py template
6. Kill process เดิม → run `python3 /root/app/run.py`

### kidbright-mai (web-adb)

1. Model files: `.bin` + `.param` ที่ `/root/model/{hash}.bin`, `/root/model/{hash}.param`
2. Upload ผ่าน ADB protocol
3. Camera config: `camera.camera.config(size=(224, 224))`

---

## Project File Structure

```
projects/{project_id}/
├── project.zip          ← upload จาก frontend
├── project.json         ← extracted, มี trainConfig, labels, projectType, currentBoard
├── dataset/
│   ├── JPEGImages/      ← รูปภาพ
│   ├── Annotations/     ← VOC XML bounding boxes
│   └── ImageSets/Main/  ← train.txt, val.txt
├── yolo_dataset/        ← สร้างโดย convert_voc_to_yolo() สำหรับ yolo11n
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   ├── labels/val/
│   └── data.yaml
├── output/
│   ├── best_map.pth     ← slim_yolo_v2 / yolo11n copy
│   ├── best_acc.pth     ← classification / voice
│   ├── model.onnx       ← ONNX export
│   ├── model.param      ← NCNN
│   ├── model.bin        ← NCNN
│   ├── model_opt.param  ← optimized NCNN
│   ├── model_opt.bin
│   ├── model_int8.param ← quantized INT8 (kidbright-mai final)
│   ├── model_int8.bin
│   ├── model.cvimodel   ← CVI model (kidbright-mai-plus final)
│   ├── model.mud        ← Model metadata (kidbright-mai-plus final)
│   ├── model.mlir       ← MLIR intermediate
│   └── yolo11n_run/     ← Ultralytics output
│       └── weights/
│           └── best.pt
└── temp/                ← inference temp images
```

---

## Key Source Files

| File | Description |
|------|-------------|
| `main.py` | Flask server, routing, conversion orchestration |
| `train_object_detection.py` | Slim YOLOv2 training (VOC dataset) |
| `train_object_detection_yolo11n.py` | YOLO11n training (Ultralytics) |
| `train_image_classification.py` | MobileNet/ResNet classification training |
| `train_voice_classification.py` | Voice CNN training (MFCC) |
| `convert.py` | torch_to_onnx(), onnx_to_ncnn(), gen_input() |
| `data/config.py` | ANCHOR_SIZE, ANCHOR_SIZE_COCO constants |
| `data/__init__.py` | BaseTransform, detection_collate |
| `data/custom.py` | CustomDetection VOC dataset loader |
| `models/slim_yolo_v2.py` | SlimYOLOv2 model (DarkNet-19 backbone) |
| `models/voice_cnn.py` | VoiceCnn custom model |
| `backbone/darknet.py` | DarkNet-19, DarkNet-53 backbone |
| `tools.py` | Loss functions, gt_creator, anchor generation |
| `utils/helper.py` | prepare_image, read_json_file, folder utils |
| `utils/message_announcer.py` | SSE message broadcasting |
| `tuna.py` | Tunnel proxy (tuna service) |

## Real Board Model Inspection (SSH 2026-04-09)

ตรวจสอบ model ที่ deploy อยู่บนบอร์ดจริง ผ่าน `nn.NN()` low-level API:

### Model Output Node Comparison

| Hash | Method | Output Nodes | Shape | nn.YOLO11 Load |
|------|--------|-------------|-------|----------------|
| `9f4bbfdd` | **A (3 Concat)** | `Concat_1_output_0` only (1 output) | `[1,2,1470,1]` | **OK** |
| `2d41b22f` | **A (3 Concat)** | `Concat_output_0`, `Concat_1_output_0`, `Concat_2_output_0` | `[1,64,1029,1]`, `[1,2,1029,1]`, `[1,4,1029,1]` | **FAIL** |
| `43596920` | **A (3 Concat)** | `Concat_output_0`, `Concat_1_output_0`, `Concat_2_output_0` | `[1,64,1470,1]`, `[1,2,1470,1]`, `[1,4,1470,1]` | **FAIL** |

### Key Finding: nn.YOLO11 จาก MaixPy รองรับ output nodes ได้จำกัด

- Model ที่มี **1 output node** → โหลดได้
- Model ที่มี **3 output nodes** → โหลดไม่ได้ (`load model failed`)
- `nn.YOLOv8` ก็โหลด model 3-output ไม่ได้เช่นกัน
- **Model ที่ทำงานได้** (`9f4bbfdd`) มี MUD `type = obb` — แต่ใช้งาน detect ปกติได้, inference 14ms บน blank image
- **Model ที่ fail** (`2d41b22f`, `43596920`) แม้จะมี/ไม่มี `type` field ใน MUD ก็ fail เหมือนกัน → ปัญหาอยู่ที่ **จำนวน output nodes ใน cvimodel** ไม่ใช่ MUD field

### Working Model Details (9f4bbfdd)

```
Input:  images, INT8, [1, 3, 224, 320]
Output: /model.23/Concat_1_output_0_Concat_f32, INT8, [1, 2, 1470, 1]

MUD:
  model_type = yolo11
  type = obb
  mean = 0, 0, 0
  scale = 0.00392156862745098, ...
  labels = coke, oleang

Inference: 14ms on blank 320x224 image
Build: 2026-02-21 11:24:57 for cv181x
```

### nn.YOLO11 API (from board)

```python
nn.YOLO11(model: str = '', dual_buff: bool = True)

# detect method:
detect(img, conf_th=0.5, iou_th=0.45, fit=FIT_CONTAIN, keypoint_th=0.5, sort=0) -> Objects

# Result attributes:
box.x, box.y, box.w, box.h, box.class_id, box.score

# Also supports: pose, seg, obb via same class
# Available methods: detect, draw_pose, draw_seg_mask, input_format, 
#   input_height, input_size, input_width, label_path, labels, load, mean, scale
```

### nn.YOLO11 vs nn.YOLOv8 (on board)

ทั้ง 2 class มี methods **เหมือนกัน**:
```
['detect', 'draw_pose', 'draw_seg_mask', 'input_format', 'input_height', 
 'input_size', 'input_width', 'label_path', 'labels', 'load', 'mean', 'scale']
```

---

## Default Detection Model Recommendation

### สำหรับ KidBright mAI Plus (MaixCAM / CV181x)

**YOLO11n** เป็นตัวเลือกเดียวที่เหมาะสม เนื่องจาก:

1. **slim_yolo_v2 ใช้ไม่ได้** — ไม่มี cvimodel conversion path สำหรับ mai-plus, runtime ใช้ `nn.load()` + `decoder.Yolo2()` ซึ่งเป็น API ของ kidbright-mai (MaixII) ไม่ใช่ MaixCAM
2. **yolo5s ใช้ไม่ได้** — ไม่มี training script ใน backend เลย
3. **nn.YOLO11 มีบนบอร์ด** — ยืนยันจาก SSH inspection ว่า runtime รองรับ
4. **Training + Conversion pipeline มีอยู่แล้ว** — `train_object_detection_yolo11n.py` + cvimodel conversion ใน `main.py`
5. **Model size เหมาะสม** — cvimodel ~2.4-2.9 MB, พอสำหรับ 128MB RAM
6. **Inference เร็ว** — 14ms per frame บนบอร์ดจริง

### Conversion ที่ทำงานได้จริงบนบอร์ด (จากการทดสอบ)

จาก model ที่ deploy อยู่บนบอร์ด พบว่า:
- **output_names ที่ใช้ใน main.py ปัจจุบัน** (`/model.23/Sigmoid_output_0` — 1 node) → ยังไม่ได้ทดสอบบนบอร์ด
- **Method A (3 Concat nodes)** → cvimodel สร้างได้ แต่ nn.YOLO11 **โหลดไม่ได้** ถ้ามี 3 output nodes
- **Model ที่โหลดได้** มี 1 output node เท่านั้น (Concat_1 only) — อาจเกิดจาก model_transform ที่ระบุ output_names ไม่ครบ หรือ toolchain ตัด node ที่เหลือออก

**ต้องทดสอบเพิ่มเติม:**
- Method B (2 nodes: dfl/conv/Conv + Sigmoid) → ยังไม่มี model บนบอร์ดที่ใช้ method นี้
- ตรวจสอบว่า MaixPy nn.YOLO11 runtime version ปัจจุบันรองรับกี่ output nodes

---

## Traced Facts

| # | Fact | Source |
|---|------|--------|
| F1 | Frontend YOLO node default value `"yolo_v11s"` ไม่ตรงกับ dropdown values ทั้ง 3 ตัว | `yolo.js:17` |
| F2 | `yolo5s` มีใน frontend dropdown แต่ไม่มี training script ใน backend | `yolo.js:20`, ไม่มีใน workspace_server |
| F3 | Backend `/train` ไม่ใช้ `train_config` จาก POST body, อ่านจาก project.json ใน ZIP | `main.py:109-112` |
| F4 | main.py yolo11n conversion ใช้ output_names เพียง 1 node: `/model.23/Sigmoid_output_0` | `main.py:323` |
| F5 | Colab notebook (Cell 18, สำเร็จ) ใช้ 2 nodes: `/model.23/dfl/conv/Conv_output_0,/model.23/Sigmoid_output_0` | Notebook Cell 18 |
| F6 | Colab notebook (Cell 22) ทดลอง 3 nodes (Method A) กับ input 224x224 | Notebook Cell 22 |
| F7 | MaixPy doc: Method B (2 nodes) แนะนำสำหรับ MaixCAM (CV181x) | Sipeed Wiki |
| F8 | main.py MUD file สำหรับ yolo11n สร้าง `type = obb` | `main.py:383` |
| F9 | MaixPy doc MUD ตัวอย่างสำหรับ detection ไม่มี field `type` ใน [extra] (MaixCAM) | Sipeed Wiki |
| F10 | mai-plus generators_ai.js ใช้ `nn.YOLO11()` โหลดจาก `.mud` file | `generators_ai.js:158` |
| F11 | main.py mobilenet path มี `--chip cv181x` แต่ yolo11n path ไม่มี | `main.py:287` vs `main.py:354-365` |
| F12 | main.py mobilenet path มี `--quant_input` แต่ yolo11n path ไม่มี | `main.py:287` vs `main.py:354-365` |
| F13 | Colab notebook yolo11n ก็ไม่มี `--chip` และ `--quant_input` เช่นกัน (แล้วสำเร็จ) | Notebook Cell 18-20 |
| F14 | Voice classification บน mai-plus ใช้ `.bin/.param` (AWNN) ไม่ใช่ `.mud` | `generators_ai.js:58-61` |
| F15 | mai-plus deployment patch `.mud` file เปลี่ยน model path ก่อน upload | `websocket-shell.js` |
| F16 | slim_yolo_v2 ไม่มี cvimodel conversion path สำหรับ kidbright-mai-plus | `main.py:273-398` |
| F17 | kidbright-mai + yolo11n ตกไป NCNN path แต่ ncnn variables ไม่ถูก define ใน yolo11n branch | `main.py:235-255` → `main.py:396-398` |
| F18 | Training imgsz, ONNX export imgsz, model_transform input_shapes ตรงกันที่ [224,320] สำหรับ yolo11n | `train_*.py:284`, `main.py:264,329` |
| F19 | Colab Version A ใช้ `[[1,3,224,320]]` ตรงกับ code | Notebook Cell 18 |
| F20 | Colab Version B ใช้ `[[1,3,224,224]]` ไม่ตรงกับ training imgsz | Notebook Cell 22 |
| F21 | บอร์ดจริงเป็น MaixCAM (CV181x), device-tree compatible = `cvitek,cv181x` | SSH `/proc/device-tree/compatible` |
| F22 | nn.YOLO11 บนบอร์ดโหลด model ที่มี 1 output node ได้ แต่ 3 output nodes ไม่ได้ | SSH `nn.NN().outputs_info()` |
| F23 | Model ที่ทำงานได้ (9f4bbfdd) มี MUD `type=obb` แต่ inference detect ปกติ 14ms | SSH test |
| F24 | nn.YOLO11 และ nn.YOLOv8 มี API methods เหมือนกันทุกประการ | SSH `dir(nn.YOLO11)` vs `dir(nn.YOLOv8)` |
| F25 | MaixCAM lib version 1.24.0, CVI Runtime 1.4.0 | SSH board |
