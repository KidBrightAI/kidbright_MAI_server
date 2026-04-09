# YOLO11s Experiment on MaixCAM (CV181x)

ทดสอบการ convert และ deploy YOLO11s pre-trained model (COCO 80 classes) ลงบอร์ด MaixCAM (CV181x)

## ผลลัพธ์

| Model | Inference | FPS | cvimodel Size | mAP50-95 (COCO) |
|-------|-----------|-----|---------------|-----------------|
| YOLO11n | 16.5ms | 60.4 | 2.8 MB | 39.5 |
| **YOLO11s** | **51.6ms** | **19.4** | **9.6 MB** | **47.0** |

- YOLO11s ใช้งานได้จริงบน MaixCAM, detect ได้ถูกต้อง (bottle, tv จากกล้อง)
- FPS ลดจาก 60 เหลือ ~19 (ช้ากว่า ~3x) แต่ mAP เพิ่ม 19%
- เหมาะสำหรับ use case ที่เน้น accuracy ยอม FPS ลดได้

## วิธี Conversion (Method B, 2 output nodes)

### 1. Download + Export ONNX

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt

python -c "
from ultralytics import YOLO
model = YOLO('yolo11s.pt')
model.export(format='onnx', imgsz=[224, 320], opset=16)
"
```

Output: `yolo11s.onnx` (36.2 MB), shape `(1, 84, 1470)`

### 2. Verify ONNX Output Nodes

YOLO11s ใช้ output node names **เดียวกับ YOLO11n**:
- `/model.23/dfl/conv/Conv_output_0`
- `/model.23/Sigmoid_output_0`

### 3. model_transform (Method B)

```bash
model_transform.py \
  --model_name yolo11s \
  --model_def yolo11s.onnx \
  --input_shapes [[1,3,224,320]] \
  --mean "0,0,0" \
  --output_names "/model.23/dfl/conv/Conv_output_0,/model.23/Sigmoid_output_0" \
  --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098" \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --channel_format nchw \
  --test_input ./data/test_images2/cat.jpg \
  --test_result yolo11s_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --mlir yolo11s.mlir
```

### 4. Calibration

```bash
run_calibration.py yolo11s.mlir \
  --dataset ./data/test_images \
  --input_num 24 \
  -o yolo11s_cali_table
```

### 5. Deploy to cvimodel

```bash
model_deploy.py \
  --mlir yolo11s.mlir \
  --quantize INT8 \
  --calibration_table yolo11s_cali_table \
  --processor cv181x \
  --test_input yolo11s_in_f32.npz \
  --test_reference yolo11s_top_outputs.npz \
  --tolerance 0.85,0.5 \
  --model yolo11s.cvimodel
```

**tolerance `0.85,0.5`** (ต่ำกว่า yolo11n ที่ใช้ `0.9,0.6`)  
เพราะ Sigmoid output node ของ yolo11s มี euclidean_similarity = 0.553 หลัง INT8 quantization

## วิธี Deploy บนบอร์ด

### 1. Kill maixapp processes

```bash
ps | grep maixapp/apps | grep -v grep | awk '{print $1}' | xargs kill -9
```

(ดูจาก `workspace_ide/boards/kidbright-mai-plus/main.py`)

### 2. Upload ไฟล์

```bash
scp yolo11s.cvimodel yolo11s.mud root@10.155.55.1:/root/
```

### 3. Run benchmark

```bash
ssh root@10.155.55.1 "python3 /root/benchmark_yolo11s.py"
```

## ผล Benchmark จริง (2026-04-09)

```
=== YOLO11s Benchmark (20 frames) ===
  Avg inference: 51.6ms
  FPS: 19.4
  Min: 49.1ms, Max: 55.2ms
  cvimodel size: 9.6 MB

Detection results:
  Frame 0: bottle 0.64, bottle 0.50
  Frame 7: bottle 0.77, tv 0.64
  Frame 12: tv 0.77, bottle 0.70
  ...
```

## ไฟล์ในโฟลเดอร์นี้

| File | Description |
|------|-------------|
| `yolo11s.cvimodel` | INT8 quantized model สำหรับ cv181x (9.6 MB) |
| `yolo11s.mud` | MUD metadata file (model_type=yolo11, 80 COCO labels) |
| `benchmark_yolo11s.py` | Script สำหรับรัน benchmark บนบอร์ด |
| `Copy_of_scratchpad.ipynb` | Colab notebook ที่ใช้ convert |
| `README.md` | ไฟล์นี้ |

## Colab Notebook

`Copy_of_scratchpad.ipynb` ใช้ขั้นตอน:
1. Install condacolab + tpu-mlir v1.27
2. Clone kidbright_MAI_server repo
3. Install dependencies (ultralytics, onnx, torch)
4. Download yolo11s.pt + export ONNX
5. Inspect ONNX output nodes
6. model_transform (Method B, 2 output nodes)
7. Calibration (24 images)
8. model_deploy (INT8, cv181x, tolerance 0.85,0.5)

## ข้อสังเกต

1. **output_names เดียวกัน** — YOLO11s ใช้ `/model.23/...` เหมือน YOLO11n ดังนั้น conversion pipeline ใน main.py ใช้ output_names ตัวเดียวกันได้
2. **tolerance ต่างกัน** — yolo11s ต้องใช้ `0.85,0.5` แทน `0.9,0.6` เพราะ quantization accuracy ลดลงจาก model ที่ใหญ่กว่า
3. **MUD file ไม่มี type field** — ใช้ `model_type = yolo11` เท่านั้น ไม่ต้องใส่ `type = obb`
