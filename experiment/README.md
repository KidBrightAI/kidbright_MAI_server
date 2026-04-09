# YOLO11s Experiment on MaixCAM (CV181x)

ทดสอบ YOLO11s ทั้ง pre-trained (COCO) และ custom training (dog/cat) บนบอร์ด MaixCAM (CV181x)

---

## 1. Pre-trained YOLO11s (COCO 80 classes)

### Benchmark บนบอร์ดจริง

| Model | Inference | FPS | cvimodel Size | mAP50-95 (COCO) |
|-------|-----------|-----|---------------|-----------------|
| YOLO11n | 16.5ms | 60.4 | 2.8 MB | 39.5 |
| **YOLO11s** | **51.6ms** | **19.4** | **9.6 MB** | **47.0** |

- YOLO11s detect ได้ถูกต้อง (bottle, tv จากกล้อง)
- FPS ลดจาก 60 เหลือ ~19 (ช้ากว่า ~3x) แต่ mAP เพิ่ม 19%

### Notebook

`Copy_of_scratchpad.ipynb` — Export pre-trained yolo11s.pt → ONNX → cvimodel

---

## 2. Custom Training: Dog/Cat (2 classes)

### Training Results (50 epochs, Tesla T4, ~2 min)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| all | 0.978 | 0.929 | **0.988** | **0.807** |
| cat | 0.963 | 0.941 | 0.988 | 0.876 |
| dog | 0.992 | 0.917 | 0.989 | 0.738 |

### Board Benchmark

```
=== Custom YOLO11s Dog/Cat Benchmark (20 frames) ===
  Avg inference: 48.4ms
  FPS: 20.7
  Min: 44.8ms, Max: 58.7ms
  Labels: ['cat', 'dog']
```

Live detection ทดสอบสำเร็จ แสดงผลบนจอบอร์ดได้

### Notebook

`dogcat_yolo11.ipynb` — Full pipeline: dataset → train → ONNX → cvimodel

### Dataset

`dataset/` — 200 images (100 cat + 100 dog), VOC format
- `JPEGImages/*.jpg` — รูปภาพ
- `Annotations/*.xml` — VOC bounding box annotations
- `ImageSets/Main/train.txt` — รายชื่อ image IDs

---

## Conversion Pipeline (Method B, 2 output nodes)

ใช้ได้กับทั้ง pre-trained และ custom trained YOLO11s/YOLO11n

### Key Parameters

```bash
# 1. Export ONNX
model.export(format='onnx', imgsz=[224, 320], opset=16)

# 2. model_transform (Method B)
model_transform.py --model_name yolo11s \
  --model_def model.onnx \
  --input_shapes [[1,3,224,320]] \
  --mean "0,0,0" \
  --output_names "/model.23/dfl/conv/Conv_output_0,/model.23/Sigmoid_output_0" \
  --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098" \
  --keep_aspect_ratio --pixel_format rgb --channel_format nchw \
  --test_input cat.jpg --test_result top_outputs.npz \
  --tolerance 0.99,0.99 --mlir model.mlir

# 3. Calibration
run_calibration.py model.mlir --dataset images/ --input_num 24 -o cali_table

# 4. Deploy
model_deploy.py --mlir model.mlir --quantize INT8 \
  --calibration_table cali_table --processor cv181x \
  --test_input in_f32.npz --test_reference top_outputs.npz \
  --tolerance 0.85,0.5 --model model.cvimodel
```

### Deploy บนบอร์ด

```bash
# Kill maixapp processes
ps | grep maixapp/apps | grep -v grep | awk '{print $1}' | xargs kill -9

# Upload
scp model.cvimodel model.mud root@10.155.55.1:/root/

# Run
ssh root@10.155.55.1 "python3 benchmark_yolo11s.py"
```

---

## ข้อสังเกตสำคัญ

1. **output_names เดียวกัน** — ทั้ง YOLO11n และ YOLO11s (pre-trained + custom) ใช้ `/model.23/dfl/conv/Conv_output_0` + `/model.23/Sigmoid_output_0`
2. **tolerance ต่างกัน** — yolo11s ต้องใช้ `0.85,0.5` แทน `0.9,0.6`
3. **MUD file** — ใช้ `model_type = yolo11` ไม่ต้องใส่ `type = obb`
4. **Kill maixapp** — ต้อง kill ก่อนรัน inference (ดูจาก `workspace_ide/boards/kidbright-mai-plus/main.py`)
5. **Custom training** — output shape เปลี่ยนตามจำนวน class (เช่น `(1, 6, 1470)` สำหรับ 2 classes)
6. **Calibration** — ใช้ training images แทน test_images ให้ผลดีกว่าสำหรับ custom model

---

## ไฟล์ในโฟลเดอร์นี้

| File | Description |
|------|-------------|
| **Pre-trained (COCO)** | |
| `yolo11s.cvimodel` | INT8 model, 80 classes (9.6 MB) |
| `yolo11s.mud` | MUD file, 80 COCO labels |
| `Copy_of_scratchpad.ipynb` | Colab notebook: export + convert |
| **Custom (Dog/Cat)** | |
| `yolo11s_dogcat.cvimodel` | INT8 model, 2 classes (10.0 MB) |
| `yolo11s_dogcat.mud` | MUD file, labels = cat, dog |
| `dogcat_yolo11.ipynb` | Colab notebook: train + convert |
| `dataset/` | VOC dataset (200 images) |
| **Utilities** | |
| `benchmark_yolo11s.py` | Benchmark script สำหรับบอร์ด |
