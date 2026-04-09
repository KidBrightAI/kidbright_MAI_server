"""
YOLO11s Benchmark Script for MaixCAM (CV181x)
Run on board via SSH: python3 benchmark_yolo11s.py

Prerequisites:
  - Kill maixapp processes first:
    ps | grep maixapp/apps | grep -v grep | awk '{print $1}' | xargs kill -9
  - yolo11s.cvimodel and yolo11s.mud must be in the same directory
"""
from maix import nn, image, camera
import time
import sys
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11s.mud")
NUM_FRAMES = 20
CONF_TH = 0.5
IOU_TH = 0.45

if len(sys.argv) > 1:
    MODEL_PATH = sys.argv[1]

print(f"Loading model: {MODEL_PATH}")
det = nn.YOLO11(model=MODEL_PATH, dual_buff=False)
print(f"Model loaded OK!")
print(f"  input size: {det.input_size()}")
print(f"  labels ({len(det.labels)}): {det.labels[:5]}...")

cam = camera.Camera(det.input_width(), det.input_height(), det.input_format())
times = []
for i in range(NUM_FRAMES):
    img = cam.read()
    t0 = time.time()
    objs = det.detect(img, conf_th=CONF_TH, iou_th=IOU_TH)
    t1 = time.time()
    dt = (t1 - t0) * 1000
    times.append(dt)
    if objs:
        for obj in objs:
            print(f"  Frame {i}: {det.labels[obj.class_id]} {obj.score:.2f}")
cam.close()

avg = sum(times) / len(times)
fps = 1000.0 / avg if avg > 0 else 0
print()
print(f"=== Benchmark ({NUM_FRAMES} frames) ===")
print(f"  Avg inference: {avg:.1f}ms")
print(f"  FPS: {fps:.1f}")
print(f"  Min: {min(times):.1f}ms, Max: {max(times):.1f}ms")
