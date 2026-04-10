# Voice Keyword Detection — POC

Custom voice keyword detection สำหรับ MaixCAM (CV181x) โดยใช้ **MEL Fbank features + Small CNN classifier**

## Pipeline

```
Audio 48KHz (mic) → decimate÷3 → 16KHz → STFT (25ms/10ms) → 80 MEL filters
  → log → normalize → 98×80 uint8 spectrogram
  → 3-channel RGB image → nn.Classifier (cvimodel INT8) → label
```

## Results

| Metric | Value |
|--------|-------|
| Keywords | sawatdee, laakorn, noise |
| Dataset | 10 samples/class (30 total) — **too small** |
| Training val_acc | 100% (overfit) |
| cvimodel size | **148 KB** (INT8) |
| MEL compute | ~130ms |
| Image conversion | ~1.6ms (using `image.from_bytes`) |
| NPU inference | ~3-8ms |
| **Total per detection** | **~140ms** |
| Sliding window hop | 250ms |
| **Real-time accuracy** | **Poor** (model overfitted, needs more data) |

## Files

| File | Description |
|------|-------------|
| `voice_kws.cvimodel` | INT8 classifier (98×80×3 input, 3 classes) |
| `voice_kws.mud` | MUD metadata (`model_type=classifier`) |
| `record_dataset.py` | Record WAV samples on board (10 per keyword + noise) |
| `live_voice_kws.py` | Continuous sliding-window detection with display UI |
| `README.md` | This file |

## Usage

### 1. Record dataset (on board)

```bash
scp record_dataset.py root@10.155.55.1:/root/
ssh root@10.155.55.1 "ps | grep maixapp/apps | grep -v grep | awk '{print \$1}' | xargs kill -9; python3 /root/record_dataset.py"
```

Output: `/tmp/voice_dataset/{sawatdee,laakorn,noise}/*.wav` (48KHz mono S16_LE, 1.5s each)

### 2. Train on Colab

(See experiment notebook — clones repo, loads `experiment/voice_dataset/*.wav`, computes MEL Fbank, trains CNN, exports cvimodel)

### 3. Live detection (on board)

```bash
scp voice_kws.cvimodel voice_kws.mud live_voice_kws.py root@10.155.55.1:/root/
ssh root@10.155.55.1 "ps | grep maixapp/apps | grep -v grep | awk '{print \$1}' | xargs kill -9; python3 /root/live_voice_kws.py"
```

UI shows:
- Title + keyword list
- Volume bar (RMS meter)
- Main detection result (large text, green if confident)
- Recent history (last 5)
- Inference time

## Known Issues & Next Steps

### 1. **Overfitting** (current blocker)
Model predicts almost always the same class regardless of input. Caused by:
- Only 10 samples per class
- No data augmentation
- Val set is only 6 samples (100% acc is meaningless)

### 2. **Fix plan**
- Record 30-50 samples per class
- Add augmentation on Colab:
  - Time shift ±100ms
  - Volume scale 0.7-1.3x
  - Background noise injection (SNR 10-20dB)
  - SpecAugment (random freq/time masking)
- Increase dropout 0.3 → 0.5
- Add weight decay + early stopping
- Verify train/inference preprocessing alignment

### 3. **Duration mismatch**
- Training WAV: 1.5s (72000 samples @ 48K)
- Inference crops to 1s (16000 samples @ 16K)
- Should either train on 1s or expand inference window to 1.5s

## Architecture Details

### CNN Model (`VoiceKwsCNN`)

```
Input: 3×98×80 (MEL spectrogram, RGB replicated)

Conv2d(3→32, 3×3) → BN → ReLU
Conv2d(32→32, 3×3) → BN → ReLU → MaxPool(2×2)     # 32×49×40
Conv2d(32→64, 3×3) → BN → ReLU
Conv2d(64→64, 3×3) → BN → ReLU → MaxPool(2×2)     # 64×24×20
Conv2d(64→128, 3×3) → BN → ReLU → GlobalAvgPool   # 128
Dropout(0.3) → Linear(128→3)                       # num_classes
```

- Params: 140,451 (~548 KB FP32, ~148 KB INT8 after quantization)

### MEL Fbank Parameters

| Param | Value |
|-------|-------|
| Sample rate | 16000 Hz |
| FFT size | 512 |
| Window | 25ms (400 samples) Hanning |
| Hop | 10ms (160 samples) |
| MEL filters | 80 |
| Duration | 98 frames (~1 second) |
| Output | 98×80 uint8 |

### Why `image.from_bytes`?

Drawing pixels via `draw_rect` loop: **1900ms** per image (too slow)
Using `image.from_bytes(w, h, format, bytes)`: **1.6ms** (1200× faster)

```python
rgb = np.stack([mel, mel, mel], axis=-1).astype(np.uint8)  # (98, 80, 3)
img = image.from_bytes(80, 98, image.Format.FMT_RGB888, rgb.tobytes())
```
