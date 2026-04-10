# Voice Keyword Detection POC — MaixCAM (CV181x)

Custom voice keyword detection บน MaixCAM ใช้ **MEL Fbank features + Small CNN classifier**
ทดสอบสำเร็จ real-time classification 2 คำ (`sawatdee`, `laakorn`) + `noise` class

## Final Results

| Metric | Value |
|--------|-------|
| Dataset | 90 WAVs (30/class × 3 classes) |
| Train/Val | 72 / 18 (stratified per class) |
| Best val_acc | **100%** (6/6 per class @ epoch 30) |
| cvimodel size | **148 KB** (INT8) |
| Per-frame inference | ~135-140ms |
| Frame rate | ~6-7 inferences/sec |
| Real-time usable | ✅ confident detections 0.90-1.00 |

## Pipeline

```
Audio 48KHz (MaixCAM mic)
  → audio.Recorder non-blocking
  → decimate ÷3 → 16KHz
  → STFT (25ms Hanning, 10ms hop)
  → 80 MEL filters + log + normalize [0-255]
  → 98×80 uint8 spectrogram
  → replicate to 3-channel RGB
  → image.from_bytes(80, 98, FMT_RGB888)
  → nn.Classifier (cvimodel INT8)
  → 3-class softmax
```

## Files

| File | Description |
|------|-------------|
| `test_voice_kbmai_plus.ipynb` | Colab notebook (full training pipeline) |
| `voice_kws.cvimodel` | INT8 classifier, 148 KB, cv181x |
| `voice_kws.mud` | MUD metadata (`model_type=classifier`) |
| `record_dataset.py` | On-board WAV recorder with display countdown |
| `live_raw.py` | **Final** real-time demo (raw, no filtering, threshold=0.90) |
| `live_voice_kws.py` | Earlier version with smoothing/stable/cooldown (slower, for reference) |

## Dataset

Located at `../voice_dataset/` (committed to repo):
```
voice_dataset/
├── sawatdee/sawatdee_00..29.wav  (30 WAVs, 1.5s each, 48KHz mono)
├── laakorn/laakorn_00..29.wav    (30 WAVs)
└── noise/noise_00..29.wav        (30 WAVs silence/ambient)
```

---

## Model Architecture

```python
class VoiceKwsCNN(nn.Module):
    # Input: 3 × 98 × 80 (RGB-replicated MEL spectrogram)
    features = Sequential(
        Conv2d(3→32, 3×3) → BN → ReLU
        Conv2d(32→32, 3×3) → BN → ReLU → MaxPool(2×2)    # 32×49×40
        Conv2d(32→64, 3×3) → BN → ReLU
        Conv2d(64→64, 3×3) → BN → ReLU → MaxPool(2×2)    # 64×24×20
        Conv2d(64→128, 3×3) → BN → ReLU
        AdaptiveAvgPool2d(1)                              # 128
    )
    classifier = Sequential(
        Dropout(0.3),
        Linear(128 → num_classes)
    )
```

**Params:** 140,451 (~548 KB FP32, **148 KB INT8 after cvimodel conversion**)

## MEL Fbank Parameters

| Param | Value |
|-------|-------|
| Sample rate | 16000 Hz (decimate from 48K ÷3) |
| FFT size | 512 |
| Window | 25ms (400 samples) Hanning |
| Hop | 10ms (160 samples) |
| MEL filters | 80 |
| Duration | 98 frames (~1 second) |
| Output | 98×80 uint8 |

---

## Usage

### 1. Record dataset (on board)

```bash
scp record_dataset.py root@10.155.55.1:/root/
ssh root@10.155.55.1 "ps | grep maixapp/apps | grep -v grep | awk '{print \$1}' | xargs kill -9; python3 /root/record_dataset.py"
```

Output: `/tmp/voice_dataset/{sawatdee,laakorn,noise}/*.wav`

### 2. Train on Colab

Open `test_voice_kbmai_plus.ipynb` in Colab, runtime = GPU (T4), run cells in order:
1. Install condacolab + tpu-mlir + torch
2. Clone repo (auto-pulls `experiment/voice_dataset`)
3. Compute MEL spectrograms (→ `/content/mel_dataset/{class}/*.png`)
4. Train CNN **with stratified split** (80/20 per class)
5. Export ONNX → cvimodel (tpu-mlir, cv181x, INT8)
6. Create MUD + download

### 3. Deploy on board

```bash
scp voice_kws.cvimodel voice_kws.mud live_raw.py root@10.155.55.1:/root/
ssh root@10.155.55.1 "ps | grep maixapp/apps | grep -v grep | awk '{print \$1}' | xargs kill -9; killall python3; python3 /root/live_raw.py"
```

Display shows (top → bottom):
- Title + realtime RMS
- Probability bars (3 classes, realtime)
- Current label (BIG text, color-coded)
- **Log section** with last N frames
- Frame counter + inference time

---

## 💡 Key Learnings

### 1. Stratified split is mandatory, not optional
**Big mistake fixed:** Initial version used `torch.utils.data.random_split(dataset, [72,18])` which
shuffles ALL 90 samples then takes 72/18. This gave **unbalanced class distribution in training
and validation sets** — laakorn ended up under-represented, causing model to confuse laakorn with
sawatdee in real-time testing (conf only 0.5-0.7, never reaching 0.9).

**Fix:** Split per class (24 train + 6 val from EACH class), then concatenate. Also added per-class
validation accuracy logging so you can SEE this problem during training.

```python
# ❌ WRONG
train_ds, val_ds = torch.utils.data.random_split(dataset, [72, 18])

# ✅ CORRECT
for label in labels:
    class_files = glob.glob(f"{MEL_DIR}/{label}/*.png")
    random.shuffle(class_files)
    split = int(0.8 * len(class_files))
    train_samples.extend([(f, i) for f in class_files[:split]])
    val_samples.extend([(f, i) for f in class_files[split:]])
```

After fix: laakorn reached conf **1.00** in real-time inference.

### 2. Dataset size: 10/class is NOT enough
- First attempt (10/class): model could only learn sawatdee, predicted almost everything as sawatdee
- 30/class: model learns all 3 classes reliably
- Likely need 50-100/class for production robustness

### 3. Per-frame image conversion: `image.from_bytes()` is 1000× faster than `draw_rect` loop
- **Slow (first try):** `img.draw_rect(x, y, 1, 1)` per pixel → ~1900ms per image
- **Fast:** `image.from_bytes(w, h, FMT_RGB888, np_array.tobytes())` → **~1.6ms per image**
- Lesson: always use bulk buffer APIs for pixel data, never per-pixel drawing in a loop.

### 4. Sliding window causes long residual
**Problem:** With 1s sliding window, after user stops speaking, the word "stays" in the buffer for
1+ seconds while being classified as the same keyword repeatedly.

**Attempted fix — reduce window to 0.7s:** Broke everything. Audio buffer became 0.7s + 0.3s
zero padding, MEL normalization stats shifted, model panicked and classified almost everything
as a single class. **Lesson: inference input distribution must match training input distribution
exactly.** Can't reduce window without retraining.

**Working fix — silence-based buffer reset:**
```python
if rms < SILENCE_RMS:
    silence_counter += 1
    if silence_counter >= 6:   # ~300ms of silence
        buf[:] = 0.0           # clear buffer → no residual
```
Residual after speech end: ~300ms (down from 1-2s).

### 5. Smoothing/stable/cooldown adds latency — not always worth it
Built elaborate post-processing (EMA smoothing, stable N-frame check, cooldown) but it added
~500-1000ms of perceived latency. For clear, confident predictions (conf > 0.9), **raw per-frame
output with a simple threshold is better**. Post-processing helps when model is noisy —
but if model is trained well, it isn't noisy enough to need it.

**Final approach:** raw classification + hard threshold (conf ≥ 0.90) + silence-reset.

### 6. Adaptive VAD: tricky to initialize correctly
Built adaptive VAD with EMA-tracked `noise_floor` × 3 multiplier. Tricky issues:
- First frames must not be corrupted by speech (need warm-up or low initial floor)
- Must cap `noise_floor` maximum to prevent drift into speech level
- Hysteresis (lower exit threshold) helps with borderline signals
- **In the end, a simple fixed RMS threshold worked just as well** for this use case

### 7. MaixCAM audio init order matters
`audio.Recorder(block=False)` must be initialized **BEFORE** `display.Display()`.
If display inits first, it holds the multimedia pipeline and PCM open fails with
"failed to open PCM" / "Resource busy".

### 8. Debug verbosely when metrics look suspicious
When real-time test showed "sawatdee every frame" with v1 model, dumping per-class val accuracy
during training would have caught the unbalanced split immediately. **val_acc=100% is meaningless
without per-class breakdown**, especially with tiny val sets.

### 9. MaixPy on-device resampling
48KHz → 16KHz: simple decimation `audio_16k = audio_48k[::3]`
No anti-aliasing filter applied — works fine for speech since we care about 0-8KHz band.
Adding a low-pass filter before decimation might help marginally but wasn't necessary.

### 10. `audio.Recorder.record(ms)` is non-blocking only after `reset(True)`
```python
r = audio.Recorder(block=False)
r.volume(100)
r.reset(True)          # MUST call before record()
data = r.record(50)    # returns ~50ms of PCM bytes
# ...
r.reset(False)         # stop stream
```

---

## Performance Summary (Final Model)

From 2-minute live test with threshold=0.90:

| Label | Count | % | Confidence range |
|-------|-------|---|------------------|
| noise | 580 | 85% | — |
| laakorn | 83 | 12% | 0.92-1.00 |
| sawatdee | 19 | 3% | 0.90-0.92 |

- ✅ Laakorn reaches **1.00 confidence** in extended blocks
- ✅ No false positives in silence
- ✅ Silence-reset buffer keeps residual < 300ms
- ⚠️ Occasional 1-frame transition glitches when switching keywords fast
- ⚠️ Sawatdee confidence lower than laakorn (0.90-0.92 vs 0.94-1.00) — could improve with more data

---

## Future Improvements

1. **More data per class** (50-100 instead of 30) — will improve generalization
2. **Data augmentation** in Colab training:
   - Time shift ±100ms (random padding)
   - Volume scaling 0.7-1.3×
   - Background noise injection (SNR 10-20 dB)
   - SpecAugment (random frequency/time masking)
3. **Different speakers** in dataset (currently single-speaker recording)
4. **Majority-vote post-processing** (2 of 3 consecutive frames must agree) to eliminate transition glitches
5. **Onset-based triggering** (start capture at speech onset, one classification per utterance) instead of sliding window
