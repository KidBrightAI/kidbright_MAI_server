# Voice classification — CPU numpy path (V831)

V831 AWNN per-tensor int8 quantization consistently collapses small-vocab
voice models regardless of preprocessing, calibration, architecture, or
QAT. The working alternative is **skip the NPU** and run the fp32 network
on the A7 Cortex via numpy. Accuracy matches PyTorch exactly and end-to-end
latency is ~130 ms post-record for a 1-second clip / ~370 ms for 3 s.

Everything else about the board (camera, LCD, ADB, GPIO, plugins) still
goes through the AWNN path — only the *voice* classify block uses the
CPU path described here.

---

## Training (Colab or any Linux with PyTorch)

Input: a project zip exported from the IDE. No data re-collection needed —
the zip already contains `dataset/sound/*.wav` which is our source of truth.
The `dataset/mfcc/*.png` (13-coef MFCC generated on-board) is **ignored**;
we regenerate 40-bin log-mel from the WAVs for better quantization-free
training.

```
# in Colab:
!git clone https://github.com/KidBrightAI/kidbright_MAI_server
%cd kidbright_MAI_server
!pip install torch torchvision numpy pillow

# upload your IDE zip (e.g. voice_trained_again.zip) as voice.zip, then:
!python run_fewshot_voice.py --project-zip voice.zip --id myvoice
#  → auto-detects duration from the first WAV (1s or 3s or anything)
#  → regenerates dataset/mfcc/ as 40-bin log-mel
#  → trains VoiceCNN (val_acc + progress prints)
#  → saves projects/myvoice/output/best_acc.pth + ONNX + centroids.json

!python export_voice_numpy.py --project-id myvoice
#  → projects/myvoice/output/model_cpu.npz (~45-450 KB)
```

Environment variables tune the architecture (defaults in parens):

| env | default | effect |
|---|---|---|
| `KBMAI_VOICE_INPUT_H` | `40` | mel filter count |
| `KBMAI_VOICE_INPUT_W` | *auto* from wav | input frames (1s → 47, 3s → 147) |
| `KBMAI_VOICE_EMB` | `128` | embedding width |
| `KBMAI_VOICE_CHANS` | `32,64,128` | conv channel triple. Smaller = faster. `8,16,32` → ~55 ms/forward with no accuracy loss on small vocabs. |

---

## On-device inference

Deploy to the board via ADB:

```
adb push projects/myvoice/output/model_cpu.npz /root/app/model_cpu.npz
adb push voice_cpu_infer.py                   /root/app/
adb push voice_end_to_end.py                  /root/app/   # demo with timing
adb push voice_loop_test.py                   /root/app/   # continuous + LCD
```

Run a single classification with per-stage timing:
```
adb shell "KBMAI_VOICE_RECORD_SEC=3 python3 /root/app/voice_end_to_end.py"
```

Run a 10-round demo with LCD display:
```
adb shell "KBMAI_VOICE_RECORD_SEC=3 KBMAI_VOICE_ROUNDS=10 python3 /root/app/voice_loop_test.py"
```

Latency budget on V831 A7 @ 1 GHz, `KBMAI_VOICE_CHANS=8,16,32`:

| stage | 1-second audio | 3-second audio |
|---|---|---|
| record | 1000 ms | 3000 ms |
| buffer → np.int16 | 4 ms | 6 ms |
| mel-spec (vectorized rfft) | 62 ms | 180 ms |
| min-max → uint8 | 3 ms | 4 ms |
| preprocess (stack+normalize) | 5 ms | 5 ms |
| **numpy forward** | **55 ms** | **170 ms** |
| **total post-record** | **130 ms** | **370 ms** |

---

## IDE integration — needs Vue/server wiring (TODO)

Current IDE still emits the old AWNN int8 path via `generators_ai.js`
(`maix3_nn_voice_*` blocks → `nn.load(".bin/.param")`). To use this CPU
path in the IDE end-to-end you need to wire:

1. **Server** (`main.py`) — when `projectType == VOICE_CLASSIFICATION`
   and `board == kidbright-mai`, skip the onnx → ncnn → int8 pipeline and
   run `run_fewshot_voice.py` + `export_voice_numpy.py` instead, serving
   `model_cpu.npz` from `GET /projects/{id}/output/model_cpu.npz`.

2. **Vue** (`src/store/server.js` → `convertModel()`) — download
   `model_cpu.npz` (and optionally `centroids.json`) when
   `workspaceStore.projectType == VOICE_CLASSIFICATION`, hand off to
   `workspaceStore.importModelFromBlob` with `ext1='npz', ext2='json'`.

3. **Board upload** (`src/engine/protocols/web-adb.js` →
   `uploadModelIfNeeded()`) — push `{hash}.npz` to `/root/model/` and the
   two inference scripts (`voice_cpu_infer.py`, mel-spec DSP) to
   `/root/app/` alongside the user's `run.py`.

4. **Code generation** (`boards/kidbright-mai/blocks/generators_ai.js`) —
   replace the `maix3_nn_voice_load` template with one that imports
   `voice_cpu_infer` + records via `pyaudio` directly, computes mel-spec,
   runs forward, then argmax. No `maix.nn.load` for voice anymore.

Each of these is a small file change. The CPU path itself is already
validated on-device — what's left is just plumbing.

---

## Why this works when AWNN int8 doesn't

See the commit history for the full story:

- `75db8ad` — normalize range match (still collapsed)
- `6d99ff2` — calibration data + size arg bugs (still collapsed)
- `ea6eda4` — switched to the numpy CPU path (first real solution)
- `69eec20`, `90f57be`, `63e9318` — optimizations (2.37 s → 130 ms)

Root cause: spnntools v0.9.6 uses per-tensor int8 weight quantization, and
the `Linear(2304, 64)` head of the original VoiceCNN compresses 2304 wildly
varying activations through a single scale. It works in float but collapses
under int8. Switching to a DS-CNN with global max pool helps, but only the
CPU fp32 path actually eliminates the collapse entirely for the small-vocab
education use case.
