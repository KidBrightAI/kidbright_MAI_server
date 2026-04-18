"""Continuous voice classification loop with V831 LCD display.

Records N times, shows on the built-in LCD:
    READY  n/total     -> prompt to speak soon
    3 2 1              -> countdown
    REC ...            -> mic is capturing
    ...                -> processing
    FORWARD / BACKWARD -> prediction (big) + margin
Expects model_cpu.npz at /root/app/model_cpu.npz for the matching duration.
"""
import sys, os, time
import numpy as np
from math import pi, floor
import pyaudio
from maix import image, display, camera

sys.path.insert(0, "/root/app")
import voice_cpu_infer as vci

# ---- config ----
RATE = 44100; CHUNK = 1024; WIDTH = 2; CHANNELS = 1
RECORD_SEC = float(os.environ.get("KBMAI_VOICE_RECORD_SEC", "3.0"))
FrameDuration = 0.040
FrameLen = int(FrameDuration * RATE)
FrameShift = int(FrameDuration * RATE / 2)
FFTLen = 2048; NFILTERS = 40
NPZ_PATH = "/root/app/model_cpu.npz"
ROUNDS = int(os.environ.get("KBMAI_VOICE_ROUNDS", "10"))

# ---- DSP ----
def hamming2(n): return 0.54 - 0.46 * np.cos(2*pi/n*np.arange(n))
WIN = hamming2(FrameLen)
def _mel_mat():
    h = FFTLen // 2
    M = np.zeros((NFILTERS, h))
    mL = 1125*np.log(1+20/700.0); mH = 1125*np.log(1+8000/700.0)
    step = int(floor((mH-mL)/NFILTERS))
    mL2H = np.arange(mL, mH, step)
    HzN = np.floor(FFTLen*(700*(np.exp(mL2H/1125)-1))/RATE)
    for f in range(NFILTERS):
        x1, x2 = HzN[f], HzN[f+1]
        if x2 <= x1: continue
        y1 = 1/(x2-x1); M[f, int(x1)] = 0.0
        for x in np.arange(x1+1, x2): M[f, int(x)] = y1*(x-x1)
        if f < NFILTERS - 1:
            x3 = HzN[f+2]
            if x3 <= x2: continue
            y2 = 1/(x2-x3)
            for x in np.arange(x2, x3+1):
                if int(x) < h: M[f, int(x)] = y2*(x-x3)
    return M
_FULL = _mel_mat()
_nz = np.any(_FULL > 0, axis=0)
_FIRST = int(np.argmax(_nz)); _LAST = int(len(_nz) - np.argmax(_nz[::-1]))
MEL_M_T = _FULL[:, _FIRST:_LAST].T.copy()

def do_mel_spec(signal):
    nframes = int((len(signal) - FrameLen) / FrameShift)
    if nframes <= 1: return np.zeros((NFILTERS, 1))
    from numpy.lib.stride_tricks import as_strided
    s = signal.astype(np.float64, copy=False)
    stride = (s.strides[0] * FrameShift, s.strides[0])
    fr = as_strided(s, shape=(nframes, FrameLen), strides=stride).copy()
    fr *= WIN
    fr[:, 1:] -= fr[:, :-1] * 0.95
    spec = np.fft.rfft(fr, FFTLen, axis=1)
    mag = np.abs(spec[:, _FIRST:_LAST]) ** 2
    mag[mag < 1e-50] = 1e-50
    return np.log(mag @ MEL_M_T).T


# ---- LCD (must go through camera.capture() — image.new() alone doesn't render) ----
camera.camera.config(size=(240, 240))
time.sleep(0.3)
_ = camera.capture()  # warm-up frame

def screen(bg, title, subtitle="", big="", big_color=(255,255,255), extras=()):
    """Render a simple LCD screen. `extras` is a list of (x, y, text, scale, color)
    for per-class prob lines under the big label."""
    img = camera.capture()
    img.draw_rectangle(0, 0, 240, 240, color=bg, thickness=-1)
    if title:
        img.draw_string(10, 10, title, scale=2, color=(255,255,255), thickness=2)
    if subtitle:
        img.draw_string(10, 45, subtitle, scale=1, color=(220,220,220), thickness=1)
    if big:
        # roughly centered; scale 3 is readable without filling the frame
        x = max(8, 120 - len(big) * 8)
        img.draw_string(x, 95, big, scale=3, color=big_color, thickness=2)
    for ex in extras:
        x, y, text, sc, col = ex
        img.draw_string(x, y, text, scale=sc, color=col, thickness=1)
    display.show(img)


def _softmax(v):
    vmax = max(v)
    exps = [np.exp(x - vmax) for x in v]
    s = sum(exps)
    return [e / s for e in exps]


# ---- load model ----
print(f"loading model from {NPZ_PATH}")
d = np.load(NPZ_PATH, allow_pickle=True)
w = {k: d[k] for k in d.files if k != "labels"}
labels = [str(x) for x in d["labels"]]
print(f"labels: {labels}   RECORD_SEC={RECORD_SEC}   ROUNDS={ROUNDS}")

screen((0,40,80), "Voice Test", f"labels: {', '.join(labels)}", big="READY?", big_color=(100,255,255))
print("\n*** GET READY — starting in 3s ***")
time.sleep(3)

# open audio stream once
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS, rate=RATE, input=True,
                frames_per_buffer=CHUNK)
# warm-up drain
stream.read(CHUNK, exception_on_overflow=False)

history = []
try:
    for rnd in range(1, ROUNDS + 1):
        # READY
        screen((0,60,0), f"Round {rnd}/{ROUNDS}", "get ready...", big="READY", big_color=(100,255,100))
        print(f"\n[round {rnd}/{ROUNDS}] READY...")
        time.sleep(1.5)

        # 3-2-1 countdown
        for c in (3, 2, 1):
            screen((80,80,0), f"Round {rnd}/{ROUNDS}", "start speaking at 0", big=str(c), big_color=(255,255,0))
            time.sleep(0.7)

        # drain pre-record audio
        for _ in range(3):
            stream.read(CHUNK, exception_on_overflow=False)

        # RECORD
        screen((150,0,0), f"Round {rnd}/{ROUNDS}", f"record {RECORD_SEC:.0f}s — SPEAK NOW", big="REC", big_color=(255,255,255))
        print(f"  [REC start]")
        t_rec0 = time.time()
        n_chunks = int(RATE / CHUNK * RECORD_SEC)
        chunks = []
        for i in range(n_chunks):
            chunks.append(stream.read(CHUNK, exception_on_overflow=False))
        t_rec1 = time.time()
        print(f"  [REC end] +{(t_rec1-t_rec0)*1000:.0f}ms")

        # PROCESS
        screen((0,60,120), f"Round {rnd}/{ROUNDS}", "mel-spec + CNN...", big="...", big_color=(100,200,255))
        signal = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float64)
        t_m0 = time.time()
        mel_spec = do_mel_spec(signal)
        lo, hi = mel_spec.min(), mel_spec.max()
        mel_img = ((mel_spec - lo) * (255.0/(hi-lo))).astype(np.uint8) if hi>lo else np.zeros_like(mel_spec, dtype=np.uint8)
        x = vci.preprocess(mel_img)
        t_m1 = time.time()
        y = vci.forward(x, w)
        t_f1 = time.time()

        # RESULT
        am = int(y.argmax())
        pred = labels[am]
        probs = _softmax(y.tolist())
        p_top = probs[am]
        margin = float(y[am] - sorted(y.tolist())[-2])
        good = p_top > 0.7
        col = (100,255,100) if good else (255,150,0)
        # per-class prob lines (up to 4 labels fit)
        extras = []
        for i, (lbl, pr) in enumerate(zip(labels, probs)):
            color = (100,255,100) if i == am else (200,200,200)
            extras.append((20, 155 + i * 22, f"{lbl:<10} {pr*100:5.1f}%", 1, color))
        screen(
            (0,0,0) if good else (60,30,0),
            f"Round {rnd}/{ROUNDS}",
            f"margin {margin:+.2f}",
            big=pred.upper(),
            big_color=col,
            extras=extras,
        )
        msg = (f"  [RESULT] pred={pred:10}  prob={p_top*100:.1f}%  margin={margin:+.2f}  "
               f"logits={y.round(2).tolist()}  "
               f"mel={int((t_m1-t_m0)*1000)}ms  fwd={int((t_f1-t_m1)*1000)}ms")
        print(msg)
        history.append((rnd, pred, p_top, margin))
        time.sleep(2.5)

finally:
    stream.stop_stream(); stream.close(); p.terminate()

# summary
print("\n=== SUMMARY ===")
for rnd, pred, prob, m in history:
    print(f"  round {rnd:2}: {pred:10}  prob={prob*100:5.1f}%  margin={m:+.2f}")
screen((0,80,0), "ALL DONE", f"{len(history)} rounds", big="DONE", big_color=(100,255,100))
