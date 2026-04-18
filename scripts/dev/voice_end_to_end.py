"""End-to-end voice classification with per-stage timing.

Record from mic -> mel-spectrogram -> preprocess -> numpy CNN forward.
Prints timestamps + durations for each stage so we can measure the real
user-perceived latency.
"""
import os, sys, time, wave
import numpy as np
from math import pi, floor
from PIL import Image

sys.path.insert(0, "/root/app")
import voice_cpu_infer as vci

# ============================== config ======================================
RATE = 44100
CHUNK = 1024
WIDTH = 2
CHANNELS = 1
RECORD_SEC = float(os.environ.get("KBMAI_VOICE_RECORD_SEC", "1.0"))

FrameDuration = 0.040
FrameLen = int(FrameDuration * RATE)
FrameShift = int(FrameDuration * RATE / 2)
FFTLen = 2048
NFILTERS = 40  # mel bins
WAV_PATH = "/root/app/voice_run.wav"
MFCC_PATH = "/root/app/mfcc_run.png"
NPZ_PATH = "/root/app/model_cpu.npz"

# ============================== DSP ========================================
def hamming2(n):
    return 0.54 - 0.46 * np.cos(2 * pi / n * np.arange(n))

WIN = hamming2(FrameLen)

def mel(nFilters, FFTLen, sampRate):
    halfFFTLen = int(floor(FFTLen / 2))
    M = np.zeros((nFilters, halfFFTLen))
    lowFreq, highFreq = 20, 8000
    melLow = 1125 * np.log(1 + lowFreq / 700.0)
    melHigh = 1125 * np.log(1 + highFreq / 700.0)
    melStep = int(floor((melHigh - melLow) / nFilters))
    melL2H = np.arange(melLow, melHigh, melStep)
    HzL2H = 700 * (np.exp(melL2H / 1125) - 1)
    HzL2HN = np.floor(FFTLen * HzL2H / sampRate)
    for filt in range(nFilters):
        x1, x2 = HzL2HN[filt], HzL2HN[filt + 1]
        if x2 <= x1: continue
        y1 = 1 / (x2 - x1)
        M[filt, int(x1)] = 0.0
        for x in np.arange(x1 + 1, x2):
            M[filt, int(x)] = y1 * (x - x1)
        if filt < nFilters - 1:
            x3 = HzL2HN[filt + 2]
            if x3 <= x2: continue
            y2 = 1 / (x2 - x3)
            for x in np.arange(x2, x3 + 1):
                if int(x) < halfFFTLen: M[filt, int(x)] = y2 * (x - x3)
    return M

_MEL_M_FULL = mel(NFILTERS, FFTLen, RATE)
# MEL filterbank is a triangular bank over [lowFreq, highFreq] Hz, so only
# bins inside that range contribute. Slice the mel matrix + the spectrum to
# that range — ~2x faster matmul on V831 without changing the output.
_nz = np.any(_MEL_M_FULL > 0, axis=0)
_MEL_FIRST = int(np.argmax(_nz))
_MEL_LAST = int(len(_nz) - np.argmax(_nz[::-1]))
MEL_M = _MEL_M_FULL[:, _MEL_FIRST:_MEL_LAST]
MEL_M_T = MEL_M.T.copy()  # contiguous for faster matmul

def doMelSpec(signal):
    """Vectorized log-mel-spectrogram.
    - batched np.fft.rfft along axis=1 (half the cost of fft)
    - truncate spectrum to the mel filterbank's non-zero bins only
    """
    lenSig = len(signal)
    nframes = int((lenSig - FrameLen) / FrameShift)
    if nframes <= 1:
        return np.zeros((NFILTERS, max(nframes, 1)))
    from numpy.lib.stride_tricks import as_strided
    s = signal.astype(np.float64, copy=False)
    strides = (s.strides[0] * FrameShift, s.strides[0])
    frames = as_strided(s, shape=(nframes, FrameLen), strides=strides).copy()
    frames *= WIN
    frames[:, 1:] -= frames[:, :-1] * 0.95
    spec = np.fft.rfft(frames, FFTLen, axis=1)         # (nframes, FFTLen/2+1)
    mag_sq = np.abs(spec[:, _MEL_FIRST:_MEL_LAST]) ** 2
    mag_sq[mag_sq < 1e-50] = 1e-50
    mel_power = mag_sq @ MEL_M_T                        # (nframes, NFILTERS)
    return np.log(mel_power).T                          # (NFILTERS, nframes)


def ts(label, t0):
    now = time.time()
    print(f"  [{time.strftime('%H:%M:%S', time.localtime(now))}.{int((now%1)*1000):03d}] "
          f"+{(now - t0)*1000:7.1f}ms  {label}")
    return now


def main():
    print(f"load model...")
    d = np.load(NPZ_PATH, allow_pickle=True)
    w = {k: d[k] for k in d.files if k != "labels"}
    labels = [str(x) for x in d["labels"]]
    print(f"labels: {labels}")

    # -- open audio stream --
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(WIDTH),
        channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK,
    )

    # warm up: drain one chunk so the first record starts clean
    stream.read(CHUNK, exception_on_overflow=False)

    print(f"\n=== speak now (recording {RECORD_SEC}s) ===\n")
    t_start = time.time()
    t_last = ts("START (mic opened, buffer warm)", t_start)

    # ---- stage 1: record (keep audio in memory, skip WAV file round-trip) ----
    n_chunks = int(RATE / CHUNK * RECORD_SEC)
    chunks = []
    for i in range(n_chunks):
        chunks.append(stream.read(CHUNK, exception_on_overflow=False))
    stream.stop_stream(); stream.close(); p.terminate()
    t_last = ts(f"RECORD done ({n_chunks} chunks = {n_chunks*CHUNK/RATE:.3f}s audio)", t_start)

    signal = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float64)
    t_last = ts(f"BUFFER decoded ({len(signal)} samples)", t_start)

    mel_spec = doMelSpec(signal)
    t_last = ts(f"MEL-SPEC computed (shape={mel_spec.shape})", t_start)

    # min-max -> 0..255 uint8, keep in memory (no PNG roundtrip)
    lo, hi = mel_spec.min(), mel_spec.max()
    if hi > lo:
        mel_img = ((mel_spec - lo) * (255.0 / (hi - lo))).astype(np.uint8)
    else:
        mel_img = np.zeros_like(mel_spec, dtype=np.uint8)
    t_last = ts(f"MEL -> uint8 array (min={lo:.2f} max={hi:.2f})", t_start)

    # ---- stage 3: preprocess (resize, to float, normalize) ----
    x = vci.preprocess(mel_img)
    t_last = ts(f"PREPROCESS done (input shape={x.shape})", t_start)

    # ---- stage 4: numpy forward ----
    y = vci.forward(x, w)
    t_last = ts(f"FORWARD done", t_start)

    pred = labels[int(y.argmax())]
    print(f"\n  prediction: {pred}   logits={y.round(2).tolist()}")
    print(f"\n  TOTAL latency: {(t_last - t_start)*1000:.1f}ms "
          f"(of which {RECORD_SEC*1000:.0f}ms is recording)")
    print(f"  PROCESSING after record ends: {(t_last - (t_start + RECORD_SEC))*1000:.1f}ms")


if __name__ == "__main__":
    main()
