"""
Simple raw sliding-window voice KWS.
No smoothing, no stable check, no VAD gate — just raw inference per frame.
Logs CSV-friendly output for analysis.
"""
from maix import nn, audio, image, display, app
import numpy as np
import time

# === Config ===
WINDOW_SEC = 1.0             # full 1 second window (matches training)
HOP_SEC = 0.10
CHUNK_MS = 50
SR = 16000

# Silence-based buffer reset (anti-residual)
SILENCE_RMS = 600            # below this = silent chunk
SILENCE_CHUNKS_TO_RESET = 6  # 6 × 50ms = 300ms of silence → clear buffer

# Detection threshold (show keyword only if confidence >= this)
DETECT_THRESHOLD = 0.90

N_FFT = 512
N_MELS = 80
HOP_LENGTH = 160
WIN_LENGTH = 400
DURATION_FRAMES = 98


def mel_filterbank(n_fft, n_mels, sr):
    fmax = sr / 2
    mel_low = 2595 * np.log10(1 + 20 / 700)
    mel_high = 2595 * np.log10(1 + fmax / 700)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        fl, fc, fr = bins[m-1], bins[m], bins[m+1]
        for k in range(fl, fc):
            fbank[m-1, k] = (k - fl) / (fc - fl + 1e-9)
        for k in range(fc, fr):
            fbank[m-1, k] = (fr - k) / (fr - fc + 1e-9)
    return fbank


MEL_BASIS = mel_filterbank(N_FFT, N_MELS, SR)
WINDOW = np.hanning(WIN_LENGTH).astype(np.float32)


def compute_mel(audio_16k):
    min_len = (DURATION_FRAMES - 1) * HOP_LENGTH + WIN_LENGTH
    if len(audio_16k) < min_len:
        audio_16k = np.pad(audio_16k, (0, min_len - len(audio_16k)))
    n_frames = min((len(audio_16k) - WIN_LENGTH) // HOP_LENGTH + 1, DURATION_FRAMES)
    mel_spec = np.zeros((n_frames, N_MELS), dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = audio_16k[start:start+WIN_LENGTH] * WINDOW
        spectrum = np.fft.rfft(frame, N_FFT)
        power = np.abs(spectrum) ** 2
        mel_spec[i] = np.dot(MEL_BASIS, power)
    mel_spec = np.log(mel_spec + 1e-9)
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-9) * 255
    return mel_spec.astype(np.uint8)


def mel_to_image(mel):
    rgb = np.stack([mel, mel, mel], axis=-1).astype(np.uint8)
    return image.from_bytes(80, 98, image.Format.FMT_RGB888, rgb.tobytes())


# === Load ===
classifier = nn.Classifier(model="/root/voice_kws.mud")
labels = classifier.labels
num_classes = len(labels)

recorder = audio.Recorder(block=False)
recorder.volume(100)
recorder.reset(True)
rec_sr = recorder.sample_rate()

disp = display.Display()
W, H = disp.width(), disp.height()

buf_size_48k = int(rec_sr * WINDOW_SEC)
buf = np.zeros(buf_size_48k, dtype=np.float32)
buf_filled = 0

# === CSV header ===
import sys
print("# Voice KWS raw log (CSV)", flush=True)
print("# Labels: " + ",".join(labels), flush=True)
print("t_ms,rms," + ",".join("p_" + l for l in labels) + ",argmax,conf,infer_ms", flush=True)

# === Display colors ===
COLOR_BG = image.Color.from_rgb(20, 20, 30)
COLOR_OK = image.Color.from_rgb(0, 255, 100)
COLOR_WARN = image.Color.from_rgb(200, 200, 50)
COLOR_SILENT = image.Color.from_rgb(120, 120, 120)
COLOR_WHITE = image.COLOR_WHITE

rms_level = 0.0
last_label = "---"
last_conf = 0.0
last_probs = np.zeros(num_classes)
last_infer_ms = 0
frame_idx = 0
t_start = time.time()
silence_counter = 0          # count consecutive silent chunks

# Log history for on-screen display (list of tuples)
LOG_MAX = 18
log_history = []  # each entry: (t_s, rms, probs, label, conf)

last_hop_time = time.time()
buffer_filled_once = False

while not app.need_exit():
    data = recorder.record(CHUNK_MS)
    if data and len(data) > 0:
        samples = np.frombuffer(bytes(data), dtype=np.int16).astype(np.float32)
        rms_level = float(np.sqrt(np.mean(samples ** 2)))

        # Silence detection + buffer reset
        if rms_level < SILENCE_RMS:
            silence_counter += 1
            if silence_counter >= SILENCE_CHUNKS_TO_RESET:
                # Clear buffer — no residual
                buf[:] = 0.0
        else:
            silence_counter = 0

        n = len(samples)
        if n >= buf_size_48k:
            buf = samples[-buf_size_48k:]
            buf_filled = buf_size_48k
        else:
            buf = np.roll(buf, -n)
            buf[-n:] = samples
            buf_filled = min(buf_filled + n, buf_size_48k)

        if buf_filled >= buf_size_48k:
            buffer_filled_once = True

    now = time.time()
    if buffer_filled_once and (now - last_hop_time) >= HOP_SEC:
        last_hop_time = now
        frame_idx += 1

        audio_norm = buf / (np.max(np.abs(buf)) + 1e-9)
        audio_16k = audio_norm[::3]

        t0 = time.time()
        mel = compute_mel(audio_16k)
        img_in = mel_to_image(mel)
        result = classifier.classify(img_in, num_classes)
        last_infer_ms = int((time.time() - t0) * 1000)

        # Build prob vector
        probs = np.zeros(num_classes)
        for idx, score in result:
            probs[idx] = float(score)

        argmax_idx = int(np.argmax(probs))
        raw_label = labels[argmax_idx]
        raw_conf = float(probs[argmax_idx])

        # Threshold filter: only show keyword if confidence >= DETECT_THRESHOLD
        if raw_label != "noise" and raw_conf < DETECT_THRESHOLD:
            # Below threshold → force noise display
            last_label = "noise"
            noise_idx = labels.index("noise") if "noise" in labels else 0
            last_conf = float(probs[noise_idx])
        else:
            last_label = raw_label
            last_conf = raw_conf
        last_probs = probs

        # CSV log line (console)
        t_ms = int((now - t_start) * 1000)
        probs_str = ",".join("%.3f" % p for p in probs)
        print("%d,%d,%s,%s,%.3f,%d" % (
            t_ms, int(rms_level), probs_str, last_label, last_conf, last_infer_ms), flush=True)

        # On-screen log history
        log_history.append((now - t_start, int(rms_level), probs.copy(), last_label, last_conf))
        if len(log_history) > LOG_MAX:
            log_history.pop(0)

    # Simple display (original UI + log at bottom)
    ui = image.Image(W, H, image.Format.FMT_RGB888)
    ui.draw_rect(0, 0, W, H, color=COLOR_BG, thickness=-1)
    ui.draw_string(10, 10, "RAW KWS (no filter)", color=COLOR_WHITE, scale=1.4)
    ui.draw_string(10, 34, "rms=" + str(int(rms_level)), color=COLOR_SILENT, scale=1.0)

    # Prob bars
    y0 = 70
    bar_h = 30
    for i, lbl in enumerate(labels):
        p = float(last_probs[i])
        y = y0 + i * (bar_h + 8)
        ui.draw_rect(100, y, W - 140, bar_h, color=image.Color.from_rgb(50, 50, 50), thickness=-1)
        bw = int(p * (W - 140))
        if bw > 0:
            bc = COLOR_OK if lbl == last_label else COLOR_SILENT
            if lbl == "noise":
                bc = COLOR_SILENT
            ui.draw_rect(100, y, bw, bar_h, color=bc, thickness=-1)
        ui.draw_string(10, y + 5, lbl, color=COLOR_WHITE, scale=1.1)
        ui.draw_string(W - 70, y + 5, "%.2f" % p, color=COLOR_WHITE, scale=1.0)

    # Big label
    main_y = y0 + num_classes * (bar_h + 8) + 20
    if last_label == "noise":
        main_color = COLOR_SILENT
    elif last_conf >= 0.7:
        main_color = COLOR_OK
    else:
        main_color = COLOR_WARN
    ui.draw_string(20, main_y, last_label.upper(), color=main_color, scale=2.5)
    ui.draw_string(20, main_y + 55, "conf " + ("%.2f" % last_conf), color=COLOR_WHITE, scale=1.1)

    # Log section at bottom (last 8 non-noise or all)
    log_y_start = main_y + 90
    ui.draw_line(5, log_y_start - 6, W - 5, log_y_start - 6, color=COLOR_SILENT, thickness=1)
    ui.draw_string(10, log_y_start - 2, "log:", color=COLOR_WARN, scale=0.9)
    row_h = 18
    available = (H - log_y_start - 20) // row_h
    shown = log_history[-available:]
    for i, entry in enumerate(shown):
        t_s, rms_v, probs_v, lbl, conf_v = entry
        y = log_y_start + 18 + i * row_h
        if lbl == "noise":
            rc = COLOR_SILENT
        elif conf_v >= 0.7:
            rc = COLOR_OK
        else:
            rc = COLOR_WARN
        line = "%5.1fs rms=%-5d %-9s %.2f" % (t_s, rms_v, lbl, conf_v)
        ui.draw_string(10, y, line, color=rc, scale=0.9)

    ui.draw_string(10, H - 18, "frame=" + str(frame_idx) + " infer=" + str(last_infer_ms) + "ms",
                   color=COLOR_SILENT, scale=0.8)
    disp.show(ui)

recorder.reset(False)
