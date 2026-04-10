from maix import nn, audio, image, display, app
import numpy as np
import time

# --- Config ---
SR = 16000
WINDOW_SEC = 1.0
HOP_SEC = 0.25
CHUNK_MS = 50
CONF_THRESHOLD = 0.6

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


classifier = nn.Classifier(model="/root/voice_kws.mud")
labels = classifier.labels
print("Labels:", labels)

recorder = audio.Recorder(block=False)
recorder.volume(100)
recorder.reset(True)
rec_sr = recorder.sample_rate()
print("Recorder:", rec_sr, "Hz")

disp = display.Display()
W, H = disp.width(), disp.height()
print("Display:", W, "x", H)

buf_size_48k = int(rec_sr * WINDOW_SEC)
buf = np.zeros(buf_size_48k, dtype=np.float32)
buf_filled = 0

last_label = "---"
last_conf = 0.0
history = []
rms_level = 0.0
last_infer_ms = 0

COLOR_BG = image.Color.from_rgb(20, 20, 30)
COLOR_OK = image.Color.from_rgb(0, 255, 100)
COLOR_WAIT = image.Color.from_rgb(200, 200, 50)
COLOR_SILENT = image.Color.from_rgb(120, 120, 120)
COLOR_WHITE = image.COLOR_WHITE


def draw_ui(img, label, conf, rms, hist, infer_ms):
    img.draw_rect(0, 0, W, H, color=COLOR_BG, thickness=-1)
    img.draw_string(10, 10, "Voice KWS Live", color=COLOR_WHITE, scale=1.5)
    img.draw_string(10, 35, "Say: sawatdee / laakorn", color=COLOR_SILENT, scale=1.0)

    # RMS bar
    bar_w = int(min(rms / 3000.0, 1.0) * (W - 40))
    img.draw_rect(20, 70, W - 40, 20, color=image.Color.from_rgb(60, 60, 60), thickness=2)
    if bar_w > 0:
        c = COLOR_OK if rms > 500 else COLOR_SILENT
        img.draw_rect(20, 70, bar_w, 20, color=c, thickness=-1)
    img.draw_string(20, 95, "vol: " + str(int(rms)), color=COLOR_WHITE, scale=1.0)

    # Result
    if conf >= CONF_THRESHOLD and label != "noise":
        rc = COLOR_OK
        rt = label.upper()
    elif conf >= CONF_THRESHOLD:
        rc = COLOR_SILENT
        rt = "--silent--"
    else:
        rc = COLOR_WAIT
        rt = "listening..."
    img.draw_string(20, 140, rt, color=rc, scale=3.0)
    img.draw_string(20, 200, "conf: " + ("%.2f" % conf), color=COLOR_WHITE, scale=1.2)

    # History
    img.draw_string(10, 240, "Recent:", color=COLOR_SILENT, scale=1.0)
    for i, h in enumerate(hist[-5:]):
        hl, hc = h
        y = 260 + i * 22
        cc = COLOR_OK if (hc >= CONF_THRESHOLD and hl != "noise") else COLOR_SILENT
        img.draw_string(20, y, hl + ": " + ("%.2f" % hc), color=cc, scale=1.0)

    img.draw_string(10, H - 25, "inference: " + str(infer_ms) + "ms", color=COLOR_SILENT, scale=1.0)


print("=== LIVE DETECTION START ===")
last_hop_time = time.time()
buffer_filled_once = False

while not app.need_exit():
    data = recorder.record(CHUNK_MS)
    if data and len(data) > 0:
        samples = np.frombuffer(bytes(data), dtype=np.int16).astype(np.float32)
        rms_level = float(np.sqrt(np.mean(samples ** 2)))

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

        audio_norm = buf / (np.max(np.abs(buf)) + 1e-9)
        audio_16k = audio_norm[::3]

        t0 = time.time()
        mel = compute_mel(audio_16k)
        img_in = mel_to_image(mel)
        result = classifier.classify(img_in, 3)
        last_infer_ms = int((time.time() - t0) * 1000)

        if result:
            idx, score = result[0]
            last_label = labels[idx]
            last_conf = float(score)
            history.append((last_label, last_conf))
            if len(history) > 20:
                history.pop(0)

            if last_conf >= CONF_THRESHOLD and last_label != "noise":
                ts = time.strftime("%H:%M:%S")
                print("  [" + ts + "] DETECTED: " + last_label + " (" + ("%.2f" % last_conf) + ")")

    ui_img = image.Image(W, H, image.Format.FMT_RGB888)
    draw_ui(ui_img, last_label, last_conf, rms_level, history, last_infer_ms)
    disp.show(ui_img)

recorder.reset(False)
print("Done!")
