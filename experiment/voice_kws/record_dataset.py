"""
Voice Dataset Recorder for MaixCAM

Records N samples per keyword using the on-board mic.
Displays instructions/countdown on screen.

Usage (on board):
    python3 record_dataset.py

Output: /tmp/voice_dataset/<keyword>/<keyword>_NN.wav
"""
from maix import audio, display, image
import time, os, wave

# --- Config ---
KEYWORDS = ["sawatdee", "laakorn"]
KEYWORD_DISPLAY = ["sawatdee", "laakorn"]
SAMPLES_PER_KW = 10
RECORD_MS = 1500
COUNTDOWN = 3
OUTPUT_DIR = "/tmp/voice_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Recorder (init BEFORE display to avoid audio lock) ---
recorder = audio.Recorder(block=False)
recorder.volume(100)
sr = recorder.sample_rate()
ch = recorder.channel()
print(f"Recorder: {sr}Hz, {ch}ch")

# --- Display ---
disp = display.Display()
W, H = disp.width(), disp.height()


def show(text, color, bg=image.COLOR_BLACK):
    img = image.Image(W, H, image.Format.FMT_RGB888)
    img.draw_rect(0, 0, W, H, color=bg, thickness=-1)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        img.draw_string(20, H//3 + i*40, line, color=color, scale=2)
    disp.show(img)


def record_wav(path, duration_ms):
    """Record using non-blocking recorder, save as WAV"""
    recorder.reset(True)
    chunks = []
    elapsed = 0
    chunk_ms = 50
    while elapsed < duration_ms:
        data = recorder.record(chunk_ms)
        if data and len(data) > 0:
            chunks.append(bytes(data))
        time.sleep(chunk_ms / 1000.0)
        elapsed += chunk_ms
    recorder.reset(False)

    all_data = b"".join(chunks)
    wf = wave.open(path, "wb")
    wf.setnchannels(ch)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(all_data)
    wf.close()
    return len(all_data)


# --- Create output dirs ---
for kw in KEYWORDS + ["noise"]:
    os.makedirs(f"{OUTPUT_DIR}/{kw}", exist_ok=True)

show("Voice Recorder\nReady!", image.COLOR_WHITE, image.COLOR_BLUE)
time.sleep(2)

# --- Record keywords ---
for ki, kw in enumerate(KEYWORDS):
    for si in range(SAMPLES_PER_KW):
        # Countdown
        for c in range(COUNTDOWN, 0, -1):
            show(f"[{KEYWORD_DISPLAY[ki]}]\n{si+1}/{SAMPLES_PER_KW}  {c}...",
                 image.COLOR_WHITE, image.COLOR_BLUE)
            time.sleep(1)

        # Record - GREEN
        show(f"SPEAK NOW!\n{KEYWORD_DISPLAY[ki]}", image.COLOR_BLACK, image.COLOR_GREEN)
        wav_path = f"{OUTPUT_DIR}/{kw}/{kw}_{si:02d}.wav"
        nbytes = record_wav(wav_path, RECORD_MS)

        show(f"OK! ({nbytes}B)", image.COLOR_WHITE)
        time.sleep(0.5)

    show(f"{KEYWORD_DISPLAY[ki]} DONE!", image.COLOR_GREEN)
    time.sleep(1)

# --- Record noise/silence ---
show("Recording noise\n(stay quiet)", image.COLOR_YELLOW)
time.sleep(1)
for si in range(SAMPLES_PER_KW):
    show(f"noise {si+1}/{SAMPLES_PER_KW}", image.COLOR_YELLOW)
    wav_path = f"{OUTPUT_DIR}/noise/noise_{si:02d}.wav"
    record_wav(wav_path, RECORD_MS)
    time.sleep(0.3)

# --- Summary ---
total = 0
for kw in KEYWORDS + ["noise"]:
    d = f"{OUTPUT_DIR}/{kw}"
    count = len([f for f in os.listdir(d) if f.endswith(".wav")])
    total += count
    print(f"  {kw}: {count} files")
print(f"\nTotal: {total} WAV files in {OUTPUT_DIR}")

show(f"DONE!\n{total} files", image.COLOR_GREEN)
disp.close()
