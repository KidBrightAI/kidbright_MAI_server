#!/usr/bin/env python
"""Regenerate MFCC PNGs from WAV with proper (min,max)->(0,255) scaling.

Board voice_mfcc.py scales with np.max() only which clips the negative
log-spectrum half of every frame to 0. Use this to rebuild the dataset
before retraining so training sees the full MFCC information.
"""
import os, sys, wave
from math import pi, floor
import numpy as np
from PIL import Image

RATE = 44100
FrameDuration = 0.040
FrameLen = int(FrameDuration * RATE)
FrameShift = int(FrameDuration * RATE / 2)
FFTLen = 2048
mfccCoefs = 13

def hamming2(n): return 0.54 - 0.46*np.cos(2*pi/n*np.arange(n))
WIN = hamming2(FrameLen)

def mel(nFilters, FFTLen, sampRate):
    halfFFTLen = int(floor(FFTLen/2))
    M = np.zeros((nFilters, halfFFTLen))
    lowFreq, highFreq = 20, 8000
    melLow = 1125*np.log(1+lowFreq/700.0)
    melHigh = 1125*np.log(1+highFreq/700.0)
    melStep = int(floor((melHigh - melLow)/nFilters))
    melL2H = np.arange(melLow, melHigh, melStep)
    HzL2H = 700*(np.exp(melL2H/1125)-1)
    HzL2HN = np.floor(FFTLen*HzL2H/sampRate)
    for filt in range(nFilters):
        x1, x2 = HzL2HN[filt], HzL2HN[filt+1]
        if x2 <= x1: continue
        y1 = 1/(x2-x1)
        M[filt, int(x1)] = 0.0
        for x in np.arange(x1+1, x2):
            M[filt, int(x)] = y1*(x-x1)
        if filt < nFilters - 1:
            x3 = HzL2HN[filt+2]
            if x3 <= x2: continue
            y2 = 1/(x2-x3)
            for x in np.arange(x2, x3+1):
                if int(x) < halfFFTLen: M[filt, int(x)] = y2*(x-x3)
    return M

def dctmtx(n):
    x, y = np.meshgrid(range(n), range(n))
    D = np.sqrt(2.0/n)*np.cos(pi*(2*x+1)*y/(2*n))
    D[0,:] = D[0,:]/np.sqrt(2)
    return D

M = mel(40, FFTLen, RATE)
D = dctmtx(40)[1:mfccCoefs+1]

def doMfcc(signal):
    lenSig = len(signal)
    nframes = int((lenSig - FrameLen) / FrameShift)
    mfcc2D = np.zeros((mfccCoefs, nframes))
    mfcc2DPow = np.zeros((40, nframes))
    minPow = 1e-50
    for fr in range(nframes - 1):
        start = fr*FrameShift
        cur = signal[start:start+FrameLen] * WIN
        cur[1:] -= cur[:-1] * 0.95
        fft = np.fft.fft(cur, FFTLen)
        fft[np.abs(fft) < minPow] = minPow
        mfcc2DPow[:, fr] = np.log(np.dot(M, np.abs(fft[:FFTLen//2])**2))
        mfcc2D[:, fr] = np.dot(D, mfcc2DPow[:, fr])
    return mfcc2D

def load_wav(path):
    with wave.open(path, 'rb') as w:
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float64)

def main():
    if len(sys.argv) < 2:
        print("usage: regen_mfcc.py <sound_dir> [out_dir]", file=sys.stderr)
        sys.exit(2)
    sound_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else sound_dir.replace("sound", "mfcc")
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for cls in sorted(os.listdir(sound_dir)):
        cls_src = os.path.join(sound_dir, cls)
        if not os.path.isdir(cls_src): continue
        cls_dst = os.path.join(out_dir, cls)
        os.makedirs(cls_dst, exist_ok=True)
        for fn in sorted(os.listdir(cls_src)):
            if not fn.endswith(".wav"): continue
            sig = load_wav(os.path.join(cls_src, fn))
            mfcc = doMfcc(sig)
            lo, hi = mfcc.min(), mfcc.max()
            if hi > lo:
                img = ((mfcc - lo) * (255.0 / (hi - lo))).astype(np.uint8)
            else:
                img = np.zeros_like(mfcc, dtype=np.uint8)
            base = os.path.splitext(fn)[0]
            Image.fromarray(img).save(os.path.join(cls_dst, f"{base}_mfcc.png"))
            n += 1
    print(f"wrote {n} MFCC PNGs to {out_dir}")

if __name__ == "__main__":
    main()
