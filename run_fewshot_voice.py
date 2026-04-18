#!/usr/bin/env python
"""Train a DSCNN voice-cnn model + export ONNX embedding + fp32 centroids.

The exported artifacts are used by the numpy CPU inference path on V831
(see voice_cpu_infer.py + voice_end_to_end.py). Centroids are saved as
json so a future few-shot enrollment UX can compare against them.

Auto-resolves repo root from __file__ so this works from anywhere
(Colab, WSL, any other host) without hard-coded paths.

Usage:
    python run_fewshot_voice.py --project-zip voice_mel.zip --id voice_fs
    # or for 3-second audio projects:
    KBMAI_VOICE_INPUT_W=147 python run_fewshot_voice.py --project-zip X.zip --id Y

Environment:
    KBMAI_VOICE_INPUT_H (default 40)   — mel bins
    KBMAI_VOICE_INPUT_W (default 47)   — frames (1s @ 50 fps; use 147 for 3s)
    KBMAI_VOICE_EMB     (default 128)  — embedding dim (pointwise conv output)
    KBMAI_VOICE_CHANS   (default 32,64,128) — conv channel widths; smaller is
                                      faster on-board (e.g. 8,16,32 for ~55ms/forward).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile

# Auto-locate the repo root (the directory this file lives in) so relative
# paths like projects/… and models/… work whether we're launched from WSL,
# Colab, or any other cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.voice_cnn import VoiceCNN, VoiceCNNEmbedding


def build_transforms(H, W):
    tf = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [128 / 255, 128 / 255, 128 / 255]),
    ])
    return tf


def split_dataset(data_dir, labels, train_split=80):
    import random as _rnd
    for label in labels:
        for sub in ("train", "valid"):
            os.makedirs(os.path.join(data_dir, sub, label), exist_ok=True)
        src = os.path.join(data_dir, label)
        if not os.path.isdir(src):
            continue
        imgs = [f for f in os.listdir(src) if f.endswith(".png")]
        _rnd.shuffle(imgs)
        n_train = int(len(imgs) * train_split / 100)
        for f in imgs[:n_train]:
            shutil.move(os.path.join(src, f), os.path.join(data_dir, "train", label, f))
        for f in imgs[n_train:]:
            shutil.move(os.path.join(src, f), os.path.join(data_dir, "valid", label, f))
        try:
            os.rmdir(src)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-zip", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    H = int(os.environ.get("KBMAI_VOICE_INPUT_H", "40"))
    W_env = os.environ.get("KBMAI_VOICE_INPUT_W")  # None -> auto-detect from WAVs
    W = int(W_env) if W_env else None

    proj_dir = os.path.join("projects", args.id)
    if os.path.exists(proj_dir):
        shutil.rmtree(proj_dir)
    os.makedirs(proj_dir)
    with zipfile.ZipFile(args.project_zip, "r") as z:
        z.extractall(proj_dir)
    with open(os.path.join(proj_dir, "project.json")) as f:
        proj = json.load(f)
    labels = sorted(l["label"] for l in proj["labels"])
    nc = len(labels)

    # Auto-regen mel-spec from wav so training features match what
    # voice_end_to_end.py computes on-board at inference time, and auto-detect
    # input W from the first wav so we don't need KBMAI_VOICE_INPUT_W.
    sound_dir = os.path.join(proj_dir, "dataset", "sound")
    mfcc_dir = os.path.join(proj_dir, "dataset", "mfcc")
    if os.path.isdir(sound_dir):
        import regen_melspec as _rms
        import wave
        # probe first wav for nframes
        for cls in sorted(os.listdir(sound_dir)):
            cdir = os.path.join(sound_dir, cls)
            if not os.path.isdir(cdir): continue
            wavs = [f for f in sorted(os.listdir(cdir)) if f.endswith(".wav")]
            if wavs:
                with wave.open(os.path.join(cdir, wavs[0]), "rb") as wf:
                    n_samples = wf.getnframes()
                FrameLen = int(0.040 * 44100); FrameShift = int(0.040 * 44100 / 2)
                detected_W = int((n_samples - FrameLen) / FrameShift)
                break
        else:
            detected_W = 47
        if W is None:
            W = detected_W
            print(f"[fewshot] auto-detected W={W} from first wav ({n_samples} samples)")
        # Regenerate mel-spec from wav (overwrites any MFCC PNGs from the IDE)
        if os.path.isdir(mfcc_dir):
            shutil.rmtree(mfcc_dir)
        print(f"[fewshot] regenerating mel-spec from {sound_dir}")
        _rms.main.__globals__.update(NFILTERS=H, FFTLen=2048, MEL_M=_rms.mel(H, 2048, 44100))
        # call regen main with argv injection
        _saved_argv = sys.argv
        sys.argv = ["regen_melspec.py", sound_dir, mfcc_dir]
        _rms.main()
        sys.argv = _saved_argv

    if W is None:
        W = 47
    # Export to child processes (VoiceCNN + train_voice pipeline read these)
    os.environ["KBMAI_VOICE_INPUT_H"] = str(H)
    os.environ["KBMAI_VOICE_INPUT_W"] = str(W)
    print(f"[fewshot] labels={labels}  H={H}  W={W}")

    data_dir = mfcc_dir
    os.makedirs(os.path.join(proj_dir, "output"), exist_ok=True)
    split_dataset(data_dir, labels, proj["trainConfig"].get("train_split", 80))

    tf = build_transforms(H, W)
    ds_train = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=tf)
    ds_val = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=tf)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VoiceCNN(num_classes=nc).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

    best_acc = 0.0
    for ep in range(args.epochs):
        net.train()
        train_loss = 0.0; n = 0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0); n += x.size(0)
        train_loss /= max(n, 1)

        net.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)
                correct += (net(x).argmax(1) == y).sum().item(); total += y.size(0)
        val_acc = 100.0 * correct / max(total, 1)
        if (ep + 1) % 20 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep+1:3d} loss={train_loss:.4f} val_acc={val_acc:.1f}")
        if ep + 1 > args.epochs // 2 and val_acc >= best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), os.path.join(proj_dir, "output", "best_acc.pth"))

    print(f"[fewshot] training done, best val_acc={best_acc:.1f}")

    # Compute per-class centroids on training set using fp32 embeddings.
    # These will be compared against board-side int8 embeddings. Cosine
    # similarity is robust to quantization noise as long as embedding DIRECTIONS
    # are preserved.
    os.makedirs(os.path.join(proj_dir, "output"), exist_ok=True)
    net.load_state_dict(torch.load(os.path.join(proj_dir, "output", "best_acc.pth"), map_location=device))
    net.eval()
    centroids = {lbl: [] for lbl in labels}
    counts = {lbl: 0 for lbl in labels}
    with torch.no_grad():
        for lbl_idx, lbl in enumerate(labels):
            cls_dir = os.path.join(data_dir, "train", lbl)
            vecs = []
            for fn in sorted(os.listdir(cls_dir)):
                img_path = os.path.join(cls_dir, fn)
                from PIL import Image as _PImg
                img = _PImg.open(img_path).convert("RGB")
                x = tf(img).unsqueeze(0).to(device)
                e = net.embed(x).cpu().numpy()[0]
                vecs.append(e)
                counts[lbl] += 1
            centroid = np.mean(np.stack(vecs), axis=0)
            centroids[lbl] = centroid.tolist()
    print(f"[fewshot] centroids: {counts}")

    with open(os.path.join(proj_dir, "output", "centroids.json"), "w") as f:
        json.dump({"labels": labels, "centroids": centroids, "dim": len(list(centroids.values())[0])}, f, indent=2)

    # Export ONNX — embedding only (no classifier head). This is the model we
    # deploy. main.py convert_model will still run the usual onnx->ncnn->int8
    # path, and the board sees a tensor output of shape (1, 128) instead of
    # (1, N).
    emb_net = VoiceCNNEmbedding(net).to("cpu")
    emb_net.eval()
    x = torch.randn(1, 3, H, W)
    onnx_out = os.path.join(proj_dir, "output", "model.onnx")
    torch.onnx.export(
        emb_net, x, onnx_out, export_params=True,
        input_names=["input0"], output_names=["output0"],
        opset_version=11,
    )
    print(f"[fewshot] ONNX (embedding) exported to {onnx_out}")
    print("[fewshot] next: run_convert.py --project-id {} and adb_deploy.py".format(args.id))

if __name__ == "__main__":
    main()
