#!/usr/bin/env python
"""Phase 3: feature-extractor + cosine nearest-centroid classifier.

The small Linear(128, N) classification head is what collapses under V831
AWNN per-tensor int8 quantization. This script trains a DSCNN voice model
end-to-end with the classifier (so features get discriminative gradients),
then exports only the 128-dim pre-classifier embedding as the deployed int8
model. Per-class centroids are computed on the board (or simulated on a
calibrated runtime) from the training MFCCs and shipped alongside the model.
On the board, classification is:

    x = model.forward(mfcc)          # int8 128-dim embedding
    s = cosine_similarity(x, centroids[N, 128])
    pred = argmax(s)

This bypasses the fragile small-FC quantization entirely — classification is
done in Python on the CPU side with float centroids.

Usage:
    python run_fewshot_voice.py --project-zip voice_mel.zip --id voice_fs
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile

HERE = "/home/comdet/kidbright_MAI_server"
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
    W = int(os.environ.get("KBMAI_VOICE_INPUT_W", "47"))

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
    print(f"[fewshot] labels={labels}  H={H}  W={W}")

    data_dir = os.path.join(proj_dir, "dataset", "mfcc")
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
