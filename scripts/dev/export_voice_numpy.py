#!/usr/bin/env python
"""Export a trained VoiceCNN (fp32) as a single .npz for numpy-only inference
on the V831 A7 CPU — bypass AWNN/spnntools entirely.

Motivation: V831 AWNN per-tensor int8 quantization collapses small-vocab voice
models regardless of preprocessing / calibration / arch. The A7 Cortex can
handle a DS-CNN in float32 via numpy in ~50-200 ms per clip (depending on
duration + channel width), which is fast enough for keyword-spotting and
guarantees parity with fp32 accuracy.

Usage:
    python export_voice_numpy.py --project-id voice_fs  # -> projects/voice_fs/output/model_cpu.npz

Environment (must match what was used at training):
    KBMAI_VOICE_INPUT_H / KBMAI_VOICE_INPUT_W / KBMAI_VOICE_EMB / KBMAI_VOICE_CHANS
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
from models.voice_cnn import VoiceCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    proj = os.path.join("projects", args.project_id)
    pth = os.path.join(proj, "output", "best_acc.pth")
    pjson = os.path.join(proj, "project.json")
    with open(pjson) as f:
        p = json.load(f)
    labels = sorted(l["label"] for l in p["labels"])
    nc = len(labels)

    os.environ.setdefault("KBMAI_VOICE_EMB", "128")
    net = VoiceCNN(num_classes=nc)
    net.load_state_dict(torch.load(pth, map_location="cpu"))
    net.eval()

    # Dump each layer's weights + bias to npz
    weights = {}
    # features: index 0=conv0, 3=conv3, 6=conv6, 10=conv(pw) if EMB layer
    # we hard-code indices matching the current VoiceCNN arch
    from collections import OrderedDict
    sd = net.state_dict()
    for k, v in sd.items():
        weights[k] = v.detach().cpu().numpy().astype(np.float32)
        print(f"  {k}: {weights[k].shape}")

    out_path = args.out or os.path.join(proj, "output", "model_cpu.npz")
    np.savez(out_path, labels=np.array(labels), **weights)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} B)")
    print(f"labels: {labels}")

if __name__ == "__main__":
    main()
