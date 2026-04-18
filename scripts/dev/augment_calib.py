"""Generate an augmented calibration set from a project's JPEGImages.

Spnntools calibrates INT8 ranges from these images; a tiny raw set (e.g. 40)
covers only a narrow pixel distribution, so post-quantize activations clip at
unseen inputs and confidence drops. Augmenting with brightness/hue/flip/blur/
crop expands the observed range so INT8 tables better match runtime inputs.

Usage:
  python scripts/dev/augment_calib.py \
      --src projects/od_bce_test_v2/dataset/JPEGImages \
      --dst projects/od_bce_test_v2/calib_aug \
      --multiplier 10
"""
from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np


def aug_brightness(img: np.ndarray) -> np.ndarray:
    delta = random.uniform(-35, 35)
    return np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def aug_contrast(img: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.7, 1.3)
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def aug_hue_saturation(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[..., 0] = (hsv[..., 0] + random.randint(-10, 10)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_blur(img: np.ndarray) -> np.ndarray:
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)


def aug_flip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def aug_crop_resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = random.uniform(0.75, 0.95)
    nh, nw = int(h * scale), int(w * scale)
    y = random.randint(0, h - nh)
    x = random.randint(0, w - nw)
    crop = img[y:y + nh, x:x + nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def aug_noise(img: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


AUGS = [
    aug_brightness, aug_contrast, aug_hue_saturation,
    aug_blur, aug_flip, aug_crop_resize, aug_noise,
]


def apply_random_chain(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    n = random.randint(1, 3)
    for fn in random.sample(AUGS, n):
        out = fn(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Source JPEGImages dir")
    ap.add_argument("--dst", required=True, help="Destination calib_aug dir (recreated)")
    ap.add_argument("--multiplier", type=int, default=10, help="Augmented copies per image")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if os.path.exists(args.dst):
        import shutil
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)

    src_files = sorted(f for f in os.listdir(args.src) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    total = 0
    for idx, fn in enumerate(src_files):
        img = cv2.imread(os.path.join(args.src, fn))
        if img is None:
            continue
        cv2.imwrite(os.path.join(args.dst, f"{idx:03d}_orig.jpg"), img)
        total += 1
        for k in range(args.multiplier - 1):
            aug = apply_random_chain(img)
            cv2.imwrite(os.path.join(args.dst, f"{idx:03d}_aug{k:02d}.jpg"), aug)
            total += 1

    print(f"[augment_calib] wrote {total} images to {args.dst}")
    print(f"[augment_calib] source {len(src_files)} images x ~{args.multiplier} = ~{len(src_files)*args.multiplier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
