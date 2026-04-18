"""Board-side numpy-only VoiceCNN inference.

Expects model_cpu.npz + a single MFCC PNG. Runs forward in fp32 and prints
the argmax label. No AWNN, no int8 — just numpy on the A7 CPU.
"""
import sys, os, time
import numpy as np
from PIL import Image


def im2col(x, kh, kw, stride=1):
    """x: (C, H, W) -> (C*kh*kw, H_out*W_out) using stride_tricks."""
    from numpy.lib.stride_tricks import as_strided
    C, H, W = x.shape
    H_out = (H - kh) // stride + 1
    W_out = (W - kw) // stride + 1
    sc, sh, sw = x.strides
    patches = as_strided(
        x,
        shape=(C, kh, kw, H_out, W_out),
        strides=(sc, sh, sw, sh * stride, sw * stride),
    )
    return patches.reshape(C * kh * kw, H_out * W_out).copy()


def conv2d(x, W, b, pad=1):
    """Conv2d: x (C_in, H, W), W (C_out, C_in, kh, kw), b (C_out,).
    Returns (C_out, H, W) with 'same' padding (pad=1) and stride=1.
    """
    C_in, H, Wd = x.shape
    C_out, _, kh, kw = W.shape
    if pad:
        xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    else:
        xp = x
    cols = im2col(xp, kh, kw)  # (C_in*kh*kw, H*W)
    Wf = W.reshape(C_out, -1)  # (C_out, C_in*kh*kw)
    y = Wf @ cols + b[:, None]
    return y.reshape(C_out, H, Wd)


def maxpool2d(x, kh, kw=None):
    kw = kw or kh
    C, H, W = x.shape
    H_out = H // kh
    W_out = W // kw
    x_crop = x[:, :H_out * kh, :W_out * kw]
    xr = x_crop.reshape(C, H_out, kh, W_out, kw)
    return xr.max(axis=(2, 4))


def maxpool_global(x, kh, kw):
    """Fixed-size global max pool with kernel (kh, kw). Produces (C, 1, 1)."""
    assert x.shape[1] == kh and x.shape[2] == kw
    return x.reshape(x.shape[0], -1).max(axis=1, keepdims=True)[:, :, None]


def relu(x): return np.maximum(x, 0)


def preprocess(src, h_out=None, w_out=None):
    """Load grayscale MFCC (path or numpy array), optionally resize, replicate
    to 3ch, and apply (x-127.5)/128 to match training's Normalize.

    If h_out/w_out are None and src is a numpy array, keep the array's shape
    (correct for the common case of feeding doMelSpec() output directly —
    the training pipeline also saw its PNG at that same shape). PNG inputs
    without explicit dims keep the PNG's existing (H, W); pass h_out/w_out
    explicitly if you want to resize, e.g. to match a shorter/longer training.
    """
    if isinstance(src, np.ndarray):
        if src.ndim == 3 and src.shape[0] == 3:
            return src.astype(np.float32)  # already (3,H,W) normalized
        if src.ndim != 2:
            raise ValueError(f"unsupported array shape {src.shape}")
        a = src.astype(np.float32)
        if h_out is not None and w_out is not None and a.shape != (h_out, w_out):
            img = Image.fromarray(a.astype(np.uint8)).resize((w_out, h_out), Image.BILINEAR)
            a = np.array(img, dtype=np.float32)
    else:
        img = Image.open(src).convert("L")
        if h_out is not None and w_out is not None:
            img = img.resize((w_out, h_out), Image.BILINEAR)
        a = np.array(img, dtype=np.float32)
    a = np.stack([a, a, a], axis=0)
    return (a - 127.5) / 128.0


def forward(x, w):
    """Full VoiceCNN forward. The global-max-pool kernel size is inferred from
    the current feature-map shape, so the same code works for input (40, 47)
    or (40, 147) depending on what the weights were trained for.
    """
    x = relu(conv2d(x, w["features.0.weight"], w["features.0.bias"], pad=1))
    x = maxpool2d(x, 2)
    x = relu(conv2d(x, w["features.3.weight"], w["features.3.bias"], pad=1))
    x = maxpool2d(x, 2)
    x = relu(conv2d(x, w["features.6.weight"], w["features.6.bias"], pad=1))
    x = maxpool2d(x, 2)
    kh, kw = x.shape[1], x.shape[2]
    x = maxpool_global(x, kh, kw)
    if "features.10.weight" in w:
        x = relu(conv2d(x, w["features.10.weight"], w["features.10.bias"], pad=0))
    x = x.flatten()
    x = w["classifier.1.weight"] @ x + w["classifier.1.bias"]
    return x


def main():
    if len(sys.argv) < 3:
        print("usage: voice_cpu_infer.py model.npz test_image.png [more.png ...]")
        sys.exit(2)
    npz = sys.argv[1]
    imgs = sys.argv[2:]
    d = np.load(npz, allow_pickle=True)
    w = {k: d[k] for k in d.files if k != "labels"}
    labels = [str(x) for x in d["labels"]]
    print(f"model: {npz}  labels={labels}")
    for p in imgs:
        x = preprocess(p)
        t0 = time.time()
        y = forward(x, w)
        dt = time.time() - t0
        pred = labels[int(y.argmax())]
        exp = os.path.basename(p).split("_", 1)[0]
        mk = "OK " if pred == exp else "BAD"
        print(f"  {mk} {os.path.basename(p)[:28]:28} exp={exp:5} pred={pred:5} logits={y.round(2).tolist()} dt={dt*1000:.0f}ms")


if __name__ == "__main__":
    main()
