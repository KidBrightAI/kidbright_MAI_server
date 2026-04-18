import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv2d, reorg_layer
from backbone import *
import numpy as np
import tools


class VoiceCNN(nn.Module):
    """IPU-safe 2D CNN for voice classification on V831.

    Uses only Conv2d + ReLU + MaxPool2d + Flatten + Linear — no BatchNorm,
    no AdaptiveAvgPool, no Dropout. Verified on MaixII V831 NPU.

    Architecture:
        Input (3, H, W) — log-mel spectrogram (H=40 mel bins, W=147 frames)
        or legacy MFCC (H=13).
        3× Conv+ReLU+MaxPool blocks extract features; a final MaxPool with
        kernel = full feature-map size acts as global max pooling to reduce
        the tensor that reaches the classifier FC to a tiny (C,) vector.
        That tiny FC is critical for INT8 on V831: per-tensor quantization
        of a wide (2304→64) FC collapses discrimination, while (128→N) keeps
        class separation intact after spnntools calibration.

    Input H is read from KBMAI_VOICE_INPUT_H env var (default 40 for log-mel).
    """
    def __init__(self, num_classes=2):
        super(VoiceCNN, self).__init__()
        import os as _os
        H = int(_os.environ.get("KBMAI_VOICE_INPUT_H", "40"))
        W = int(_os.environ.get("KBMAI_VOICE_INPUT_W", "47"))
        EMB = int(_os.environ.get("KBMAI_VOICE_EMB", "128"))
        # After 3 MaxPool2d(2) on (H, W): (H//8, W//8)
        h3 = max(1, H // 8)
        w3 = max(1, W // 8)
        # Final pointwise conv lets us pick embedding width independent of the
        # conv backbone — useful when AWNN chokes on large channel counts at
        # the output tensor.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.MaxPool2d(kernel_size=(h3, w3)),
            nn.Conv2d(128, EMB, 1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(EMB, num_classes),
        )

    def forward(self, x, target=None):
        x = self.features(x)
        return self.classifier(x)

    def embed(self, x):
        """Return the 128-dim pre-classifier embedding (after global max pool + flatten).
        Used by the few-shot centroid-based inference path which bypasses the
        fragile per-tensor int8 quantization of the small Linear(128, N) head.
        """
        x = self.features(x)
        return x.flatten(1)


class VoiceCNNEmbedding(nn.Module):
    """Wrapper that exposes the (C, 1, 1) feature tensor as the forward output.

    Used for ONNX export + spnntools int8 conversion when we want to deploy
    the feature extractor (not the classifier) and do centroid/cosine
    classification in Python on the board side.

    AWNN on V831 segfaults on 2D output tensors, so we keep the spatial
    dimensions (1, 1) and reshape on the board side.
    """
    def __init__(self, core):
        super().__init__()
        self.features = core.features

    def forward(self, x, target=None):
        return self.features(x)


class VoiceCnn(nn.Module):
    def __init__(self, device, code, input_size=None, num_classes=2, trainable=False):
        super(VoiceCnn, self).__init__()
        code = ",\n".join(code.split("\n"))
        self.device = device
        self.input_size = input_size #147x13 pixels
        print("input_size: ", input_size)
        self.num_classes = num_classes
        self.trainable = trainable
        fc = "torch.nn.LazyLinear(out_features = " + str(num_classes) + ", bias=True)"
        code = code + fc
        print("code: ", code)
        self.pred = eval("nn.Sequential("+ code + ")")
        print("self.pred: ", self.pred)

    def forward(self, x, target=None):
        # x: [B, 1, H, W]
        B, C, H, W = x.shape
        print("x.shape: ", x.shape)
        # pred: [B, 512]
        pred = self.pred(x)
        return pred
