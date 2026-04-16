import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv2d, reorg_layer
from backbone import *
import numpy as np
import tools

class Voice1DCNN(nn.Module):
    """1D CNN that treats MFCC as time-series: (B, 3, 13, 147) → reshape (B, 39, 147) → Conv1d"""
    def __init__(self, num_classes=2):
        super(Voice1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(39, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, target=None):
        B, C, H, W = x.shape
        x = x.reshape(B, C * H, W)  # (B, 3*13, 147) = (B, 39, 147)
        x = self.features(x)
        x = self.classifier(x)
        return x


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