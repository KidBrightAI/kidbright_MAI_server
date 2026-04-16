import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv2d, reorg_layer
from backbone import *
import numpy as np
import tools

class VoiceCNN(nn.Module):
    """2D CNN for MFCC spectrogram classification (NCNN compatible)"""
    def __init__(self, num_classes=2):
        super(VoiceCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, target=None):
        return self.net(x)


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