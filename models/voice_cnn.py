import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv2d, reorg_layer
from backbone import *
import numpy as np
import tools

class VoiceCnn(nn.Module):
    def __init__(self, device, code, input_size=None, num_classes=2, trainable=False):
        super(VoiceCnn, self).__init__()
        code = ",\n".join(code.split("\n"))
        self.device = device
        self.input_size = input_size #147x13 pixels
        self.num_classes = num_classes
        self.trainable = trainable
        fc = "torch.nn.LazyLinear(out_features = " + str(num_classes) + ", bias=True)"
        code = code + fc
        self.pred = eval("nn.Sequential("+ code + ")")
        print("self.pred: ", self.pred)
        #self.pred = eval(code)
        #eval("        self.pred = nn.Sequential(" + code + ")")
    def forward(self, x, target=None):
        # x: [B, 1, H, W]
        B, C, H, W = x.shape
        # pred: [B, 512]
        pred = self.pred(x)
        return pred