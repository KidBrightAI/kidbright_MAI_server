"""Minimal Quantization-Aware Training helpers matching ncnn/AWNN INT8 scheme.

QAT here simulates the symmetric per-tensor INT8 quantization that spnntools
applies post-training, so the network learns weights that survive the
calibrate+quantize step with less confidence drop on the V831 board.

Scope: weights + input activations of every Conv2d get fake-quantized. Backward
uses a straight-through estimator. BatchNorm is intentionally NOT folded into
conv during QAT — spnntools does that during its optimize pass. This is an MVP;
if confidence gain is material we can add BN folding + per-output-channel weight
quant later.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FakeQuantSTE(torch.autograd.Function):
    """Symmetric per-tensor int8 fake quantize with straight-through gradient."""

    @staticmethod
    def forward(ctx, x, scale):
        q = torch.clamp(torch.round(x / scale), -127.0, 127.0) * scale
        return q

    @staticmethod
    def backward(ctx, g):
        return g, None


def _symmetric_scale(t: torch.Tensor) -> torch.Tensor:
    """Scale = max(|t|) / 127, clamped to avoid div-by-zero."""
    return (t.detach().abs().max() / 127.0).clamp(min=1e-8)


class QConv2d(nn.Conv2d):
    """Conv2d with symmetric per-tensor INT8 fake quant on weights + input act.

    Activation scale is tracked with an EMA of per-batch max(|x|). This keeps
    the observation stable without needing a warm-up calibration pass, which
    would complicate training_task. Weights use per-step max(|W|) (no running
    average — weights change slowly).
    """

    def __init__(self, *args, act_momentum: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("act_scale", torch.tensor(1e-4))
        self.register_buffer("_qat_steps", torch.tensor(0, dtype=torch.long))
        self.act_momentum = act_momentum

    @staticmethod
    def from_conv(conv: nn.Conv2d) -> "QConv2d":
        kernel = conv.kernel_size
        q = QConv2d(
            conv.in_channels, conv.out_channels, kernel,
            stride=conv.stride, padding=conv.padding,
            dilation=conv.dilation, groups=conv.groups,
            bias=(conv.bias is not None),
        )
        q.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            q.bias.data.copy_(conv.bias.data)
        return q

    def _update_act_scale(self, x: torch.Tensor) -> None:
        new_s = _symmetric_scale(x)
        if self._qat_steps.item() == 0:
            self.act_scale.copy_(new_s)
        else:
            self.act_scale.mul_(1 - self.act_momentum).add_(new_s * self.act_momentum)
        self._qat_steps.add_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                self._update_act_scale(x)
        act_s = self.act_scale.clamp(min=1e-8)
        w_s = _symmetric_scale(self.weight)
        x_q = _FakeQuantSTE.apply(x, act_s)
        w_q = _FakeQuantSTE.apply(self.weight, w_s)
        return F.conv2d(
            x_q, w_q, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


def prepare_qat_model(model: nn.Module) -> nn.Module:
    """Recursively replace every nn.Conv2d in model with QConv2d (in place)."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, QConv2d):
            setattr(model, name, QConv2d.from_conv(child))
        else:
            prepare_qat_model(child)
    return model


def apply_qat_if_enabled(model: nn.Module) -> nn.Module:
    """Check env KBMAI_USE_QAT=1 and wrap model for QAT. No-op otherwise."""
    if os.environ.get("KBMAI_USE_QAT") == "1":
        n_before = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        prepare_qat_model(model)
        n_q = sum(1 for m in model.modules() if isinstance(m, QConv2d))
        print(f"[QAT] enabled — wrapped {n_q}/{n_before} Conv2d layers")
    return model
