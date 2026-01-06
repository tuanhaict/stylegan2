import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRAConv2d(nn.Module):
    def __init__(self, conv, rank=8, alpha=1.0):
        super().__init__()
        self.conv = conv
        self.rank = rank
        self.scale = alpha / rank

        out_c, in_c, k, _ = conv.weight.shape
        self.A = nn.Parameter(torch.randn(rank, in_c * k * k) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_c, rank))

        for p in conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        W = self.conv.weight
        delta = (self.B @ self.A).view(W.shape)
        return F.conv2d(
            x,
            W + self.scale * delta,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding
        )
