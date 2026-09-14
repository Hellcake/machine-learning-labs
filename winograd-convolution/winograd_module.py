# winograd_module.py
import torch, torch.nn as nn
from winograd_f23 import winograd_conv2d_f23

class WinogradConv2dF23(nn.Module):
    def __init__(self, cin, cout, bias=True, device="cpu", dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin, 3, 3, device=device, dtype=dtype)*0.01)
        self.bias   = nn.Parameter(torch.zeros(cout, device=device, dtype=dtype)) if bias else None
    def forward(self, x):
        return winograd_conv2d_f23(x, self.weight, self.bias)
