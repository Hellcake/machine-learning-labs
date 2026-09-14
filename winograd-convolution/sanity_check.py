import torch, torch.nn.functional as F
from winograd_f23 import winograd_conv2d_f23

device = "mps" if torch.backends.mps.is_available() else "cpu"
N, Cin, Cout, H, W = 1, 3, 8, 223, 301
x = torch.randn(N, Cin, H, W, device=device)
w = torch.randn(Cout, Cin, 3, 3, device=device)
b = torch.randn(Cout, device=device)

y_ref = F.conv2d(x, w, b, padding=1)           # baseline
y_wino = winograd_conv2d_f23(x, w, b)          # Winograd

err = (y_ref - y_wino).abs()
print("max abs err:", err.max().item(), "  mean abs err:", err.mean().item())
