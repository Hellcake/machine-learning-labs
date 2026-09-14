# mini_net_demo.py
import torch, torch.nn as nn
from winograd_module import WinogradConv2dF23

net = nn.Sequential(
    WinogradConv2dF23(3, 32),
    nn.ReLU(),
    nn.AvgPool2d(2),
    WinogradConv2dF23(32, 64),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(64, 10)  # псевдо-классификация на 10 классов (рандомные веса)
)

x = torch.randn(1,3,224,224)
y = net(x)
print(y.shape)  # torch.Size([1, 10])
