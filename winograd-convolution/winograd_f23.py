# winograd_f23.py
import torch, torch.nn.functional as F

def winograd_conv2d_f23(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor|None=None):
    """
    x: [N,Cin,H,W], w: [Cout,Cin,3,3], stride=1, pad=1 (эквивалент стандартной Conv2d)
    Возвращает: [N,Cout,H,W]
    """
    device, dtype = x.device, x.dtype
    # Матрицы из Lavin/wincnn (F(2,3))
    BT = torch.tensor([[1,0,-1,0],
                       [0,1, 1,0],
                       [0,-1,1,0],
                       [0,-1,0,1]], dtype=dtype, device=device)   # = B^T
    G  = torch.tensor([[1,0,0],
                       [0.5,0.5,0.5],
                       [0.5,-0.5,0.5],
                       [0,0,1]], dtype=dtype, device=device)
    AT = torch.tensor([[1,1,1,0],
                       [0,1,-1,1]], dtype=dtype, device=device)    # = A^T

    N,Cin,H,W = x.shape
    Cout = w.shape[0]

    extra_h = (2 - (H % 2)) % 2
    extra_w = (2 - (W % 2)) % 2
    x_pad = F.pad(x, (1, 1 + extra_w, 1, 1 + extra_h))  # (l,r,t,b)

    tiles = x_pad.unfold(2, 4, 2).unfold(3, 4, 2)       # [N,Cin,nH,nW,4,4]
    nH, nW = tiles.shape[2], tiles.shape[3]
    T = nH * nW
    tiles = tiles.contiguous().view(N, Cin, T, 4, 4)

    V = (BT @ tiles.view(-1,4,4) @ BT.T).view(N, Cin, T, 4, 4)

    U = (G @ w.view(Cout*Cin,3,3) @ G.T).view(Cout, Cin, 4, 4)

    M = (V.unsqueeze(1) * U.unsqueeze(0).unsqueeze(3)).sum(dim=2)  # [N,Cout,T,4,4]

    Yt = (AT @ M.view(-1,4,4) @ AT.T).view(N, Cout, nH, nW, 2, 2)

    outH, outW = nH*2, nW*2
    Y = torch.empty((N, Cout, outH, outW), dtype=dtype, device=device)
    Y[:, :, ::2, ::2]   = Yt[:, :, :, :, 0, 0]
    Y[:, :, ::2, 1::2]  = Yt[:, :, :, :, 0, 1]
    Y[:, :, 1::2, ::2]  = Yt[:, :, :, :, 1, 0]
    Y[:, :, 1::2, 1::2] = Yt[:, :, :, :, 1, 1]

    Y = Y[..., :H, :W]  # обрезаем добивки
    if bias is not None:
        Y = Y + bias.view(1, -1, 1, 1)
    return Y
