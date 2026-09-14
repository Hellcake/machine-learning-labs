# lab_showcase.py
import argparse, time, math, torch, torch.nn.functional as F

def has_mps():
    return torch.backends.mps.is_available()

def mps_sync():
    if has_mps():
        torch.mps.synchronize()

def matrices_f23(device, dtype):
    # Матрицы F(2x2,3x3) из примеров Lavin (wincnn)
    BT = torch.tensor([[1,0,-1,0],
                       [0,1, 1,0],
                       [0,-1,1,0],
                       [0,-1,0,1]], dtype=dtype, device=device)
    G  = torch.tensor([[1,0,0],
                       [0.5,0.5,0.5],
                       [0.5,-0.5,0.5],
                       [0,0,1]], dtype=dtype, device=device)
    AT = torch.tensor([[1,1,1,0],
                       [0,1,-1,1]], dtype=dtype, device=device)
    return AT, BT, G

def winograd_conv2d_f23(x, w, bias=None):
    device, dtype = x.device, x.dtype
    AT, BT, G = matrices_f23(device, dtype)
    N,Cin,H,W = x.shape
    Cout = w.shape[0]
    # паддинг и добивка до кратности 2
    extra_h = (2 - (H % 2)) % 2
    extra_w = (2 - (W % 2)) % 2
    x_pad = F.pad(x, (1, 1 + extra_w, 1, 1 + extra_h))
    tiles = x_pad.unfold(2,4,2).unfold(3,4,2)              # [N,Cin,nH,nW,4,4]
    nH, nW = tiles.shape[2], tiles.shape[3]
    T = nH*nW
    V = (BT @ tiles.contiguous().view(-1,4,4) @ BT.T).view(N, Cin, T, 4, 4)
    U = (G @ w.view(Cout*Cin,3,3) @ G.T).view(Cout, Cin, 4, 4)
    M = (V.unsqueeze(1) * U.unsqueeze(0).unsqueeze(3)).sum(dim=2)   # [N,Cout,T,4,4]
    Yt = (AT @ M.view(-1,4,4) @ AT.T).view(N, Cout, nH, nW, 2, 2)
    outH, outW = nH*2, nW*2
    Y = torch.empty((N, Cout, outH, outW), dtype=dtype, device=device)
    Y[:, :, ::2, ::2]   = Yt[:, :, :, :, 0, 0]
    Y[:, :, ::2, 1::2]  = Yt[:, :, :, :, 0, 1]
    Y[:, :, 1::2, ::2]  = Yt[:, :, :, :, 1, 0]
    Y[:, :, 1::2, 1::2] = Yt[:, :, :, :, 1, 1]
    Y = Y[..., :H, :W]
    if bias is not None:
        Y = Y + bias.view(1, -1, 1, 1)
    return Y

def print_matrices(args):
    AT, BT, G = matrices_f23(args.device, torch.float32)
    print("A^T (F(2,3)):\n", AT.cpu().numpy())
    print("B^T (F(2,3)):\n", BT.cpu().numpy())
    print("G (F(2,3)):\n", G.cpu().numpy())
    # арифметическая экономия
    m, r = 2, 3
    direct = m*m*r*r     # 36
    winograd = (m+r-1)*(m+r-1)  # 16
    print(f"\nУмножений на 2x2 выход: прямая свёртка = {direct}, Winograd = {winograd}  (выигрыш ×{direct/winograd:.2f})")

def check_equivalence(args):
    torch.manual_seed(0)
    dev = args.device
    dtype = torch.float32
    for H,W in [(64,64),(127,193),(223,301)]:
        x = torch.randn(1, 3, H, W, device=dev, dtype=dtype)
        w = torch.randn(args.cout, 3, 3, 3, device=dev, dtype=dtype)
        b = torch.randn(args.cout, device=dev, dtype=dtype)
        y_ref  = F.conv2d(x, w, b, padding=1)
        y_wino = winograd_conv2d_f23(x, w, b)
        err = (y_ref - y_wino).abs()
        print(f"[{H}x{W}] max|err|={err.max().item():.3e}, mean|err|={err.mean().item():.3e}")

def bench(args):
    torch.manual_seed(0)
    dev = args.device
    dtype = torch.float16 if args.fp16 else torch.float32
    shapes = [(1,3,720,1280),(1,3,480,640),(1,3,256,256)]
    for N,Cin,H,W in shapes:
        x = torch.randn(N, Cin, H, W, device=dev, dtype=dtype)
        w = torch.randn(args.cout, Cin, 3, 3, device=dev, dtype=dtype)
        b = torch.randn(args.cout, device=dev, dtype=dtype)
        # прогрев
        for _ in range(5):
            _ = F.conv2d(x, w, b, padding=1); mps_sync()
            _ = winograd_conv2d_f23(x, w, b); mps_sync()
        # замер
        def timeit(fn, iters=30):
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = fn(); mps_sync()
            t1 = time.perf_counter()
            return (t1-t0)/iters*1000
        t_conv  = timeit(lambda: F.conv2d(x, w, b, padding=1))
        t_wino  = timeit(lambda: winograd_conv2d_f23(x, w, b))
        # точность (на fp16 будет хуже)
        err = (F.conv2d(x, w, b, padding=1) - winograd_conv2d_f23(x, w, b)).abs()
        print(f"[{H}x{W}, Cout={args.cout}, dtype={str(dtype).split('.')[-1]}, dev={dev}]  "
              f"conv2d={t_conv:.2f} ms   wino={t_wino:.2f} ms   speedup ×{t_conv/max(t_wino,1e-6):.2f}   "
              f"max|err|={err.max().item():.3e}")

def bad_cases(args):
    dev = args.device
    # 1) Шахматка (жёсткие перепады)
    H,W = 256,256
    base = torch.zeros((H,W), device=dev)
    base[::2,::2] = 1.0; base[1::2,1::2] = 1.0
    img = base.unsqueeze(0).repeat(3,1,1).unsqueeze(0)  # [1,3,H,W]
    # 2) «жёсткие» веса
    w = torch.sign(torch.randn(args.cout, 3, 3, 3, device=dev)) * 5.0
    for dtype in [torch.float32, torch.float16]:
        x = img.to(dtype)
        y_ref  = F.conv2d(x, w.to(dtype), padding=1)
        y_wino = winograd_conv2d_f23(x, w.to(dtype))
        err = (y_ref - y_wino).abs()
        print(f"[checkerboard, dtype={str(dtype).split('.')[-1]}]  max|err|={err.max().item():.3e}, mean|err|={err.mean().item():.3e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Winograd (F(2,3)) showcase: матрицы, эквивалентность, бенчмарк, плохие кейсы")
    p.add_argument("--device", default=("mps" if has_mps() else "cpu"), choices=["cpu","mps"])
    p.add_argument("--cout", type=int, default=16)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("matrices", help="Печать A^T, B^T, G и арифметической экономии")
    sub.add_parser("check", help="Проверка эквивалентности с F.conv2d (ошибка)")
    bp = sub.add_parser("bench", help="Бенчмарк conv2d vs Winograd")
    bp.add_argument("--fp16", action="store_true", help="Измерить в float16 (на MPS ускоряет, но точность хуже)")

    sub.add_parser("badcases", help="Плохие входы/веса: рост ошибки")

    args = p.parse_args()
    if args.cmd == "matrices":
        print_matrices(args)
    elif args.cmd == "check":
        check_equivalence(args)
    elif args.cmd == "bench":
        bench(args)
    elif args.cmd == "badcases":
        bad_cases(args)
