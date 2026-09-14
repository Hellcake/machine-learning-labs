# demo_realtime_v2.py
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from winograd_f23 import winograd_conv2d_f23

def has_mps(): return torch.backends.mps.is_available()
def mps_sync():
    if has_mps(): torch.mps.synchronize()

def to_bgr(im):
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im

def resize_to_h(im, h):
    im = to_bgr(im)
    scale = h / im.shape[0]
    return cv2.resize(im, (int(im.shape[1]*scale), h))

def make_grid(imgs, cols=2, h=360, pad=6, bg=(20,20,20)):
    tiles = [resize_to_h(im, h) for im in imgs if im is not None]
    if not tiles: return None
    cell_w = max(t.shape[1] for t in tiles)
    cells = []
    for t in tiles:
        if t.shape[1] < cell_w:
            pad_r = np.full((h, cell_w - t.shape[1], 3), bg, dtype=t.dtype)
            t = np.hstack([t, pad_r])
        cells.append(t)
    while len(cells) % cols != 0:
        cells.append(np.full((h, cell_w, 3), bg, dtype=cells[0].dtype))
    rows = [np.hstack(cells[i:i+cols]) for i in range(0, len(cells), cols)]
    if len(rows) == 1:
        return rows[0]
    sep = np.full((pad, rows[0].shape[1], 3), bg, dtype=rows[0].dtype)
    view = rows[0]
    for r in rows[1:]:
        view = np.vstack([view, sep, r])
    return view

def make_edge_weights(cout=16, cin=3, device="cpu", dtype=torch.float32):
    k = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
    w = np.stack([np.stack([k]*cin, 0)]*cout, 0)
    return torch.tensor(w, device=device, dtype=dtype)

def make_sobel_weights(cout=16, cin=3, device="cpu", dtype=torch.float32, axis="x"):
    sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
    k = sx if axis.lower()=="x" else sy
    w = np.stack([np.stack([k]*cin, 0)]*cout, 0)  # [Cout,Cin,3,3]
    return torch.tensor(w, device=device, dtype=dtype)

def build_weights(mode: str, cout: int, cin: int, device="cpu", dtype=torch.float32):
    mode = mode.lower()
    if mode == "laplace":
        return make_edge_weights(cout, cin, device, dtype)
    if mode == "sobelx":
        return make_sobel_weights(cout, cin, device, dtype, axis="x")
    if mode == "sobely":
        return make_sobel_weights(cout, cin, device, dtype, axis="y")
    # "rand" или неизвестное — случайные
    return torch.randn(cout, cin, 3, 3, device=device, dtype=dtype) * 0.01

def apply_fx(frame_bgr: np.ndarray, fx: str) -> np.ndarray:
    fx = (fx or "none").lower()
    out = frame_bgr
    if fx == "blur":
        out = cv2.GaussianBlur(out, (5,5), 0)
    elif fx == "sharpen":
        out = cv2.addWeighted(out, 1.5, cv2.GaussianBlur(out, (0,0), 1.0), -0.5, 0)
    elif fx == "noise":
        noise = np.random.normal(0, 10, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif fx == "bright+":
        out = cv2.convertScaleAbs(out, alpha=1.0, beta=+40)
    elif fx == "bright-":
        out = cv2.convertScaleAbs(out, alpha=1.0, beta=-40)
    elif fx == "contrast":
        out = cv2.convertScaleAbs(out, alpha=1.3, beta=0)
    elif fx == "down2x":
        h, w = out.shape[:2]
        out = cv2.resize(out, (w//2, h//2), interpolation=cv2.INTER_AREA)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def gray_u8(t: torch.Tensor):  # t in [0..1]
    if t.dim() == 3: t = t[0]
    t = t.clamp(0,1).detach().cpu().numpy()
    return (t*255.0).astype(np.uint8)

def norm_minmax(t: torch.Tensor):
    mn = t.amin(dim=(-2,-1), keepdim=True)
    mx = t.amax(dim=(-2,-1), keepdim=True)
    return (t - mn) / (mx - mn + 1e-8)

def colorize(z01: torch.Tensor, cmap=cv2.COLORMAP_TURBO):
    z8 = gray_u8(z01)
    return cv2.applyColorMap(z8, cmap)

def hstack_same_h(imgs, h=360):
    rs = []
    for im in imgs:
        if im is None: continue
        if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        scale = h / im.shape[0]
        rs.append(cv2.resize(im, (int(im.shape[1]*scale), h)))
    return np.hstack(rs) if rs else None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Камера не открывается. ")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    dev   = "mps" if has_mps() else "cpu"
    dtype = torch.float32
    cout, cin = 16, 3
    view_mode = "mean"  
    ch = 0  
    print_patch = False  
    save_npys   = False            

    filter_mode = "rand"   # "rand" | "laplace" | "sobelx" | "sobely"
    fx_mode     = "none"   # "none" | "blur" | "sharpen" | "noise" | "bright+" | "bright-" | "contrast" | "down2x"

    weights = build_weights(filter_mode, cout, cin, device=dev, dtype=dtype)
    bias    = torch.zeros(cout, device=dev, dtype=dtype)

    show_compare = False
    use_abs      = True
    norm_mode    = "minmax"
    fps = 0.0
    layout = "row"

    cv2.namedWindow("Winograd realtime", cv2.WINDOW_NORMAL)

    while True:
        ok, frame_bgr = cap.read()
        if not ok: break

        frame_bgr_fx = apply_fx(frame_bgr, fx_mode)

        rgb = cv2.cvtColor(frame_bgr_fx, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
        x = x.to(dtype); w = weights.to(dev, dtype); b = bias.to(dev, dtype)

        t0 = time.perf_counter()
        y_wino = winograd_conv2d_f23(x, w, b)                  # [1,cout,H,W]
        mps_sync()
        t1 = time.perf_counter()

        if view_mode == "mean":
            feat = y_wino.mean(1)
        else:
            ch = max(0, min(ch, cout-1))
            feat = y_wino[:, ch:ch+1, ...].squeeze(1)
        if use_abs: feat = feat.abs()
        if norm_mode == "minmax": vis_w = colorize(norm_minmax(feat))
        else:                     vis_w = cv2.cvtColor(gray_u8(feat), cv2.COLOR_GRAY2BGR)

        row1 = hstack_same_h([frame_bgr_fx, vis_w], h=360)
        # численные метрики для HUD (заполним, если compare=ON)
        max_err = mean_err = rmse = rel_mean = rel_max = cos_sim = None

        row2 = None
        if show_compare:
            y_ref = F.conv2d(x, w, b, padding=1)
            mps_sync()

            # тот же режим просмотра, что и для Winograd (mean или конкретный канал)
            if view_mode == "mean":
                feat_ref = y_ref.mean(1)
            else:
                ch = max(0, min(ch, cout-1))
                feat_ref = y_ref[:, ch:ch+1, ...].squeeze(1)

            if use_abs:
                feat_ref = feat_ref.abs()

            vis_ref = colorize(norm_minmax(feat_ref)) if norm_mode == "minmax" \
                    else cv2.cvtColor(gray_u8(feat_ref), cv2.COLOR_GRAY2BGR)

            # --- численная разность и метрики ---
            diff    = y_ref - y_wino                # [1,Cout,H,W]
            absdiff = diff.abs()                    # |разность|
            err     = absdiff.mean(1).squeeze(0)    # [H,W] для хитмапы (среднее по каналам)

            max_err  = float(absdiff.max())
            mean_err = float(absdiff.mean())
            rmse     = float((diff.pow(2).mean()).sqrt())

            rel      = absdiff / (y_ref.abs() + 1e-8)
            rel_mean = float(rel.mean())
            rel_max  = float(rel.max())

            num = float((y_ref.flatten() * y_wino.flatten()).sum())
            den = float(y_ref.flatten().norm() * y_wino.flatten().norm() + 1e-12)
            cos_sim = num / den

            # печать 8x8 фрагмента матрицы разности для выбранного канала (по запросу)
            if print_patch:
                Hc, Wc = diff.shape[-2:]
                h0, w0 = Hc//2 - 4, Wc//2 - 4
                patch = diff[0, ch, h0:h0+8, w0:w0+8].detach().cpu().numpy()
                np.set_printoptions(precision=3, suppress=True, linewidth=120)
                print(f"\nDiff patch (ch={ch}, center 8x8):\n{patch}\n")
                print_patch = False

            # сборка рядов
            row1 = hstack_same_h([frame_bgr_fx, vis_w, vis_ref], h=360)
            errmap = colorize(norm_minmax(err.unsqueeze(0)))  # [H,W,3]
            row2   = cv2.resize(errmap, (row1.shape[1], 180))

        else:
            row2 = None

        view = row1 if row2 is None else np.vstack([row1, row2])
        tiles = [frame_bgr_fx, vis_w]
        if show_compare:
            tiles.extend([vis_ref, errmap])

        if layout == "row":
            view = hstack_same_h(tiles, h=360)  
        else:
            view = make_grid(tiles, cols=2, h=360)

        # HUD
        fps = 0.9*fps + 0.1*(1.0/max(time.perf_counter()-t0,1e-6))
        hud = (f"Device={dev} | dtype={str(dtype).split('.')[-1]} | Cout={cout} | "
               f"filter={filter_mode} | fx={fx_mode} | "
               f"compare={'ON' if show_compare else 'OFF'} | norm={norm_mode} | abs={'ON' if use_abs else 'OFF'} | "
               f"{(t1-t0)*1000:.2f} ms | ~{fps:4.1f} FPS | view={view_mode}{'' if view_mode=='mean' else f'[{ch}]'} | layout={layout}" )
        if max_err is not None:  # compare=ON
            hud += (f" | max|err|={max_err:.2e} mean|err|={mean_err:.2e} "
                    f"rmse={rmse:.2e} rel={rel_mean:.2e}/{rel_max:.2e} cos={cos_sim:.5f}")
        cv2.putText(view, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow("Winograd realtime", view)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break
        elif k == ord('c'):
            show_compare = not show_compare

        # ===== выбор фильтра ядра =====
        elif k == ord('w'):       # случайные веса
            filter_mode = "rand"
            weights = build_weights(filter_mode, cout, cin, device=dev, dtype=dtype)
        elif k == ord('e'):       # Лаплас (контуры)
            filter_mode = "laplace"
            weights = build_weights(filter_mode, cout, cin, device=dev, dtype=dtype)
        elif k == ord('x'):       # Собель X (вертикальные границы)
            filter_mode = "sobelx"
            weights = build_weights(filter_mode, cout, cin, device=dev, dtype=dtype)
        elif k == ord('y'):       # Собель Y (горизонтальные границы)
            filter_mode = "sobely"
            weights = build_weights(filter_mode, cout, cin, device=dev, dtype=dtype)

        # ===== эффекты на входе =====
        elif k == ord('f'):  
            order = ["none","blur","sharpen","noise","bright+","bright-","contrast","down2x"]
            fx_mode = order[(order.index(fx_mode)+1) % len(order)]

        elif k == ord('d'):
            dtype = torch.float16 if dtype == torch.float32 else torch.float32
        elif k == ord('m'):
            dev = "cpu" if dev == "mps" else ("mps" if has_mps() else "cpu"); mps_sync()
        elif k == ord('n'):
            norm_mode = "clamp" if norm_mode=="minmax" else "minmax"
        elif k == ord('a'):
            use_abs = not use_abs
        elif k == ord('g'):
            layout = "grid" if layout == "row" else "row"

        elif k in (ord('+'), ord('=')):
            cout = min(cout+8, 64)
            weights = build_weights(filter_mode, cout, cin, device=dev, dtype=dtype)
            bias = torch.zeros(cout, device=dev, dtype=dtype)
        elif k in (ord('-'), ord('_')):
            cout = max(cout-8, 1)
            weights = build_weights(filter_mode, cout, cin, device=dev, dtype=dtype)
            bias = torch.zeros(cout, device=dev, dtype=dtype)
            
        elif k == ord('p'):  
            view_mode = "ch" if view_mode == "mean" else "mean"
        elif k == ord(']'):  
            ch = (ch + 1) % cout
        elif k == ord('['):  
            ch = (ch - 1) % cout
        elif k == ord('r'):  # распечатать 8x8 фрагмент матрицы разности для выбранного канала
            print_patch = True

        elif k == ord('s'):  # сохранить тензоры в .npy (для отчёта/ноута)
            np.save("y_ref.npy",  y_ref.detach().cpu().numpy())
            np.save("y_wino.npy", y_wino.detach().cpu().numpy())
            np.save("diff.npy",   diff.detach().cpu().numpy())
            print("Saved: y_ref.npy, y_wino.npy, diff.npy")


    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
