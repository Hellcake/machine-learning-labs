# demo_realtime.py
import time, cv2, numpy as np, torch, torch.nn.functional as F
from winograd_f23 import winograd_conv2d_f23

def has_mps():
    return torch.backends.mps.is_available()

def mps_sync():
    if has_mps():
        torch.mps.synchronize()

def make_edge_weights(cout=16, cin=3, device="cpu", dtype=torch.float32):
    # Лапласиан: подчёркивает контуры
    k = np.array([[0,-1,0],
                  [-1,4,-1],
                  [0,-1,0]], dtype=np.float32)
    w = np.stack([k]*cin, axis=0)                 # [Cin,3,3]
    w = np.stack([w]*cout, axis=0)                # [Cout,Cin,3,3]
    return torch.tensor(w, device=device, dtype=dtype)

def to_uint8_gray(t: torch.Tensor):
    # t: [H,W] или [1,H,W]
    if t.dim() == 3: t = t[0]
    t = t.clamp(0,1).detach().cpu().numpy()
    return (t*255.0).astype(np.uint8)

def colorize_heatmap(z: torch.Tensor):
    # z: [H,W], предполагаем >=0, нормализуем
    z = z.detach().cpu().numpy()
    if z.size == 0:
        return np.zeros((1,1,3), np.uint8)
    z = z / (z.max() + 1e-8)
    z8 = (z*255).astype(np.uint8)
    return cv2.applyColorMap(z8, cv2.COLORMAP_JET)

def put_hud(img, text, y=18):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру. Проверь разрешения macOS: System Settings → Privacy & Security → Camera.")

    dev = "cpu" if has_mps() else "cpu"
    dtype = torch.float32
    cout, cin = 16, 3
    weights = torch.randn(cout, cin, 3, 3, device=dev, dtype=dtype) * 0.01
    bias = torch.zeros(cout, device=dev, dtype=dtype)

    show_compare = False  # показывать сравнение с обычной conv2d
    use_edge = False      # использовать лапласиан вместо случайных весов
    t_prev, fps = time.perf_counter(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0

        # при смене dtype — приводим и вход/веса
        x = x.to(dtype)
        w = weights.to(dev, dtype)
        b = bias.to(dev, dtype)

        t0 = time.perf_counter()
        y_wino = winograd_conv2d_f23(x, w, b)            # [1,cout,H,W]
        mps_sync()
        t1 = time.perf_counter()

        vis_wino = to_uint8_gray(y_wino.mean(1))         # [H,W]
        vis_wino = cv2.cvtColor(vis_wino, cv2.COLOR_GRAY2BGR)

        if show_compare:
            y_ref = F.conv2d(x, w, b, padding=1)
            mps_sync()
            err = (y_ref - y_wino).abs().mean(1).squeeze(0)   # [H,W]
            diff = colorize_heatmap(err)
            vis_ref = to_uint8_gray(y_ref.mean(1))
            vis_ref = cv2.cvtColor(vis_ref, cv2.COLOR_GRAY2BGR)
            # Склеиваем: [Input | Winograd | Conv2d | |Diff|]
            vis_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # оригинал в RGB
            vis_in = cv2.cvtColor(vis_in, cv2.COLOR_RGB2BGR)  # back to BGR for cv2
            row1 = np.hstack([cv2.resize(vis_in, (0,0), fx=0.5, fy=0.5),
                              cv2.resize(vis_wino, (0,0), fx=0.5, fy=0.5),
                              cv2.resize(vis_ref, (0,0), fx=0.5, fy=0.5)])
            row2 = cv2.resize(diff, (row1.shape[1], row1.shape[0]))
            view = np.vstack([row1, row2])
        else:
            view = vis_wino

        # FPS
        t_now = time.perf_counter()
        dt = t_now - t_prev
        t_prev = t_now
        fps = 0.9*fps + 0.1*(1.0/max(dt,1e-6))

        put_hud(view, f"Device: {dev} | DType: {str(dtype).split('.')[-1]} | Cout={cout} | Compare={'ON' if show_compare else 'OFF'}")
        put_hud(view, f"FPS ~ {fps:5.1f} | Winograd time: {(t1-t0)*1000:.2f} ms", y=view.shape[0]-10)

        cv2.imshow("Winograd realtime", view)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break
        elif k == ord('c'):
            show_compare = not show_compare
        elif k == ord('w'):
            weights = torch.randn(cout, cin, 3, 3, device=dev, dtype=dtype) * 0.01
            use_edge = False
        elif k == ord('e'):
            weights = make_edge_weights(cout, cin, device=dev, dtype=dtype)
            use_edge = True
        elif k == ord('d'):
            dtype = torch.float16 if dtype == torch.float32 else torch.float32
        elif k == ord('m'):
            dev = "cpu" if dev == "mps" else ("mps" if has_mps() else "cpu")
            mps_sync()
        elif k == ord('+') or k == ord('='):
            cout = min(cout+8, 64)
            weights = torch.randn(cout, cin, 3, 3, device=dev, dtype=dtype) * 0.01
            bias = torch.zeros(cout, device=dev, dtype=dtype)
            if use_edge:
                weights = make_edge_weights(cout, cin, device=dev, dtype=dtype)
        elif k == ord('-') or k == ord('_'):
            cout = max(cout-8, 1)
            weights = torch.randn(cout, cin, 3, 3, device=dev, dtype=dtype) * 0.01
            bias = torch.zeros(cout, device=dev, dtype=dtype)
            if use_edge:
                weights = make_edge_weights(cout, cin, device=dev, dtype=dtype)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
