import time
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# --------------------------------------------------------------------------- #
#                             ПАРСЕР АРГУМЕНТОВ                               #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description='Классическая сеть Хопфилда для EMNIST Letters')
    p.add_argument('--num_classes',     type=int, default=3,
                   help='Сколько ПЕРВЫХ букв взять (игнорируется, если указан --letters)')
    p.add_argument('--letters',         type=str, default="ACFZ",
                   help='Строка нужных букв, например "ACFZ"')
    p.add_argument('--gs', action='store_true',
                   help='Включить ортогонализацию Грамма–Шмидта для прототипов')
    p.add_argument('--test_variants',   type=int, default=100,
                   help='Сколько шумных вариаций эталона на уровень шума')
    p.add_argument('--display_samples', type=int, default=9,
                   help='Сколько классов показать при визуализации')
    p.add_argument('--max_iter',        type=int, default=10,
                   help='Максимум итераций динамики сети')
    return p.parse_args()


# --------------------------------------------------------------------------- #
#                          ДИНАМИКА СЕТИ ХОПФИЛДА                             #
# --------------------------------------------------------------------------- #
def retrieve(state: torch.Tensor, W: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
    """Асинхронная динамика Хопфилда."""
    s = state.clone()
    for _ in range(max_iter):
        new = torch.sign(W @ s)
        new[new == 0] = 1
        if torch.equal(new, s):
            break
        s = new
    return s


# --------------------------------------------------------------------------- #
#                  ОРТОГОНАЛИЗАЦИЯ ГРАММА–ШМИДТА ПО СТРОКАМ                   #
# --------------------------------------------------------------------------- #
def gram_schmidt_rows(mat: torch.Tensor) -> torch.Tensor:
    """
    Ортогонализирует строки матрицы (k × N) и нормирует их до длины 1.
    Работает на CPU и GPU.
    """
    q_rows = []
    for v in mat:
        u = v.clone()
        for q in q_rows:
            u = u - torch.dot(u, q) * q
        norm = torch.norm(u)
        q_rows.append(u / (norm + 1e-8))
    return torch.stack(q_rows)


# --------------------------------------------------------------------------- #
#                               ШУМОВЫЕ ФУНКЦИИ                               #
# --------------------------------------------------------------------------- #
def noise_bit_flip(p, level):
    p2 = p.clone()
    num = p2.numel()
    n_flip = int(level * num)
    idx = torch.randperm(num, device=p.device)[:n_flip]
    flat = p2.view(-1)
    flat[idx] *= -1
    return flat.view(p2.shape)


def noise_gaussian(p, level):
    p2 = p.clone().view(-1)
    noise = torch.randn_like(p2) * level
    v = p2.float() + noise
    return torch.where(v >= 0, torch.tensor(1., device=p.device),
                       torch.tensor(-1., device=p.device)).view(p.shape)


def noise_dropout(p, level):
    p2 = p.clone().view(-1)
    num = p2.numel()
    n_drop = int(level * num)
    idx = torch.randperm(num, device=p.device)[:n_drop]
    flat = p2.clone()
    flat[idx] = 1
    return flat.view(p.shape)


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    t_global = time.time()

    # --------------------------- Устройство ---------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Устройство: {device}')

    # --------------------------- Подмножество букв --------------------------
    if args.letters:
        chosen_orig = [ord(ch.upper()) - 65 for ch in args.letters if ch.strip()]
        if any(x < 0 or x > 25 for x in chosen_orig):
            raise ValueError('В EMNIST есть только буквы A–Z')
    else:
        chosen_orig = list(range(args.num_classes))
    args.num_classes = len(chosen_orig)
    orig2new = {o: i for i, o in enumerate(chosen_orig)}
    tick_labels = [chr(65 + o) for o in chosen_orig]
    print(f'Классы: {tick_labels} (всего {args.num_classes})')

    # ----------------------------- Загрузка EMNIST -------------------------
    t0 = time.time()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, -1, [1, 2])),  # повернуть
        transforms.Lambda(lambda x: torch.flip(x, [2])),          # зеркально
    ])
    emnist = torchvision.datasets.EMNIST(root='data', split='letters',
                                         train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(emnist, batch_size=len(emnist), shuffle=False)
    imgs, lbls = next(iter(loader))
    lbls = (lbls - 1).to(device)                          # 1-26 → 0-25
    imgs = imgs.view(-1, 28 * 28).to(device)

    # ----------------------- Фильтрация/перекодировка ----------------------
    map_tensor = torch.full((26,), -1, device=device, dtype=torch.long)
    for o, n in orig2new.items():
        map_tensor[o] = n
    lbls = map_tensor[lbls]
    mask = lbls >= 0
    imgs, lbls = imgs[mask], lbls[mask]
    print(f'[Загрузка данных] {time.time() - t0:.2f} s')

    # ---------------------- Формируем прототипы ---------------------------
    t0 = time.time()
    refs, ref_lbls = [], []
    for c in range(args.num_classes):
        idx = (lbls == c).nonzero(as_tuple=False).view(-1)[0]
        p = imgs[idx]
        b = torch.where(p > 0.5, torch.tensor(1., device=device),
                        torch.tensor(-1., device=device))
        refs.append(b.float())
        ref_lbls.append(c)
    P = torch.stack(refs)                            # (classes × N)
    ref_lbls = torch.tensor(ref_lbls, device=device)
    print(f'[Эталоны] {P.shape[0]} шт. за {time.time() - t0:.2f} s')

    # ---------------- Ортогонализация (по желанию) ------------------------
    if args.gs:
        print('[GS] Ортогонализируем прототипы (Грамм–Шмидт)…')
        P = gram_schmidt_rows(P)                     # уже float, unit-length

    # --------------------- Матрица весов (Hebb) ---------------------------
    t0 = time.time()
    W = P.T @ P
    W.fill_diagonal_(0)
    print(f'[Матрица W] {"GS-Hebb" if args.gs else "Hebb"} за {time.time() - t0:.2f} s')

    # --------------------- Baseline (без шума) ----------------------------
    preds0 = []
    true0 = np.arange(args.num_classes)
    for ref in P:
        out = retrieve(ref, W, max_iter=args.max_iter)
        j = torch.argmax(P @ out).item()
        preds0.append(ref_lbls[j].item())
    acc0 = np.mean((np.array(preds0) == true0).astype(float))
    print(f'Baseline accuracy: {acc0 * 100:.2f}%')

    cm0 = confusion_matrix(true0, preds0, labels=list(range(args.num_classes)))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm0, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title('Confusion matrix (baseline)')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout(); plt.show()

    # -------------------- Робастность к шуму ------------------------------
    noise_funcs = {'BitFlip': noise_bit_flip, 'Gaussian': noise_gaussian, 'Dropout': noise_dropout}
    noise_levels = np.linspace(0, 0.5, 6)
    results = {name: [] for name in noise_funcs}

    for name, func in noise_funcs.items():
        for lvl in noise_levels:
            preds = []
            for c, ref in enumerate(P):
                for _ in range(args.test_variants):
                    noisy = func(ref, lvl)
                    out = retrieve(noisy, W, max_iter=args.max_iter)
                    j = torch.argmax(P @ out).item()
                    preds.append(ref_lbls[j].item())
            true = np.repeat(np.arange(args.num_classes), args.test_variants)
            acc = np.mean((np.array(preds) == true).astype(float))
            results[name].append(acc)
        print('[Eval', name, ']', ', '.join(f'{int(l*100)}%:{a*100:.2f}%' for l, a in zip(noise_levels, results[name])))

    # --------------------- График робастности -----------------------------
    plt.figure(figsize=(6, 4))
    for name, accs in results.items():
        plt.plot(noise_levels, accs, marker='o', label=name)
    plt.xlabel('Уровень шума');  plt.ylabel('Точность')
    plt.title('Классическая сеть Хопфилда' + (' + GS' if args.gs else ''))
    plt.legend();  plt.grid(True);  plt.tight_layout();  plt.show()

    # ----------------- Визуализация восстановления ------------------------
    n_vis = min(args.display_samples, args.num_classes)
    noise_level = 0.3
    plt.figure(figsize=(12, 4 * n_vis))
    for i in range(n_vis):
        orig = P[i].cpu().view(28, 28).numpy()
        noisy = noise_dropout(P[i], noise_level).cpu().view(28, 28).numpy()
        rec = retrieve(noise_dropout(P[i], noise_level), W, max_iter=args.max_iter).cpu().view(28, 28).numpy()

        ax = plt.subplot(n_vis, 3, 3 * i + 1)
        ax.imshow(orig, cmap='gray');   ax.set_title('Эталон');          ax.axis('off')
        ax = plt.subplot(n_vis, 3, 3 * i + 2)
        ax.imshow(noisy, cmap='gray');  ax.set_title(f'Шум {int(noise_level*100)}%'); ax.axis('off')
        ax = plt.subplot(n_vis, 3, 3 * i + 3)
        ax.imshow(rec, cmap='gray');    ax.set_title('Восстановлено');   ax.axis('off')
    plt.tight_layout(); plt.show()

    print(f'[Всего времени] {time.time() - t_global:.2f} s')


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    main()