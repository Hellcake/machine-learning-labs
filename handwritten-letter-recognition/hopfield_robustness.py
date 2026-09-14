
import time
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ---------------------- Разбор аргументов -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Улучшенная сеть Хопфилда для EMNIST Letters')
    parser.add_argument('--num_classes',     type=int, default=4,   help='Количество первых букв (игнорируется, если указан --letters)')
    parser.add_argument('--letters',         type=str, default="ACFZ", help='Строка требуемых букв, например "ACFZ"')
    parser.add_argument('--K',               type=int, default=5,    help='Число прототипов на класс')
    parser.add_argument('--test_samples',    type=int, default=4000, help='Число тестовых образцов для оценки')
    parser.add_argument('--display_samples', type=int, default=12,    help='Число изображений для визуализации')
    parser.add_argument('--max_iter',        type=int, default=10,   help='Максимум итераций в извлечении')
    return parser.parse_args()


# -------------------- Динамика сети Хопфилда --------------------------
def retrieve(state, W, max_iter=10):
    """Асинхронная (двоичная) динамика; прекращается, когда состояние перестает меняться."""
    s = state.clone()
    for _ in range(max_iter):
        new = torch.sign(W @ s)
        new[new == 0] = 1
        if torch.equal(new, s):
            break
        s = new
    return s

def retrieve_with_trace(state, W, max_iter=10):
    s = state.clone()
    trace = [s.clone()]  # сохраняем начальное состояние
    for _ in range(max_iter):
        new = torch.sign(W @ s)
        new[new == 0] = 1
        if torch.equal(new, s):
            break
        s = new
        trace.append(s.clone())  # сохраняем каждое новое состояние
    return s, trace


# ---------------------- Шумовые функции -------------------------------
def noise_bit_flip(p, level):
    p2 = p.clone()
    num = p2.numel()
    n_flip = int(level * num)
    idx = torch.randperm(num, device=p.device)[:n_flip]
    flat = p2.view(-1)
    flat[idx] *= -1
    return flat.view(p2.shape)


def noise_gaussian(p, level):
    p2 = ((p + 1) / 2).clone().view(-1)  # переводим {-1, +1} → {0, 1}
    noise = torch.randn_like(p2) * level
    v = p2 + noise
    v = torch.clamp(v, 0, 1)
    return torch.where(v >= 0.5, torch.ones_like(p2), -torch.ones_like(p2)).view(p.shape)


def noise_dropout(p, level):
    p2 = p.clone().view(-1)
    num = p2.numel()
    n_drop = int(level * num)
    idx = torch.randperm(num, device=p.device)[:n_drop]
    flat = p2.clone()
    flat[idx] = 1  # фон (+1)
    return flat.view(p.shape)


# ------------------------------ main ----------------------------------
def main():
    args = parse_args()
    t_global = time.time()

    # Устройство: GPU или CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Устройство: {device}')

    # Какие буквы берём?
    if args.letters:
        chosen_orig = [ord(ch.upper()) - 65 for ch in args.letters if ch.strip()]
        if any(x < 0 or x > 25 for x in chosen_orig):
            raise ValueError('В EMNIST есть только буквы A–Z')
    else:
        chosen_orig = list(range(args.num_classes))
    args.num_classes = len(chosen_orig)
    orig2new = {o: i for i, o in enumerate(chosen_orig)}
    tick_labels = [chr(65 + o) for o in chosen_orig]
    print(f'Классы: {tick_labels} (всего {args.num_classes}), прототипов на класс: {args.K}')

    # -------------------- Загрузка EMNIST Letters ----------------------
    t0 = time.time()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, -1, [1, 2])),
        transforms.Lambda(lambda x: torch.flip(x, [2])),
    ])
    train = torchvision.datasets.EMNIST(root='data', split='letters', train=True,
                                        download=True, transform=transform)
    test = torchvision.datasets.EMNIST(root='data', split='letters', train=False,
                                       download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=len(train), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=False)
    train_imgs, train_lbls = next(iter(train_loader))
    test_imgs, test_lbls = next(iter(test_loader))

    train_lbls = (train_lbls - 1).to(device)  # 1–26 → 0–25
    test_lbls = (test_lbls - 1).to(device)
    train_imgs = train_imgs.view(-1, 28 * 28).to(device)
    test_imgs = test_imgs.view(-1, 28 * 28).to(device)

    # -------------------------------- Фильтр ---------------------------
    map_tensor = torch.full((26,), -1, device=device, dtype=torch.long)
    for o, n in orig2new.items():
        map_tensor[o] = n

    train_lbls = map_tensor[train_lbls]
    test_lbls = map_tensor[test_lbls]

    mask_train = train_lbls >= 0
    mask_test = test_lbls >= 0
    train_imgs, train_lbls = train_imgs[mask_train], train_lbls[mask_train]
    test_imgs, test_lbls = test_imgs[mask_test], test_lbls[mask_test]
    print(f'[Загрузка данных] {time.time() - t0:.2f} s')

    # ------------------ Прототипы (K случайных на класс) ---------------
    t0 = time.time()
    prototypes, proto_lbls = [], []
    for c in range(args.num_classes):
        idxs = (train_lbls == c).nonzero(as_tuple=False).view(-1)
        perm = torch.randperm(len(idxs), device=device)
        sel = idxs[perm[:args.K]]
        for i in sel:
            p = train_imgs[i]
            b = torch.where(p > 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
            prototypes.append(b)
            proto_lbls.append(c)
    P = torch.stack(prototypes)
    proto_lbls = torch.tensor(proto_lbls, device=device)
    print(f'[Прототипы] {P.shape[0]} шт. за {time.time() - t0:.2f} s')

    # ---------------- Матрица весов (псевдообратное правило) -----------
    t0 = time.time()
    M = P @ P.T
    M_inv = torch.pinverse(M)
    W = P.T @ M_inv @ P
    W.fill_diagonal_(0)
    print(f'[Матрица W] построена за {time.time() - t0:.2f} s')



    # ---------------- Подготовка тестовой выборки ----------------------
    t0 = time.time()
    perm = torch.randperm(len(test_lbls), device=device)
    sel = perm[:args.test_samples]
    testP = test_imgs[sel]
    testL = test_lbls[sel]
    print(f'[Тест] {len(testL)} образцов за {time.time() - t0:.2f} s')

    # ---------------------- Оценка без шума ----------------------------
    preds0 = []
    for x in testP:
        inp0 = torch.where(x > 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
        out0 = retrieve(inp0, W, max_iter=args.max_iter)
        sims0 = P @ out0
        j0 = torch.argmax(sims0).item()
        preds0.append(proto_lbls[j0].item())
    preds0 = torch.tensor(preds0, device=device)
    acc0 = (preds0 == testL).float().mean().item()
    print(f'Baseline accuracy: {acc0 * 100:.2f}%')

    cm0 = confusion_matrix(testL.cpu().numpy(), preds0.cpu().numpy(), labels=list(range(args.num_classes)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm0, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title('Confusion matrix (baseline)')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    plt.show()

    # ------------------- Робастность к шуму ----------------------------
    noise_funcs = {
        'BitFlip': noise_bit_flip,
        'Gaussian': noise_gaussian,
        'Dropout': noise_dropout,
    }
    noise_levels = np.linspace(0, 0.5, 6)
    results = {name: [] for name in noise_funcs}

    for name, func in noise_funcs.items():
        t_start = time.time()
        for level in noise_levels:
            preds = []
            for x in testP:
                inp = torch.where(x > 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
                noisy = func(inp, level)
                out = retrieve(noisy, W, max_iter=args.max_iter)
                sims = (P @ out)
                j = torch.argmax(sims).item()
                preds.append(proto_lbls[j].item())
            acc = np.mean((np.array(preds) == testL.cpu().numpy()).astype(float))
            results[name].append(acc)
        print(f'[Eval {name}] {time.time() - t_start:.2f} s')

    print('Точности при разных видах шума:')
    for name, accs in results.items():
        acc_str = ', '.join(f'{int(level*100)}%: {acc * 100:.2f}%' for level, acc in zip(noise_levels, accs))
        print(f'{name}: {acc_str}')

    plt.figure(figsize=(6, 4))
    for name, accs in results.items():
        plt.plot(noise_levels, accs, marker='o', label=name)
    plt.xlabel('Уровень шума')
    plt.ylabel('Точность')
    plt.title('Робастность сети Хопфилда')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    # Визуализируем процесс "вспоминания"
    idx = 0  # индекс изображения из testP
    inp = torch.where(testP[idx] > 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
    noisy = noise_bit_flip(inp, 0.3)

    _, trace = retrieve_with_trace(noisy, W, max_iter=10)

    plt.figure(figsize=(12, len(trace) * 2))
    for i, s in enumerate(trace):
        img = s.cpu().view(28, 28).numpy()
        ax = plt.subplot(1, len(trace), i + 1)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Шаг {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # ------------------- Визуализация восстановления -------------------
    n = min(args.display_samples, len(testL))
    noise_level = 0.1
    plt.figure(figsize=(12, 12))
    for i in range(n):
        orig = testP[i].cpu().view(28, 28).numpy()
        inp = torch.where(testP[i] > 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
        noisy = noise_bit_flip(inp, noise_level).cpu().view(28, 28).numpy()
        rec = retrieve(noise_bit_flip(inp, noise_level), W, max_iter=args.max_iter).cpu().view(28, 28).numpy()

        ax = plt.subplot(n, 3, 3 * i + 1)
        ax.imshow(orig, cmap='gray'); ax.set_title('Оригинал'); ax.axis('off')
        ax = plt.subplot(n, 3, 3 * i + 2)
        ax.imshow(noisy, cmap='gray'); ax.set_title(f'Шум {int(noise_level * 100)}%'); ax.axis('off')
        ax = plt.subplot(n, 3, 3 * i + 3)
        ax.imshow(rec, cmap='gray'); ax.set_title('Восстановлено'); ax.axis('off')
    plt.tight_layout(); plt.show()

    print(f'[Всего времени] {time.time() - t_global:.2f} s')


if __name__ == '__main__':
    main()
