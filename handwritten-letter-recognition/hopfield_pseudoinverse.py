import time
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Разбор аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description='Улучшенная сеть Хопфилда для EMNIST Letters')
    parser.add_argument('--num_classes',     type=int,   default=10,   help='Количество букв (A=0, B=1, ...)')
    parser.add_argument('--K',               type=int,   default=6,   help='Число прототипов на класс')
    parser.add_argument('--test_samples',    type=int,   default=2000, help='Число тестовых образцов для оценки')
    parser.add_argument('--display_samples', type=int,   default=6,   help='Число изображений для визуализации')
    parser.add_argument('--max_iter',        type=int,   default=10,  help='Максимум итераций в извлечении')
    parser.add_argument(
    '--letters',
    type=str,
    default=None,
    help='Строка требуемых букв, например "ACFZ"; если указана, параметр '
         '--num_classes игнорируется'
)
    return parser.parse_args()

# Функция динамики сети Хопфилда (извлечение)
def retrieve(state, W, max_iter=10):
    s = state.clone()
    for _ in range(max_iter):
        new = torch.sign(W @ s)
        new[new == 0] = 1
        if torch.equal(new, s):
            break
        s = new
    return s

# Шумовые функции
# 1. Переворот случайных битов (бит-флип)
def noise_bit_flip(p, level):
    p2 = p.clone()
    num = p2.numel()
    n_flip = int(level * num)
    idx = torch.randperm(num, device=p.device)[:n_flip]
    flat = p2.view(-1)
    flat[idx] *= -1
    return flat.view(p2.shape)

# 2. Гауссовский шум + порог
def noise_gaussian(p, level):
    # level — относительная дисперсия
    p2 = p.clone().view(-1)
    noise = torch.randn_like(p2) * level
    v = p2.float() + noise
    # порог по нулю: >0 -> +1, <=0 -> -1
    return torch.where(v >= 0, torch.tensor(1., device=p.device), torch.tensor(-1., device=p.device)).view(p.shape)

# 3. Пропуск пикселей (dropout)
def noise_dropout(p, level):
    p2 = p.clone().view(-1)
    num = p2.numel()
    n_drop = int(level * num)
    idx = torch.randperm(num, device=p.device)[:n_drop]
    flat = p2.clone()
    flat[idx] = 1  # заполняем +1 (фон)
    return flat.view(p.shape)

# Основная функция

def main():
    args = parse_args()
    t_global = time.time()

    # Устройство: GPU или CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Устройство: {device}, классов: {args.num_classes}, прототипов на класс: {args.K}')

    # 1. Загрузка EMNIST Letters
    t0 = time.time()
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.rot90(x, -1, [1,2])),
    transforms.Lambda(lambda x: torch.flip(x, [2])),
    ])
    train = torchvision.datasets.EMNIST(root='data', split='letters', train=True, download=True, transform=transform)
    test  = torchvision.datasets.EMNIST(root='data', split='letters', train=False,download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=len(train), shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test,  batch_size=len(test),  shuffle=False)
    train_imgs, train_lbls = next(iter(train_loader))
    test_imgs,  test_lbls  = next(iter(test_loader))
    # Сдвиг меток: 1-26 -> 0-25
    train_lbls = (train_lbls - 1).to(device)
    test_lbls  = (test_lbls  - 1).to(device)
    train_imgs = train_imgs.view(-1, 28*28).to(device)
    test_imgs  = test_imgs.view(-1, 28*28).to(device)
    print(f'[Загрузка данных] {time.time()-t0:.2f}s')

    # 2. Построение прототипов: K случайных примеров на класс
    t0 = time.time()
    prototypes = []
    proto_lbls = []
    for c in range(args.num_classes):
        idxs = (train_lbls == c).nonzero(as_tuple=False).view(-1)
        perm = torch.randperm(len(idxs), device=device)
        sel = idxs[perm[:args.K]]
        for i in sel:
            p = train_imgs[i]
            b = torch.where(p > 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
            prototypes.append(b)
            proto_lbls.append(c)
    P = torch.stack(prototypes)        # (patterns, N)
    proto_lbls = torch.tensor(proto_lbls, device=device)
    print(f'[Прототипы] {P.shape[0]} шт. собрано за {time.time()-t0:.2f}s')

    # 3. Построение матрицы весов: псевдообратное правило
    t0 = time.time()
    M = P @ P.T                       # (patterns x patterns)
    M_inv = torch.pinverse(M)
    W = P.T @ M_inv @ P               # (N x N)
    W.fill_diagonal_(0)
    print(f'[Матрица W] псевдообратное правило за {time.time()-t0:.2f}s')

    # 4. Подготовка тестовой выборки
    t0 = time.time()
    mask = test_lbls < args.num_classes
    idxs = mask.nonzero(as_tuple=False).view(-1)
    perm = torch.randperm(len(idxs), device=device)
    sel = idxs[perm[:args.test_samples]]
    testP = test_imgs[sel]
    testL = test_lbls[sel]
    print(f'[Тест] {len(testL)} образцов отобрано за {time.time()-t0:.2f}s')

    # Список шумовых функций и уровней
    noise_funcs = {
        'BitFlip': noise_bit_flip,
        'Gaussian': noise_gaussian,
        'Dropout': noise_dropout
    }
    noise_levels = np.linspace(0, 0.5, 6)
    results = {name: [] for name in noise_funcs}

        # 5. Базовая классификация без шума и матрица ошибок
    print('--- Базовая классификация (без шума) ---')
    preds0 = []
    for x in testP:
        inp0 = torch.where(x > 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
        out0 = retrieve(inp0, W, max_iter=args.max_iter)
        sims0 = P @ out0
        j0 = torch.argmax(sims0).item()
        preds0.append(proto_lbls[j0].item())
    preds0 = torch.tensor(preds0, device=device)
    acc0 = (preds0 == testL).float().mean().item()
    print(f'Baseline accuracy (0 noise): {acc0*100:.2f}%')

    # confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm0 = confusion_matrix(testL.cpu().numpy(), preds0.cpu().numpy(), labels=list(range(args.num_classes)))
    ticks = [chr(65+i) for i in range(args.num_classes)]
    plt.figure(figsize=(6,5))
    sns.heatmap(cm0, annot=True, fmt='d', cmap='Blues', xticklabels=ticks, yticklabels=ticks)
    plt.title('Confusion matrix (baseline)')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout(); plt.show()

    # 6. Оценка точности при разных видах шума
    for name, func in noise_funcs.items():
        t_start = time.time()
        for level in noise_levels:
            preds = []
            for x in testP:
                # биполяризация входа
                inp = torch.where(x > 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
                # добавление шума
                noisy = func(inp, level)
                # восстановление
                out = retrieve(noisy, W, max_iter=args.max_iter)
                # классификация ближайшим прототипом
                sims = (P @ out)
                j = torch.argmax(sims).item()
                preds.append(proto_lbls[j].item())
            acc = np.mean((np.array(preds) == testL.cpu().numpy()).astype(float))
            results[name].append(acc)
        print(f'[Eval {name}] {time.time()-t_start:.2f}s')

            # 6. Текстовый вывод точностей для каждого шума и модели
    print('Точности при разных видах шума:')
    for name, accs in results.items():
        acc_str = ', '.join(f'{level*100:.0f}%:{acc*100:.2f}%' for level, acc in zip(noise_levels, accs))
        print(f'{name}: {acc_str}')

    # 7. Построение графика точности по шуму точности по шуму
    plt.figure(figsize=(6,4))
    for name, accs in results.items():
        plt.plot(noise_levels, accs, marker='o', label=name)
    plt.xlabel('Уровень шума')
    plt.ylabel('Точность')
    plt.title('Робастность сети Хопфилда при разных шумовых моделях')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 7. Визуализация примеров восстановления
    n = min(args.display_samples, len(testL))
    noise_level = 0.3
    plt.figure(figsize=(12,12))
    for i in range(n):
        orig = testP[i].cpu().view(28,28).numpy()
        inp = torch.where(testP[i]>0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
        # шум
        noisy = noise_dropout(inp, noise_level).cpu().view(28,28).numpy()
        # восстановление
        rec = retrieve(noise_dropout(inp, noise_level), W, max_iter=args.max_iter).cpu().view(28,28).numpy()

        ax = plt.subplot(n,3,3*i+1)
        ax.imshow(orig, cmap='gray'); ax.set_title('Оригинал'); ax.axis('off')
        ax = plt.subplot(n,3,3*i+2)
        ax.imshow(noisy, cmap='gray'); ax.set_title(f'Шум {int(noise_level*100)}%'); ax.axis('off')
        ax = plt.subplot(n,3,3*i+3)
        ax.imshow(rec, cmap='gray'); ax.set_title('Восстановлено'); ax.axis('off')
    plt.tight_layout(); plt.show()

    print(f'[Всего времени] {time.time()-t_global:.2f}s')

if __name__ == '__main__':
    main()
