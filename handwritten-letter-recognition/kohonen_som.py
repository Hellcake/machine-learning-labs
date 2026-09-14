#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Organising Map (сеть Кохонена) для распознавания рукописных букв EMNIST.
Поддерживает выбор подмножества букв, полный русский комментарий и привычный
запуск через  `if __name__ == "__main__": main()`.
"""

# ---------- Импорт библиотек -------------------------------------------------
import argparse
import time
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ---------- Параметры командной строки --------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Разбираем аргументы. Все параметры можно задать из консоли.
    Пример:
        python som_emnist_letters.py --letters ACFZ --width 12 --height 12 --epochs 10
    """
    p = argparse.ArgumentParser(
        description="Сеть Кохонена (Self-Organising Map) для EMNIST Letters"
    )
    # Выбор классов
    p.add_argument(
        "--letters",
        type=str,
        default="ACFZ",
        help='Строка нужных букв, например "ACFZ" (по умолчанию — все 26)',
    )
    # Параметры карты
    p.add_argument("--width", type=int, default=20, help="Ширина SOM-карты")
    p.add_argument("--height", type=int, default=20, help="Высота SOM-карты")
    # Обучение
    p.add_argument("--epochs", type=int, default=15, help="Число эпох обучения SOM")
    p.add_argument("--batch", type=int, default=256, help="Размер батча")
    p.add_argument(
        "--init_lr",
        type=float,
        default=0.5,
        help="Начальная скорость обучения (alpha) для SOM",
    )
    p.add_argument(
        "--init_sigma",
        type=float,
        default=None,
        help="Начальный радиус соседства; если None — возьмётся max(width, height) / 2",
    )
    # Визуализация
    p.add_argument(
        "--display_samples",
        type=int,
        default=12,
        help="Сколько изображений отображать в гриде «оригинал/предсказание»",
    )
    p.add_argument(
        "--max_noise",
        type=float,
        default=0.3,
        help="Максимальный уровень шума для оценки робастности (0 — без шума)",
    )
    return p.parse_args()


# ---------- Класс SOM --------------------------------------------------------
class SOM(torch.nn.Module):
    """
    Простая реализация двумерной SOM на PyTorch.
    • m, n  — размеры решётки
    • dim   — размерность входного вектора (здесь 784)
    """

    def __init__(self, m: int, n: int, dim: int, device: torch.device):
        super().__init__()
        self.m = m
        self.n = n
        self.dim = dim
        self.device = device

        # Весовые векторы узлов карты: (nodes, dim)
        self.weights = torch.randn(m * n, dim, device=device)

        # Координаты узлов на решётке, пригодятся для расчёта соседства
        self.coords = self._create_coords().to(device)  # shape (nodes, 2)

    def _create_coords(self) -> torch.Tensor:
        """Возвращает тензор [[0,0], [0,1], …] размером (m*n, 2)."""
        xs, ys = torch.meshgrid(
            torch.arange(self.m), torch.arange(self.n), indexing="ij"
        )
        return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1).float()

    # --------------------------------------------------------------------- #
    #                    Основные операции SOM                              #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def find_bmu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Находит Best Matching Unit (индекс узла, чей вес ближе всего к x).
        x:       (batch, dim)
        returns: (batch,) индексы BMU
        """
        # Используем формулу ||w||^2 + ||x||^2 – 2 w·x для экономии памяти
        x2 = (x**2).sum(dim=1, keepdim=True)  # (batch, 1)
        w2 = (self.weights**2).sum(dim=1).unsqueeze(0)  # (1, nodes)
        dist = x2 + w2 - 2.0 * (x @ self.weights.t())  # (batch, nodes)
        bmu_idx = torch.argmin(dist, dim=1)  # (batch,)
        return bmu_idx

    @torch.no_grad()
    def update(self, x: torch.Tensor, bmu_idx: torch.Tensor, alpha: float, sigma: float):
        """
        Обновляет веса узлов по правилу Кохонена.
        x        : (batch, dim) — входные векторы
        bmu_idx  : (batch,)     — индексы BMU для каждого x
        alpha    : скорость обучения
        sigma    : радиус соседства
        """
        batch = x.size(0)
        nodes = self.m * self.n

        # --- Маска соседства ------------------------------------------------
        # Координаты BMU выбранных узлов
        bmu_coords = self.coords[bmu_idx]  # (batch, 2)
        # Расстояния до всех остальных узлов (broadcast)
        dxy = (
            self.coords.unsqueeze(0) - bmu_coords.unsqueeze(1)
        )  # (batch, nodes, 2)
        dist_sq = (dxy**2).sum(dim=2)  # (batch, nodes)
        # Расчёт гауссового соседства
        neighborhood = torch.exp(-dist_sq / (2 * sigma * sigma))  # (batch, nodes)

        # --- Адаптация весов -------------------------------------------------
        # Δw = α · h · (x - w)
        # Переводим x в shape (batch, 1, dim) чтобы broadcastилось
        x_expanded = x.unsqueeze(1)  # (batch, 1, dim)
        w = self.weights.unsqueeze(0)  # (1, nodes, dim)
        h = neighborhood.unsqueeze(2)  # (batch, nodes, 1)

        delta = alpha * h * (x_expanded - w)  # (batch, nodes, dim)
        # Усредняем по батчу, чтобы одно обновление
        # было похоже на «mini-batch» правило
        self.weights += delta.mean(dim=0)

    # --------------------------------------------------------------------- #
    #                        Вспомогательные методы                         #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def assign_labels(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Подписывает каждый узел SOM меткой большинства.
        data   : (N, dim)  тензор изображений {-1, +1}
        labels : (N,)      ground-truth метки 0..C-1
        returns: (nodes,)  метка узла (-1 если пустой)
        """
        nodes = self.m * self.n
        assigned = torch.full((nodes,), -1, device=self.device, dtype=torch.long)

        # Для каждой точки узнаём её BMU
        bmu = self.find_bmu(data)  # (N,)
        for node in range(nodes):
            mask = bmu == node
            if mask.any():
                # Берём моду (метку большинства)
                vals, counts = torch.unique(labels[mask], return_counts=True)
                assigned[node] = vals[torch.argmax(counts)]
        return assigned

    @torch.no_grad()
    def predict(self, x: torch.Tensor, node_labels: torch.Tensor) -> torch.Tensor:
        """
        Предсказывает класс для входного батча x на основе уже подписанных узлов.
        Возвращает (batch,) тензор предсказаний.
        """
        bmu = self.find_bmu(x)  # (batch,)
        return node_labels[bmu]


# ---------- Функции шума (как в Хопфилде) ------------------------------------
def noise_bit_flip(x: torch.Tensor, level: float) -> torch.Tensor:
    """Искажение «битовым» шумом: меняем знак у level·N случайных пикселей."""
    x2 = x.clone()
    num_flip = int(level * x2.numel())
    idx = torch.randperm(x2.numel(), device=x.device)[:num_flip]
    flat = x2.view(-1)
    flat[idx] *= -1
    return flat.view(x2.shape)


# ---------- Основной pipeline -----------------------------------------------
def main() -> None:
    args = parse_args()
    t_global = time.time()

    # ---- Устройство -----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # ---- Какие буквы берём? --------------------------------------------
    if args.letters.strip():
        chosen_orig = [ord(ch.upper()) - 65 for ch in args.letters if ch.strip()]
        if any(x < 0 or x > 25 for x in chosen_orig):
            raise ValueError("В EMNIST есть только буквы A–Z")
    else:
        chosen_orig = list(range(26))  # все буквы
    num_classes = len(chosen_orig)
    orig2new = {o: i for i, o in enumerate(chosen_orig)}
    tick_labels = [chr(65 + o) for o in chosen_orig]
    print(
        f"Выбраны буквы: {''.join(tick_labels)}  (классов: {num_classes})",
        f"\nРазмер SOM-карты: {args.height}×{args.width}",
    )

    # ---- Загрузка EMNIST Letters ---------------------------------------
    t0 = time.time()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # (1, 28, 28), значения [0..1]
            transforms.Lambda(
                lambda x: torch.rot90(x, -1, [1, 2])
            ),  # поворот как в вашем примере
            transforms.Lambda(lambda x: torch.flip(x, [2])),  # отражение
        ]
    )
    train_ds = torchvision.datasets.EMNIST(
        root="data", split="letters", train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.EMNIST(
        root="data", split="letters", train=False, download=True, transform=transform
    )

    # Склеиваем в тензоры весь датасет сразу (EMNIST небольшой)
    train_imgs = train_ds.data.float().unsqueeze(1) / 255.0
    test_imgs = test_ds.data.float().unsqueeze(1) / 255.0
    train_lbls = train_ds.targets
    test_lbls = test_ds.targets
    # Приводим 1-26 → 0-25
    train_lbls -= 1
    test_lbls -= 1

    # Фильтруем выбранные буквы
    map_tensor = torch.full((26,), -1, dtype=torch.long)
    for o, n in orig2new.items():
        map_tensor[o] = n
    train_lbls = map_tensor[train_lbls]
    test_lbls = map_tensor[test_lbls]
    mask_train = train_lbls >= 0
    mask_test = test_lbls >= 0
    train_imgs = train_imgs[mask_train]
    train_lbls = train_lbls[mask_train]
    test_imgs = test_imgs[mask_test]
    test_lbls = test_lbls[mask_test]

    # Переводим в векторы {-1, +1}
    train_vec = torch.where(
        train_imgs > 0.5, torch.tensor(1.0), torch.tensor(-1.0)
    ).view(-1, 28 * 28)
    test_vec = torch.where(
        test_imgs > 0.5, torch.tensor(1.0), torch.tensor(-1.0)
    ).view(-1, 28 * 28)

    # На устройство
    train_vec = train_vec.to(device)
    test_vec = test_vec.to(device)
    train_lbls = train_lbls.to(device)
    test_lbls = test_lbls.to(device)

    print(f"[Данные] загружены за {time.time() - t0:.2f} с.")

    # ---- Создаём и обучаем SOM -----------------------------------------
    som = SOM(args.height, args.width, 28 * 28, device).to(device)
    print("Обучение SOM…")
    nodes = args.height * args.width
    init_sigma = args.init_sigma or max(args.height, args.width) / 2
    for epoch in range(args.epochs):
        t_ep = time.time()
        # Экспоненциальное затухание скорости и радиуса
        alpha = args.init_lr * math.exp(-epoch / args.epochs)
        sigma = init_sigma * math.exp(-epoch / args.epochs)
        # Мини-батчевое обучение
        perm = torch.randperm(len(train_vec), device=device)
        for i in range(0, len(train_vec), args.batch):
            batch_idx = perm[i : i + args.batch]
            x = train_vec[batch_idx]
            bmu = som.find_bmu(x)  # (batch,)
            som.update(x, bmu, alpha, sigma)
        print(
            f"  Эпоха {epoch+1:02d}/{args.epochs}: "
            f"α={alpha:.3f}, σ={sigma:.2f}, "
            f"{time.time() - t_ep:.1f} с"
        )

    # ---- Подписываем узлы и тестируем ----------------------------------
    print("Подписываю узлы карты метками…")
    node_labels = som.assign_labels(train_vec, train_lbls)  # (nodes,)

    # Предсказания на чистых данных
    with torch.no_grad():
        preds_clean = som.predict(test_vec, node_labels)
    acc_clean = (preds_clean == test_lbls).float().mean().item()
    print(f"Точность без шума: {acc_clean*100:.2f}%")

    # Матрица ошибок
    cm = confusion_matrix(
        test_lbls.cpu().numpy(),
        preds_clean.cpu().numpy(),
        labels=list(range(num_classes)),
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    plt.title("Матрица ошибок (SOM, без шума)")
    plt.xlabel("Предсказано")
    plt.ylabel("Истина")
    plt.tight_layout()
    plt.show()

    # ---- Робастность (при желании) -------------------------------------
    if args.max_noise > 0:
        noise_levels = np.linspace(0.0, args.max_noise, 6)
        accs = []
        for level in noise_levels:
            with torch.no_grad():
                noisy = torch.stack(
                    [noise_bit_flip(v, level) for v in test_vec], dim=0
                )  # (N, 784)
                preds = som.predict(noisy, node_labels)
            acc = (preds == test_lbls).float().mean().item()
            accs.append(acc)
        plt.figure(figsize=(6, 4))
        plt.plot(noise_levels, accs, marker="o")
        plt.title("Робастность SOM к битовому шуму")
        plt.xlabel("Уровень шума")
        plt.ylabel("Точность")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ---- Визуализация нескольких примеров ------------------------------
    n = min(args.display_samples, len(test_vec))
    plt.figure(figsize=(9, n * 3))
    for i in range(n):
        orig = test_vec[i].cpu().view(28, 28).numpy()
        noisy = noise_bit_flip(test_vec[i], 0.1).cpu().view(28, 28).numpy()
        rec_label = node_labels[som.find_bmu(test_vec[i].unsqueeze(0))[0]].item()
        title_rec = f"Пред: {tick_labels[rec_label]}" if rec_label >= 0 else "Нет узла"

        # Оригинал
        ax = plt.subplot(n, 3, 3 * i + 1)
        ax.imshow(orig, cmap="gray")
        ax.set_title("Оригинал")
        ax.axis("off")
        # С шумом
        ax = plt.subplot(n, 3, 3 * i + 2)
        ax.imshow(noisy, cmap="gray")
        ax.set_title("Шум 10%")
        ax.axis("off")
        # BMU-прототип
        ax = plt.subplot(n, 3, 3 * i + 3)
        proto = som.weights[som.find_bmu(test_vec[i].unsqueeze(0))[0]]
        ax.imshow(proto.cpu().view(28, 28).numpy(), cmap="gray")
        ax.set_title(title_rec)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Всего времени работы: {time.time() - t_global:.1f} с.")


# ---------- Запуск как скрипта ----------------------------------------------
if __name__ == "__main__":
    main()
