
### Пример использования

"""```bash
python3 mountaincar_dqn_full.py \
  --episodes 500 --eval_episodes 20 --seed 42 \
  --hill_freq 3.0 --hill_amp 1.0 \
  --force 0.001 --gravity 0.0025 \
  --render_demo --gif_path mc_final_demo.gif --plot_path mc_training_curve.png
```
"""

from __future__ import annotations

import argparse
import math
import random
from typing import List, Tuple, Optional

import matplotlib

matplotlib.use("Agg")  # используем бэкенд без GUI
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MountainCarEnv:
    """Параметризуемая среда MountainCar.

    Поддерживает настройку профиля холмов (частота и амплитуда),
    коэффициента гравитации на склоне (slope_gain), диапазонов
    положений/скоростей, начального диапазона, силы мотора, силы
    «уклона», а также максимального числа шагов до завершения
    эпизода.
    """

    def __init__(
        self,
        position_range: Tuple[float, float] = (-1.2, 0.6),
        velocity_range: Tuple[float, float] = (-0.07, 0.07),
        start_range: Tuple[float, float] = (-0.6, -0.4),
        goal_position: float = 0.5,
        force: float = 0.001,
        gravity: float = 0.0025,
        max_steps: int = 200,
        hill_freq: float = 3.0,
        hill_amp: float = 1.0,
        slope_gain: Optional[float] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.position_min, self.position_max = position_range
        self.velocity_min, self.velocity_max = velocity_range
        self.start_min, self.start_max = start_range
        self.goal_position = goal_position
        self.force = force
        self.gravity = gravity
        self.max_steps = max_steps
        # параметры профиля холмов
        self.hill_freq = hill_freq
        self.hill_amp = hill_amp
        # slope_gain определяет, насколько сильно «уклон» влияет на
        # изменение скорости.  По умолчанию равен 1.0 (то есть сила
        # гравитации зависит только от параметра ``gravity`` и
        # выражения ``cos(hill_freq * position)``), что повторяет
        # стандартную реализацию gym.  Пользователь может задать
        # собственный ``slope_gain`` через аргументы CLI; это может
        # понадобиться при изменении профиля холма.
        self.slope_gain = slope_gain if slope_gain is not None else 1.0
        self.rng = rng or np.random.RandomState()
        # внутренние переменные состояния
        self.position = 0.0
        self.velocity = 0.0
        self.steps = 0

    def reset(self) -> np.ndarray:
        """Сбрасывает состояние в начальное положение."""
        self.position = float(self.rng.uniform(self.start_min, self.start_max))
        self.velocity = 0.0
        self.steps = 0
        return np.array([self.position, self.velocity], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Выполняет дискретное действие.

        Допустимые действия: 0 (толкнуть влево), 1 (ничего не делать),
        2 (толкнуть вправо).  Возвращает: (следующее состояние,
        награда, флаг завершения, info).
        """
        action = int(action)
        assert 0 <= action < 3, f"Неверное действие: {action}"
        # вычисляем ускорение от мотора: 0→-1, 1→0, 2→+1
        acc_motor = (action - 1) * self.force
        # вклад «уклона»: -gravity * slope_gain * cos(freq * pos)
        acc_slope = (
            -self.gravity * self.slope_gain * math.cos(self.hill_freq * self.position)
        )
        # обновляем скорость
        self.velocity += acc_motor + acc_slope
        self.velocity = max(min(self.velocity, self.velocity_max), self.velocity_min)
        # обновляем позицию
        self.position += self.velocity
        if self.position <= self.position_min:
            self.position = self.position_min
            self.velocity = 0.0
        self.position = max(min(self.position, self.position_max), self.position_min)
        self.steps += 1
        # проверяем достижение цели
        done = bool(self.position >= self.goal_position or self.steps >= self.max_steps)
        reward = -1.0
        return (
            np.array([self.position, self.velocity], dtype=np.float32),
            reward,
            done,
            {},
        )

    def render(self, ax: Optional[plt.Axes] = None, show: bool = True) -> plt.Axes:
        """Отрисовывает профиль холма и текущее положение машинки.

        :param ax: существующий Axes; если None, создаётся новый
        :param show: если True — вызывает pause для обновления
        :return: объект Axes
        """
        import numpy as _np
        import matplotlib.pyplot as _plt

        xs = _np.linspace(self.position_min, self.position_max, 600)
        ys = self.hill_amp * _np.sin(self.hill_freq * xs)
        if ax is None:
            _, ax = _plt.subplots()
        ax.cla()
        ax.plot(xs, ys, color="black")
        car_y = self.hill_amp * math.sin(self.hill_freq * self.position)
        ax.scatter([self.position], [car_y], s=80, color="red")
        ax.set_xlim(self.position_min, self.position_max)
        # вертикальные границы немного выше амплитуды, чтобы было место
        amp = abs(self.hill_amp)
        ax.set_ylim(-amp * 1.2, amp * 1.2)
        ax.set_xlabel("Позиция")
        ax.set_ylabel("Высота")
        ax.set_title(f"pos={self.position:.3f}, vel={self.velocity:.3f}")
        if show:
            _plt.pause(0.001)
        return ax


class ReplayBuffer:
    """Буфер воспоминаний для хранения и выборки переходов.

    Данные хранятся в списках, выборка возвращается в виде
    тензоров PyTorch.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.states)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
        else:
            self.states[self.position] = state
            self.actions[self.position] = action
            self.rewards[self.position] = reward
            self.next_states[self.position] = next_state
            self.dones[self.position] = done
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(self.states) >= batch_size, "Недостаточно элементов в буфере"
        indices = random.sample(range(len(self.states)), batch_size)
        states = torch.tensor(
            np.array([self.states[i] for i in indices]), dtype=torch.float32
        )
        actions = torch.tensor([self.actions[i] for i in indices], dtype=torch.int64)
        rewards = torch.tensor([self.rewards[i] for i in indices], dtype=torch.float32)
        next_states = torch.tensor(
            np.array([self.next_states[i] for i in indices]), dtype=torch.float32
        )
        dones = torch.tensor([self.dones[i] for i in indices], dtype=torch.float32)
        return states, actions, rewards, next_states, dones


class QNetwork(nn.Module):
    """Полносвязная сеть для аппроксимации Q(s,a)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # инициализация
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)
        # выходной слой инициализируем маленькими значениями
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Агент Double DQN с ε-жадной политикой и мягким обновлением."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        replay_capacity: int,
        batch_size: int,
        gamma: float,
        lr: float,
        eps_start: float,
        eps_end: float,
        eps_decay_steps: int,
        tau: float,
        huber_delta: float,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.tau = tau
        self.huber_delta = huber_delta
        self.device = device
        self.rng = random.Random(seed)
        # сети
        self.q_online = QNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.q_target = QNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()
        # оптимизатор
        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=self.lr)
        # буфер
        self.replay = ReplayBuffer(replay_capacity)
        # счётчик шагов
        self.steps_done = 0

    def select_action(self, state: np.ndarray) -> int:
        eps_threshold = max(
            self.eps_end,
            self.eps_start
            - (self.eps_start - self.eps_end)
            * (self.steps_done / self.eps_decay_steps),
        )
        self.steps_done += 1
        if self.rng.random() < eps_threshold:
            # randrange выдаёт 0..action_dim-1
            return self.rng.randrange(self.action_dim)
        # greedy действие
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_online(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(state, action, reward, next_state, done)

    def update(self) -> None:
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        # Q(s,a)
        q_values = self.q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_online(next_states).argmax(dim=1)
            next_q_target = (
                self.q_target(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
        targets = rewards + self.gamma * (1.0 - dones) * next_q_target
        # Huber loss
        loss = F.smooth_l1_loss(q_values, targets, beta=self.huber_delta)
        # оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_online.parameters(), max_norm=10.0)
        self.optimizer.step()
        # мягкое обновление целевой сети
        with torch.no_grad():
            for target_param, online_param in zip(
                self.q_target.parameters(), self.q_online.parameters()
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * online_param.data)

    def save(self, path: str) -> None:
        torch.save(self.q_online.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.q_online.load_state_dict(state_dict)
        self.q_target.load_state_dict(state_dict)


def train_agent(
    env: MountainCarEnv,
    agent: DQNAgent,
    episodes: int,
    initial_random_steps: int = 1000,
    ma_window: int = 50,
) -> Tuple[List[float], List[int], List[int], List[float]]:
    """Обучает агента в течение указанного числа эпизодов.

    Эта функция возвращает четыре списка одинаковой длины:
    * returns — суммарная награда за эпизод (cumulative reward);
    * lengths — длина эпизода (количество шагов);
    * successes — индикатор успеха (1, если цель достигнута до max_steps, иначе 0);
    * moving_avgs — скользящее среднее суммарной награды по окну ma_window.

    :param env: среда
    :param agent: агент
    :param episodes: количество эпизодов обучения
    :param initial_random_steps: количество случайных шагов для предварительного заполнения буфера
    :param ma_window: размер окна для расчёта скользящего среднего
    :returns: (returns, lengths, successes, moving_avgs)
    """
    returns: List[float] = []
    lengths: List[int] = []
    successes: List[int] = []
    moving_avgs: List[float] = []
    # разогрев буфера
    state = env.reset()
    for _ in range(initial_random_steps):
        action = agent.rng.randrange(agent.action_dim)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()
    # цикл обучения
    for ep in range(episodes):
        state = env.reset()
        ep_return = 0.0
        ep_len = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            ep_return += reward
            ep_len += 1
        returns.append(ep_return)
        lengths.append(ep_len)
        # успех, если завершение произошло до истечения max_steps
        success_flag = 1 if ep_len < env.max_steps else 0
        successes.append(success_flag)
        # скользящее среднее
        window = returns[-ma_window:]
        moving_avgs.append(sum(window) / len(window))
        # периодический вывод прогресса
        if (ep + 1) % max(1, episodes // 10) == 0:
            recent = returns[-10:] if len(returns) >= 10 else returns
            avg_ret = sum(recent) / len(recent)
            eps_threshold = max(
                agent.eps_end,
                agent.eps_start
                - (agent.eps_start - agent.eps_end)
                * (agent.steps_done / agent.eps_decay_steps),
            )
            print(
                f"Episode {ep + 1}/{episodes}: avg_return={avg_ret:.2f}, steps={ep_len}, eps={eps_threshold:.3f}"
            )
    return returns, lengths, successes, moving_avgs


def evaluate_agent(
    env: MountainCarEnv, agent: DQNAgent, episodes: int
) -> Tuple[float, float, float]:
    """Оценивает агента на эпизодах без исследования.

    Возвращает среднюю суммарную награду, среднее число шагов до завершения,
    а также долю успешных эпизодов (успех — достижение цели до max_steps).
    """
    total_return = 0.0
    total_steps = 0
    successes = 0
    for _ in range(episodes):
        state = env.reset()
        ep_return = 0.0
        ep_len = 0
        done = False
        while not done:
            state_t = torch.tensor(
                state, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)
            with torch.no_grad():
                action = int(torch.argmax(agent.q_online(state_t), dim=1).item())
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_return += reward
            ep_len += 1
        total_return += ep_return
        total_steps += ep_len
        if ep_len < env.max_steps:
            successes += 1
    avg_return = total_return / episodes
    avg_steps = total_steps / episodes
    success_rate = successes / episodes
    return avg_return, avg_steps, success_rate


def save_demo_gif(
    env: MountainCarEnv,
    agent: DQNAgent,
    path: str = "mountaincar_demo.gif",
    fps: int = 30,
) -> None:
    """Сохраняет GIF с демонстрацией работы агента."""
    fig, ax = plt.subplots()
    state = env.reset()
    done = False

    def update_frame(_: int):
        nonlocal state, done
        env.render(ax=ax, show=False)
        if done:
            return []
        state_t = torch.tensor(
            state, dtype=torch.float32, device=agent.device
        ).unsqueeze(0)
        with torch.no_grad():
            action = int(torch.argmax(agent.q_online(state_t), dim=1).item())
        next_state, _, done, _ = env.step(action)
        state = next_state
        return []

    ani = animation.FuncAnimation(
        fig, update_frame, frames=env.max_steps, interval=1000 / fps, blit=False
    )
    ani.save(path, writer="pillow", fps=fps)
    print(f"GIF сохранён в {path}")


# Дополнительные функции для анализа


def compute_sample_efficiency(
    moving_avgs: List[float], threshold: float
) -> Optional[int]:
    """Возвращает номер эпизода, на котором скользящее среднее суммарной награды
    впервые достигло (или превысило) заданный порог.  Если порог не достигнут,
    возвращает None.

    :param moving_avgs: список скользящих средних суммарной награды
    :param threshold: значение порога (например, -110 для MountainCar)
    :returns: индекс (1-based) эпизода или None
    """
    for idx, val in enumerate(moving_avgs, start=1):
        if val >= threshold:
            return idx
    return None


def save_metrics_csv(
    path: str,
    returns: List[float],
    lengths: List[int],
    successes: List[int],
    moving_avgs: List[float],
) -> None:
    """Сохраняет метрики обучения в CSV-файл.

    Формат: episode,return,steps,success,moving_avg
    """
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "steps", "success", "moving_avg"])
        for i, (r, l, s, ma) in enumerate(
            zip(returns, lengths, successes, moving_avgs), start=1
        ):
            writer.writerow([i, r, l, s, ma])


def plot_training_curve(
    returns: List[float],
    plot_path: str,
    window: int = 10,
    show: bool = False,
) -> None:
    """Строит и сохраняет график суммарной награды."""
    if len(returns) == 0:
        return
    episodes = np.arange(len(returns))
    returns = np.array(returns)
    # скользящее среднее
    if len(returns) >= window:
        smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
        smooth_x = np.arange(len(smooth)) + (window - 1) / 2
    else:
        smooth = returns
        smooth_x = episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, returns, label="Return per episode", alpha=0.4)
    plt.plot(smooth_x, smooth, label=f"Smoothed (window={window})", linewidth=2.0)
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards")
    plt.title("Training curve: MountainCar Double DQN")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    if show:
        plt.show()
    print(f"График обучения сохранён в {plot_path}")


# Новая функция: строит наложенные кривые обучения для нескольких запусков
def plot_training_overlay(
    per_seed_returns: List[List[float]],
    aggregated_returns: List[float],
    plot_path: str,
    window: int = 10,
    seed_labels: Optional[List[str]] = None,
    show: bool = False,
) -> None:
    """Строит график со всеми кривыми обучения и усреднённой кривой.

    :param per_seed_returns: список списков суммарной награды (по сид-ам)
    :param aggregated_returns: усреднённые значения награды
    :param plot_path: путь для сохранения графика
    :param window: размер окна сглаживания
    :param seed_labels: метки для подписей (по умолчанию индекс запуска)
    :param show: показать график на экране
    """
    import numpy as _np

    if not per_seed_returns:
        return
    episodes = _np.arange(len(per_seed_returns[0]))
    plt.figure(figsize=(10, 6))
    # Цветовая карта для разных сидов
    from itertools import cycle

    color_cycle = cycle(plt.cm.tab10.colors)
    # Рисуем кривые по сид-ам
    for idx, ret in enumerate(per_seed_returns):
        color = next(color_cycle)
        # сглаживаем
        ret_arr = _np.array(ret)
        if len(ret_arr) >= window:
            smooth = _np.convolve(ret_arr, _np.ones(window) / window, mode="valid")
            smooth_x = _np.arange(len(smooth)) + (window - 1) / 2
        else:
            smooth = ret_arr
            smooth_x = episodes
        label = (
            seed_labels[idx]
            if seed_labels and idx < len(seed_labels)
            else f"Run {idx + 1}"
        )
        plt.plot(smooth_x, smooth, label=label, alpha=0.5, linewidth=1.5)
    # Рисуем усреднённую кривую
    if aggregated_returns:
        agg_arr = _np.array(aggregated_returns)
        if len(agg_arr) >= window:
            agg_smooth = _np.convolve(agg_arr, _np.ones(window) / window, mode="valid")
            agg_x = _np.arange(len(agg_smooth)) + (window - 1) / 2
        else:
            agg_smooth = agg_arr
            agg_x = episodes
        plt.plot(agg_x, agg_smooth, label="Average", color="black", linewidth=2.5)
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards")
    plt.title("Training curves: overlay per run and average")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    if show:
        plt.show()
    print(f"График наложенных кривых сохранён в {plot_path}")


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Double DQN с расширяемой средой MountainCar"
    )
    # Общие параметры
    parser.add_argument(
        "--episodes", type=int, default=500, help="Количество эпизодов для обучения"
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=20, help="Количество эпизодов для оценки"
    )
    parser.add_argument("--seed", type=int, default=42, help="Начальный seed")
    # Параметры окружения
    parser.add_argument(
        "--position_min", type=float, default=-1.2, help="Минимальная позиция"
    )
    parser.add_argument(
        "--position_max", type=float, default=0.6, help="Максимальная позиция"
    )
    parser.add_argument(
        "--velocity_min", type=float, default=-0.07, help="Минимальная скорость"
    )
    parser.add_argument(
        "--velocity_max", type=float, default=0.07, help="Максимальная скорость"
    )
    parser.add_argument(
        "--start_min", type=float, default=-0.6, help="Нижняя граница стартовой позиции"
    )
    parser.add_argument(
        "--start_max",
        type=float,
        default=-0.4,
        help="Верхняя граница стартовой позиции",
    )
    parser.add_argument("--goal_position", type=float, default=0.5, help="Позиция цели")
    parser.add_argument("--force", type=float, default=0.001, help="Сила мотора")
    parser.add_argument(
        "--gravity", type=float, default=0.0025, help="Сила гравитации (масштаб уклона)"
    )
    parser.add_argument(
        "--max_steps", type=int, default=200, help="Максимальное число шагов в эпизоде"
    )
    parser.add_argument(
        "--hill_freq", type=float, default=3.0, help="Частота холмов (частота синуса)"
    )
    parser.add_argument("--hill_amp", type=float, default=1.0, help="Амплитуда холмов")
    parser.add_argument(
        "--slope_gain",
        type=float,
        default=None,
        help="Коэффициент наклона: по умолчанию (None) равен 1.0, что повторяет стандартную динамику. "
        "Установите другое значение для усиления или ослабления влияния уклона.",
    )
    # Параметры DQN
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Размер скрытых слоёв"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Скорость обучения")
    parser.add_argument("--gamma", type=float, default=0.99, help="Дисконт-фактор")
    parser.add_argument("--batch_size", type=int, default=64, help="Размер мини-батча")
    parser.add_argument(
        "--replay_capacity",
        type=int,
        default=50_000,
        help="Ёмкость буфера воспоминаний",
    )
    parser.add_argument(
        "--eps_start", type=float, default=1.0, help="Начальное значение epsilon"
    )
    parser.add_argument(
        "--eps_end", type=float, default=0.05, help="Минимальное значение epsilon"
    )
    parser.add_argument(
        "--eps_decay_steps",
        type=int,
        default=20_000,
        help="Количество шагов для линейного спада epsilon",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Коэффициент Polyak для обновления целевой сети",
    )
    parser.add_argument(
        "--huber_delta", type=float, default=1.0, help="Параметр дельта для Huber-loss"
    )
    parser.add_argument(
        "--initial_random_steps",
        type=int,
        default=1000,
        help="Количество случайных шагов для заполнения буфера",
    )
    parser.add_argument(
        "--ma_window",
        type=int,
        default=50,
        help="Размер окна для скользящего среднего награды",
    )
    parser.add_argument(
        "--solved_threshold",
        type=float,
        default=-110.0,
        help="Порог для sample efficiency (когда moving_avg ≥ threshold)",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="metrics.csv",
        help="Путь для сохранения CSV с метриками обучения",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Список начальных сидов, разделённых запятыми (для многократных запусков)",
    )
    # Параметры вывода
    parser.add_argument(
        "--render_demo", action="store_true", help="Сохранить GIF-демо после обучения"
    )
    parser.add_argument(
        "--gif_path",
        type=str,
        default="mountaincar_demo.gif",
        help="Путь для GIF-демонстрации",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS для GIF")
    parser.add_argument(
        "--plot_path",
        type=str,
        default="training_curve.png",
        help="Путь для графика обучения",
    )
    parser.add_argument(
        "--show_plot", action="store_true", help="Отображать график на экране"
    )
    # Устройство
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Устройство для вычислений",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Путь к сохранённым весам (.pth). Если задан — обучать не будем, а загрузим модель."
    )

    return parser.parse_args()


def choose_device(device_str: str) -> torch.device:
    """Выбирает устройство на основе строки (auto/cpu/cuda/mps)."""
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "mps":
        # для Apple Silicon
        return torch.device(
            "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    print(f"Используем устройство: {device}")
    # подготовка списка сидов
    if args.seeds is not None and args.seeds.strip():
        # парсим список целых чисел
        seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seed_list = [args.seed]

    aggregated_eval_returns: List[float] = []
    aggregated_eval_steps: List[float] = []
    aggregated_eval_success_rates: List[float] = []
    aggregated_sample_efficiencies: List[int] = []
    # будем хранить списки возвратов, скользящих средних, длины эпизодов и флаги успеха
    # для каждого сидa, чтобы потом построить общий график обучения и агрегированные метрики.
    per_seed_returns: List[List[float]] = []
    per_seed_moving_avgs: List[List[float]] = []
    per_seed_lengths: List[List[int]] = []
    per_seed_successes: List[List[int]] = []

    # запуск для каждого сида
    for idx, seed in enumerate(seed_list):
        print(f"\n==== Запуск {idx + 1}/{len(seed_list)} (seed={seed}) ====")
        # инициализация среды и агента
        rng = np.random.RandomState(seed)
        env = MountainCarEnv(
            position_range=(args.position_min, args.position_max),
            velocity_range=(args.velocity_min, args.velocity_max),
            start_range=(args.start_min, args.start_max),
            goal_position=args.goal_position,
            force=args.force,
            gravity=args.gravity,
            max_steps=args.max_steps,
            hill_freq=args.hill_freq,
            hill_amp=args.hill_amp,
            slope_gain=args.slope_gain,
            rng=rng,
        )
        agent = DQNAgent(
            state_dim=2,
            action_dim=3,
            hidden_dim=args.hidden_size,
            replay_capacity=args.replay_capacity,
            batch_size=args.batch_size,
            gamma=args.gamma,
            lr=args.lr,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay_steps=args.eps_decay_steps,
            tau=args.tau,
            huber_delta=args.huber_delta,
            device=device,
            seed=seed,
        )
        if args.load_path:
            agent.load(args.load_path)
            print(f"Модель загружена из {args.load_path}")
            # если episodes > 0, можно дообучить; если 0, обучение будет пропущено
        # обучение
        returns, lengths, successes, moving_avgs = train_agent(
            env,
            agent,
            args.episodes,
            initial_random_steps=args.initial_random_steps,
            ma_window=args.ma_window,
        )
        # sample efficiency
        sample_eff = compute_sample_efficiency(moving_avgs, args.solved_threshold)
        sample_eff_str = (
            str(sample_eff)
            if sample_eff is not None
            else f">= {args.episodes} (порог не достигнут)"
        )
        # оценка
        eval_ret, eval_steps, eval_success_rate = evaluate_agent(
            env, agent, args.eval_episodes
        )
        print(
            f"Оценка (seed={seed}) на {args.eval_episodes} эпизодах: средняя награда = {eval_ret:.2f}, "
            f"среднее число шагов = {eval_steps:.2f}, успех = {eval_success_rate * 100:.1f}%"
        )
        print(
            f"Sample efficiency (episode где moving_avg ≥ {args.solved_threshold:.2f}) : {sample_eff_str}"
        )
        # добавляем в агрегаторы
        aggregated_eval_returns.append(eval_ret)
        aggregated_eval_steps.append(eval_steps)
        aggregated_eval_success_rates.append(eval_success_rate)
        if sample_eff is not None:
            aggregated_sample_efficiencies.append(sample_eff)
        # сохранение индивидуальной модели
        model_path = (
            f"dqn_mountaincar_full_seed{seed}.pth"
            if len(seed_list) > 1
            else "dqn_mountaincar_full.pth"
        )
        agent.save(model_path)
        print(f"Модель сохранена в {model_path}")
        # сохранение метрик
        metrics_path = (
            args.metrics_path.replace(".csv", f"_seed{seed}.csv")
            if len(seed_list) > 1
            else args.metrics_path
        )
        save_metrics_csv(metrics_path, returns, lengths, successes, moving_avgs)
        print(f"Метрики сохранены в {metrics_path}")
        # график обучения
        if args.plot_path:
            plot_path = (
                args.plot_path.replace(".png", f"_seed{seed}.png")
                if len(seed_list) > 1
                else args.plot_path
            )
            plot_training_curve(
                returns, plot_path, window=args.ma_window, show=args.show_plot
            )
        # накапливаем для общего графика и агрегированных метрик
        per_seed_returns.append(returns)
        per_seed_moving_avgs.append(moving_avgs)
        per_seed_lengths.append(lengths)
        per_seed_successes.append(successes)
        # GIF
        if args.render_demo:
            gif_path = (
                args.gif_path.replace(".gif", f"_seed{seed}.gif")
                if len(seed_list) > 1
                else args.gif_path
            )
            save_demo_gif(env, agent, path=gif_path, fps=args.fps)
    # вывод агрегированных результатов, если несколько сидов
    if len(seed_list) > 1:
        import statistics

        def mean_std(values: List[float]) -> Tuple[float, float]:
            return (
                sum(values) / len(values),
                statistics.stdev(values) if len(values) > 1 else 0.0,
            )

        mean_ret, std_ret = mean_std(aggregated_eval_returns)
        mean_steps, std_steps = mean_std(aggregated_eval_steps)
        mean_succ, std_succ = mean_std(aggregated_eval_success_rates)
        if aggregated_sample_efficiencies:
            mean_seff, std_seff = mean_std(aggregated_sample_efficiencies)
            seff_info = f"{mean_seff:.1f} ± {std_seff:.1f} эпизодов"
        else:
            seff_info = "порог не достигнут ни в одном запуске"
        print("\n==== Сводная статистика по сид-ам ====")
        print(
            f"Средняя награда (eval): {mean_ret:.2f} ± {std_ret:.2f}; "
            f"Среднее число шагов (eval): {mean_steps:.2f} ± {std_steps:.2f}; "
            f"Успех: {mean_succ * 100:.1f}% ± {std_succ * 100:.1f}%"
        )
        print(
            f"Sample efficiency (эпизод где moving_avg ≥ {args.solved_threshold:.2f}): {seff_info}"
        )

        # построение общего графика обучения (усредняем кривые по сид-ам)
        try:
            import numpy as _np

            # проверяем, что каждая кривая имеет длину, равную числу эпизодов
            if all(len(lst) == args.episodes for lst in per_seed_returns):
                all_returns = _np.array(per_seed_returns)  # (n_seeds, episodes)
                aggregated_returns = all_returns.mean(axis=0).tolist()
                all_moving = _np.array(per_seed_moving_avgs)
                aggregated_moving_avg = all_moving.mean(axis=0).tolist()
                all_lengths_arr = _np.array(per_seed_lengths)
                aggregated_lengths = all_lengths_arr.mean(axis=0).tolist()
                all_success_arr = _np.array(per_seed_successes)
                aggregated_successes = all_success_arr.mean(axis=0).tolist()
                # сохранение CSV и графика
                if args.metrics_path:
                    agg_metrics_path = args.metrics_path.replace(
                        ".csv", "_aggregated.csv"
                    )
                    save_metrics_csv(
                        agg_metrics_path,
                        aggregated_returns,
                        aggregated_lengths,
                        aggregated_successes,
                        aggregated_moving_avg,
                    )
                    print(f"Агрегированные метрики сохранены в {agg_metrics_path}")
                if args.plot_path:
                    # сохраняем график усреднённой кривой
                    agg_plot_path = args.plot_path.replace(".png", "_aggregated.png")
                    plot_training_curve(
                        aggregated_returns,
                        agg_plot_path,
                        window=args.ma_window,
                        show=args.show_plot,
                    )
                    print(f"Общий график обучения сохранён в {agg_plot_path}")
                    # сохраняем график наложенных кривых
                    overlay_plot_path = args.plot_path.replace(".png", "_overlay.png")
                    # метки для сидов — строки seed_list преобразуем в str
                    seed_labels = [str(s) for s in seed_list]
                    plot_training_overlay(
                        per_seed_returns,
                        aggregated_returns,
                        overlay_plot_path,
                        window=args.ma_window,
                        seed_labels=seed_labels,
                        show=args.show_plot,
                    )
        except Exception as e:
            print(f"Не удалось построить агрегированный график: {e}")


if __name__ == "__main__":
    main()
