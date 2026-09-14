"""
Полная и расширяемая реализация обучения агента Double DQN в среде
MountainCar с использованием PyTorch.  Скрипт предоставляет
параметризуемую среду, гибкие настройки гиперпараметров, сохранение
графиков обучения и генерацию GIF-демонстрации выученной стратегии.

## Основные возможности

* **Параметризуемая среда**: можно изменять диапазоны позиции и
  скорости, стартовый диапазон, положение цели, силу мотора, силу
  «уклона», максимальную длину эпизода, частоту и амплитуду холмов,
  а также коэффициент `slope_gain`, влияющий на силу гравитационного
  компонента.  По умолчанию `slope_gain` равен 1.0, что воспроизводит
  стандартную реализацию MountainCar (из OpenAI Gym) — изменение
  амплитуды и частоты холмов не влияет на «силу» наклона.  При желании
  можно задать собственное значение `slope_gain` для усложнения или
  упрощения динамики.

* **Double DQN**: обучение с использованием буфера воспоминаний,
  целевой сети, ε-жадной политики и мягкого обновления.  Вставлена
  защита от переоценки (Double DQN) и Huber-loss для устойчивости.

* **Графики обучения**: после завершения обучения скрипт строит
  кривую суммарной награды по эпизодам и её скользящее среднее,
  сохраняет график в PNG и выводит его на экран (при наличии GUI).

* **Генерация GIF**: по желанию можно сохранить GIF-анимацию,
  демонстрирующую, как обученный агент решает задачу.

* **Запуск из командной строки**: все параметры (окружения и
  обучения) задаются через аргументы CLI; также можно выбрать
  устройство (CPU, CUDA, MPS) или позволить скрипту определить его
  автоматически.

### Пример использования

```bash
python3 mountaincar_dqn_full.py \
  --episodes 500 --eval_episodes 20 --seed 42 \
  --hill_freq 3.0 --hill_amp 1.0 \
  --force 0.001 --gravity 0.0025 \
  --render_demo --gif_path mc_final_demo.gif --plot_path mc_training_curve.png
```

По окончании работы в консоли будет выведена статистика (средняя
награда и число шагов на оценочных эпизодах), файл GIF с траекторией
сохранится в `mc_final_demo.gif`, а график обучения — в
`mc_training_curve.png`.
"""

from __future__ import annotations
"""
Это адаптированная версия скрипта Double DQN для классической среды
MountainCar из библиотеки Gym/Gymnasium.  В отличие от варианта с
кастомной средой, здесь используется готовая среда ``MountainCar-v0``
с фиксированной динамикой (сила мотора, гравитация, диапазоны
состояний и т.д.).  Скрипт сохраняет все ключевые возможности: он
поддерживает гибкие настройки гиперпараметров, собирает и сохраняет
метрики обучения, строит график кривой обучения, вычисляет sample
efficiency и по желанию сохраняет GIF-демонстрацию.

Если вы передаёте флаг ``--render_demo``, среда будет создана с
``render_mode='rgb_array'``, чтобы можно было получить кадры для
анимации.  Учтите, что готовая среда не позволяет изменять
физические параметры (амплитуду холмов, силу мотора, гравитацию и
т.п.), поэтому эти аргументы CLI будут игнорироваться.
"""

import argparse
import math
import random
from typing import List, Tuple, Optional

import matplotlib

# Используем бэкенд без GUI для сохранения графиков и GIF
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Попытка импортировать gymnasium, при отсутствии — gym
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym  # type: ignore
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False

# -----------------------------------------------------------------------------
# Определение основных компонентов: реплей-буфер, нейросеть и агент.
# Среда MountainCar будет создана в ``main()`` с помощью gym.make.

# -----------------------------------------------------------------------------
# Обратите внимание: класс ``MountainCarEnv`` из кастомной реализации
# удалён.  В этой версии скрипта используется стандартная среда
# ``MountainCar-v0`` из Gym/Gymnasium.  Аргументы, связанные с
# параметризацией холмов (freq, amp, force, gravity и пр.), будут
# проигнорированы, поскольку готовая среда использует фиксированные
# значения: сила мотора = 0.001, гравитация = 0.0025, начальный
# диапазон позиции = [-0.6, -0.4], макс. длина эпизода = 200 шагов и
# цель по позиции = 0.5【73180758583950†L224-L255】.


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

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
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

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(self.states) >= batch_size, "Недостаточно элементов в буфере"
        indices = random.sample(range(len(self.states)), batch_size)
        states = torch.tensor(np.array([self.states[i] for i in indices]), dtype=torch.float32)
        actions = torch.tensor([self.actions[i] for i in indices], dtype=torch.int64)
        rewards = torch.tensor([self.rewards[i] for i in indices], dtype=torch.float32)
        next_states = torch.tensor(np.array([self.next_states[i] for i in indices]), dtype=torch.float32)
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
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
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
        device: torch.device = torch.device('cpu'),
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
            self.eps_start - (self.eps_start - self.eps_end) * (self.steps_done / self.eps_decay_steps),
        )
        self.steps_done += 1
        if self.rng.random() < eps_threshold:
            # randrange выдаёт 0..action_dim-1
            return self.rng.randrange(self.action_dim)
        # greedy действие
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_online(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay.add(state, action, reward, next_state, done)

    def update(self) -> None:
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        # Q(s,a)
        q_values = self.q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_online(next_states).argmax(dim=1)
            next_q_target = self.q_target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
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
            for target_param, online_param in zip(self.q_target.parameters(), self.q_online.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * online_param.data)

    def save(self, path: str) -> None:
        torch.save(self.q_online.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.q_online.load_state_dict(state_dict)
        self.q_target.load_state_dict(state_dict)


def train_agent(
    env: gym.Env,
    agent: DQNAgent,
    episodes: int,
    initial_random_steps: int = 1000,
    ma_window: int = 50,
) -> Tuple[List[float], List[int], List[int], List[float]]:
    """Обучает агента в течение указанного числа эпизодов на стандартной среде Gym.

    Возвращаются четыре списка одинаковой длины:
    * ``returns`` — суммарная награда за эпизод;
    * ``lengths`` — длина эпизода (количество шагов);
    * ``successes`` — индикатор достижения цели (1, если агент достиг флага до истечения лимита, иначе 0);
    * ``moving_avgs`` — скользящее среднее суммарной награды (окно ``ma_window``).

    При работе с Gym/Gymnasium необходимо корректно обрабатывать сигнатуры методов
    ``reset`` и ``step``: ``reset`` может возвращать кортеж (obs, info), а
    ``step`` — пятерку (obs, reward, terminated, truncated, info).  Здесь
    поддерживаются оба варианта.
    """
    returns: List[float] = []
    lengths: List[int] = []
    successes: List[int] = []
    moving_avgs: List[float] = []
    # начальный разогрев буфера случайными действиями
    reset_result = env.reset()
    # ``reset`` может возвращать (obs, info) — выделяем наблюдение
    state = reset_result[0] if isinstance(reset_result, tuple) else reset_result  # type: ignore
    for _ in range(initial_random_steps):
        action = agent.rng.randrange(agent.action_dim)
        step_result = env.step(action)
        # поддерживаем старые и новые сигнатуры
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _info = step_result  # type: ignore
            done = bool(terminated or truncated)
        else:
            next_state, reward, done, _info = step_result  # type: ignore
        agent.store_transition(state, action, reward, next_state, done)
        if done:
            reset_result = env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result  # type: ignore
        else:
            state = next_state
    # цикл обучения
    for ep in range(episodes):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result  # type: ignore
        ep_return = 0.0
        ep_len = 0
        done = False
        terminated_flag = False
        while not done:
            action = agent.select_action(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _info = step_result  # type: ignore
                done = bool(terminated or truncated)
                terminated_flag = bool(terminated)
            else:
                next_state, reward, done, _info = step_result  # type: ignore
                terminated_flag = bool(done)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            ep_return += reward
            ep_len += 1
        returns.append(ep_return)
        lengths.append(ep_len)
        # успех — если эпизод завершился достижением цели (terminated), а не тайм-аутом
        success_flag = 1 if terminated_flag else 0
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
                agent.eps_start - (agent.eps_start - agent.eps_end) * (agent.steps_done / agent.eps_decay_steps),
            )
            print(
                f"Episode {ep + 1}/{episodes}: avg_return={avg_ret:.2f}, steps={ep_len}, eps={eps_threshold:.3f}"
            )
    return returns, lengths, successes, moving_avgs


def evaluate_agent(env: gym.Env, agent: DQNAgent, episodes: int) -> Tuple[float, float, float]:
    """Оценивает агента на эпизодах без исследования.

    Возвращает среднюю суммарную награду, среднее число шагов до завершения
    (до терминального состояния или тайм-аута) и долю эпизодов, где цель
    была достигнута (terminated).  Работает как с Gym, так и с Gymnasium.
    """
    total_return: float = 0.0
    total_steps: int = 0
    successes: int = 0
    for _ in range(episodes):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result  # type: ignore
        ep_return: float = 0.0
        ep_len: int = 0
        done = False
        terminated_flag = False
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                action = int(torch.argmax(agent.q_online(state_t), dim=1).item())
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _info = step_result  # type: ignore
                done = bool(terminated or truncated)
                terminated_flag = bool(terminated)
            else:
                next_state, reward, done, _info = step_result  # type: ignore
                terminated_flag = bool(done)
            state = next_state
            ep_return += reward
            ep_len += 1
        total_return += ep_return
        total_steps += ep_len
        if terminated_flag:
            successes += 1
    avg_return = total_return / episodes
    avg_steps = total_steps / episodes
    success_rate = successes / episodes
    return avg_return, avg_steps, success_rate


def save_demo_gif(
    env: gym.Env,
    agent: DQNAgent,
    path: str = "mountaincar_demo.gif",
    fps: int = 30,
) -> None:
    """Сохраняет GIF с демонстрацией работы агента в стандартной среде Gym.

    При создании среды в ``main()`` необходимо указать ``render_mode='rgb_array'``.
    ``env.render()`` тогда будет возвращать RGB-кадр.  Эта функция
    собирает кадры во время прохождения одного эпизода жадной политикой
    агента и сохраняет их в GIF.
    """
    # Список кадров
    frames: List[np.ndarray] = []
    reset_result = env.reset()
    state = reset_result[0] if isinstance(reset_result, tuple) else reset_result  # type: ignore
    done = False
    # Проходим эпизод жадной политикой (без исследования)
    while not done:
        # Получаем текущий кадр.  Метод render() возвращает None, если render_mode None
        frame = env.render()  # type: ignore
        if frame is not None:
            frames.append(frame)
        # выбираем greedy-действие
        state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            action = int(torch.argmax(agent.q_online(state_t), dim=1).item())
        # совершаем шаг
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, _r, terminated, truncated, _info = step_result  # type: ignore
            done = bool(terminated or truncated)
        else:
            next_state, _r, done, _info = step_result  # type: ignore
        state = next_state
    # Добавляем последний кадр после завершения
    frame = env.render()  # type: ignore
    if frame is not None:
        frames.append(frame)
    if len(frames) == 0:
        print("Предупреждение: среда не возвращает кадры (render_mode=None). GIF не сохранён.")
        return
    # Создаём анимацию
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])
    ax.axis('off')
    def update(i: int):
        im.set_data(frames[i])
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)
    ani.save(path, writer='pillow', fps=fps)
    print(f"GIF сохранён в {path}")


# Дополнительные функции для анализа

def compute_sample_efficiency(moving_avgs: List[float], threshold: float) -> Optional[int]:
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
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'return', 'steps', 'success', 'moving_avg'])
        for i, (r, l, s, ma) in enumerate(zip(returns, lengths, successes, moving_avgs), start=1):
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
        smooth = np.convolve(returns, np.ones(window) / window, mode='valid')
        smooth_x = np.arange(len(smooth)) + (window - 1) / 2
    else:
        smooth = returns
        smooth_x = episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, returns, label='Return per episode', alpha=0.4)
    plt.plot(smooth_x, smooth, label=f'Smoothed (window={window})', linewidth=2.0)
    plt.xlabel('Episode')
    plt.ylabel('Sum of rewards')
    plt.title('Training curve: MountainCar Double DQN')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    if show:
        plt.show()
    print(f"График обучения сохранён в {plot_path}")


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки для работы с классической Gym-средой."""
    parser = argparse.ArgumentParser(
        description=(
            "Double DQN для классической среды MountainCar из Gym/Gymnasium. "
            "Физические параметры среды (сила мотора, гравитация, профиль холмов) "
            "заданы фиксированными значениями и не могут быть изменены через CLI."
        )
    )
    # Обучение и оценка
    parser.add_argument('--episodes', type=int, default=500, help='Количество эпизодов для обучения')
    parser.add_argument('--eval_episodes', type=int, default=20, help='Количество эпизодов для оценки')
    parser.add_argument('--seed', type=int, default=42, help='Начальный seed')
    parser.add_argument('--seeds', type=str, default=None, help='Список начальных сидов, разделённых запятыми (для многократных запусков)')
    # Параметр max_steps для обёртки TimeLimit: при ненулевом значении заменяет стандартный лимит (200)
    parser.add_argument('--max_steps', type=int, default=200, help='Лимит шагов в эпизоде (обёртка TimeLimit)')
    # Параметры DQN
    parser.add_argument('--hidden_size', type=int, default=128, help='Размер скрытых слоёв нейросети')
    parser.add_argument('--lr', type=float, default=1e-3, help='Скорость обучения (learning rate)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Дисконт-фактор γ')
    parser.add_argument('--batch_size', type=int, default=64, help='Размер мини-батча для обучения')
    parser.add_argument('--replay_capacity', type=int, default=50_000, help='Ёмкость буфера воспоминаний')
    parser.add_argument('--eps_start', type=float, default=1.0, help='Начальное значение ε')
    parser.add_argument('--eps_end', type=float, default=0.05, help='Минимальное значение ε')
    parser.add_argument('--eps_decay_steps', type=int, default=20_000, help='Количество шагов для линейного спада ε')
    parser.add_argument('--tau', type=float, default=0.005, help='Коэффициент Polyak для обновления целевой сети')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='Параметр δ для Huber-loss')
    parser.add_argument('--initial_random_steps', type=int, default=1000, help='Количество случайных шагов для заполнения буфера')
    parser.add_argument('--ma_window', type=int, default=50, help='Размер окна для скользящего среднего награды')
    parser.add_argument('--solved_threshold', type=float, default=-110.0, help='Порог для sample efficiency (когда moving_avg ≥ threshold)')
    parser.add_argument('--metrics_path', type=str, default='metrics.csv', help='Путь для сохранения CSV с метриками обучения')
    # Визуализация
    parser.add_argument('--render_demo', action='store_true', help='Сохранить GIF-демо после обучения')
    parser.add_argument('--gif_path', type=str, default='mountaincar_demo.gif', help='Путь для GIF-демонстрации')
    parser.add_argument('--fps', type=int, default=30, help='FPS для GIF-анимации')
    parser.add_argument('--plot_path', type=str, default='training_curve.png', help='Путь для графика обучения')
    parser.add_argument('--show_plot', action='store_true', help='Отобразить график на экране')
    # Устройство
    parser.add_argument('--device', type=str, default='cpu', choices=['auto', 'cpu', 'cuda', 'mps'], help='Устройство для вычислений')
    return parser.parse_args()


def choose_device(device_str: str) -> torch.device:
    """Выбирает устройство на основе строки (auto/cpu/cuda/mps)."""
    if device_str == 'cpu':
        return torch.device('cpu')
    if device_str == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_str == 'mps':
        # для Apple Silicon
        return torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
    # auto
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    print(f"Используем устройство: {device}")
    # подготовка списка сидов
    if args.seeds is not None and args.seeds.strip():
        # парсим список целых чисел
        seed_list = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    else:
        seed_list = [args.seed]

    aggregated_eval_returns: List[float] = []
    aggregated_eval_steps: List[float] = []
    aggregated_eval_success_rates: List[float] = []
    aggregated_sample_efficiencies: List[int] = []

    # запуск для каждого сида
    for idx, seed in enumerate(seed_list):
        print(f"\n==== Запуск {idx + 1}/{len(seed_list)} (seed={seed}) ====")
        # Создание стандартной среды MountainCar.  Если требуется GIF, используем render_mode="rgb_array".
        if not GYM_AVAILABLE:
            raise RuntimeError(
                "Библиотека gym/gymnasium не найдена. Установите её, чтобы запустить классическую среду."
            )
        render_mode = 'rgb_array' if args.render_demo else None
        try:
            # Используем gymnasium, если импортирован
            env = gym.make('MountainCar-v0', render_mode=render_mode)
        except Exception:
            # fallback на gym
            import gym as gym_legacy  # type: ignore
            env = gym_legacy.make('MountainCar-v0', render_mode=render_mode)  # type: ignore
        # Установка лимита эпизода, если пользователь указал max_steps, отличное от дефолта
        try:
            max_eps_default = env.spec.max_episode_steps  # type: ignore
        except Exception:
            max_eps_default = None
        if args.max_steps and max_eps_default and args.max_steps != max_eps_default:
            from gym.wrappers import TimeLimit  # type: ignore
            env = TimeLimit(env, max_episode_steps=args.max_steps)  # type: ignore
        # Установка seed среды для воспроизводимости (если поддерживается)
        try:
            env.reset(seed=seed)
        except Exception:
            pass
        # Создание агента
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
        # Обучение агента
        returns, lengths, successes, moving_avgs = train_agent(
            env,
            agent,
            args.episodes,
            initial_random_steps=args.initial_random_steps,
            ma_window=args.ma_window,
        )
        # Расчёт sample efficiency: на каком эпизоде moving_avg достигает порога
        sample_eff = compute_sample_efficiency(moving_avgs, args.solved_threshold)
        sample_eff_str = (
            str(sample_eff)
            if sample_eff is not None
            else f">= {args.episodes} (порог не достигнут)"
        )
        # Оценка агента
        eval_ret, eval_steps, eval_success_rate = evaluate_agent(env, agent, args.eval_episodes)
        print(
            f"Оценка (seed={seed}) на {args.eval_episodes} эпизодах: средняя награда = {eval_ret:.2f}, "
            f"среднее число шагов = {eval_steps:.2f}, успех = {eval_success_rate * 100:.1f}%"
        )
        print(
            f"Sample efficiency (episode где moving_avg ≥ {args.solved_threshold:.2f}): {sample_eff_str}"
        )
        # Агрегирование результатов для нескольких сидов
        aggregated_eval_returns.append(eval_ret)
        aggregated_eval_steps.append(eval_steps)
        aggregated_eval_success_rates.append(eval_success_rate)
        if sample_eff is not None:
            aggregated_sample_efficiencies.append(sample_eff)
        # Сохранение модели
        model_path = (
            f"dqn_mountaincar_full_seed{seed}.pth"
            if len(seed_list) > 1
            else "dqn_mountaincar_full.pth"
        )
        agent.save(model_path)
        print(f"Модель сохранена в {model_path}")
        # Сохранение метрик
        metrics_path = (
            args.metrics_path.replace('.csv', f'_seed{seed}.csv')
            if len(seed_list) > 1
            else args.metrics_path
        )
        save_metrics_csv(metrics_path, returns, lengths, successes, moving_avgs)
        print(f"Метрики сохранены в {metrics_path}")
        # Сохранение графика обучения
        if args.plot_path:
            plot_path = (
                args.plot_path.replace('.png', f'_seed{seed}.png')
                if len(seed_list) > 1
                else args.plot_path
            )
            plot_training_curve(returns, plot_path, window=args.ma_window, show=args.show_plot)
        # Сохранение GIF с демонстрацией
        if args.render_demo:
            gif_path = (
                args.gif_path.replace('.gif', f'_seed{seed}.gif')
                if len(seed_list) > 1
                else args.gif_path
            )
            save_demo_gif(env, agent, path=gif_path, fps=args.fps)
    # вывод агрегированных результатов, если несколько сидов
    if len(seed_list) > 1:
        import statistics
        def mean_std(values: List[float]) -> Tuple[float, float]:
            return (sum(values) / len(values), statistics.stdev(values) if len(values) > 1 else 0.0)
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


if __name__ == '__main__':
    main()