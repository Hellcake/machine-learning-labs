"""
Simple tabular Q-learning baseline for the MountainCar environment.

This script implements a classic tabular Q‑learning agent to solve the
MountainCar control problem.  It discretizes the continuous state space
into a finite number of bins for position and velocity and then
maintains a Q‑table of size `(num_position_bins, num_velocity_bins, n_actions)`.
The agent learns via the standard Q‑learning update rule:

    Q(s,a) ← Q(s,a) + α * (r + γ * max_a' Q(s', a') − Q(s,a))

where α is the learning rate and γ is the discount factor.  An
ε‑greedy policy is used for exploration/exploitation during training.

After training the agent for a specified number of episodes, the script
evaluates the learned policy without exploration and prints metrics such
as average return and average episode length.

Usage:

    python3 mountaincar_q_learning_baseline.py --episodes 1000 --eval_episodes 20 \
        --num_bins 30 --alpha 0.1 --gamma 0.99 --eps_start 1.0 --eps_end 0.05 \
        --eps_decay_steps 20000 --seed 42

The environment is imported from ``mountaincar_dqn_full`` so that the same
customisable MountainCar dynamics are used for fair comparison with the
Double DQN implementation.  Alternatively, you may uncomment the gym
version to use the standard ``MountainCar-v0`` environment from the
Gymnasium library (note that the state discretisation assumes the
state bounds of the standard environment).

"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np

# --- Environment import logic ---
# Prefer a pure NumPy implementation of MountainCar (from mountaincar_dqn)
# which does not depend on PyTorch.  If that import fails, fall back to
# gymnasium's standard MountainCar-v0.  We do not use the torch-based
# mountaincar_dqn_full here, because that brings in PyTorch which may not
# be installed.  The choice of environment is controlled by
# --use_custom_env at runtime.
try:
    from mountaincar_dqn import MountainCarEnv  # type: ignore
    _CUSTOM_ENV_AVAILABLE = True
except ImportError:
    import gymnasium as gym  # type: ignore
    _CUSTOM_ENV_AVAILABLE = False


@dataclass
class QLearningConfig:
    """Hyperparameters for the Q‑learning baseline."""
    episodes: int = 500
    eval_episodes: int = 20
    num_bins: int = 30
    alpha: float = 0.1
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20000
    seed: int = 42
    # Environment parameters (only used with custom MountainCarEnv)
    position_min: float = -1.2
    position_max: float = 0.6
    velocity_min: float = -0.07
    velocity_max: float = 0.07
    start_min: float = -0.6
    start_max: float = -0.4
    goal_position: float = 0.5
    force: float = 0.001
    gravity: float = 0.0025
    max_steps: int = 200
    hill_freq: float = 3.0
    hill_amp: float = 1.0
    slope_gain: Optional[float] = None


class TabularQLearningAgent:
    """Tabular Q‑learning agent for discretised MountainCar."""

    def __init__(self, config: QLearningConfig, env) -> None:
        self.config = config
        self.env = env
        self.n_actions = 3  # MountainCar has 3 discrete actions
        self.num_bins = config.num_bins
        # Determine state bounds from environment.  For our custom environment,
        # we inspect attributes position_min/max and velocity_min/max.  For
        # gym environments, we use observation_space.low/high.
        if hasattr(env, "position_min") and hasattr(env, "velocity_min"):
            # Custom numpy environment
            self.pos_bounds = (env.position_min, env.position_max)
            self.vel_bounds = (env.velocity_min, env.velocity_max)
        else:
            # Standard Gymnasium environment
            self.pos_bounds = (env.observation_space.low[0], env.observation_space.high[0])
            self.vel_bounds = (env.observation_space.low[1], env.observation_space.high[1])
        # Initialize Q‑table to zeros: shape (pos_bins, vel_bins, actions)
        self.q_table = np.zeros((self.num_bins, self.num_bins, self.n_actions), dtype=np.float32)
        # Exploration parameters
        self.eps_start = config.eps_start
        self.eps_end = config.eps_end
        self.eps_decay_steps = config.eps_decay_steps
        # For reproducibility
        self.rng = random.Random(config.seed)
        # Running count of steps to compute epsilon
        self.global_step = 0

    def discretise_state(self, state: np.ndarray) -> Tuple[int, int]:
        """Convert continuous position and velocity to discrete bin indices.

        Each dimension is divided into ``num_bins`` equally sized intervals.

        Args:
            state: A numpy array of shape (2,) containing [position, velocity].

        Returns:
            A tuple (pos_bin, vel_bin) with values in [0, num_bins-1].
        """
        pos, vel = state
        pos_min, pos_max = self.pos_bounds
        vel_min, vel_max = self.vel_bounds
        # Compute bin widths
        pos_bin_width = (pos_max - pos_min) / self.num_bins
        vel_bin_width = (vel_max - vel_min) / self.num_bins
        # Map position and velocity into bins
        pos_bin = int((pos - pos_min) / pos_bin_width)
        vel_bin = int((vel - vel_min) / vel_bin_width)
        # Clip to ensure indices are within [0, num_bins-1]
        pos_bin = max(0, min(self.num_bins - 1, pos_bin))
        vel_bin = max(0, min(self.num_bins - 1, vel_bin))
        return pos_bin, vel_bin

    def get_epsilon(self) -> float:
        """Linearly anneal epsilon from eps_start to eps_end over eps_decay_steps."""
        eps = self.eps_start - (self.eps_start - self.eps_end) * (
            min(self.global_step, self.eps_decay_steps) / self.eps_decay_steps
        )
        return max(self.eps_end, eps)

    def choose_action(self, state_disc: Tuple[int, int]) -> int:
        """Select an action using ε‑greedy strategy based on discretised state."""
        eps = self.get_epsilon()
        if self.rng.random() < eps:
            # Exploration: random action
            return self.rng.randrange(self.n_actions)
        # Exploitation: pick action with highest Q
        pos_bin, vel_bin = state_disc
        return int(np.argmax(self.q_table[pos_bin, vel_bin]))

    def update_q(self,
                 state_disc: Tuple[int, int],
                 action: int,
                 reward: float,
                 next_state_disc: Tuple[int, int],
                 done: bool) -> None:
        """Apply Q‑learning update rule to a single transition."""
        pos_bin, vel_bin = state_disc
        next_pos_bin, next_vel_bin = next_state_disc
        current_q = self.q_table[pos_bin, vel_bin, action]
        # Compute TD target: r + γ * max_{a'} Q(s', a') if not done
        if done:
            target = reward
        else:
            target = reward + self.config.gamma * np.max(self.q_table[next_pos_bin, next_vel_bin])
        # Update
        self.q_table[pos_bin, vel_bin, action] = current_q + self.config.alpha * (
            target - current_q
        )

    def train(self, episodes: int) -> Tuple[List[float], List[int], List[bool]]:
        """Train the agent for the given number of episodes.

        Returns:
            A tuple of lists: (returns, lengths, successes) recording the sum of
            rewards, the number of steps, and whether the agent succeeded in each episode.
        """
        returns: List[float] = []
        lengths: List[int] = []
        successes: List[bool] = []
        for ep in range(episodes):
            # Reset environment; Gymnasium returns (obs, info), our custom env returns obs
            reset_out = self.env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            state_disc = self.discretise_state(state)
            done = False
            ep_return = 0.0
            ep_steps = 0
            ep_success = False
            # Loop until done
            while not done:
                self.global_step += 1
                action = self.choose_action(state_disc)
                # Take a step
                # Step environment; Gymnasium returns (obs, reward, terminated, truncated, info), our env returns (obs, reward, done, info)
                step_out = self.env.step(action)
                if len(step_out) == 4:
                    # custom env
                    next_state, reward, done, _ = step_out
                else:
                    # gym env
                    next_state, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated
                ep_return += reward
                ep_steps += 1
                # Discretise next state
                next_state_disc = self.discretise_state(next_state)
                # If reached goal in custom env, reward is -1 each step; success if done before max_steps
                if done:
                    # Determine success.  In our custom env, reaching goal
                    # corresponds to position >= goal_position.  In gym env,
                    # 'terminated' flag indicates reaching goal, 'truncated' is due to time limit.
                    if len(step_out) == 4:
                        ep_success = next_state[0] >= getattr(self.env, "goal_position", 0.5)
                    else:
                        ep_success = terminated
                # Q‑learning update
                self.update_q(state_disc, action, reward, next_state_disc, done)
                # Move to next state
                state_disc = next_state_disc
            returns.append(ep_return)
            lengths.append(ep_steps)
            successes.append(ep_success)
        return returns, lengths, successes

    def evaluate(self, episodes: int) -> Tuple[float, float, float]:
        """Evaluate the learned policy without exploration.

        Returns:
            (avg_return, avg_length, success_rate)
        """
        total_return = 0.0
        total_length = 0
        success_count = 0
        for _ in range(episodes):
            # Reset environment; handle both custom and gym env
            reset_out = self.env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            state_disc = self.discretise_state(state)
            done = False
            ep_return = 0.0
            ep_steps = 0
            ep_success = False
            while not done:
                # Greedy action: choose best action
                pos_bin, vel_bin = state_disc
                action = int(np.argmax(self.q_table[pos_bin, vel_bin]))
                step_out = self.env.step(action)
                if len(step_out) == 4:
                    next_state, reward, done, _ = step_out
                else:
                    next_state, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated
                ep_return += reward
                ep_steps += 1
                state_disc = self.discretise_state(next_state)
                if done:
                    if len(step_out) == 4:
                        ep_success = next_state[0] >= getattr(self.env, "goal_position", 0.5)
                    else:
                        ep_success = terminated
            total_return += ep_return
            total_length += ep_steps
            success_count += int(ep_success)
        avg_return = total_return / episodes
        avg_length = total_length / episodes
        success_rate = success_count / episodes
        return avg_return, avg_length, success_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabular Q-learning baseline for MountainCar")
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--eval_episodes', type=int, default=20, help='Number of evaluation episodes')
    parser.add_argument('--num_bins', type=int, default=30, help='Number of bins per dimension for discretisation')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps_start', type=float, default=1.0, help='Initial epsilon for ε-greedy')
    parser.add_argument('--eps_end', type=float, default=0.05, help='Final epsilon')
    parser.add_argument('--eps_decay_steps', type=int, default=20000, help='Steps over which epsilon decays linearly')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # Custom environment parameters
    parser.add_argument('--use_custom_env', action='store_true', help='Use custom MountainCarEnv instead of gym')
    parser.add_argument('--position_min', type=float, default=-1.2, help='Minimum position (custom env)')
    parser.add_argument('--position_max', type=float, default=0.6, help='Maximum position (custom env)')
    parser.add_argument('--velocity_min', type=float, default=-0.07, help='Minimum velocity (custom env)')
    parser.add_argument('--velocity_max', type=float, default=0.07, help='Maximum velocity (custom env)')
    parser.add_argument('--start_min', type=float, default=-0.6, help='Minimum start position (custom env)')
    parser.add_argument('--start_max', type=float, default=-0.4, help='Maximum start position (custom env)')
    parser.add_argument('--goal_position', type=float, default=0.5, help='Goal position (custom env)')
    parser.add_argument('--force', type=float, default=0.001, help='Force applied (custom env)')
    parser.add_argument('--gravity', type=float, default=0.0025, help='Gravity factor (custom env)')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum steps per episode (custom env)')
    parser.add_argument('--hill_freq', type=float, default=3.0, help='Frequency of hills (custom env)')
    parser.add_argument('--hill_amp', type=float, default=1.0, help='Amplitude of hills (custom env)')
    parser.add_argument('--slope_gain', type=float, default=None, help='Slope gain (custom env)')
    args = parser.parse_args()

    # Build configuration
    config = QLearningConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        num_bins=args.num_bins,
        alpha=args.alpha,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        seed=args.seed,
        position_min=args.position_min,
        position_max=args.position_max,
        velocity_min=args.velocity_min,
        velocity_max=args.velocity_max,
        start_min=args.start_min,
        start_max=args.start_max,
        goal_position=args.goal_position,
        force=args.force,
        gravity=args.gravity,
        max_steps=args.max_steps,
        hill_freq=args.hill_freq,
        hill_amp=args.hill_amp,
        slope_gain=args.slope_gain,
    )

    # Create environment.  If --use_custom_env is set and the pure NumPy
    # MountainCar implementation is available, instantiate it with the
    # parameters that it supports.  Otherwise, fall back to gymnasium.
    if args.use_custom_env and _CUSTOM_ENV_AVAILABLE:
        # The pure NumPy MountainCarEnv accepts position_range,
        # velocity_range, start_range, goal_position, force, gravity,
        # max_steps and rng.  It does not support hill_freq, hill_amp or
        # slope_gain.
        env = MountainCarEnv(
            position_range=(config.position_min, config.position_max),
            velocity_range=(config.velocity_min, config.velocity_max),
            start_range=(config.start_min, config.start_max),
            goal_position=config.goal_position,
            force=config.force,
            gravity=config.gravity,
            max_steps=config.max_steps,
            rng=np.random.RandomState(config.seed),
        )
    else:
        # Use standard gymnasium environment.  This environment does not
        # support extra parameters like hill_freq and hill_amp.  We wrap it
        # with TimeLimit to enforce a maximum episode length if the user
        # specifies a value different from the default.
        if not _CUSTOM_ENV_AVAILABLE:
            # gymnasium should already be imported at top-level if custom env
            # isn't available.
            pass
        env = gym.make("MountainCar-v0")
        # Wrap in TimeLimit if max_steps differs from default (default is 200)
        if config.max_steps != 200:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_steps)
    # Create agent
    agent = TabularQLearningAgent(config, env)
    # Train
    returns, lengths, successes = agent.train(config.episodes)
    # Evaluate
    avg_return, avg_length, success_rate = agent.evaluate(config.eval_episodes)
    print(
        f"Training finished. Avg return over {config.eval_episodes} eval episodes: {avg_return:.2f}, "
        f"Avg length: {avg_length:.2f}, Success rate: {success_rate * 100:.1f}%"
    )


if __name__ == "__main__":
    main()