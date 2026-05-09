"""
Tabular Q-Learning agent for Flappy Bird.

Uses an epsilon-greedy exploration strategy with linear epsilon decay.
The Q-table is stored as a Python defaultdict so unseen states are
automatically initialised to zero.
"""

import numpy as np
import pickle
import os
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Parameters
    ----------
    alpha       Learning rate.
    gamma       Discount factor.
    epsilon     Initial exploration probability.
    epsilon_min Minimum epsilon after decay.
    epsilon_decay  Linear decay per episode.
    n_actions   Number of discrete actions (default 2: no-op / flap).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.001,
        n_actions: int = 2,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions     = n_actions

        # Q(s, a) initialised to 0 for unseen state-action pairs
        self.q_table: dict = defaultdict(lambda: np.zeros(n_actions))

    # ── action selection ─────────────────────────────────────────────────────

    def choose_action(self, state: tuple) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def choose_action_greedy(self, state: tuple) -> int:
        """Pure greedy (for evaluation / rendering)."""
        return int(np.argmax(self.q_table[state]))

    # ── learning ─────────────────────────────────────────────────────────────

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ):
        """Single Q-learning update step."""
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])

        td_error = target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Call once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        data = {
            "q_table":       dict(self.q_table),
            "alpha":         self.alpha,
            "gamma":         self.gamma,
            "epsilon":       self.epsilon,
            "epsilon_min":   self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "n_actions":     self.n_actions,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[Agent] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            alpha=data["alpha"],
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
            n_actions=data["n_actions"],
        )
        agent.q_table = defaultdict(lambda: np.zeros(agent.n_actions), data["q_table"])
        return agent

    def q_table_size(self) -> int:
        return len(self.q_table)
