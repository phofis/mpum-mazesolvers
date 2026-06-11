from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict
import numpy as np
from maze_dataset import Coord
from numpy.typing import NDArray

from maze_solvers.utils import Maze


class Strategy(Enum):
    EPSILON_GREEDY = 1
    SOFTMAX = 2
    PURSUIT = 3
    UCB = 4


class ExplorationStrategy(ABC):
    @abstractmethod
    def pick_action(self, state: Coord, q_table: NDArray, hyperparams: Dict) -> np.intp:
        pass

    def on_episode_start(self) -> None:
        pass

    def after_step(
        self,
        state: Coord,
        action: int,
        reward: float,
        next_state: Coord,
        q_table: NDArray,
        hyperparams: Dict,
    ) -> None:
        pass


class Greedy(ExplorationStrategy):
    def pick_action(self, state: Coord, q_table: NDArray, hyperparams: Dict) -> np.intp:
        row, col = state
        q_values = q_table[:, row, col]
        max_actions = np.flatnonzero(q_values == q_values.max())
        return np.random.choice(max_actions)


class EpsilonGreedy(ExplorationStrategy):
    def pick_action(self, state: Coord, q_table: NDArray, hyperparams: Dict) -> np.intp:
        eps = hyperparams["epsilon"]
        row, col = state
        q_values = q_table[:, row, col]
        if np.random.rand() < eps:
            return np.random.randint(0, 4)
        max_actions = np.flatnonzero(q_values == q_values.max())
        return np.random.choice(max_actions)


class Softmax(ExplorationStrategy):
    def pick_action(self, state: Coord, q_table: NDArray, hyperparams: Dict) -> np.intp:
        row, col = state
        q_values = q_table[:, row, col]
        T = hyperparams["temperature"]
        softmax_q_values = np.exp(q_values / T)
        softmax_q_values_norm = softmax_q_values / np.sum(softmax_q_values)
        return np.random.choice(np.arange(4), p=softmax_q_values_norm)


class Pursuit(ExplorationStrategy):
    def pick_action(self, state: Coord, q_table: NDArray, hyperparams: Dict) -> np.intp:
        row, col = state
        return np.random.choice(np.arange(4), p=self.pi_table[:, row, col])

    def __init__(self, maze: Maze) -> None:
        self.pi_table = np.ones_like(maze.connections) / 4

    def after_step(
        self,
        state: Coord,
        action: int,
        reward: float,
        next_state: Coord,
        q_table: NDArray,
        hyperparams: Dict,
    ) -> None:
        row, col = state
        beta = hyperparams["beta"]
        q_values = q_table[:, row, col]
        greedy = q_values.argmax()
        for a in range(q_values.shape[0]):
            target = 1.0 if a == greedy else 0.0
            self.pi_table[a, row, col] += beta * (target - self.pi_table[a, row, col])


class UCB(ExplorationStrategy):
    def __init__(self, maze: Maze) -> None:
        self.available = maze.connections
        self.counts = np.zeros((4, maze.rows, maze.cols), dtype=np.int64)

    def pick_action(self, state: Coord, q_table: NDArray, hyperparams: Dict) -> np.intp:
        row, col = state
        q_values = q_table[:, row, col]
        C = hyperparams["C"]
        available = np.flatnonzero(self.available[:, row, col])
        not_tried = available[self.counts[available, row, col] == 0]
        if len(not_tried) > 0:
            return np.random.choice(not_tried)
        if C == 0:
            scores = q_values[available]
        else:
            n_s = self.counts[:, row, col].sum()
            bonuses = (
                100 * C * np.sqrt(2 * np.log(n_s) / self.counts[available, row, col])
            )
            scores = q_values[available] + bonuses

        max_actions = available[scores == scores.max()]
        return np.random.choice(max_actions)

    def after_step(
        self,
        state: Coord,
        action: int,
        reward: float,
        next_state: Coord,
        q_table: NDArray,
        hyperparams: Dict,
    ) -> None:
        row, col = state
        self.counts[action, row, col] += 1
