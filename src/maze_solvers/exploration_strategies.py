from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict
import numpy as np
from maze_dataset import Coord
from numpy.typing import NDArray

from maze_solvers.utils import Maze
from maze_solvers.masking import sample_valid, masked_argmax, masked_max, masked_softmax_sample

class Strategy(Enum):
    EPSILON_GREEDY = 1
    SOFTMAX = 2
    PURSUIT = 3
    UCB = 4
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict
import numpy as np
from maze_dataset import Coord
from numpy.typing import NDArray

from maze_solvers.utils import Maze
from maze_solvers.masking import sample_valid, masked_argmax, masked_max, masked_softmax_sample

class Strategy(Enum):
    EPSILON_GREEDY = 1
    SOFTMAX = 2
    PURSUIT = 3
    UCB = 4


class ExplorationStrategy(ABC):
    @abstractmethod
    def pick_action(self, state: Coord, q_table: NDArray, hyperparams: Dict, mask: NDArray) -> np.intp:
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
        mask: NDArray
    ) -> None:
        pass


class Greedy(ExplorationStrategy):
    def pick_action(self, state, q_table, hyperparams, mask):
        row, col = state
        return masked_argmax(
            q_table[:, row, col],
            mask[:, row, col]
        )
class EpsilonGreedy(ExplorationStrategy):
    def pick_action(self, state: Coord, q_table: NDArray, hyperparams: Dict, mask: NDArray) -> np.intp:
        row, col = state
        eps = hyperparams["epsilon"]
        q_values = q_table[:, row, col]
        action_mask = mask[:, row, col]

        if np.random.rand() < eps:
            return sample_valid(action_mask)

        return masked_argmax(q_values, action_mask)

class Softmax(ExplorationStrategy):
    def pick_action(self, state: Coord, q_table: NDArray, hyperparams: Dict, mask: NDArray) -> np.intp:
        row, col = state
        q_values = q_table[:, row, col]
        action_mask = mask[:, row, col]
        T = hyperparams["temperature"]
        return masked_softmax_sample(q_values, action_mask, T)


class Pursuit(ExplorationStrategy):
    def __init__(self, maze: Maze) -> None:
        self.pi_table = np.ones_like(maze.connections) / 4

    def pick_action(self, state, q_table, hyperparams, mask):
        row, col = state

        probs = self.pi_table[:, row, col] * mask[:, row, col]
        probs = probs / np.sum(probs)

        return np.random.choice(np.arange(4), p=probs)

    def after_step(self, state, action, reward, next_state, q_table, hyperparams, mask):
        row, col = state
        beta = hyperparams["beta"]

        q_values = q_table[:, row, col]
        action_mask = mask[:, row, col]

        greedy = masked_argmax(q_values, action_mask)

        for a in range(q_values.shape[0]):
            target = 1.0 if a == greedy else 0.0
            self.pi_table[a, row, col] += beta * (target - self.pi_table[a, row, col])

        self.pi_table[:, row, col] /= np.sum(self.pi_table[:, row, col])
class UCB(ExplorationStrategy):
    def __init__(self, maze: Maze) -> None:
        self.counts = np.zeros((4, maze.rows, maze.cols), dtype=np.int64)

    def pick_action(self, state: Coord, q_table: NDArray, hyperparams: Dict, mask: NDArray) -> np.intp:
        row, col = state
        q_values = q_table[:, row, col]
        C = hyperparams["C"]
        available = np.flatnonzero(mask[:, row, col])
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
        mask: NDArray
    ) -> None:
        row, col = state
        self.counts[action, row, col] += 1


