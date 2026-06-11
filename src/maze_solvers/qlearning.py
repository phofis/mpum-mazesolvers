from typing import Dict, List, Tuple
from maze_solvers.utils import Maze
import numpy as np
from numpy.typing import NDArray
from maze_dataset.constants import Coord
from maze_solvers.maze_env import History, Rewards, Env
from maze_solvers.exploration_strategies import (
    ExplorationStrategy,
    EpsilonGreedy,
    Greedy,
    Softmax,
    Pursuit,
    UCB,
    Strategy,
)

default_hyperparams = {
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 0.25,
    "episodes": 500,
    "max_steps": 1000,
    "early_stop": 10,  # stop training after k consecutive finds of optimal path, -1 if no early stopping
    "min_epsilon": 0.01,
    "epsilon_decay": 0.995,
    "temperature": 50,
    "min_t": 0.5,
    "t_decay": 0.95,
    "beta": 0.1,
    "C": 1.0,
}


class Qlearning:
    def __init__(
        self,
        maze: Maze,
        hyperparams: Dict = {},
        strategy: Strategy = Strategy.EPSILON_GREEDY,
    ) -> None:
        self.hyperparams = hyperparams
        for key, val in default_hyperparams.items():
            if key not in self.hyperparams:
                self.hyperparams[key] = val
        self.env = Env(maze, Rewards())
        self.q_table = np.zeros((4, maze.rows, maze.cols))  # 4 actions, rows, columns
        self.exploration: ExplorationStrategy = EpsilonGreedy()
        self.history = History()
        self.k = 0
        self.stopped_at_epoch = 0
        if strategy == Strategy.SOFTMAX:
            self.exploration = Softmax()
        elif strategy == Strategy.PURSUIT:
            self.exploration = Pursuit(maze)
        elif strategy == Strategy.UCB:
            self.exploration = UCB(maze)

    def _update_hyperparams(self) -> None:
        self.hyperparams["epsilon"] = max(
            self.hyperparams["epsilon"] * self.hyperparams["epsilon_decay"],
            self.hyperparams["min_epsilon"],
        )
        self.hyperparams["temperature"] = max(
            self.hyperparams["temperature"] * self.hyperparams["t_decay"],
            self.hyperparams["min_t"],
        )

    def _try_early_stop(self, path) -> bool:
        if len(path) == self.env.maze.quickest_path_len:
            self.k += 1
        else:
            self.k = 0
        if self.k == self.hyperparams["early_stop"]:
            return True
        else:
            return False

    def train(self) -> None:
        for episode in range(self.hyperparams["episodes"]):
            S = self.env.reset()
            self.exploration.on_episode_start()
            path = [self.env.maze.start_pos]
            rewards = []
            actions = []
            foundExit = False
            while not foundExit and self.env.steps < self.hyperparams["max_steps"]:
                A = self.exploration.pick_action(S, self.q_table, self.hyperparams)
                R, S_prim, foundExit = self.env.step(A)

                row, col = S
                row_prim, col_prim = S_prim
                self.q_table[A, row, col] += self.hyperparams["alpha"] * (
                    R
                    + self.hyperparams["gamma"]
                    * self.q_table[:, row_prim, col_prim].max()
                    - self.q_table[A, row, col]
                )

                rewards.append(R)
                path.append(S_prim)
                actions.append(A)
                self.exploration.after_step(
                    S, A, R, S_prim, self.q_table, self.hyperparams
                )
                S = S_prim
                self.env.steps += 1
                # print(self.env.steps)
            self.history.update(episode, rewards, path, actions)
            self._update_hyperparams()
            self.stopped_at_epoch = episode
            # early stop
            if self._try_early_stop(path):
                return

    def predict(self) -> Tuple[List[int], List[Coord], List[int], bool, int]:
        rewards = []
        path = []
        actions = []
        foundExit = False
        S = self.env.reset()
        exploration = Greedy()
        while not foundExit:
            A = exploration.pick_action(S, self.q_table, self.hyperparams)
            R, S_prim, foundExit = self.env.step(A)
            rewards.append(R)
            path.append(S_prim)
            actions.append(A)
            S = S_prim
        return (
            rewards,
            path,
            actions,
            len(path) == self.env.maze.quickest_path_len,
            self.stopped_at_epoch
        )
