from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray

from maze_dataset.constants import Coord
from maze_solvers.utils import Maze
from maze_solvers.maze_env import Env, Rewards, History
from maze_solvers.exploration_strategies import (
    ExplorationStrategy,
    EpsilonGreedy,
    Greedy,
    Softmax,
    Pursuit,
    UCB,
    Strategy,
)
from maze_solvers.agent import Agent


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
    "masking" : True
}



class SARSA(Agent):
    def __init__(
        self,
        maze: Maze,
        hyperparams: Dict = {},
        strategy: Strategy = Strategy.EPSILON_GREEDY,
    ) -> None:

        self.hyperparams = hyperparams
        for k, v in default_hyperparams.items():
            if k not in self.hyperparams:
                self.hyperparams[k] = v

        self.env = Env(maze, Rewards(), self.hyperparams)
        self.q_table = np.zeros((4, maze.rows, maze.cols))

        self.history = History()
        self.k = 0
        self.stopped_at_epoch = 0
        self.exploration: ExplorationStrategy = EpsilonGreedy()
        if strategy == Strategy.SOFTMAX:
            self.exploration = Softmax()
        elif strategy == Strategy.PURSUIT:
            self.exploration = Pursuit(maze)
        elif strategy == Strategy.UCB:
            self.exploration = UCB(maze)


    def _try_early_stop(self, path) -> bool:
        if len(path) == self.env.maze.quickest_path_len:
            self.k += 1
        else:
            self.k = 0
        if self.k == self.hyperparams["early_stop"]:
            return True
        else:
            return False
    def _update_hyperparams(self) -> None:
        self.hyperparams["epsilon"] = max(
            self.hyperparams["epsilon"] * self.hyperparams["epsilon_decay"],
            self.hyperparams["min_epsilon"],
        )
        self.hyperparams["temperature"] = max(
            self.hyperparams["temperature"] * self.hyperparams["t_decay"],
            self.hyperparams["min_t"],
        )
    

    def train(self) -> None:
        for episode in range(self.hyperparams["episodes"]):

            S = self.env.reset()
            A = self.exploration.pick_action(S, self.q_table, self.hyperparams, self.env.mask)
            rewards = []
            path = [S]
            actions = []

            done = False

            while not done and self.env.steps < self.hyperparams["max_steps"]:

                R, S_prim, done = self.env.step(A)
                A_prim = self.exploration.pick_action(S_prim, self.q_table, self.hyperparams, self.env.mask)

                r, c = S
                r2, c2 = S_prim

                if done:
                    target = R
                else:   
                    target = R + self.hyperparams["gamma"] * self.q_table[A_prim, r2, c2]

                self.q_table[A, r, c] += self.hyperparams["alpha"] * (
                    target - self.q_table[A, r, c]
                )

                rewards.append(R)
                path.append(S_prim)
                actions.append(A)
                self.exploration.after_step(
                    S, A, R, S_prim, self.q_table, self.hyperparams, self.env.mask
                )

                S, A = S_prim, A_prim
                self.env.steps += 1

            self.history.update(episode, rewards, path, actions)
            self._update_hyperparams()
            self.stopped_at_epoch = episode
            # early stop
            if self._try_early_stop(path):
                return
            

    def predict(self) -> Tuple[List[int], List[Coord], List[int], bool, int]:
        S = self.env.reset()
        path = [S]
        rewards = []
        actions = []

        done = False
        policy = Greedy()

        while not done:
            A = policy.pick_action(S, self.q_table, self.hyperparams, self.env.mask)

            R, S_prim, done = self.env.step(A)

            rewards.append(R)
            actions.append(A)
            path.append(S_prim)

            S = S_prim

        return (
            rewards,
            path,
            actions,
            len(path) == self.env.maze.quickest_path_len,
            self.hyperparams.get("episodes", 0),
        )