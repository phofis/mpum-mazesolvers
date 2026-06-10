from maze_solvers.utils import Maze
import numpy as np
from numpy.typing import NDArray
from maze_dataset.constants import Coord
from maze_solvers.maze_env import Rewards, Env

default_hyperparams = {
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 0.25,
    "episodes": 1000,
    "max_steps": 1000,
    "min_epsilon": 0.01,
    "epsilon_decay": 0.995,
    "temperature": 5,
    "min_t": 0.1,
    "t_decay": 0.95,
    "beta": 0.1,
}


def epsilon_greedy(
    state: Coord,
    q_table: NDArray[np.float64],
    hyperparams: dict
) -> np.intp:
    row, col = state
    if np.random.rand() < hyperparams["epsilon"]:
        return np.random.randint(0, 4)
    q_values = q_table[:, row, col]
    max_actions = np.flatnonzero(q_values == q_values.max())
    return np.random.choice(max_actions)

def softmax_exploration(
    state: Coord,
    q_table: NDArray[np.float64],
    hyperparams: dict
) -> np.intp:
    T = hyperparams["temperature"]
    row, col = state
    q_values = q_table[:, row, col]
    sotfmax_q_values = np.exp(q_values / T)
    sotfmax_q_values_norm = sotfmax_q_values / np.sum(sotfmax_q_values)
    return np.random.choice(np.arange(4), p=sotfmax_q_values_norm)
     

def pursuit(
    state: Coord,
    pi_table: NDArray[np.float64],
) -> np.intp:
    row, col = state
    return np.random.choice(np.arange(4), p=pi_table[:, row, col])




class Qlearning:
    def __init__(self, maze: Maze, hyperparams: dict) -> None:
        self.hyperparams = hyperparams
        for key, val in default_hyperparams.items():
            if key not in self.hyperparams:
                self.hyperparams[key] = val
        self.env = Env(maze, Rewards())
        self.q_table = np.zeros((4, maze.rows, maze.cols)) # 4 actions, rows, columns


    def _pick_action(self, state: Coord) -> np.intp:
        strategy = self.hyperparams.get("exploration", "pursuit")
        if strategy == "epsilon_greedy":
            return epsilon_greedy(state, self.q_table, self.hyperparams)
        if strategy == "softmax":
            return softmax_exploration(state, self.q_table, self.hyperparams)
        return pursuit(state, self.pi_table)

    def _update_hyperparams(self) -> None:
        self.hyperparams["epsilon"] = max(self.hyperparams["epsilon"] * self.hyperparams["epsilon_decay"], self.hyperparams["min_epsilon"])
        self.hyperparams["temperature"] = max(self.hyperparams["temperature"] * self.hyperparams["t_decay"], self.hyperparams["min_t"])
            

    def run(self) -> None:
        for episode in range(self.hyperparams["episodes"]):
            S = self.env.reset()
            path = []
            rewards = []
            actions = []
            foundExit = False
            while not foundExit and self.env.steps < self.hyperparams["max_steps"]:
                A = self._pick_action(S)
                R, S_prim, foundExit = self.env.step(A)

                row, col = S
                row_prim, col_prim = S_prim
                self.q_table[A, row, col] += self.hyperparams["alpha"] * (
                    R
                    + self.hyperparams["gamma"] * self.q_table[:, row_prim, col_prim].max()
                    - self.q_table[A, row, col]
                )

                rewards.append(R)
                path.append(S_prim)
                actions.append(A)
                S = S_prim
                steps += 1


            self.env.history.update(episode, rewards, path, actions)
            self._update_hyperparams()
        
            
