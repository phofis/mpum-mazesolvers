from maze_solvers.utils import Maze, STEP_PENALTY, WALL_PENALTY, WIN_REWARD, maze_step
import numpy as np
from numpy.typing import NDArray
from maze_dataset.constants import Coord

default_hyperparams = {
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 0.25,
    "episodes": 1000,
    "max_steps": 1000,
    "min_epsilon": 0.01,
    "epsilon_decay": 0.995,
}


def epsilon_greedy(
    state: Coord,
    q_table: NDArray[np.float64],
    step: np.intp,
    hyperparams: dict
) -> np.intp:
    row, col = state
    if np.random.rand() < hyperparams["epsilon"]:
        return np.random.randint(0, 4)
    q_values = q_table[:, row, col]
    max_actions = np.flatnonzero(q_values == q_values.max())
    return np.random.choice(max_actions)


def make_step(
    action: int,
    state: Coord,
    connections: NDArray[np.bool_],
    end_pos: Coord,
) -> tuple[int, Coord]:
    row, col = state
    if connections[action, row, col]:
        next_state = maze_step(action, state)
        if np.array_equal(next_state, end_pos):
            return WIN_REWARD, next_state
        return STEP_PENALTY, next_state
    return WALL_PENALTY, state


class Qlearing:
    def __init__(self, maze: Maze, hyperparams: dict) -> None:
        self.hyperparams = hyperparams
        for key, val in default_hyperparams.items():
            if key not in self.hyperparams:
                self.hyperparams[key] = val
        self.start_pos = maze.start_pos
        self.end_pos = maze.end_pos
        self.connections = maze.connections
        self.quickest_path_len = maze.quickest_path_len
        self.q_table = np.zeros((4, maze.rows, maze.cols)) # 4 actions, rows, columns
        self.history = {
            "rewards": {},
            "steps": {},
        }

    def _pick_action(self, state: Coord, step: int) -> np.intp:
        return epsilon_greedy(state, self.q_table, self.hyperparams)

    def run(self) -> None:
        for episode in range(self.hyperparams["episodes"]):
            S = self.start_pos.copy()
            steps = 0
            path = []
            rewards = []
            while not np.array_equal(S, self.end_pos) and steps < self.hyperparams["max_steps"]:
                A = self._pick_action(S, steps)
                R, S_prim = make_step(A, S, self.connections, self.end_pos)

                row, col = S
                row_prim, col_prim = S_prim
                self.q_table[A, row, col] += self.hyperparams["alpha"] * (
                    R
                    + self.hyperparams["gamma"] * self.q_table[:, row_prim, col_prim].max()
                    - self.q_table[A, row, col]
                )
                
                rewards.append(R)
                path.append(S_prim)
                S = S_prim
                steps += 1
            self.hyperparams["epsilon"] = np.max(self.hyperparams["epsilon"] * self.hyperparams["epsilon_decay"], self.hyperparams["min_epsilon"])
            self.history["rewards"][episode] = rewards
            self.history["steps"][episode] = path
        
            
