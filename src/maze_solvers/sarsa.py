
from maze_dataset import Coord
from maze_solvers.maze_env import Env, Rewards
from maze_solvers.utils import Maze
import numpy as np
from numpy.typing import NDArray
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
    hyperparams: dict
) -> np.intp:
    row, col = state
    if np.random.rand() < hyperparams["epsilon"]:
        return np.random.randint(0, 4)
    q_values = q_table[:, row, col]
    max_actions = np.flatnonzero(q_values == q_values.max())
    return np.random.choice(max_actions)
class SARSA:
    def __init__(self, maze: Maze, hyperparams: dict) -> None:
        self.hyperparams = hyperparams
        for key,val in default_hyperparams.items(): 
            if key not in self.hyperparams:
                self.hyperparams[key] = val
        self.env = Env(maze, Rewards())
        self.q_table = np.zeros((4, maze.rows, maze.cols)) # 4 actions, rows, columns
        self.history = {
            "rewards": [],
            "steps": [],
            "epsilon": [],
            "episodes": [],
        }

    def _pick_action(self, state: Coord):
        return epsilon_greedy(state, self.q_table, self.hyperparams)
    def run(self) -> None:
        for episode in range(self.hyperparams["episodes"]):
            S = self.env.reset()
            A = self._pick_action(S)
            steps = 0
            rewards = []
            path = []
            while not np.array_equal(S, self.env.end_pos) and steps < self.hyperparams["max_steps"]:
                R, S_prim, done = self.env.step(A)
                rewards.append(R)
                path.append(S)
                A_prim = self._pick_action(S_prim)

                row, col = S
                row_prim, col_prim = S_prim
                self.q_table[A, row, col] += self.hyperparams["alpha"] \
                   * (R + self.hyperparams["gamma"] \
                   * self.q_table[A_prim, row_prim, col_prim] 
                   - self.q_table[A, row, col]) \
                
                S = S_prim
                A = A_prim
                steps += 1
                if done:
                    break
            
            self.history["rewards"].append(rewards)
            self.history["steps"].append(path)
    def test(self) -> None:
        S = self.env.reset()
        path = [S]
        while not np.array_equal(S, self.env.end_pos):
            row, col = S
            A = np.argmax(self.q_table[:, row, col])
            R, S_prim, done = self.env.step(A)
            path.append(S_prim)
            S = S_prim
            if done:
                break
        print("Test path:", path, "Total reward:", sum([self.env.rewards.STEP_PENALTY for _ in path[:-1]]) + self.env.rewards.WIN_REWARD, "Steps:", len(path)-1)
        

    
    