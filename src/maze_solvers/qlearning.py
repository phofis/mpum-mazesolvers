from maze_solvers.utils import Maze
import numpy as np

default_hyperparams = {
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 0.25,
    "episodes": 1000,
    "max_steps": 1000,
    "min_epsilon": 0.01,
    "epsilon_decay": 0.995,
}

def epsilon_greedy(state: tuple, q_table: tuple, hyperparams: dict) -> int:
    if np.random.rand() < hyperparams["epsilon"]:
        return np.random.randint(0,4)
    else:
        return np.argmax(q_table[:, state])



class Qlearing:
    def __init__(self, maze: Maze, hyperparams: dict) -> None:
        self.hyperparams = hyperparams
        for key,val in default_hyperparams:
            if key not in self.hyperparams:
                self.hyperparams[key] = val
        self.start_pos = maze.start_pos
        self.end_pos = maze.end_pos
        self.connections = maze.connections
        self.q_table = np.zeros((4, maze.rows, maze.cols)) # 4 actions, rows, columns
        self.history = {
            "rewards": [],
            "steps": [],
            "epsilon": [],
            "episodes": [],
        }

    def _pick_action(self, state: tuple):
        return epsilon_greedy(state, self.q_table, self.hyperparams)

    
    def run(self) -> None:
        for episode in range(self.hyperparams["episodes"]):
            S = self.start_pos
            steps = 0
            while S is not self.end_pos and steps < self.hyperparams["max_steps"]:
                action = self._pick_action
                



    
    


        