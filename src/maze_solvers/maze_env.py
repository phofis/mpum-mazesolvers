import numpy as np
from maze_dataset.constants import Coord
from numpy.typing import NDArray

from maze_solvers.utils import Maze, maze_step
class rewards:
    STEP_PENALTY = -1
    WALL_PENALTY = -5
    WIN_REWARD = 100
    def __init__(self, step_penalty = STEP_PENALTY, wall_penalty = WALL_PENALTY, win_reward = WIN_REWARD) -> None:
        self.STEP_PENALTY = step_penalty
        self.WALL_PENALTY = wall_penalty
        self.WIN_REWARD = win_reward
class env:
    def __init__(self, maze : Maze, rewards : rewards) -> None:
        self.maze = maze
        self.state = maze.start_pos
        self.rewards = rewards
        self.end_pos = maze.end_pos
        self.steps = 0

    def step(self, action : int) -> tuple[int, Coord, bool]:
        row, col = self.state
        if self.maze.connections[action, row, col]:
            self.state = maze_step(action, self.state)
            self.steps += 1
            if np.array_equal(self.state, self.end_pos):
                return self.rewards.WIN_REWARD, self.state, True
            return self.rewards.STEP_PENALTY, self.state, False
        return self.rewards.WALL_PENALTY, self.state, False
    def reset(self) -> Coord:
        self.state = self.maze.start_pos
        self.steps = 0
        return self.state