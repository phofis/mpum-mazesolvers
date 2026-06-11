from typing import Dict, List, Tuple
import numpy as np
from maze_dataset.constants import Coord
from numpy.typing import NDArray

from maze_solvers.utils import Maze, maze_step


class Rewards:
    def __init__(self, step_penalty=-1, wall_penalty=-5, win_reward=100) -> None:
        self.STEP_PENALTY = step_penalty
        self.WALL_PENALTY = wall_penalty
        self.WIN_REWARD = win_reward


class History:
    def __init__(self) -> None:
        self.rewards: Dict[int, List[int]] = {}
        self.steps: Dict[int, List[Coord]] = {}
        self.actions: Dict[int, List[int]] = {}

    def update(
        self, epoch: int, rewards: List[int], path: List[Coord], actions: List[int]
    ) -> None:
        self.rewards[epoch] = rewards
        self.steps[epoch] = path
        self.actions[epoch] = actions


class Env:
    def __init__(self, maze: Maze, rewards: Rewards, masking: bool) -> None:
        self.maze = maze
        self.state = maze.start_pos
        self.rewards = rewards
        self.end_pos = maze.end_pos
        self.steps = 0
        self.masking = masking
        self.mask = self.maze.connections if masking else np.ones_like(self.maze.connections, dtype=bool)

    def step(self, action: int) -> Tuple[int, Coord, bool]:
        row, col = self.state
        if self.maze.connections[action, row, col]:
            self.state = maze_step(action, self.state)
            if np.array_equal(self.state, self.end_pos):
                return self.rewards.WIN_REWARD, self.state, True
            return self.rewards.STEP_PENALTY, self.state, False
        return self.rewards.WALL_PENALTY, self.state, False
    def action_mask(self, state: Coord, masking : bool) -> NDArray[np.bool_]:
        row, col = state
        if(masking):
            return self.maze.connections[:, row, col] 
        return np.ones(4, dtype=bool)
    def reset(self) -> Coord:
        self.state = self.maze.start_pos
        self.steps = 0
        return self.state
