from typing import Dict
from abc import ABC, abstractmethod
from maze_solvers.exploration_strategies import ExplorationStrategy
from maze_solvers.utils import Maze
class Agent(ABC):
    @abstractmethod
    def __init__(self, maze: Maze, strategy: ExplorationStrategy, hyperparams: Dict) -> None:
        pass
    @abstractmethod
    def train(self) -> None:
        pass
    @abstractmethod
    def predict(self) -> tuple:
        pass
    