from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.maze.lattice_maze import SolvedMaze
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.plotting import MazePlot
from maze_solvers.exploration_strategies import Strategy
from maze_solvers.utils import Maze
import matplotlib.pyplot as plt
import numpy as np

from maze_solvers.qlearning import Qlearning

cfg: MazeDatasetConfig = MazeDatasetConfig(
    name="test",  # name is only for you to keep track of things
    grid_n=6,  # number of rows/columns in the lattice
    n_mazes=4,  # number of mazes to generate
    maze_ctor=LatticeMazeGenerators.gen_wilson,  # algorithm to generate the maze
    # additional parameters to pass to the maze generation algorithm
)
from maze_dataset.plotting import MazePlot

dataset: MazeDataset = MazeDataset.from_config(cfg)
maze: SolvedMaze = dataset[1]


q: Qlearning = Qlearning(
    maze=Maze.from_solvedmaze(maze), hyperparams={}, strategy=Strategy.SOFTMAX
)
q.train()


for i in range(len(q.env.history.steps)):
    print(
        f"epoch {i} steps {len(q.env.history.steps[i])} reward {np.sum(q.env.history.rewards[i])}"
    )

fig = MazePlot(maze).plot()
plt.show()
