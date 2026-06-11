from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.maze.lattice_maze import SolvedMaze
from maze_dataset.generation import LatticeMazeGenerators
from maze_solvers.exploration_strategies import Strategy
from maze_solvers.plotting import plotHistory
from maze_solvers.utils import Maze

from maze_solvers.qlearning import Qlearning

cfg: MazeDatasetConfig = MazeDatasetConfig(
    name="test",  # name is only for you to keep track of things
    grid_n=20,  # number of rows/columns in the lattice
    n_mazes=10,  # number of mazes to generate
    maze_ctor=LatticeMazeGenerators.gen_wilson,  # algorithm to generate the maze
    # additional parameters to pass to the maze generation algorithm
)

dataset: MazeDataset = MazeDataset.from_config(cfg)

m = dataset[2]
q = Qlearning(maze=Maze.from_solvedmaze(m), strategy=Strategy.SOFTMAX)
q.train()
r, p, a, b, epoch = q.predict()
print(len(p), b, epoch)
plotHistory(q.history,m)
