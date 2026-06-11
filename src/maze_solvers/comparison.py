from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators

from maze_solvers.exploration_strategies import Strategy
from maze_solvers.plotting import plotHistory
from maze_solvers.qlearning import Qlearning
from maze_solvers.sarsa import SARSA
from maze_solvers.utils import Maze

cfg: MazeDatasetConfig = MazeDatasetConfig(
    name="test", # name is only for you to keep track of things
    grid_n=20, # number of rows/columns in the lattice
    n_mazes=4, # number of mazes to generate
    maze_ctor=LatticeMazeGenerators.gen_wilson, # algorithm to generate the maze
)

dataset: MazeDataset = MazeDataset.from_config(cfg)

maze1 = dataset[1]
# q = Qlearning(Maze.from_solvedmaze(maze1), strategy=Strategy.PURSUIT)
# q.train()
# plotHistory(q.history, maze1, output_dir="plotting/maze1/qlearing_nomasking")
# sarsa = SARSA(Maze.from_solvedmaze(maze1), strategy=Strategy.PURSUIT)
# sarsa.train()
# plotHistory(sarsa.history, maze1, output_dir="plotting/maze1/sarsa_nomasking")

maze0 = dataset[0]
q = Qlearning(Maze.from_solvedmaze(maze0), strategy=Strategy.PURSUIT)
q.train()
plotHistory(q.history, maze0, output_dir="plotting/maze0/qlearing_nomasking")
q = Qlearning(Maze.from_solvedmaze(maze0), hyperparams={"masking":True}, strategy=Strategy.PURSUIT)
q.train()
plotHistory(q.history, maze0, output_dir="plotting/maze0/qlearing_masking")

maze2 = dataset[2]
q = Qlearning(Maze.from_solvedmaze(maze2), strategy=Strategy.EPSILON_GREEDY)
q.train()
plotHistory(q.history, maze2, output_dir="plotting/maze2/qlearing_epsilongreedy")
q = Qlearning(Maze.from_solvedmaze(maze2), strategy=Strategy.SOFTMAX)
q.train()
plotHistory(q.history, maze2, output_dir="plotting/maze2/qlearing_softmax")
q = Qlearning(Maze.from_solvedmaze(maze2), strategy=Strategy.PURSUIT)
q.train()
plotHistory(q.history, maze2, output_dir="plotting/maze2/qlearing_pursuit")
q = Qlearning(Maze.from_solvedmaze(maze2), strategy=Strategy.UCB)
q.train()
plotHistory(q.history, maze2, output_dir="plotting/maze2/qlearing_ucb")