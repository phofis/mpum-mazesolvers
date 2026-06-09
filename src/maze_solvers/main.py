from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.maze.lattice_maze import SolvedMaze
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.plotting import MazePlot
import matplotlib.pyplot as plt
import numpy as np

cfg: MazeDatasetConfig = MazeDatasetConfig(
	name="test", # name is only for you to keep track of things
	grid_n=3, # number of rows/columns in the lattice
	n_mazes=4, # number of mazes to generate
	maze_ctor=LatticeMazeGenerators.gen_dfs, # algorithm to generate the maze
    maze_ctor_kwargs=dict(do_forks=False), # additional parameters to pass to the maze generation algorithm
)

dataset: MazeDataset = MazeDataset.from_config(cfg)
maze = dataset[0]

start_node = maze.start_pos
end_node = maze.end_pos

maze_connection_list = maze.connection_list

print(maze.grid_shape)

print(maze.as_ascii())
