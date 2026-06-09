from dataclasses import dataclass
import numpy as np
from maze_dataset.maze.lattice_maze import SolvedMaze

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class Maze:
    def __init__(self, rows: int, cols:int, connections, start_pos, end_pos) -> None:
        self.rows = rows
        self.cols = cols
        self.connections = connections
        self.start_pos = start_pos
        self.end_pos = end_pos
    
    def from_solvedmaze(solved_maze: SolvedMaze):
        return Maze(
            rows=solved_maze.grid_shape[0],
            cols=solved_maze.grid_shape[1],
            connections=solved_maze.connection_list,
            start_pos=solved_maze.start_pos,
            end_pos=solved_maze.end_pos
        )

def connection_list_to_4dir(connection_list):
    """
    build 4-direction mask from a maze.connection_list

    connection_list has shape (2, n, m) where index 0 stores downward
    edges and index 1 stores rightward edges.

    returns an array of shape (4, n, m) with axis 0 ordered as
    up, down, left, right. True = edge
    """
    down = connection_list[0]
    right = connection_list[1]

    four_dir = np.zeros((4, *down.shape), dtype=np.bool_)

    four_dir[DOWN] = down
    four_dir[UP, 1:, :] = down[:-1, :]
    four_dir[RIGHT] = right
    four_dir[LEFT, :, 1:] = right[:, :-1]

    return four_dir