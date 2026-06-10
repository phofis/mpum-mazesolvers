import numpy as np
from numpy.typing import NDArray
from maze_dataset.constants import Coord
from maze_dataset.maze.lattice_maze import SolvedMaze

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

STEP_PENALTY = -1
WALL_PENALTY = -5
WIN_REWARD = 100

DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int8)


class Maze:
    def __init__(
        self,
        rows: int,
        cols: int,
        connections: NDArray[np.bool_],
        start_pos: Coord,
        end_pos: Coord,
        quick_path_len: int
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.connections = connections
        self.start_pos = np.asarray(start_pos, dtype=np.intp)
        self.end_pos = np.asarray(end_pos, dtype=np.intp)
        self.quickest_path_len = quick_path_len

    @staticmethod
    def from_solvedmaze(solved_maze: SolvedMaze) -> "Maze":
        return Maze(
            rows=solved_maze.grid_shape[0],
            cols=solved_maze.grid_shape[1],
            connections=connection_list_to_4dir(solved_maze.connection_list),
            start_pos=solved_maze.start_pos,
            end_pos=solved_maze.end_pos,
            quick_path_len=len(solved_maze.find_shortest_path(solved_maze.start_pos,solved_maze.end_pos))
        )


def connection_list_to_4dir(connection_list: NDArray[np.bool_]) -> NDArray[np.bool_]:
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


def maze_step(action: int, state: Coord) -> Coord:
    return state + DELTAS[action]
