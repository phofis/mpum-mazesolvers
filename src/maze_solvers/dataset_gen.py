from pathlib import Path

from maze_dataset import MazeDatasetConfig
from maze_dataset.dataset.collected_dataset import (
    MazeDatasetCollection,
    MazeDatasetCollectionConfig,
)
from maze_dataset.generation import LatticeMazeGenerators

GRID_SIZES = [5, 10, 20]
N_MAZES = 50
GENERATORS = [
    ("wilson", LatticeMazeGenerators.gen_wilson, {}),
    ("dfs_percolation", LatticeMazeGenerators.gen_dfs_percolation, {"p": 0.5}),
    ("kruskal", LatticeMazeGenerators.gen_kruskal, {}),
]
COLLECTION_NAME = "maze_solvers_benchmark"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_configs() -> list[MazeDatasetConfig]:
    configs: list[MazeDatasetConfig] = []
    for grid_n in GRID_SIZES:
        for name, ctor, kwargs in GENERATORS:
            configs.append(
                MazeDatasetConfig(
                    name=f"g{grid_n}_{name}",
                    grid_n=grid_n,
                    n_mazes=N_MAZES,
                    maze_ctor=ctor,
                    maze_ctor_kwargs=kwargs,
                )
            )
    return configs


def generate_collection(verbose: bool = True) -> MazeDatasetCollection:
    cfg = MazeDatasetCollectionConfig(
        name=COLLECTION_NAME,
        maze_dataset_configs=build_configs(),
    )
    return MazeDatasetCollection.generate(cfg, verbose=verbose)


def save_dataset(
    collection: MazeDatasetCollection,
    path: Path | None = None,
) -> Path:
    output = project_root() / "dataset"
    collection.save(output)
    return output


def load_dataset(path: Path | None = None) -> MazeDatasetCollection:
    dataset_path = project_root() /"dataset.zanj"
    return MazeDatasetCollection.read(dataset_path)


def main() -> None:
    collection = generate_collection(verbose=True)
    path = save_dataset(collection)
    summary = collection.cfg.summary()
    print(f"Saved {len(collection)} mazes to {path}")
    print(f"Collection: {summary['fname']}")
    for sub_summary in summary["cfg_summaries"]:
        print(
            f"  {sub_summary['name']}: "
            f"grid={sub_summary['grid_n']}, "
            f"n={sub_summary['n_mazes']}, "
            f"generator={sub_summary['maze_ctor_name']}"
        )


if __name__ == "__main__":
    main()
