import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from joblib import Parallel, delayed
from maze_dataset.maze.lattice_maze import SolvedMaze

from maze_solvers.dataset_gen import load_dataset, project_root
from maze_solvers.exploration_strategies import Strategy
from maze_solvers.qlearning import Qlearning, default_hyperparams
from maze_solvers.sarsa import SARSA
from maze_solvers.utils import Maze
from maze_solvers.agent import Agent

STRATEGIES = list(Strategy)
DATASET_PATH = project_root() / "dataset.zanj"
RESULTS_DIR = Path("results")
N_JOBS = -1


@dataclass(frozen=True)
class RunConfig:
    grid_n: int
    generator: str
    maze_idx: int
    strategy: Strategy


@dataclass
class RunResult:
    grid_n: int
    generator: str
    maze_idx: int
    strategy: str
    optimal: bool
    path_len: int
    optimal_len: int
    episodes_trained: int
    optimality_ratio: float
    path_regret: int

    @property
    def group_key(self) -> tuple:
        return (self.strategy, self.grid_n, self.generator)


def run_single(config: RunConfig, solved_maze: SolvedMaze, agent: Agent) -> RunResult:
    maze = Maze.from_solvedmaze(solved_maze)
    agent = agent(maze, strategy=config.strategy)
    agent.train()
    _rewards, path, _actions, optimal, epoch = agent.predict()
    path_len = len(path)
    optimal_len = maze.quickest_path_len
    return RunResult(
        grid_n=config.grid_n,
        generator=config.generator,
        maze_idx=config.maze_idx,
        strategy=config.strategy.name,
        optimal=optimal,
        path_len=path_len,
        optimal_len=optimal_len,
        episodes_trained=epoch + 1,
        optimality_ratio=path_len / optimal_len,
        path_regret=path_len - optimal_len,
    )


def build_jobs(collection) -> list[tuple[RunConfig, SolvedMaze]]:
    jobs: list[tuple[RunConfig, SolvedMaze]] = []
    for sub_ds in collection.maze_datasets:
        generator = sub_ds.cfg.maze_ctor.__name__.removeprefix("gen_")
        grid_n = sub_ds.cfg.grid_n
        for maze_idx, solved_maze in enumerate(sub_ds):
            for strategy in STRATEGIES:
                jobs.append(
                    (
                        RunConfig(
                            grid_n=grid_n,
                            generator=generator,
                            maze_idx=maze_idx,
                            strategy=strategy,
                        ),
                        solved_maze,
                    )
                )
    return jobs


def aggregate_results(results: list[RunResult]) -> dict[str, dict[tuple, dict]]:
    grouped: dict[str, dict[tuple, list[RunResult]]] = {
        "strategy": defaultdict(list),
        "strategy_grid": defaultdict(list),
        "strategy_generator": defaultdict(list),
    }
    for result in results:
        grouped["strategy"][result.strategy].append(result)
        grouped["strategy_grid"][(result.strategy, result.grid_n)].append(result)
        grouped["strategy_generator"][(result.strategy, result.generator)].append(
            result
        )

    summaries: dict[str, dict[tuple, dict]] = {}
    for group_name, buckets in grouped.items():
        summaries[group_name] = {}
        for key, bucket in sorted(buckets.items()):
            n = len(bucket)
            summaries[group_name][key] = {
                "n": n,
                "success_rate": sum(r.optimal for r in bucket) / n,
                "mean_optimality_ratio": sum(r.optimality_ratio for r in bucket) / n,
                "mean_path_regret": sum(r.path_regret for r in bucket) / n,
                "mean_episodes": sum(r.episodes_trained for r in bucket) / n,
            }
    return summaries


def print_summary(summaries: dict[str, dict[tuple, dict]]) -> None:
    print("\n=== By strategy ===")
    for strategy, stats in summaries["strategy"].items():
        print(
            f"{strategy:16s}  success={stats['success_rate']:.1%}  "
            f"optimality={stats['mean_optimality_ratio']:.3f}  "
            f"regret={stats['mean_path_regret']:.2f}  "
            f"episodes={stats['mean_episodes']:.1f}"
        )

    print("\n=== By strategy × grid size ===")
    for (strategy, grid_n), stats in summaries["strategy_grid"].items():
        print(
            f"{strategy:16s} grid={grid_n:2d}  success={stats['success_rate']:.1%}  "
            f"optimality={stats['mean_optimality_ratio']:.3f}  "
            f"episodes={stats['mean_episodes']:.1f}"
        )

    print("\n=== By strategy × generator ===")
    for (strategy, generator), stats in summaries["strategy_generator"].items():
        print(
            f"{strategy:16s} {generator:16s}  success={stats['success_rate']:.1%}  "
            f"optimality={stats['mean_optimality_ratio']:.3f}  "
            f"episodes={stats['mean_episodes']:.1f}"
        )


def save_results_csv(results: list[RunResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "strategy_comparison.csv"
    fieldnames = list(asdict(results[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return path


def main() -> None:
    collection = load_dataset(DATASET_PATH)
    jobs = build_jobs(collection)
    print(
        f"Loaded {len(collection)} mazes from {DATASET_PATH} "
        f"({len(jobs)} training runs, hyperparams={default_hyperparams})"
    )
    results: list[RunResult] = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(run_single)(config, maze, SARSA) for config, maze in jobs
    )

    summaries = aggregate_results(results)
    print_summary(summaries)

    csv_path = save_results_csv(results, RESULTS_DIR)
    print(f"\nWrote per-run results to {csv_path}")


if __name__ == "__main__":
    main()
