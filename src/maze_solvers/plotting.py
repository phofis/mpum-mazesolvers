import io
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from maze_dataset import SolvedMaze
from maze_dataset.constants import Coord
from maze_dataset.plotting import MazePlot
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from maze_solvers.maze_env import History

_LABEL_FONT: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None


def _path_key(path: list[Coord]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(c) for c in coord) for coord in path)


def _get_label_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    global _LABEL_FONT
    if _LABEL_FONT is None:
        try:
            font_path = fm.findfont("DejaVu Sans Bold", fallback_to_default=False)
            _LABEL_FONT = ImageFont.truetype(font_path, 12)
        except (OSError, ValueError):
            _LABEL_FONT = ImageFont.load_default()
    return _LABEL_FONT


def _save_reference_maze(solved_maze: SolvedMaze, output_dir: Path) -> None:
    mp = MazePlot(solved_maze)
    mp.plot(title="Reference maze")
    mp.fig.savefig(output_dir / "reference_maze.png", bbox_inches="tight", dpi=100)
    plt.close(mp.fig)


def _save_metric_plot(
    epochs: list[int],
    values: list[float],
    ylabel: str,
    filename: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(epochs, values)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    fig.savefig(output_dir / filename, bbox_inches="tight", dpi=100)
    plt.close(fig)


def _gif_epochs(epochs: list[int], step: int = 5) -> list[int]:
    if not epochs:
        return []
    sampled = epochs[::step]
    last = epochs[-1]
    if last not in sampled:
        sampled.append(last)
    return sampled


def _step_indices(path_len: int, h: int) -> list[int]:
    if path_len == 0:
        return []
    indices = list(range(0, path_len, h))
    last = path_len - 1
    if last not in indices:
        indices.append(last)
    return indices


def _render_maze_base(solved_maze: SolvedMaze, path: list[Coord]) -> Image.Image:
    mp = MazePlot(solved_maze)
    mp.add_predicted_path(path)
    mp.plot(plain=True)

    buf = io.BytesIO()
    mp.fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(mp.fig)
    buf.seek(0)
    image = Image.open(buf).copy()
    buf.close()
    return image


def _render_maze_base_worker(
    args: tuple[tuple[tuple[int, ...], ...], list[Coord], SolvedMaze],
) -> tuple[tuple[tuple[int, ...], ...], Image.Image]:
    matplotlib.use("Agg")
    path_key, path, solved_maze = args
    return path_key, _render_maze_base(solved_maze, path)


def _render_maze_bases_parallel(
    solved_maze: SolvedMaze,
    path_by_key: dict[tuple[tuple[int, ...], ...], list[Coord]],
    max_workers: int | None = None,
) -> dict[tuple[tuple[int, ...], ...], Image.Image]:
    if not path_by_key:
        return {}

    workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    if workers <= 1 or len(path_by_key) <= 1:
        return {
            key: _render_maze_base(solved_maze, path)
            for key, path in path_by_key.items()
        }

    tasks = [(key, path, solved_maze) for key, path in path_by_key.items()]
    cache: dict[tuple[tuple[int, ...], ...], Image.Image] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for path_key, image in executor.map(
            _render_maze_base_worker, tasks, chunksize=4
        ):
            cache[path_key] = image
    return cache


def _add_label(image: Image.Image, label: str) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font = _get_label_font()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    padding = 6
    x = int(img.width * 0.02)
    y = int(img.height * 0.02)
    draw.rounded_rectangle(
        [
            x - padding,
            y - padding,
            x + text_w + padding,
            y + text_h + padding,
        ],
        radius=4,
        fill=(0, 0, 0, 179),
    )
    draw.text((x, y), label, fill="white", font=font)
    return img


@dataclass(frozen=True)
class LabeledFrame:
    path_key: tuple[tuple[int, ...], ...]
    label: str


def _register_path(
    path: list[Coord],
    label: str,
    frames: list[LabeledFrame],
    path_by_key: dict[tuple[tuple[int, ...], ...], list[Coord]],
) -> None:
    key = _path_key(path)
    path_by_key.setdefault(key, path)
    frames.append(LabeledFrame(path_key=key, label=label))


def _collect_training_frames(
    history: History,
    epochs: list[int],
) -> tuple[list[LabeledFrame], dict[tuple[tuple[int, ...], ...], list[Coord]]]:
    frames: list[LabeledFrame] = []
    path_by_key: dict[tuple[tuple[int, ...], ...], list[Coord]] = {}
    for epoch in epochs:
        _register_path(
            history.steps[epoch],
            f"Epoch {epoch}",
            frames,
            path_by_key,
        )
    return frames, path_by_key


def _collect_animation_frames(
    history: History,
    epochs: list[int],
    step_interval: int,
) -> tuple[list[LabeledFrame], dict[tuple[tuple[int, ...], ...], list[Coord]]]:
    frames: list[LabeledFrame] = []
    path_by_key: dict[tuple[tuple[int, ...], ...], list[Coord]] = {}
    for epoch in epochs:
        path = history.steps[epoch]
        total_steps = len(path) - 1
        for step_idx in _step_indices(len(path), step_interval):
            _register_path(
                path[: step_idx + 1],
                f"Epoch {epoch}, step {step_idx}/{total_steps}",
                frames,
                path_by_key,
            )
    return frames, path_by_key


def _assemble_labeled_frames(
    frames: list[LabeledFrame],
    base_cache: dict[tuple[tuple[int, ...], ...], Image.Image],
) -> list[Image.Image]:
    return [_add_label(base_cache[frame.path_key], frame.label) for frame in frames]


def _save_gif(
    images: list[Image.Image],
    output_path: Path,
    duration_ms: int = 100,
) -> None:
    if not images:
        return
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def _save_training_gif(
    history: History,
    solved_maze: SolvedMaze,
    epochs: list[int],
    output_dir: Path,
    *,
    base_cache: dict[tuple[tuple[int, ...], ...], Image.Image] | None = None,
    max_workers: int | None = None,
) -> None:
    frames, path_by_key = _collect_training_frames(history, epochs)
    cache = dict(base_cache or {})
    missing = {k: path for k, path in path_by_key.items() if k not in cache}
    cache.update(_render_maze_bases_parallel(solved_maze, missing, max_workers))
    images = _assemble_labeled_frames(frames, cache)
    _save_gif(images, output_dir / "training_paths.gif")


def plotPathAnimation(
    history: History,
    solved_maze: SolvedMaze,
    output_dir: str | Path = "plotting",
    *,
    epoch_interval: int = 10,
    step_interval: int = 5,
    filename: str = "path_animation.gif",
    frame_duration_ms: int = 25,
    base_cache: dict[tuple[tuple[int, ...], ...], Image.Image] | None = None,
    max_workers: int | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = _gif_epochs(sorted(history.steps.keys()), step=epoch_interval)
    frames, path_by_key = _collect_animation_frames(
        history, epochs, step_interval
    )
    cache = dict(base_cache or {})
    missing = {k: path for k, path in path_by_key.items() if k not in cache}
    cache.update(_render_maze_bases_parallel(solved_maze, missing, max_workers))
    images = _assemble_labeled_frames(frames, cache)

    output_path = output_dir / filename
    _save_gif(images, output_path, frame_duration_ms)
    return output_path


def _save_path_gifs(
    history: History,
    solved_maze: SolvedMaze,
    output_dir: Path,
    *,
    training_epoch_interval: int = 5,
    animation_epoch_interval: int = 10,
    animation_step_interval: int = 5,
    training_duration_ms: int = 100,
    animation_duration_ms: int = 25,
    max_workers: int | None = None,
) -> None:
    epochs = sorted(history.steps.keys())
    training_epochs = _gif_epochs(epochs, training_epoch_interval)
    animation_epochs = _gif_epochs(epochs, animation_epoch_interval)

    training_frames, training_paths = _collect_training_frames(
        history, training_epochs
    )
    animation_frames, animation_paths = _collect_animation_frames(
        history, animation_epochs, animation_step_interval
    )

    path_by_key = {**training_paths, **animation_paths}
    base_cache = _render_maze_bases_parallel(
        solved_maze, path_by_key, max_workers
    )

    training_images = _assemble_labeled_frames(training_frames, base_cache)
    animation_images = _assemble_labeled_frames(animation_frames, base_cache)

    _save_gif(training_images, output_dir / "training_paths.gif", training_duration_ms)
    _save_gif(
        animation_images,
        output_dir / "path_animation.gif",
        animation_duration_ms,
    )


def plotHistory(
    history: History,
    solvedMaze: SolvedMaze,
    output_dir: str | Path = "plotting",
    *,
    max_workers: int | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = sorted(history.steps.keys())
    path_lengths = [len(history.steps[epoch]) for epoch in epochs]
    total_rewards = [np.sum(history.rewards.get(epoch, [])) for epoch in epochs]

    _save_reference_maze(solvedMaze, output_dir)
    _save_metric_plot(
        epochs,
        path_lengths,
        "Path length",
        "path_length_vs_epoch.png",
        output_dir,
    )
    _save_metric_plot(
        epochs,
        total_rewards,
        "Total reward",
        "reward_vs_epoch.png",
        output_dir,
    )
    _save_path_gifs(history, solvedMaze, output_dir, max_workers=max_workers)

    return output_dir
