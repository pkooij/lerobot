# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pick a small, fixed set of demos per LIBERO task: FLEX's few-shot demonstration set D_demo.

The same selection must feed the SFT run that produces the starting policy and FLEX's demo buffer.
Print it for `lerobot-train --dataset.episodes=...`:

    python -m lerobot.rl.flex.select_demos --suite libero_spatial --demos_per_task 10
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
from collections.abc import Iterable, Sequence

import numpy as np


def normalize_task(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


def libero_task_descriptions(suite: str) -> list[str]:
    """Language instructions of a LIBERO suite, in task-id order (needs the `libero` package)."""
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[suite]()
    return [task_suite.get_task(i).language for i in range(task_suite.n_tasks)]


def episode_task_strings(repo_id: str, revision: str | None = None) -> list[list[str]]:
    """Task string(s) of every episode of a dataset, in episode-index order.

    Uses the episode metadata's ``tasks`` column when present; otherwise (e.g. ``lerobot/libero``)
    reads each episode's first-frame ``task_index`` from the frame table (no videos are downloaded).
    """
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta = LeRobotDatasetMetadata(repo_id, revision=revision)
    if "tasks" in meta.episodes.column_names:
        return [list(t) for t in meta.episodes["tasks"]]
    frames = LeRobotDataset(repo_id, revision=revision, download_videos=False).hf_dataset
    task_index = np.asarray(frames.with_format("numpy")["task_index"]).reshape(-1)
    index_to_task = {int(i): str(t) for t, i in meta.tasks["task_index"].items()}
    starts = np.asarray(meta.episodes["dataset_from_index"], dtype=np.int64)
    return [[index_to_task[int(task_index[s])]] for s in starts]


def select_demo_episodes(
    episode_tasks: Sequence[Sequence[str] | None],
    task_descriptions: Iterable[str],
    demos_per_task: int,
    seed: int,
) -> list[int]:
    """Choose ``demos_per_task`` episodes for each task description (seeded, so SFT and FLEX agree).

    ``episode_tasks[i]`` is the list of task strings of dataset episode ``i`` (see `episode_task_strings`).
    """
    by_task: dict[str, list[int]] = {}
    for ep_idx, tasks in enumerate(episode_tasks):
        if tasks:
            by_task.setdefault(normalize_task(tasks[0]), []).append(ep_idx)

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for description in task_descriptions:
        candidates = by_task.get(normalize_task(description))
        if not candidates:
            raise ValueError(f"No demo episodes for task {description!r}")
        if len(candidates) < demos_per_task:
            raise ValueError(
                f"Task {description!r} has {len(candidates)} demos, fewer than demos_per_task={demos_per_task}"
            )
        selected += sorted(rng.choice(candidates, size=demos_per_task, replace=False).tolist())
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--demos_per_task", type=int, default=10)
    parser.add_argument("--repo_id", default="lerobot/libero")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seed", type=int, default=1000, help="Must match FLEX's --seed.")
    args = parser.parse_args()

    # LIBERO prints "[info] ..." to stdout on import; keep stdout to the episode list so that
    # EPISODES=$(python -m lerobot.rl.flex.select_demos ...) can be passed straight to lerobot-train.
    with contextlib.redirect_stdout(sys.stderr):
        episodes = select_demo_episodes(
            episode_task_strings(args.repo_id, args.revision),
            libero_task_descriptions(args.suite),
            args.demos_per_task,
            args.seed,
        )
    print("[" + ",".join(str(e) for e in episodes) + "]")


if __name__ == "__main__":
    main()
