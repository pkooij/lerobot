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
"""Practice: run the current policy in the simulator and record every attempt with its outcome.

Each round's attempts go into their own LeRobotDataset (images stored as PNG frames), in exactly the
schema of the demo dataset, so practice and demos load through the same dataset/processor path.
Only envs that emit canonical observations (``observation.state``, ``observation.images.*`` as HWC
uint8, already oriented like the demos) are supported, which is what LIBERO does.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.utils import prepare_canonical_observation
from lerobot.processor import policy_output_to_transition, transition_to_policy_action
from lerobot.utils.constants import ACTION

from .advantages import EpisodeRecord

logger = logging.getLogger(__name__)


def practice_features(demo_features: dict[str, dict], camera_keys: list[str], state_key: str) -> dict:
    """Dataset features for practice data: the demo schema, with cameras stored as images (not video)."""
    features: dict[str, dict] = {}
    for key in [*camera_keys, state_key, ACTION]:
        ft = dict(demo_features[key])
        if ft["dtype"] == "video":
            ft = {"dtype": "image", "shape": tuple(ft["shape"]), "names": ft.get("names")}
        features[key] = ft
    return features


ACTION_IS_PAD = "action_is_pad"


def chunk_features(features: dict[str, dict], chunk: int) -> dict:
    """Store a whole executed chunk per frame: ``action`` flattened to ``chunk * action_dim`` values, and
    ``action_is_pad`` (1.0 for steps after the episode ended, filled with the last action)."""
    out = dict(features)
    (action_dim,) = tuple(out[ACTION]["shape"])
    out[ACTION] = {"dtype": "float32", "shape": (chunk * action_dim,), "names": None}
    out[ACTION_IS_PAD] = {"dtype": "float32", "shape": (chunk,), "names": None}
    return out


def env_practice_features(observation: dict[str, Any], action_dim: int) -> dict:
    """Practice features read off one (batched) env observation, for runs without a demo dataset."""
    features: dict[str, dict] = {}
    for key, value in observation.items():
        arr = np.asarray(value)[0]
        if arr.ndim == 3 and arr.dtype == np.uint8:
            features[key] = {
                "dtype": "image",
                "shape": tuple(arr.shape),
                "names": ["height", "width", "channels"],
            }
        elif arr.ndim == 1:
            features[key] = {"dtype": "float32", "shape": tuple(arr.shape), "names": None}
    features[ACTION] = {"dtype": "float32", "shape": (action_dim,), "names": None}
    return features


def create_practice_dataset(root: Path, round_idx: int, fps: int, features: dict) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=f"flex/practice_round_{round_idx:03d}",
        fps=fps,
        features=features,
        root=root,
        use_videos=False,
        image_writer_threads=8,
    )


def _successes(info: dict[str, Any], n_envs: int) -> np.ndarray:
    """Per-env success flags from a vector env's step info (mirrors `lerobot_eval.rollout`)."""
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            flags = final_info.get("is_success", np.zeros(n_envs, dtype=bool))
            return np.asarray(flags, dtype=bool).reshape(n_envs)
        return np.array(
            [bool(item.get("is_success", False)) if isinstance(item, dict) else False for item in final_info]
        )
    if "is_success" in info:
        return np.asarray(info["is_success"], dtype=bool).reshape(n_envs)
    return np.zeros(n_envs, dtype=bool)


@torch.no_grad()
def practice_task(
    env: gym.vector.VectorEnv,
    task_key: str,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    dataset: LeRobotDataset,
    round_idx: int,
    record_keys: list[str],
    instruction_fn: Callable[[str], str] | None = None,
    chunk: int = 0,
) -> list[EpisodeRecord]:
    """Run one batch of attempts (one per sub-env) on one task and append them to ``dataset``.

    ``chunk > 0`` records only every ``chunk``-th frame (the chunk starts), each with the ``chunk`` actions
    executed from it (see `chunk_features`)."""
    if not env.metadata.get("canonical_observations", False):
        raise ValueError("FLEX practice needs an env with canonical observations (e.g. LIBERO).")
    policy.eval()
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    n = env.num_envs
    observation, _ = env.reset()
    task = list(env.call("task_description"))[0]
    prompt = instruction_fn(task) if instruction_fn is not None else task
    max_steps = int(env.call("_max_episode_steps")[0])

    buffers: list[list[dict[str, np.ndarray]]] = [[] for _ in range(n)]
    done = np.zeros(n, dtype=bool)
    success = np.zeros(n, dtype=bool)
    for step in range(max_steps):
        raw = observation
        obs = prepare_canonical_observation(observation)
        obs["task"] = [prompt] * n
        obs = preprocessor(env_preprocessor(obs))
        action = transition_to_policy_action(policy_output_to_transition(policy.select_action(obs)))
        action = postprocessor(action)
        env_action = env_postprocessor({ACTION: action})[ACTION]
        action_np = action.to("cpu").numpy()

        for i in np.nonzero(~done)[0]:
            if chunk and step % chunk:
                buffers[i][-1]["_actions"].append(action_np[i].astype(np.float32))
                continue
            frame = {}
            for key in record_keys:
                value = np.asarray(raw[key][i])
                frame[key] = value.astype(np.float32) if value.dtype == np.float64 else value
            if chunk:
                frame["_actions"] = [action_np[i].astype(np.float32)]
            else:
                frame[ACTION] = action_np[i].astype(np.float32)
            buffers[i].append(frame)

        observation, _, terminated, truncated, info = env.step(env_action.to("cpu").numpy())
        succ = _successes(info, n)
        finished = (np.asarray(terminated) | np.asarray(truncated) | succ) & ~done
        if step == max_steps - 1:
            finished |= ~done
        success |= succ & ~done
        done |= finished
        if done.all():
            break

    records = []
    for i in range(n):
        for frame in buffers[i]:
            if chunk:
                acts = frame.pop("_actions")
                pad = np.zeros(chunk, dtype=np.float32)
                pad[len(acts) :] = 1.0
                acts = acts + [acts[-1]] * (chunk - len(acts))
                frame[ACTION] = np.stack(acts).reshape(-1)
                frame[ACTION_IS_PAD] = pad
            dataset.add_frame({**frame, "task": task})
        dataset.save_episode()
        records.append(
            EpisodeRecord(
                source="practice",
                round=round_idx,
                task_key=task_key,
                success=bool(success[i]),
                length=len(buffers[i]),
                stride=chunk or 1,
            )
        )
    logger.info(
        "practice %s: %d/%d success, lengths %s",
        task_key,
        int(success.sum()),
        n,
        [len(b) for b in buffers],
    )
    return records


def practice_round(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    dataset: LeRobotDataset,
    round_idx: int,
    record_keys: list[str],
    batches_per_task: int = 1,
    **kwargs: Any,
) -> list[EpisodeRecord]:
    """Practice every task ``batches_per_task`` times (``num_envs`` attempts each). Records are in dataset
    episode order. Each batch resets the envs, so the attempts start from different init states."""
    records: list[EpisodeRecord] = []
    for suite, tasks in envs.items():
        for task_id, env in tasks.items():
            for _ in range(batches_per_task):
                records += practice_task(
                    env,
                    f"{suite}/{task_id}",
                    dataset=dataset,
                    round_idx=round_idx,
                    record_keys=record_keys,
                    **kwargs,
                )
    dataset.finalize()
    return records
