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
"""FLEX on LIBERO: a fine-tuned flow-matching VLA improves from its own practice, with no human corrections.

Follows Sereact's FLEX poster ("FLEX: Adaptive Experience Replay for Sample-Efficient Flow Policy
Gradients in Vision-Language-Action Models", R2RL workshop, CoRL 2026). Each iteration c:

1. Data collection - roll out pi_theta_c, ``episodes_per_task`` attempts per task, success/failure only.
2. Replay buffer   - keep the last ``replay_window`` iterations; data from d iterations ago has weight
                     2 ** (-d / h_s).
3. Critic V_phi    - fit on the buffer (cross-fitted); q_c = clip(E_c / E*, q_min, 1) from its explained
                     variance.
4. Advantages      - A = G - V_phi(s), standardized within each task.
5. Actor update    - L = q_c * L_PG + beta * E_demo[l_theta], with L_PG the FPO++ flow policy gradient on
                     transitions sampled ~ |A| (+ uniform), and the consensus (cautious) optimizer.

Example (after the SFT stage, see docs/source/flex_libero.mdx):

    python -m lerobot.rl.flex.train_flex \\
        --policy.path=<user>/smolvla_libero_spatial_10demos --policy.n_action_steps=10 \\
        --env.type=libero --env.task=libero_spatial --env.observation_height=256 --env.observation_width=256 \\
        --demos_per_task=10 --rounds=10 --episodes_per_task=5
"""

import copy
import json
import logging
import time
from collections import defaultdict
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.optim.cautious import CautiousAdamW
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import bind_relative_anchor
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from . import advantages as adv
from .advantages import EpisodeRecord
from .configuration_flex import PG_MODES, FlexConfig
from .critic import FrameEncoder, FrameFeatures, build_critic_inputs, encode_dataset, fit_cross_fitted_critic
from .flow_pg import cfm_loss, flow_pg_loss
from .practice import (
    ACTION_IS_PAD,
    chunk_features,
    create_practice_dataset,
    env_practice_features,
    practice_features,
    practice_round,
)
from .select_demos import episode_task_strings, select_demo_episodes

logger = logging.getLogger(__name__)

RECAP_POSITIVE = "positive"
RECAP_NEGATIVE = "negative"


def recap_prompt(task: str, tag: str) -> str:
    return f"{task.rstrip()} Advantage: {tag}."


@dataclass
class MemoryRound:
    """One iteration's practice data in the replay buffer."""

    round: int
    records: list[EpisodeRecord]
    train_ds: LeRobotDataset  # opened with the policy's delta_timestamps
    features: FrameFeatures  # critic inputs, in dataset position order


class WeightedView(torch.utils.data.Dataset):
    """A dataset restricted to the keys the policy trains on, with each frame's standardized advantage
    (``flex_adv``) and, for RECAP, an advantage tag appended to the instruction."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        keys: list[str],
        advantages: np.ndarray | None = None,
        tags: list[str] | None = None,
        chunk: int = 0,
    ):
        if advantages is None:
            advantages = np.zeros(len(dataset), dtype=np.float32)
        if len(advantages) != len(dataset):
            raise ValueError(f"{len(advantages)} advantages for a dataset of {len(dataset)} frames")
        self.dataset = dataset
        self.keys = keys
        self.advantages = np.asarray(advantages, dtype=np.float32)
        self.tags = tags
        self.chunk = chunk

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        out = {k: v for k, v in item.items() if k in self.keys or k.endswith("_is_pad")}
        if self.chunk:  # recorded chunks (see practice.chunk_features): (chunk * dim,) -> (chunk, dim)
            out[ACTION] = out[ACTION].reshape(self.chunk, -1)
            out[ACTION_IS_PAD] = out[ACTION_IS_PAD].bool()
        out["task"] = item["task"] if self.tags is None else recap_prompt(item["task"], self.tags[idx])
        out["flex_adv"] = torch.tensor(self.advantages[idx])
        return out


class InstructionWrapper:
    """Wrap a policy preprocessor so every observation's instruction is rewritten (RECAP prompting)."""

    def __init__(self, preprocessor: Any, tag: str):
        self._pre = preprocessor
        self._tag = tag

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        if "task" in observation:
            tasks = observation["task"]
            observation = dict(observation)
            observation["task"] = (
                [recap_prompt(t, self._tag) for t in tasks]
                if isinstance(tasks, list)
                else recap_prompt(tasks, self._tag)
            )
        return self._pre(observation)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pre, name)


def upcast_trainable(policy: torch.nn.Module) -> int:
    """Store the trainable weights in fp32; frozen ones keep their dtype. Returns how many were cast.

    SmolVLA loads its action expert in bf16, where an Adam step of ~lr per weight is below the rounding step
    of most weights: at lr 1e-5 only 16% of them change. The layers cast their inputs to their weight
    dtype, so the expert then also computes in fp32."""
    cast = 0
    for p in policy.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            cast += p.numel()
    return cast


def share_frozen_weights(reference: torch.nn.Module, policy: torch.nn.Module) -> int:
    """Point the reference's frozen parameters at the policy's (they never change), so a large frozen
    backbone (pi05's 2.9B-parameter VLM) is held once. Returns the number of parameters shared."""
    shared = 0
    for (name, p_ref), (name_pol, p) in zip(
        reference.named_parameters(), policy.named_parameters(), strict=True
    ):
        if name != name_pol:
            raise RuntimeError(f"Parameter order differs: {name} vs {name_pol}")
        if not p.requires_grad:
            p_ref.data = p.data
            shared += p.numel()
    return shared


def _episode_frame_slices(records: list[EpisodeRecord]) -> list[slice]:
    starts = np.concatenate([[0], np.cumsum([r.length for r in records])])
    return [slice(int(starts[i]), int(starts[i + 1])) for i in range(len(records))]


def _in_use(mem_round: int, current_round: int, replay: bool) -> bool:
    return replay or mem_round == current_round


def actor_frames(mem: MemoryRound, cfg: FlexConfig, horizon: int) -> np.ndarray:
    """Frames of one round the actor learns from, in dataset position order.

    The policy samples a chunk every ``horizon`` steps (from ``policy.reset()`` at frame 0) and executes its
    first ``horizon`` actions. Only at those frames is the recorded ``horizon``-step action a sample of
    ``pi(. | s_t)``; anywhere else it is the tail of one chunk joined to the head of the next, which was
    sampled from a later observation. The policy gradient therefore uses the chunk-start frames only. The
    supervised modes imitate the executed trajectory and keep every frame.
    """
    frame_index = np.asarray(mem.features.frame_index)
    if cfg.mode not in PG_MODES or horizon <= 1 or cfg.record_chunks:
        return np.ones(len(frame_index), dtype=bool)
    return frame_index % horizon == 0


def contrast_frames(memory: list[MemoryRound], cfg: FlexConfig, current_round: int) -> list[np.ndarray]:
    """Per round, the frames of tasks whose actor window holds both a success and a failure.

    In a task the policy always fails (or always solves) in the window, every return is the same, so
    ``A = G - V`` is the critic's error alone, and per-task standardization would scale that noise to unit
    variance. Such tasks get no policy gradient (their demos still enter the BC term).
    """
    if cfg.mode not in PG_MODES or not cfg.require_contrast:
        return [np.ones(sum(r.length for r in mem.records), dtype=bool) for mem in memory]
    outcomes: dict[str, set[bool]] = defaultdict(set)
    for mem in memory:
        if _in_use(mem.round, current_round, cfg.actor_replay):
            for rec in mem.records:
                outcomes[rec.task_key].add(rec.success)
    keep = {task for task, seen in outcomes.items() if len(seen) == 2}
    return [np.concatenate([np.full(r.length, r.task_key in keep) for r in mem.records]) for mem in memory]


def judge(
    memory: list[MemoryRound],
    cfg: FlexConfig,
    task_index: dict[str, int],
    max_steps: dict[str, int],
    current_round: int,
    device: torch.device,
    horizon: int,
) -> tuple[list[np.ndarray], float, dict[str, float]]:
    """Fit the critic (on every frame), return per-round standardized advantages ``A = G - V``, ``q_c`` and
    stats. Advantages are standardized over the frames the actor learns from (see `actor_frames`)."""
    image, state, task_ids, time_frac, returns, weights, ep_ids, outcome, critic_on, actor_on = (
        [] for _ in range(10)
    )
    ep_counter = 0
    for mem in memory:
        if len(mem.features.episode_index) != sum(r.length for r in mem.records):
            raise RuntimeError(f"Round {mem.round}: feature count does not match recorded frames")
        image.append(mem.features.image)
        state.append(mem.features.state)
        stale = adv.staleness_weights(np.array([mem.round]), current_round, cfg.staleness_half_life)[0]
        use_critic = _in_use(mem.round, current_round, cfg.critic_replay)
        use_actor = _in_use(mem.round, current_round, cfg.actor_replay)
        for rec in mem.records:
            n = rec.length
            task_ids.append(np.full(n, task_index[rec.task_key]))
            time_frac.append(np.arange(n) * rec.stride / max_steps[rec.task_key])
            returns.append(adv.value_targets(n, rec.success, cfg.gamma**rec.stride))
            weights.append(np.full(n, stale))
            ep_ids.append(np.full(n, ep_counter))
            outcome.append(np.full(n, rec.success))
            critic_on.append(np.full(n, use_critic))
            ep_counter += 1
        actor_on.append(actor_frames(mem, cfg, horizon) & use_actor)

    task_ids_a = np.concatenate(task_ids)
    returns_a = np.concatenate(returns)
    outcome_a = np.concatenate(outcome)
    critic_mask = np.concatenate(critic_on)
    actor_mask = np.concatenate(actor_on)
    inputs = build_critic_inputs(
        torch.cat(image), torch.cat(state), task_ids_a, len(task_index), np.concatenate(time_frac)
    )
    fit = fit_cross_fitted_critic(
        inputs,
        returns_a,
        np.concatenate(weights),
        np.concatenate(ep_ids),
        train_mask=critic_mask,
        n_folds=cfg.critic_folds,
        steps=cfg.critic_steps,
        batch_size=cfg.critic_batch_size,
        lr=cfg.critic_lr,
        hidden=cfg.critic_hidden,
        device=device,
        seed=cfg.seed + current_round,
    )
    values = fit.oof_values

    # Critic quality: explained variance of the return, out of fold.
    ev = adv.explained_variance(values[critic_mask], returns_a[critic_mask])
    q_c = adv.critic_scale(ev, cfg.ev_target, cfg.q_min) if cfg.critic_scaling else 1.0

    raw = returns_a - values
    contrast = np.concatenate(contrast_frames(memory, cfg, current_round))
    groups = task_ids_a if cfg.per_task_std else np.zeros_like(task_ids_a)
    std_adv = adv.per_task_standardize(raw, groups, fit_mask=actor_mask & contrast)
    std_adv[~contrast] = 0.0

    per_round, offset = [], 0
    for mem in memory:
        n = sum(r.length for r in mem.records)
        per_round.append(std_adv[offset : offset + n])
        offset += n

    task_aurocs = [
        a
        for t in np.unique(task_ids_a)
        if np.isfinite(
            a := adv.binary_auroc(
                values[(task_ids_a == t) & critic_mask], outcome_a[(task_ids_a == t) & critic_mask]
            )
        )
    ]
    stats = {
        "critic/explained_variance": ev,
        "critic/q_c": q_c,
        "critic/auroc_within_task": float(np.mean(task_aurocs)) if task_aurocs else float("nan"),
        "critic/oof_bce": fit.oof_bce,
        "critic/train_bce": fit.train_loss,
        "critic/value_mean": float(values[critic_mask].mean()),
        "adv/raw_std": float(raw[actor_mask].std()),
        "adv/frac_positive": float((std_adv[actor_mask] > 0).mean()),
        "adv/tasks_without_contrast": float(len(np.unique(task_ids_a[actor_mask & ~contrast]))),
    }
    return per_round, q_c, stats


def build_loaders(
    cfg: FlexConfig,
    demo_ds: LeRobotDataset | None,
    memory: list[MemoryRound],
    round_adv: list[np.ndarray],
    current_round: int,
    keys: list[str],
    horizon: int,
) -> tuple[torch.utils.data.DataLoader | None, torch.utils.data.DataLoader | None, dict[str, float]]:
    """Practice loader (replay, sampled by staleness x advantage priority over `actor_frames`) and demo
    loader, each sized for one iteration of ``steps_per_round`` updates."""
    used = [
        (mem, a, c)
        for mem, a, c in zip(memory, round_adv, contrast_frames(memory, cfg, current_round), strict=True)
        if _in_use(mem.round, current_round, cfg.actor_replay)
    ]
    views: list[torch.utils.data.Dataset] = []
    stale_l, adv_l, success_l, frames_l = [], [], [], []
    all_adv = np.concatenate([a for _, a, _ in used])
    threshold = np.quantile(all_adv, 1.0 - cfg.recap_positive_frac) if cfg.mode == "recap" else 0.0
    for mem, a, contrast in used:
        tags = None
        if cfg.mode == "recap":
            tags = [RECAP_POSITIVE if x >= threshold else RECAP_NEGATIVE for x in a]
        views.append(WeightedView(mem.train_ds, keys, a, tags, chunk=horizon if cfg.record_chunks else 0))
        stale = adv.staleness_weights(np.array([mem.round]), current_round, cfg.staleness_half_life)[0]
        stale_l.append(np.full(len(a), stale))
        adv_l.append(a)
        success_l.append(np.concatenate([np.full(r.length, r.success) for r in mem.records]))
        frames_l.append(actor_frames(mem, cfg, horizon) & contrast)
    stale_a = np.concatenate(stale_l)
    frames_a = np.concatenate(frames_l)
    if cfg.mode in PG_MODES:
        # Priorities are normalized over the frames that can be drawn.
        probs = np.zeros(len(frames_a))
        probs[frames_a] = adv.sampling_probs(
            np.concatenate(adv_l)[frames_a], stale_a[frames_a], cfg.uniform_mix, cfg.prioritized_sampling
        )
    elif cfg.mode == "filtered_bc":
        probs = stale_a * np.concatenate(success_l)
    else:  # recap
        probs = stale_a

    pin = torch.cuda.is_available()
    practice_loader = None
    if probs.sum() > 0:
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(probs / probs.sum(), dtype=torch.double),
            num_samples=cfg.steps_per_round * cfg.batch_size,
            replacement=True,
        )
        practice_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(views),
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=pin,
            drop_last=True,
        )

    demo_loader = None
    if demo_ds is not None and (cfg.mode not in PG_MODES or cfg.beta > 0):
        demo_tags = [RECAP_POSITIVE] * len(demo_ds) if cfg.mode == "recap" else None
        demo_view = WeightedView(demo_ds, keys, None, demo_tags)
        demo_loader = torch.utils.data.DataLoader(
            demo_view,
            batch_size=cfg.demo_batch_size,
            sampler=torch.utils.data.RandomSampler(
                demo_view, replacement=True, num_samples=cfg.steps_per_round * cfg.demo_batch_size
            ),
            num_workers=cfg.num_workers,
            pin_memory=pin,
            drop_last=True,
        )

    p = probs / probs.sum() if probs.sum() > 0 else probs
    stats = {
        "buffer/frames": float(len(probs)),
        "buffer/actor_frames": float(frames_a.sum()),
        "buffer/rounds": float(len(used)),
        "sample/ess_frac": float(1.0 / (len(p) * (p**2).sum())) if p.sum() > 0 else 0.0,
        "sample/current_round_mass": float(p[-len(adv_l[-1]) :].sum()) if p.sum() > 0 else 0.0,
    }
    return practice_loader, demo_loader, stats


def improve(
    policy: torch.nn.Module,
    reference: torch.nn.Module | None,
    preprocessor: Any,
    optimizer: torch.optim.Optimizer,
    practice_loader: torch.utils.data.DataLoader | None,
    demo_loader: torch.utils.data.DataLoader | None,
    cfg: FlexConfig,
    q_c: float,
    horizon: int,
    device: torch.device,
    round_idx: int,
) -> dict[str, float]:
    """One iteration of actor updates.

    PG modes: ``q_c * L_PG(practice) + beta * l_theta(demos)``. The policy runs in eval mode (no dropout)
    so the ratio against ``reference`` is exactly 1 before the first step. Supervised modes:
    flow matching on (filtered or tagged) practice plus demos.
    """
    pg = cfg.mode in PG_MODES
    if pg and reference is None:
        raise ValueError("Policy-gradient modes need a reference policy")
    policy.train(not pg or cfg.pg_train_mode)
    if reference is not None:
        reference.train(cfg.pg_train_mode)
    params = [p for p in policy.parameters() if p.requires_grad]

    def amp() -> AbstractContextManager:
        if cfg.use_amp and device.type == "cuda":
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        return nullcontext()

    rng = np.random.default_rng(cfg.seed * 1000 + round_idx)
    practice_iter = iter(practice_loader) if practice_loader is not None else None
    demo_iter = iter(demo_loader) if demo_loader is not None else None
    if practice_iter is None and demo_iter is None:
        return {}
    logs: dict[str, list[float]] = defaultdict(list)
    t0 = time.perf_counter()
    for step in range(cfg.steps_per_round):
        loss = torch.zeros((), device=device)
        if practice_iter is not None:
            batch = next(practice_iter)
            a = batch.pop("flex_adv").to(device, non_blocking=True).float()
            batch = preprocessor(batch)
            if pg:
                assert reference is not None
                pg_loss, pg_stats = flow_pg_loss(
                    policy,
                    reference,
                    batch,
                    a,
                    n_mc=cfg.n_mc,
                    clip_eps=cfg.clip_eps,
                    kind=cfg.surrogate,
                    seed=int(rng.integers(2**31 - 1)),
                    horizon=horizon,
                    reduction=cfg.loss_reduction,
                    amp=amp,
                )
                loss = loss + q_c * pg_loss
                for k, v in pg_stats.items():
                    logs[k].append(v)
            else:
                with amp():
                    fm = cfm_loss(policy, batch, "mean").mean()
                loss = loss + fm
                logs["fm/practice_loss"].append(float(fm.detach()))
        if demo_iter is not None:
            batch = next(demo_iter)
            batch.pop("flex_adv")
            batch = preprocessor(batch)
            with amp():
                bc = cfm_loss(policy, batch, cfg.loss_reduction if pg else "mean").mean()
            loss = loss + (cfg.beta if pg else 1.0) * bc
            logs["bc/demo_loss"].append(float(bc.detach()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip_norm > 0:
            logs["improve/grad_norm"].append(
                float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm))
            )
        optimizer.step()
        logs["improve/loss"].append(float(loss.detach()))
        if isinstance(optimizer, CautiousAdamW):
            logs["improve/cautious_kept_frac"].append(optimizer.last_kept_fraction)
        if (step + 1) % 100 == 0:
            logger.info(
                "improve step %d/%d loss %.4f (%.2f it/s)",
                step + 1,
                cfg.steps_per_round,
                np.mean(logs["improve/loss"][-100:]),
                (step + 1) / (time.perf_counter() - t0),
            )
    policy.eval()
    if reference is not None:
        reference.eval()
    return {k: float(np.mean(v)) for k, v in logs.items()} | {"improve/q_c": q_c}


def evaluate(
    cfg: FlexConfig, policy, preprocessor, postprocessor, env_pre, env_post, envs=None
) -> dict[str, Any]:
    """Evaluate on fresh envs, or on ``envs`` (left open) when given."""
    from lerobot.scripts.lerobot_eval import eval_policy_all

    eval_envs = envs or make_env(cfg.env, n_envs=cfg.eval_batch_size, use_async_envs=cfg.use_async_envs)
    pre = InstructionWrapper(preprocessor, RECAP_POSITIVE) if cfg.mode == "recap" else preprocessor
    with torch.no_grad():
        info = eval_policy_all(
            envs=eval_envs,
            policy=policy,
            env_preprocessor=env_pre,
            env_postprocessor=env_post,
            preprocessor=pre,
            postprocessor=postprocessor,
            n_episodes=cfg.eval_episodes_per_task,
            start_seed=cfg.seed,
            max_parallel_tasks=1,
        )
    if envs is None:
        close_envs(eval_envs)
    overall = info["overall"]
    per_task = {
        f"{t['task_group']}/{t['task_id']}": 100.0 * float(np.mean(t["metrics"]["successes"]))
        for t in info["per_task"]
        if t.get("metrics", {}).get("successes")
    }
    return {
        "eval/pc_success": overall["pc_success"],
        "eval/pc_success_ci95": overall["pc_success_ci95"],
        "eval/n_episodes": overall["n_episodes"],
        "eval/per_task": per_task,
    }


def _summarize_practice(records: list[EpisodeRecord]) -> dict[str, float]:
    succ = [r for r in records if r.success]
    return {
        "practice/success_rate": len(succ) / max(1, len(records)),
        "practice/n_episodes": float(len(records)),
        "practice/mean_success_length": float(np.mean([r.length for r in succ])) if succ else float("nan"),
    }


@parser.wrap()
def train_flex(cfg: FlexConfig) -> None:
    init_logging()
    cfg.validate()
    logger.info(pformat(asdict(cfg)))
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"

    def log(metrics: dict[str, Any]) -> None:
        logger.info(pformat(metrics))
        with metrics_path.open("a") as f:
            f.write(json.dumps(metrics, default=float) + "\n")

    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    # Practice frames are stored in the demos' schema, so the sim must render at the demos' resolution.
    meta = LeRobotDatasetMetadata(cfg.demo_repo_id, revision=cfg.demo_revision) if cfg.demo_repo_id else None
    demo_cams = [k for k in meta.camera_keys if k.startswith(OBS_IMAGES)] if meta else []
    if meta and demo_cams and hasattr(cfg.env, "observation_height"):
        height, width = (int(x) for x in meta.features[demo_cams[0]]["shape"][:2])
        if (cfg.env.observation_height, cfg.env.observation_width) != (height, width):
            logger.info(
                "Rendering the sim at the demos' %dx%d (was %dx%d)",
                height,
                width,
                cfg.env.observation_height,
                cfg.env.observation_width,
            )
            cfg.env.observation_height, cfg.env.observation_width = height, width

    (out / "flex_config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    # --- Envs, policy, processors ---
    envs_per_task = cfg.practice_envs_per_task or cfg.episodes_per_task
    practice_envs = make_env(cfg.env, n_envs=envs_per_task, use_async_envs=cfg.use_async_envs)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    task_descriptions = {}
    first_obs: dict[str, Any] = {}
    for suite, tasks in practice_envs.items():
        for tid, env in tasks.items():
            # Sim envs only report their task description after a reset.
            first_obs, _ = env.reset(seed=cfg.seed)
            task_descriptions[f"{suite}/{tid}"] = list(env.call("task_description"))[0]
    max_steps = {
        f"{suite}/{tid}": int(env.call("_max_episode_steps")[0])
        for suite, tasks in practice_envs.items()
        for tid, env in tasks.items()
    }
    task_index = {key: i for i, key in enumerate(sorted(task_descriptions))}

    policy = make_policy(cfg.policy, env_cfg=cfg.env)
    policy.eval()
    if cfg.fp32_trainable:
        upcast_trainable(policy)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    # Relative-action policies (pi05): hold each chunk's state anchor until the chunk is used up.
    bind_relative_anchor(policy, preprocessor)
    horizon = int(getattr(cfg.policy, "n_action_steps", 1))
    chunk = horizon if cfg.record_chunks else 0

    # --- Demos (fixed offline buffer) ---
    demo_ds: LeRobotDataset | None = None
    delta_timestamps: dict[str, list[float]] | None = None
    if meta is not None:
        assert cfg.demo_repo_id is not None
        demo_episodes = cfg.demo_episodes or select_demo_episodes(
            episode_task_strings(cfg.demo_repo_id, cfg.demo_revision),
            [task_descriptions[k] for k in sorted(task_descriptions)],
            cfg.demos_per_task,
            cfg.seed,
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, meta)
        demo_ds = LeRobotDataset(
            cfg.demo_repo_id,
            episodes=demo_episodes,
            delta_timestamps=delta_timestamps,
            revision=cfg.demo_revision,
            video_backend=cfg.video_backend,
        )
        camera_keys = [k for k in meta.camera_keys if k.startswith(OBS_IMAGES)]
        feats = practice_features(meta.features, camera_keys, OBS_STATE)
        fps = meta.fps
        logger.info(
            "Demo buffer: %d episodes, %d frames; cameras %s", len(demo_episodes), len(demo_ds), camera_keys
        )
    else:
        assert cfg.policy.output_features is not None
        (action_dim,) = cfg.policy.output_features[ACTION].shape
        feats = env_practice_features(first_obs, action_dim)
        camera_keys = [k for k in feats if k.startswith(OBS_IMAGES)]
        fps = cfg.env.fps
        logger.info("No demos; practice cameras %s", camera_keys)
    if cfg.record_chunks:
        feats = chunk_features(feats, chunk)
        # Each recorded frame already holds its chunk of actions.
        delta_timestamps = None
    record_keys = [*camera_keys, OBS_STATE]
    train_keys = [*record_keys, ACTION]

    # --- Optimizer (state persists across rounds: the cautious gate compares against the running trend) ---
    params = [p for p in policy.parameters() if p.requires_grad]
    opt_cls = CautiousAdamW if cfg.optimizer == "cautious_adamw" else torch.optim.AdamW
    optimizer = opt_cls(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    encoder = FrameEncoder().to(device)
    # Trust-region anchor of the flow policy gradient: the policy as it was when this iteration's data
    # was collected (synced at the start of every actor update).
    reference = None
    if cfg.mode in PG_MODES:
        reference = copy.deepcopy(policy)
        logger.info("Reference shares %d frozen parameters", share_frozen_weights(reference, policy))
        reference.eval().requires_grad_(False)
    window = cfg.replay_window if (cfg.actor_replay or cfg.critic_replay) else 1
    eval_envs = practice_envs if cfg.eval_reuse_practice_envs else None

    def run_eval() -> dict[str, Any]:
        return evaluate(cfg, policy, preprocessor, postprocessor, env_pre, env_post, envs=eval_envs)

    def sync(paths: list[str]) -> None:
        if not cfg.hub_sync_repo:
            return
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(cfg.hub_sync_repo, private=True, exist_ok=True)
        prefix = out.name
        try:
            api.upload_folder(
                repo_id=cfg.hub_sync_repo,
                folder_path=out,
                path_in_repo=prefix,
                allow_patterns=paths,
                commit_message=f"{prefix}: sync",
            )
        except Exception as e:  # keep training if the Hub is unreachable
            logger.warning("Hub sync failed: %s", e)

    if cfg.eval_at_start:
        log({"round": -1, **run_eval()})
        sync(["metrics.jsonl", "flex_config.json"])

    practice_pre = InstructionWrapper(preprocessor, RECAP_POSITIVE) if cfg.mode == "recap" else preprocessor
    memory: list[MemoryRound] = []
    for r in range(cfg.rounds):
        t_round = time.perf_counter()
        # 1. Data collection
        root = out / "practice" / f"round_{r:03d}"
        writer = create_practice_dataset(root, r, fps, feats)
        records = practice_round(
            practice_envs,
            writer,
            r,
            record_keys,
            batches_per_task=cfg.episodes_per_task // envs_per_task,
            chunk=chunk,
            policy=policy,
            preprocessor=practice_pre,
            postprocessor=postprocessor,
            env_preprocessor=env_pre,
            env_postprocessor=env_post,
        )
        (root / "flex_episodes.json").write_text(json.dumps([asdict(rec) for rec in records], indent=2))
        metrics: dict[str, Any] = {"round": r, **_summarize_practice(records)}

        # 2. Replay buffer
        repo_id = f"flex/practice_round_{r:03d}"
        plain_ds = LeRobotDataset(repo_id, root=root)
        features = encode_dataset(
            plain_ds, camera_keys, OBS_STATE, encoder, device, cfg.encoder_batch_size, cfg.num_workers
        )
        train_ds = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)
        memory.append(MemoryRound(round=r, records=records, train_ds=train_ds, features=features))
        memory = memory[-window:]

        # 3. Critic and 4. advantages
        round_adv, q_c, judge_stats = judge(memory, cfg, task_index, max_steps, r, device, horizon)
        metrics.update(judge_stats)

        # 5. Actor update
        practice_loader, demo_loader, mix_stats = build_loaders(
            cfg, demo_ds, memory, round_adv, r, train_keys, horizon
        )
        metrics.update(mix_stats)
        if reference is not None:
            reference.load_state_dict(policy.state_dict())
        metrics.update(
            improve(
                policy,
                reference,
                preprocessor,
                optimizer,
                practice_loader,
                demo_loader,
                cfg,
                q_c,
                horizon,
                device,
                r,
            )
        )

        last = r == cfg.rounds - 1
        if (r + 1) % cfg.checkpoint_every == 0 or last:
            ckpt = out / "checkpoints" / f"round_{r:03d}" / "pretrained_model"
            policy.save_pretrained(ckpt)
            preprocessor.save_pretrained(ckpt)
            postprocessor.save_pretrained(ckpt)

        if (r + 1) % cfg.eval_every == 0 or last:
            metrics.update(run_eval())
        metrics["round_s"] = time.perf_counter() - t_round
        log(metrics)
        sync(["metrics.jsonl", "flex_config.json", "practice/*/flex_episodes.json"])

    close_envs(practice_envs)
    sync([f"checkpoints/round_{cfg.rounds - 1:03d}/**"])
    logger.info("FLEX done. Metrics in %s", metrics_path)


def main() -> None:
    train_flex()


if __name__ == "__main__":
    main()
