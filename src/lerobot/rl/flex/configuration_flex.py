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
"""Configuration of the FLEX loop (`python -m lerobot.rl.flex.train_flex`)."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot import envs, policies  # noqa: F401  (registers env and policy config choices)
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnvConfig

logger = logging.getLogger(__name__)

FLEX_MODES = ("flex", "fpo", "filtered_bc", "recap")
PG_MODES = ("flex", "fpo")


def _default_env() -> envs.EnvConfig:
    # LIBERO-Spatial is the suite of the poster's ablation table. 256x256 frames match the lerobot/libero
    # demos, so practice and demo frames share one format.
    return LiberoEnvConfig(task="libero_spatial", observation_height=256, observation_width=256)


@dataclass
class FlexConfig:
    """Data collection -> replay buffer -> critic -> advantages -> actor update, for ``rounds`` iterations.

    ``mode`` picks the learning rule applied to the same practice data:
      - ``flex``: FLEX. ``q_c * L_PG + beta * BC(demos)`` with a flow policy gradient on a staleness-weighted,
        advantage-prioritized replay buffer, and the consensus (cautious) optimizer.
      - ``fpo``: the FPO++ baseline. On-policy flow policy gradient: no replay, no critic scaling, no BC
        term, uniform sampling, global advantage normalization, plain AdamW.
      - ``filtered_bc``: flow matching on demos + successful attempts (rejection-sampling fine-tuning).
      - ``recap``: advantage-conditioned flow matching ("Advantage: positive/negative." appended to the
        instruction, prompted "positive" at test time), in the spirit of pi*0.6's RECAP.

    Each FLEX component can also be switched off on its own to reproduce the poster's ablations.
    """

    env: envs.EnvConfig = field(default_factory=_default_env)
    # The starting policy: --policy.path=<hub id or local pretrained_model dir> (a fine-tuned checkpoint).
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int = 1000
    mode: str = "flex"

    # --- Demonstrations (D_demo, fixed) ---
    # None: no demos. Needs a policy-gradient mode with beta=0 and record_chunks; the practice schema then
    # comes from the env's observations.
    demo_repo_id: str | None = "lerobot/libero"
    demo_revision: str | None = None
    # Demos per task drawn for the env's tasks. Use the same value (and seed) as the SFT run that produced
    # --policy.path, see `python -m lerobot.rl.flex.select_demos`.
    demos_per_task: int = 10
    # Explicit demo episode indices; overrides demos_per_task when set.
    demo_episodes: list[int] | None = None
    video_backend: str | None = None

    # --- 1. Data collection ---
    rounds: int = 10
    # Attempts per task per iteration. 5 x 10 LIBERO tasks = the poster's 50 trials per cycle.
    episodes_per_task: int = 5
    # Simulators run in parallel per task (0 = episodes_per_task). Each one holds a GPU render context and
    # every task keeps its own, so 20 x 10 LIBERO tasks runs the H100 out of memory; 5 runs 20 in 4 batches.
    practice_envs_per_task: int = 0
    use_async_envs: bool = False
    # Record only the frames where the policy samples a chunk (every n_action_steps), each with the chunk of
    # actions it executed. The policy gradient only uses those frames, and long episodes (RoboLab: up to
    # 4,500 steps) would otherwise store every camera frame. The critic then sees one frame per chunk.
    record_chunks: bool = False
    # Evaluate on the practice envs instead of building new ones (served envs allow one session at a time).
    eval_reuse_practice_envs: bool = False

    # --- 2. Replay buffer ---
    # Iterations kept in the replay window P_c (the current one plus replay_window - 1 older ones).
    replay_window: int = 4
    # Staleness weight 2 ** (-d / h_s) for data from d iterations ago; <= 0 disables (ablation).
    staleness_half_life: float = 1.0
    # Ablations "replay buffer": train the actor / the critic on the current iteration's data only.
    actor_replay: bool = True
    critic_replay: bool = True

    # --- 3. Critic V_phi ---
    gamma: float = 0.995
    critic_folds: int = 5
    critic_steps: int = 1500
    critic_batch_size: int = 1024
    critic_lr: float = 3e-4
    critic_hidden: int = 512
    encoder_batch_size: int = 256
    # q_c = clip(E_c / E*, q_min, 1), E_c the critic's (out-of-fold) explained variance.
    ev_target: float = 0.5
    q_min: float = 0.1
    # Ablation "critic-scaled PG loss": q_c = 1.
    critic_scaling: bool = True

    # --- 4. Advantages ---
    # Ablation "per-task standardization": standardize over all tasks at once instead.
    per_task_std: bool = True
    # No policy gradient for a task whose actor window has only successes or only failures: its advantages
    # are critic noise, which per-task standardization would scale up. Not in the poster (ours).
    require_contrast: bool = True
    # p_i ~ (1 - uniform_mix) * |A_i| / mean|A| + uniform_mix. Ablation: uniform sampling.
    prioritized_sampling: bool = True
    uniform_mix: float = 0.5

    # --- 5. Actor update ---
    steps_per_round: int = 500
    # Practice states per step; each is evaluated at n_mc (tau, eps) pairs, so the policy sees
    # batch_size * n_mc samples per forward.
    batch_size: int = 16
    n_mc: int = 8
    # Clip range of the PPO surrogate; "aspo" uses FPO++'s quadratic penalty for negative advantages.
    clip_eps: float = 0.05
    surrogate: str = "ppo"
    # CFM loss per sample in the ratio and BC term: "mean" over chunk steps and action dims, or "sum" (FPO++).
    # On SmolVLA/LIBERO the summed log-ratio has a numerical floor of ~0.2 after any weight change, so
    # "sum" needs a clip_eps above that (e.g. 0.3); "mean" divides the log-ratio by chunk steps x dims.
    loss_reduction: str = "mean"
    # Weight of the behavior-cloning term on demos; 0 disables it (ablation).
    beta: float = 0.5
    demo_batch_size: int = 16
    lr: float = 1e-5
    weight_decay: float = 1e-10
    grad_clip_norm: float = 10.0
    # "cautious_adamw" (FLEX's consensus update) or "adamw" (ablation).
    optimizer: str = "cautious_adamw"
    betas: tuple[float, float] = (0.9, 0.95)
    # recap: fraction of practice frames tagged "positive" (top quantile by advantage).
    recap_positive_frac: float = 0.3
    use_amp: bool = True
    # Keep trainable weights in fp32 (SmolVLA loads them in bf16, which rounds most small Adam steps away).
    fp32_trainable: bool = True
    # Run the policy and its reference in train mode during policy-gradient updates (both identically, so
    # the ratio stays exact). For policies without dropout whose gradient checkpointing only runs in train
    # mode, such as pi05.
    pg_train_mode: bool = False
    # Save a checkpoint every this many rounds (the last round is always saved).
    checkpoint_every: int = 1
    # Upload metrics (every round) and the final checkpoint to this Hub repo (private model repo).
    hub_sync_repo: str | None = None
    num_workers: int = 4

    # --- Evaluation ---
    eval_at_start: bool = True
    eval_every: int = 5
    eval_episodes_per_task: int = 50
    eval_batch_size: int = 10

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy") or []
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)

    def apply_mode_preset(self) -> None:
        """``fpo`` is FLEX with every FLEX component removed."""
        if self.mode == "fpo":
            self.actor_replay = self.critic_replay = False
            self.critic_scaling = self.per_task_std = self.prioritized_sampling = False
            self.require_contrast = False
            self.beta = 0.0
            self.optimizer = "adamw"

    def validate(self) -> None:
        if self.policy is None or self.policy.pretrained_path is None:
            raise ValueError("FLEX improves a trained policy: pass --policy.path=<checkpoint>.")
        self.check()
        if not self.job_name:
            self.job_name = f"flex_{self.mode}_{self.policy.type}_{getattr(self.env, 'task', self.env.type)}"
        if not self.output_dir:
            now = dt.datetime.now()
            self.output_dir = Path("outputs/flex") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"

    def check(self) -> None:
        """Validate the learning settings (no policy needed) and apply the mode preset."""
        if self.mode not in FLEX_MODES:
            raise ValueError(f"mode must be one of {FLEX_MODES}, got {self.mode!r}")
        self.apply_mode_preset()
        if self.optimizer not in ("cautious_adamw", "adamw"):
            raise ValueError(f"optimizer must be 'cautious_adamw' or 'adamw', got {self.optimizer!r}")
        if self.surrogate not in ("ppo", "aspo"):
            raise ValueError(f"surrogate must be 'ppo' or 'aspo', got {self.surrogate!r}")
        if self.loss_reduction not in ("sum", "mean"):
            raise ValueError(f"loss_reduction must be 'sum' or 'mean', got {self.loss_reduction!r}")
        if self.rounds < 1 or self.episodes_per_task < 1 or self.replay_window < 1 or self.n_mc < 1:
            raise ValueError("rounds, episodes_per_task, replay_window and n_mc must be >= 1")
        if self.practice_envs_per_task < 0 or (
            self.practice_envs_per_task and self.episodes_per_task % self.practice_envs_per_task
        ):
            raise ValueError("practice_envs_per_task must be 0 or divide episodes_per_task")
        if self.demo_repo_id is None and (
            self.mode not in PG_MODES or self.beta > 0 or not self.record_chunks
        ):
            raise ValueError(
                "Without demos (demo_repo_id=None) use a policy-gradient mode, beta=0 and record_chunks"
            )
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        if not 0.0 <= self.uniform_mix <= 1.0:
            raise ValueError("uniform_mix must be in [0, 1]")
        if not 0.0 < self.q_min <= 1.0 or self.ev_target <= 0:
            raise ValueError("Need 0 < q_min <= 1 and ev_target > 0")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]
