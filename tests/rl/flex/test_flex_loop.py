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
"""End-to-end test of one FLEX iteration (collect -> replay buffer -> critic -> advantages -> actor update)
without a simulator or a VLA: a fake canonical-observation env and a tiny linear "policy" stand in for
LIBERO and SmolVLA."""

import copy

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("torchvision")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from gymnasium import spaces  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.rl.flex.advantages import EpisodeRecord  # noqa: E402
from lerobot.rl.flex.configuration_flex import FlexConfig  # noqa: E402
from lerobot.rl.flex.critic import FrameEncoder, encode_dataset, fit_cross_fitted_critic  # noqa: E402
from lerobot.rl.flex.practice import (  # noqa: E402
    chunk_features,
    create_practice_dataset,
    env_practice_features,
    practice_features,
    practice_round,
)
from lerobot.rl.flex.train_flex import (  # noqa: E402
    RECAP_POSITIVE,
    InstructionWrapper,
    MemoryRound,
    WeightedView,
    actor_frames,
    build_loaders,
    contrast_frames,
    improve,
    judge,
    share_frozen_weights,
    upcast_trainable,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

pytestmark = pytest.mark.dataset

CAM = "observation.images.image"
IMG = 16
MAX_STEPS = 8
FPS = 10


class FakeCanonicalEnv(gym.Env):
    """Even sub-envs succeed at step 5; odd ones time out. Observations are already canonical."""

    metadata = {"render_modes": [], "canonical_observations": True}

    def __init__(self, idx: int):
        self.idx = idx
        self.task_description = "pick up the cube"
        self._max_episode_steps = MAX_STEPS
        self.observation_space = spaces.Dict(
            {
                CAM: spaces.Box(0, 255, (IMG, IMG, 3), np.uint8),
                OBS_STATE: spaces.Box(-1, 1, (8,), np.float32),
            }
        )
        self.action_space = spaces.Box(-1, 1, (7,), np.float32)
        self.t = 0

    def _obs(self):
        img = np.full((IMG, IMG, 3), 40 * (self.idx % 2) + 5 * self.t, dtype=np.uint8)
        state = np.full(8, self.t / MAX_STEPS + (self.idx % 2), dtype=np.float32)
        return {CAM: img, OBS_STATE: state}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {"is_success": False}

    def step(self, action):
        self.t += 1
        success = self.idx % 2 == 0 and self.t >= 5
        return self._obs(), float(success), success, self.t >= MAX_STEPS, {"is_success": success}


class Identity:
    def __call__(self, x):
        return x

    def reset(self):
        pass


class LinearPolicy(torch.nn.Module):
    """Predicts the action chunk from the state; `forward(reduction='none')` like the VLA policies."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(8, 7)

    def reset(self):
        pass

    def select_action(self, obs):
        return self.net(obs[OBS_STATE].float())

    def forward(self, batch, reduction="mean"):
        pred = self.net(batch[OBS_STATE].float().reshape(len(batch[ACTION]), -1)[:, :8])
        per_sample = ((pred.unsqueeze(1) - batch[ACTION]) ** 2).mean(dim=(1, 2))
        return (per_sample if reduction == "none" else per_sample.mean()), {}


@pytest.fixture
def demo_features():
    return {
        CAM: {"dtype": "video", "shape": (IMG, IMG, 3), "names": ["height", "width", "channel"]},
        OBS_STATE: {"dtype": "float32", "shape": (8,), "names": None},
        ACTION: {"dtype": "float32", "shape": (7,), "names": None},
    }


def _practice(tmp_path, demo_features, round_idx, policy, batches_per_task=1):
    envs = {"fake": {0: gym.vector.SyncVectorEnv([lambda i=i: FakeCanonicalEnv(i) for i in range(4)])}}
    feats = practice_features(demo_features, [CAM], OBS_STATE)
    root = tmp_path / f"round_{round_idx:03d}"
    writer = create_practice_dataset(root, round_idx, FPS, feats)
    records = practice_round(
        envs,
        writer,
        round_idx,
        [CAM, OBS_STATE],
        batches_per_task=batches_per_task,
        policy=policy,
        preprocessor=Identity(),
        postprocessor=Identity(),
        env_preprocessor=Identity(),
        env_postprocessor=Identity(),
    )
    envs["fake"][0].close()
    return root, records


def test_practice_records_attempts_with_outcomes(tmp_path, demo_features):
    root, records = _practice(tmp_path, demo_features, 0, LinearPolicy())
    assert [r.success for r in records] == [True, False, True, False]
    assert [r.length for r in records] == [5, MAX_STEPS, 5, MAX_STEPS]

    ds = LeRobotDataset("flex/practice_round_000", root=root)
    assert len(ds) == sum(r.length for r in records)
    assert ds[0][CAM].shape == (3, IMG, IMG)
    assert ds[0]["task"] == "pick up the cube"

    chunked = LeRobotDataset(
        "flex/practice_round_000", root=root, delta_timestamps={ACTION: [i / FPS for i in range(4)]}
    )
    item = chunked[3]  # 2 frames before the end of a 5-frame episode
    assert item[ACTION].shape == (4, 7)
    assert item["action_is_pad"].tolist() == [False, False, True, True]


def test_practice_runs_each_task_in_batches(tmp_path, demo_features):
    root, records = _practice(tmp_path, demo_features, 0, LinearPolicy(), batches_per_task=2)
    assert [r.success for r in records] == [True, False, True, False] * 2
    ds = LeRobotDataset("flex/practice_round_000", root=root)
    assert ds.meta.total_episodes == 8 and len(ds) == sum(r.length for r in records)
    with pytest.raises(ValueError, match="practice_envs_per_task"):
        FlexConfig(episodes_per_task=20, practice_envs_per_task=3).check()


def test_critic_cross_fitting_learns_separable_outcomes():
    rng = np.random.default_rng(0)
    n_eps, length = 40, 10
    success = rng.random(n_eps) < 0.5
    x = torch.as_tensor(
        np.repeat(success[:, None], length, 1).reshape(-1, 1)
        + 0.3 * rng.standard_normal((n_eps * length, 4)),
        dtype=torch.float32,
    )
    y = np.repeat(success, length).astype(np.float32)
    fit = fit_cross_fitted_critic(
        x,
        y,
        np.ones_like(y),
        np.repeat(np.arange(n_eps), length),
        n_folds=4,
        steps=200,
        batch_size=64,
        lr=1e-2,
        hidden=32,
        device=torch.device("cpu"),
        seed=0,
    )
    assert fit.oof_values.min() >= 0 and fit.oof_values.max() <= 1
    from lerobot.rl.flex.advantages import binary_auroc

    assert binary_auroc(fit.oof_values, y) > 0.9


@pytest.mark.parametrize("mode", ["flex", "fpo", "filtered_bc", "recap"])
def test_one_round_judge_and_improve(tmp_path, demo_features, mode):
    policy = LinearPolicy()
    root, records = _practice(tmp_path, demo_features, 0, policy)
    repo_id = "flex/practice_round_000"
    delta = {ACTION: [i / FPS for i in range(4)]}

    encoder = FrameEncoder(image_size=32, pretrained=False)
    features = encode_dataset(
        LeRobotDataset(repo_id, root=root), [CAM], OBS_STATE, encoder, torch.device("cpu"), num_workers=0
    )
    memory = [
        MemoryRound(
            round=0,
            records=records,
            train_ds=LeRobotDataset(repo_id, root=root, delta_timestamps=delta),
            features=features,
        )
    ]
    cfg = FlexConfig(
        mode=mode,
        critic_folds=2,
        critic_steps=50,
        critic_batch_size=16,
        critic_hidden=32,
        steps_per_round=5,
        batch_size=4,
        n_mc=2,
        demo_batch_size=2,
        num_workers=0,
        use_amp=False,
    )
    cfg.check()
    horizon = 2
    round_adv, q_c, stats = judge(
        memory,
        cfg,
        {"fake/0": 0},
        {"fake/0": MAX_STEPS},
        current_round=0,
        device=torch.device("cpu"),
        horizon=horizon,
    )
    assert round_adv[0].shape == (len(memory[0].train_ds),)
    assert cfg.q_min <= q_c <= 1.0 if cfg.critic_scaling else q_c == 1.0
    # Standardized within the (single) task, over the frames the actor learns from.
    learn = actor_frames(memory[0], cfg, horizon)
    assert abs(float(round_adv[0][learn].mean())) < 1e-4

    # The practice dataset doubles as the demo buffer here; only the keys matter.
    demo_ds = LeRobotDataset(repo_id, root=root, delta_timestamps=delta)
    practice_loader, demo_loader, mix = build_loaders(
        cfg, demo_ds, memory, round_adv, 0, [CAM, OBS_STATE, ACTION], horizon
    )
    assert practice_loader is not None
    # The policy gradient only sees frames where the policy sampled a chunk.
    weights = np.asarray(practice_loader.sampler.weights)
    chunk_start = features.frame_index % horizon == 0
    if mode in ("flex", "fpo"):
        assert (weights[~chunk_start] == 0).all() and (weights[chunk_start] > 0).all()
        assert mix["buffer/actor_frames"] == chunk_start.sum()
    else:
        assert mix["buffer/actor_frames"] == len(chunk_start)
    assert (demo_loader is None) == (mode == "fpo")  # FPO++ has no BC term
    batch = next(iter(practice_loader))
    assert set(batch) >= {CAM, OBS_STATE, ACTION, "action_is_pad", "task", "flex_adv"}
    if mode == "recap":
        assert all(t.endswith("positive.") or t.endswith("negative.") for t in batch["task"])
    assert mix["buffer/rounds"] == 1

    before = policy.net.weight.detach().clone()
    reference = copy.deepcopy(policy).requires_grad_(False) if mode in ("flex", "fpo") else None
    out = improve(
        policy,
        reference,
        Identity(),
        torch.optim.AdamW(policy.parameters(), lr=1e-2),
        practice_loader,
        demo_loader,
        cfg,
        q_c,
        horizon=2,
        device=torch.device("cpu"),
        round_idx=0,
    )
    assert np.isfinite(out["improve/loss"])
    if reference is not None:
        assert out["pg/ratio_mean"] > 0
    assert not torch.equal(before, policy.net.weight.detach())


def test_instruction_wrapper_rewrites_tasks():
    seen = {}

    class Pre:
        def __call__(self, obs):
            seen.update(obs)
            return obs

        def reset(self):
            seen["reset"] = True

    wrapped = InstructionWrapper(Pre(), RECAP_POSITIVE)
    wrapped({"task": ["open the drawer"]})
    assert seen["task"] == ["open the drawer Advantage: positive."]
    wrapped.reset()
    assert seen["reset"]


def test_cli_entry_point_parses_flex_config():
    """`parser.wrap` reads the config class from the annotation, so it must be a class, not a string."""
    import inspect

    import draccus

    from lerobot.rl.flex import train_flex as train_flex_module

    assert inspect.signature(train_flex_module.train_flex).parameters["cfg"].annotation is FlexConfig
    # --env.type resets the env to LIBERO's defaults (360 px), so FLEX commands pass the render size too.
    args = ["--mode=fpo", "--rounds=2", "--env.type=libero", "--env.task=libero_spatial"]
    args += ["--env.observation_height=256", "--env.observation_width=256"]
    cfg = draccus.parse(FlexConfig, args=args)
    assert (cfg.mode, cfg.rounds, cfg.env.task, cfg.env.observation_height) == (
        "fpo",
        2,
        "libero_spatial",
        256,
    )


def test_tasks_without_outcome_contrast_get_no_policy_gradient():
    def rec(r, task, success, length=3):
        return EpisodeRecord("practice", r, f"s/{task}", success, length)

    # Task 0 always fails in the window, task 1 has both outcomes, task 2 only succeeds.
    memory = [
        MemoryRound(0, [rec(0, 0, False), rec(0, 1, True), rec(0, 2, True)], None, None),
        MemoryRound(1, [rec(1, 0, False), rec(1, 1, False), rec(1, 2, True)], None, None),
    ]
    masks = contrast_frames(memory, FlexConfig(mode="flex"), current_round=1)
    assert [m.tolist() for m in masks] == [[False] * 3 + [True] * 3 + [False] * 3] * 2
    # Without actor replay only the current round counts: task 1 then has failures only.
    no_replay = FlexConfig(mode="flex", actor_replay=False)
    assert not any(m.any() for m in contrast_frames(memory, no_replay, current_round=1))
    for cfg in (FlexConfig(mode="flex", require_contrast=False), FlexConfig(mode="filtered_bc")):
        assert all(m.all() for m in contrast_frames(memory, cfg, current_round=1))
    fpo = FlexConfig(mode="fpo")
    fpo.check()
    assert not fpo.require_contrast  # the FPO++ baseline has none of FLEX's additions


def test_upcast_trainable_casts_only_trainable_weights():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)).to(torch.bfloat16)
    model[0].requires_grad_(False)
    cast = upcast_trainable(model)
    assert cast == 4 * 2 + 2
    assert model[0].weight.dtype == torch.bfloat16
    assert model[1].weight.dtype == torch.float32 and model[1].bias.dtype == torch.float32
    assert upcast_trainable(model) == 0
    assert FlexConfig().fp32_trainable


def test_practice_records_one_frame_per_chunk_without_demos(tmp_path):
    chunk = 3
    envs = {"fake": {0: gym.vector.SyncVectorEnv([lambda i=i: FakeCanonicalEnv(i) for i in range(2)])}}
    obs, _ = envs["fake"][0].reset(seed=0)
    feats = chunk_features(env_practice_features(obs, 7), chunk)
    assert feats[CAM]["dtype"] == "image" and feats[ACTION]["shape"] == (chunk * 7,)
    root = tmp_path / "round_000"
    records = practice_round(
        envs,
        create_practice_dataset(root, 0, FPS, feats),
        0,
        [CAM, OBS_STATE],
        chunk=chunk,
        policy=LinearPolicy(),
        preprocessor=Identity(),
        postprocessor=Identity(),
        env_preprocessor=Identity(),
        env_postprocessor=Identity(),
    )
    envs["fake"][0].close()
    # 5 steps -> chunks at t=0,3; 8 steps -> t=0,3,6.
    assert [(r.success, r.length, r.stride) for r in records] == [(True, 2, chunk), (False, 3, chunk)]
    ds = LeRobotDataset("flex/practice_round_000", root=root)
    assert ds.hf_dataset["frame_index"][:5] == [0, 1, 0, 1, 2]
    view = WeightedView(ds, [CAM, OBS_STATE, ACTION], chunk=chunk)
    item = view[1]  # the success episode's second chunk: steps 3 and 4, then padding
    assert item[ACTION].shape == (chunk, 7)
    assert item["action_is_pad"].tolist() == [False, False, True]
    assert torch.equal(item[ACTION][2], item[ACTION][1])  # padded with the last executed action
    assert ds[1][OBS_STATE][0].item() == pytest.approx(3 / MAX_STEPS)  # recorded at t=3


def test_reference_shares_frozen_weights():
    policy = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    policy[0].requires_grad_(False)
    reference = copy.deepcopy(policy).requires_grad_(False)
    assert share_frozen_weights(reference, policy) == 20
    assert reference[0].weight.data_ptr() == policy[0].weight.data_ptr()
    assert reference[1].weight.data_ptr() != policy[1].weight.data_ptr()
