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
from lerobot.rl.flex.configuration_flex import FlexConfig  # noqa: E402
from lerobot.rl.flex.critic import FrameEncoder, encode_dataset, fit_cross_fitted_critic  # noqa: E402
from lerobot.rl.flex.practice import create_practice_dataset, practice_features, practice_round  # noqa: E402
from lerobot.rl.flex.train_flex import (  # noqa: E402
    RECAP_POSITIVE,
    InstructionWrapper,
    MemoryRound,
    build_loaders,
    improve,
    judge,
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


def _practice(tmp_path, demo_features, round_idx, policy):
    envs = {"fake": {0: gym.vector.SyncVectorEnv([lambda i=i: FakeCanonicalEnv(i) for i in range(4)])}}
    feats = practice_features(demo_features, [CAM], OBS_STATE)
    root = tmp_path / f"round_{round_idx:03d}"
    writer = create_practice_dataset(root, round_idx, FPS, feats)
    records = practice_round(
        envs,
        writer,
        round_idx,
        [CAM, OBS_STATE],
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
    round_adv, q_c, stats = judge(
        memory, cfg, {"fake/0": 0}, {"fake/0": MAX_STEPS}, current_round=0, device=torch.device("cpu")
    )
    assert round_adv[0].shape == (len(memory[0].train_ds),)
    assert cfg.q_min <= q_c <= 1.0 if cfg.critic_scaling else q_c == 1.0
    assert abs(float(round_adv[0].mean())) < 1e-4  # standardized within the (single) task

    # The practice dataset doubles as the demo buffer here; only the keys matter.
    demo_ds = LeRobotDataset(repo_id, root=root, delta_timestamps=delta)
    practice_loader, demo_loader, mix = build_loaders(
        cfg, demo_ds, memory, round_adv, 0, [CAM, OBS_STATE, ACTION]
    )
    assert practice_loader is not None
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
