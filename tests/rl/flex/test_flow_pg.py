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
import copy

import pytest
import torch

from lerobot.rl.flex.flow_pg import cfm_loss, flow_pg_loss, mask_beyond_horizon, repeat_batch, surrogate
from lerobot.utils.constants import ACTION, OBS_STATE

CHUNK, ADIM = 4, 2


class NoisyFlowPolicy(torch.nn.Module):
    """A toy flow-matching policy: its CFM loss draws (noise, time) from the global RNG, like the VLAs."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(3 + CHUNK * ADIM + 1, CHUNK * ADIM)

    def forward(self, batch, reduction="mean"):
        a = batch[ACTION]
        noise = torch.randn_like(a)
        t = torch.rand(a.shape[0], 1, 1)
        x_t = t * noise + (1 - t) * a
        inp = torch.cat([batch[OBS_STATE], x_t.flatten(1), t.flatten(1)], dim=1)
        v = self.net(inp).view_as(a)
        losses = (v - (noise - a)) ** 2
        mask = (~batch["action_is_pad"]).unsqueeze(-1).float()
        per_sample = (losses * mask).sum((1, 2)) / (mask.sum((1, 2)) * ADIM).clamp_min(1)
        return (per_sample if reduction == "none" else per_sample.mean()), {}


def _batch(b=5):
    torch.manual_seed(0)
    return {
        OBS_STATE: torch.randn(b, 3),
        ACTION: torch.randn(b, CHUNK, ADIM),
        "action_is_pad": torch.zeros(b, CHUNK, dtype=torch.bool),
        "task": [f"t{i}" for i in range(b)],
    }


def test_surrogates():
    ratio = torch.tensor([0.5, 1.0, 1.5, 1.5, 0.5])
    a = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0])
    ppo = surrogate(ratio, a, 0.2, "ppo")
    torch.testing.assert_close(ppo, torch.tensor([0.5, 1.0, 1.2, -1.5, -0.8]))
    aspo = surrogate(ratio, a, 0.2, "aspo")
    torch.testing.assert_close(aspo[:3], ppo[:3])  # positive advantages: plain PPO
    torch.testing.assert_close(aspo[3], torch.tensor(-1.5 - 1.0 / 0.4 * 0.25))  # SPO quadratic penalty
    with pytest.raises(ValueError):
        surrogate(ratio, a, 0.2, "trpo")


def test_repeat_and_mask():
    b = _batch(2)
    rep = repeat_batch(b, 3)
    assert rep[ACTION].shape == (6, CHUNK, ADIM)
    assert rep["task"] == ["t0", "t1"] * 3
    torch.testing.assert_close(rep[ACTION][2:4], b[ACTION])
    masked = mask_beyond_horizon(b, 2)
    assert masked["action_is_pad"][0].tolist() == [False, False, True, True]
    assert not b["action_is_pad"].any()  # input untouched


def test_cfm_sum_scales_the_policy_mean_by_valid_entries():
    policy, b = NoisyFlowPolicy(), mask_beyond_horizon(_batch(), 3)
    torch.manual_seed(1)
    mean = cfm_loss(policy, b, "mean")
    torch.manual_seed(1)
    total = cfm_loss(policy, b, "sum")
    torch.testing.assert_close(total, mean * 3 * ADIM)


def test_ratio_is_one_against_an_identical_reference_and_gradient_follows_the_advantage():
    policy = NoisyFlowPolicy()
    reference = copy.deepcopy(policy).requires_grad_(False)
    b, adv = _batch(), torch.tensor([1.0, -1.0, 0.5, 2.0, -0.5])
    loss, stats = flow_pg_loss(
        policy, reference, b, adv, n_mc=4, clip_eps=0.05, kind="ppo", seed=3, horizon=2
    )
    assert stats["pg/ratio_mean"] == pytest.approx(1.0, abs=1e-6)
    assert stats["pg/clip_frac"] == 0.0
    assert float(loss.detach()) == pytest.approx(-float(adv.mean()), abs=1e-6)
    loss.backward()
    assert policy.net.weight.grad is not None and policy.net.weight.grad.abs().sum() > 0

    # One step along the gradient raises the surrogate: positive-advantage samples get likelier.
    opt = torch.optim.SGD(policy.parameters(), lr=1e-3)
    opt.step()
    loss2, stats2 = flow_pg_loss(
        policy, reference, b, adv, n_mc=4, clip_eps=0.05, kind="ppo", seed=3, horizon=2
    )
    assert float(loss2.detach()) < float(loss.detach())
    assert stats2["pg/abs_log_ratio"] > 0
