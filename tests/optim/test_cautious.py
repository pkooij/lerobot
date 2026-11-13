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
import torch

from lerobot.optim.cautious import CautiousAdamW
from lerobot.optim.optimizers import CautiousAdamWConfig, OptimizerConfig


def test_first_step_matches_adamw_up_to_mask_rescale():
    # On the first step the Adam update is sign(g) * lr, so every coordinate agrees with its gradient.
    torch.manual_seed(0)
    p_c = torch.nn.Parameter(torch.randn(10))
    p_a = torch.nn.Parameter(p_c.detach().clone())
    grad = torch.randn(10)
    opt_c = CautiousAdamW([p_c], lr=0.1, weight_decay=0.0)
    opt_a = torch.optim.AdamW([p_a], lr=0.1, weight_decay=0.0)
    p_c.grad, p_a.grad = grad.clone(), grad.clone()
    opt_c.step()
    opt_a.step()
    start = p_a.detach() + 0.1 * torch.sign(grad)  # undo AdamW's step to get the start point
    delta_c = p_c.detach() - start
    delta_a = p_a.detach() - start
    torch.testing.assert_close(delta_c, delta_a * 10 / 11, rtol=1e-4, atol=1e-6)
    assert opt_c.last_kept_fraction == 1.0


def test_disagreeing_coordinate_holds_still():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = CautiousAdamW([p], lr=0.1, betas=(0.9, 0.999), weight_decay=0.0)
    # Build momentum pointing towards +grad on both coordinates.
    for _ in range(5):
        p.grad = torch.tensor([1.0, 1.0])
        opt.step()
    before = p.detach().clone()
    # A small gradient reversal on coordinate 0: momentum still says "+", the fresh gradient says "-".
    p.grad = torch.tensor([-0.1, 1.0])
    opt.step()
    after = p.detach()
    assert after[0] == before[0]
    assert after[1] < before[1]
    assert opt.last_kept_fraction == 0.5


def test_converges_on_quadratic():
    torch.manual_seed(0)
    target = torch.randn(20)
    p = torch.nn.Parameter(torch.zeros(20))
    opt = CautiousAdamW([p], lr=0.05, weight_decay=0.0)
    for _ in range(500):
        opt.zero_grad()
        loss = ((p - target) ** 2).sum()
        loss.backward()
        opt.step()
    assert torch.allclose(p.detach(), target, atol=1e-2)


def test_config_registered_and_builds():
    cfg_cls = OptimizerConfig.get_choice_class("cautious_adamw")
    assert cfg_cls is CautiousAdamWConfig
    opt = CautiousAdamWConfig(lr=1e-3).build([torch.nn.Parameter(torch.zeros(3))])
    assert isinstance(opt, CautiousAdamW)
    assert opt.param_groups[0]["lr"] == 1e-3
