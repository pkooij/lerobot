"""PI052 backend options must not require changes to the shared optimizer config."""

import pytest
import torch

from lerobot.policies.pi052.configuration_pi052 import PI052Config
from lerobot.policies.pi052.modeling_pi052 import PI052Policy


@pytest.mark.parametrize("backbone_scale", [1.0, 0.5])
def test_adamw_backend_options_stay_in_policy_groups(backbone_scale):
    policy = PI052Policy.__new__(PI052Policy)
    torch.nn.Module.__init__(policy)
    policy.weight = torch.nn.Parameter(torch.ones(2))
    policy.config = PI052Config(
        device="cpu",
        backbone_lr_scale=backbone_scale,
        optimizer_foreach=False,
        optimizer_fused=False,
    )
    preset = policy.config.get_optimizer_preset()
    assert not hasattr(preset, "fused")
    optimizer = preset.build(policy.get_optim_params())
    assert all(group["fused"] is False and group["foreach"] is False for group in optimizer.param_groups)
    policy.weight.sum().backward()
    optimizer.step()
    assert torch.isfinite(policy.weight).all()
    assert (policy.weight < 1).all()


def test_conditioning_group_covers_time_and_adaptive_norms_once():
    from torch import nn

    policy = PI052Policy.__new__(PI052Policy)
    nn.Module.__init__(policy)
    policy.model = nn.Module()
    policy.model.time_mlp_in = nn.Linear(2, 2)
    policy.model.time_mlp_out = nn.Linear(2, 2)
    policy.model.action_out_proj = nn.Linear(2, 2)
    backbone = policy.model.paligemma_with_expert = nn.Module()
    backbone.gemma_expert = nn.Module()
    backbone.gemma_expert.input_layernorm = nn.Module()
    backbone.gemma_expert.input_layernorm.dense = nn.Linear(2, 6)
    backbone.paligemma = nn.Linear(2, 2)
    policy.config = PI052Config(
        device="cpu",
        conditioning_lr_scale=0.1,
        conditioning_grad_clip_norm=0.1,
        action_expert_grad_clip_norm=1,
        action_expert_lr_scale=0.5,
    )
    groups = {g["name"]: g for g in policy.get_optim_params()}
    conditioning = groups["conditioning"]
    expected = [
        *policy.model.time_mlp_in.parameters(),
        *policy.model.time_mlp_out.parameters(),
        *backbone.gemma_expert.input_layernorm.dense.parameters(),
    ]
    assert {id(p) for p in conditioning["params"]} == {id(p) for p in expected}
    assert conditioning["lr"] == pytest.approx(policy.config.optimizer_lr * 0.5 * 0.1)
    assert groups["action_expert"]["lr"] == pytest.approx(policy.config.optimizer_lr * 0.5)
    assert groups["backbone"]["lr"] == pytest.approx(policy.config.optimizer_lr)
    grouped = [id(p) for g in groups.values() for p in g["params"]]
    assert len(grouped) == len(set(grouped)) == len(list(policy.parameters()))
    assert conditioning["grad_clip_norm"] == 0.1
    assert groups["action_expert"]["grad_clip_norm"] == 1


@pytest.mark.parametrize("name", ["conditioning_grad_clip_norm", "action_expert_grad_clip_norm"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_stability_limits_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        PI052Config(device="cpu", **{name: value})
