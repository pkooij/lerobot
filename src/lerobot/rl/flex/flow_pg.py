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
"""Flow policy gradient (FPO / FPO++) for flow-matching VLAs, the ``L_PG`` term of FLEX.

A flow policy has no tractable likelihood, so FPO replaces the PPO likelihood ratio with one built from
conditional flow-matching (CFM) losses. For each action, draw ``N_mc`` pairs ``(tau_i, eps_i)`` and use the
*same* pairs under the reference policy and the current one:

    rho_i = exp(l_ref(a, tau_i, eps_i; o) - l_theta(a, tau_i, eps_i; o))

FPO++ (arXiv:2602.02481) keeps one ratio per pair instead of averaging the losses first, and replaces the
negative-advantage branch of PPO's clip with a quadratic (SPO) penalty ("ASPO"); for manipulation it keeps
plain PPO clipping. The policy only has to expose ``forward(batch, reduction="none")`` returning per-sample
CFM losses; the pairs are matched by replaying the RNG state, so no policy internals are touched.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

import torch

from lerobot.utils.constants import ACTION

SURROGATES = ("ppo", "aspo")


def surrogate(ratio: torch.Tensor, adv: torch.Tensor, clip_eps: float, kind: str = "ppo") -> torch.Tensor:
    """Per-sample objective to *maximize*: PPO's clipped surrogate, or FPO++'s ASPO."""
    ppo = torch.minimum(ratio * adv, ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv)
    if kind == "ppo":
        return ppo
    if kind == "aspo":
        spo = ratio * adv - adv.abs() / (2.0 * clip_eps) * (ratio - 1.0) ** 2
        return torch.where(adv >= 0, ppo, spo)
    raise ValueError(f"surrogate must be one of {SURROGATES}, got {kind!r}")


def repeat_batch(batch: dict[str, Any], n: int) -> dict[str, Any]:
    """Tile every per-sample entry ``n`` times along the batch dimension (copy 0 first, then copy 1, ...)."""
    bsize = len(batch[ACTION])
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == bsize:
            out[key] = value.repeat(n, *([1] * (value.ndim - 1)))
        elif isinstance(value, list) and len(value) == bsize:
            out[key] = value * n
        else:
            out[key] = value
    return out


def mask_beyond_horizon(batch: dict[str, Any], horizon: int | None) -> dict[str, Any]:
    """Mark chunk steps ``>= horizon`` as padding: only the executed part of a chunk is the action."""
    if horizon is None or "action_is_pad" not in batch:
        return batch
    pad = batch["action_is_pad"]
    beyond = torch.arange(pad.shape[1], device=pad.device) >= horizon
    return {**batch, "action_is_pad": pad | beyond.unsqueeze(0)}


def cfm_loss(policy: torch.nn.Module, batch: dict[str, Any], reduction: str = "sum") -> torch.Tensor:
    """Per-sample CFM loss ``l``. ``"sum"`` sums over valid chunk steps and action dims (FPO++'s
    ``||v - (a - eps)||^2``); ``"mean"`` keeps the policy's average over them."""
    per_sample, _ = policy.forward(dict(batch), reduction="none")
    per_sample = per_sample.float()
    if reduction == "mean":
        return per_sample
    if reduction != "sum":
        raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")
    action = batch[ACTION]
    steps = action.shape[1] if action.ndim == 3 else 1
    if "action_is_pad" in batch:
        valid = (~batch["action_is_pad"]).sum(dim=1).float()
    else:
        valid = torch.full_like(per_sample, float(steps))
    return per_sample * valid * action.shape[-1]


@contextmanager
def _seeded(seed: int, device: torch.device) -> Iterator[None]:
    """Run the block from a fixed RNG state without disturbing the global one."""
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def flow_pg_loss(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    batch: dict[str, Any],
    adv: torch.Tensor,
    *,
    n_mc: int,
    clip_eps: float,
    kind: str,
    seed: int,
    horizon: int | None,
    reduction: str = "sum",
    amp: Callable[[], AbstractContextManager] = nullcontext,
    max_log_ratio: float = 20.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``L_PG = -E[mean_i psi(rho_i, A)]`` over ``n_mc`` matched (tau, eps) pairs per sample.

    ``reference`` is the policy before this iteration's update (the trust-region anchor); both models
    see the same noise and flow times because the RNG is reseeded with ``seed`` before each forward.
    """
    batch = mask_beyond_horizon(repeat_batch(batch, n_mc), horizon)
    adv_rep = adv.float().repeat(n_mc)
    device = adv.device
    with torch.no_grad(), _seeded(seed, device), amp():
        l_ref = cfm_loss(reference, batch, reduction)
    with _seeded(seed, device), amp():
        l_new = cfm_loss(policy, batch, reduction)
    log_ratio = (l_ref - l_new).clamp(-max_log_ratio, max_log_ratio)
    ratio = log_ratio.exp()
    objective = surrogate(ratio, adv_rep, clip_eps, kind)
    loss = -objective.mean()
    with torch.no_grad():
        stats = {
            "pg/loss": float(loss.detach()),
            "pg/ratio_mean": float(ratio.mean()),
            "pg/abs_log_ratio": float(log_ratio.abs().mean()),
            "pg/clip_frac": float(((ratio - 1.0).abs() > clip_eps).float().mean()),
            "pg/cfm_loss": float(l_new.mean()),
        }
    return loss, stats
