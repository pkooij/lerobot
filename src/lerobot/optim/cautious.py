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
"""Cautious AdamW (C-AdamW).

Liang et al., "Cautious Optimizers: Improving Training with One Line of Code", arXiv:2411.16085.

An Adam update ``u`` is applied to a coordinate only where it agrees in sign with the current
gradient ``g`` (``u * g > 0``): a parameter moves only when the fresh evidence (the gradient) and the
recent trend (the momentum-driven Adam direction) point the same way. Masked coordinates hold still.
The surviving coordinates are rescaled by ``numel / (n_kept + 1)`` so the average step size is kept.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


class CautiousAdamW(torch.optim.Optimizer):
    """AdamW with a sign-agreement mask on the update (decoupled weight decay is not masked).

    ``last_kept_fraction`` is the fraction of all coordinates the last step updated, for logging.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self.last_kept_fraction: float = 1.0

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        kept_count: torch.Tensor | None = None
        total = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CautiousAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay (AdamW), applied unmasked.
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                update = (exp_avg / bias_correction1) / denom

                # The cautious mask: keep coordinates where the update agrees with the fresh gradient.
                mask = (update * grad > 0).to(update.dtype)
                n_kept = mask.sum()
                mask.mul_(mask.numel() / (n_kept + 1.0))
                p.add_(update * mask, alpha=-lr)

                # Accumulate on device; one host sync per step instead of one per parameter.
                kept_count = n_kept if kept_count is None else kept_count + n_kept.to(kept_count.device)
                total += mask.numel()

        if kept_count is not None and total > 0:
            self.last_kept_fraction = float(kept_count.item()) / total
        return loss
