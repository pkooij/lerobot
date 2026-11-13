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
"""FLEX's critic ``V_phi``: a small model that estimates the return from any moment of an attempt.

Frames are embedded once by a frozen image encoder; the critic itself is an MLP over
``[camera embeddings, state, task one-hot, time fraction]`` trained with binary cross-entropy on the
discounted success return ``G`` (in [0, 1]). The FLEX poster does not specify the architecture; this is
our choice. It is *cross-fitted*: episodes are split into folds and each episode is scored by the model
that never saw it, so ``A = G - V`` is not shrunk by the critic memorizing outcomes, and the explained
variance that scales the policy gradient (``q_c``) is measured out of sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class FrameEncoder(nn.Module):
    """Frozen ImageNet ResNet-18 mapping a float [0, 1] (B, 3, H, W) image to a 512-d embedding."""

    out_dim = 512

    def __init__(self, image_size: int = 224, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone.eval().requires_grad_(False)
        self.image_size = image_size
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        images = F.interpolate(
            images, size=(self.image_size, self.image_size), mode="bilinear", antialias=True
        )
        return self.backbone((images - self.mean) / self.std)


@dataclass
class FrameFeatures:
    """Per-frame critic inputs for one dataset, in dataset position order."""

    image: torch.Tensor  # (N, n_cameras * 512) float16
    state: torch.Tensor  # (N, state_dim) float32
    episode_index: np.ndarray  # (N,)
    frame_index: np.ndarray  # (N,)


@torch.inference_mode()
def encode_dataset(
    dataset: torch.utils.data.Dataset,
    camera_keys: list[str],
    state_key: str,
    encoder: FrameEncoder,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4,
) -> FrameFeatures:
    """Embed every frame of a dataset that was opened without ``delta_timestamps``."""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    images, states, episodes, frames = [], [], [], []
    for batch in loader:
        cams = [encoder(batch[key].to(device, non_blocking=True)).half().cpu() for key in camera_keys]
        images.append(torch.cat(cams, dim=-1))
        states.append(batch[state_key].float().reshape(len(cams[0]), -1))
        episodes.append(batch["episode_index"].numpy().reshape(-1))
        frames.append(batch["frame_index"].numpy().reshape(-1))
    return FrameFeatures(
        image=torch.cat(images),
        state=torch.cat(states),
        episode_index=np.concatenate(episodes),
        frame_index=np.concatenate(frames),
    )


class CriticMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_critic_inputs(
    image: torch.Tensor,
    state: torch.Tensor,
    task_ids: np.ndarray,
    n_tasks: int,
    time_frac: np.ndarray,
) -> torch.Tensor:
    """Concatenate the critic's per-frame inputs. The time fraction matters: LIBERO episodes time out,
    so the same scene is worth less late in an attempt."""
    task_onehot = F.one_hot(torch.as_tensor(task_ids, dtype=torch.long), num_classes=n_tasks).float()
    time_feat = torch.as_tensor(time_frac, dtype=torch.float32).unsqueeze(-1)
    return torch.cat([image.float(), state.float(), task_onehot, time_feat], dim=-1)


@torch.inference_mode()
def _predict(model: nn.Module, x_all: torch.Tensor, idx: np.ndarray, device: torch.device) -> np.ndarray:
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
    preds = torch.cat([torch.sigmoid(model(x_all[chunk])) for chunk in idx_t.split(8192)])
    return preds.float().cpu().numpy()


@dataclass
class CriticFitResult:
    oof_values: np.ndarray  # (N,) V(s_t) in [0, 1]: out of fold for trained frames, fold mean otherwise
    train_loss: float
    oof_bce: float


def fit_cross_fitted_critic(
    inputs: torch.Tensor,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    episode_ids: np.ndarray,
    *,
    train_mask: np.ndarray | None = None,
    n_folds: int,
    steps: int,
    batch_size: int,
    lr: float,
    hidden: int,
    device: torch.device,
    seed: int,
) -> CriticFitResult:
    """Train ``n_folds`` critics on disjoint episode folds and score every frame out of fold.

    ``episode_ids`` must be unique per episode across the whole buffer (not per dataset). Frames outside
    ``train_mask`` (default: all frames train) are not trained on and get the mean of the fold models.
    """
    n = inputs.shape[0]
    train = np.ones(n, dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    if not train.any():
        raise ValueError("No frames to fit the critic on")
    rng = np.random.default_rng(seed)
    unique_eps = np.unique(episode_ids[train])
    n_folds = max(2, min(n_folds, len(unique_eps)))
    fold_of_ep = dict(zip(rng.permutation(unique_eps), np.arange(len(unique_eps)) % n_folds, strict=True))
    fold = np.array([fold_of_ep[e] if t else -1 for e, t in zip(episode_ids, train, strict=True)])
    extra_idx = np.nonzero(~train)[0]
    extra_sum = np.zeros(len(extra_idx), dtype=np.float64)

    # Standardize inputs with statistics of the whole buffer (no labels involved, so no leakage).
    mean = inputs.mean(dim=0, keepdim=True)
    std = inputs.std(dim=0, keepdim=True).clamp_min(1e-4)
    x_all = ((inputs - mean) / std).to(device)
    y_all = torch.as_tensor(targets, dtype=torch.float32, device=device)

    oof = np.zeros(n, dtype=np.float32)
    train_losses = []
    torch_gen = torch.Generator(device="cpu").manual_seed(seed)
    for k in range(n_folds):
        train_idx = np.nonzero(train & (fold != k))[0]
        test_idx = np.nonzero(fold == k)[0]
        if len(test_idx) == 0:
            continue
        model = CriticMLP(x_all.shape[1], hidden=hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        probs = torch.as_tensor(sample_weights[train_idx], dtype=torch.float64)
        probs = probs / probs.sum()
        train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)
        model.train()
        last = 0.0
        for _ in range(steps):
            pick = train_idx_t[torch.multinomial(probs, batch_size, replacement=True, generator=torch_gen)]
            pick = pick.to(device)
            logits = model(x_all[pick])
            loss = F.binary_cross_entropy_with_logits(logits, y_all[pick])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = float(loss.detach())
        train_losses.append(last)
        model.eval()
        oof[test_idx] = _predict(model, x_all, test_idx, device)
        if len(extra_idx):
            extra_sum += _predict(model, x_all, extra_idx, device)

    if len(extra_idx):
        oof[extra_idx] = (extra_sum / len(train_losses)).astype(np.float32)
    eps = 1e-6
    clipped = np.clip(oof[train], eps, 1 - eps)
    y = targets[train]
    bce = -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
    oof_bce = float(np.average(bce, weights=sample_weights[train]))
    return CriticFitResult(oof_values=oof, train_loss=float(np.mean(train_losses)), oof_bce=oof_bce)
