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
"""The numeric core of FLEX: returns, advantages, critic quality, staleness and sampling weights.

Plain numpy, so it can be unit-tested without a simulator or a policy. Notation follows the FLEX poster
(Nguyen, Hoeg, Tuscher, "FLEX: Adaptive Experience Replay for Sample-Efficient Flow Policy Gradients in
Vision-Language-Action Models", R2RL workshop, CoRL 2026):

- advantage ``A = G - V(s)``, standardized within each task: ``A_i = (A_i - mu_task) / sigma_task``;
- critic-scaled policy-gradient loss, ``q_c = clip(E_c / E*, q_min, 1)`` with ``E_c`` the critic's
  explained variance;
- staleness weight ``w = 2 ** (-d / h_s)`` for data collected ``d`` iterations ago;
- advantage-prioritized sampling, ``p_i ~ |A_i|`` mixed with a uniform distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeRecord:
    """One episode in the replay buffer (a practice attempt)."""

    source: str  # "practice"
    round: int  # practice round (FLEX iteration) that produced it
    task_key: str  # e.g. "libero_spatial/3"
    success: bool
    length: int  # recorded frames
    stride: int = 1  # env steps per recorded frame (n_action_steps when only chunk starts are recorded)


def value_targets(length: int, success: bool, gamma: float) -> np.ndarray:
    """Return ``G_t`` of every frame for a sparse success reward at the last step:
    ``success * gamma ** (steps until the episode ends)``.

    With ``gamma < 1`` the return also rewards finishing sooner. ``gamma = 1`` gives the plain outcome.
    """
    steps_to_go = np.arange(length - 1, -1, -1, dtype=np.float64)
    return (float(success) * gamma**steps_to_go).astype(np.float32)


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (ties averaged). Returns NaN when only one class is present."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    i = 0
    while i < scores.size:  # average ranks over ties
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def explained_variance(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """``1 - Var(target - pred) / Var(target)``; NaN when the target is constant."""
    target = np.asarray(target, dtype=np.float64)
    var = float(target.var())
    if target.size == 0 or var < eps:
        return float("nan")
    return float(1.0 - np.var(target - np.asarray(pred, dtype=np.float64)) / var)


def critic_scale(ev: float, ev_target: float, q_min: float) -> float:
    """``q_c = clip(E_c / E*, q_min, 1)``. An unmeasurable critic (NaN) gets ``q_min``."""
    if not np.isfinite(ev):
        return float(q_min)
    return float(np.clip(ev / ev_target, q_min, 1.0))


def standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zero-mean, unit-std copy (zeros for a constant input)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    std = float(x.std())
    if std < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x.mean()) / std).astype(np.float32)


def per_task_standardize(
    adv: np.ndarray, task_ids: np.ndarray, fit_mask: np.ndarray | None = None, eps: float = 1e-6
) -> np.ndarray:
    """``(A - mu_task) / sigma_task`` within each task, with the moments taken over ``fit_mask`` frames.

    Tasks differ in learnability: without this, a task whose returns vary a lot would dominate the
    policy gradient.
    """
    adv = np.asarray(adv, dtype=np.float64)
    task_ids = np.asarray(task_ids)
    fit = np.ones(adv.shape, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    out = np.zeros_like(adv, dtype=np.float32)
    for t in np.unique(task_ids):
        sel = task_ids == t
        ref = adv[sel & fit]
        if ref.size == 0:
            continue
        std = float(ref.std())
        out[sel] = 0.0 if std < eps else ((adv[sel] - ref.mean()) / std).astype(np.float32)
    return out


def staleness_weights(rounds: np.ndarray, current_round: int, half_life: float) -> np.ndarray:
    """``w = 2 ** (-d / h_s)`` with ``d`` the age in iterations. ``half_life <= 0`` disables (all ones)."""
    age = np.maximum(0, current_round - np.asarray(rounds, dtype=np.float64))
    if half_life <= 0:
        return np.ones_like(age)
    return np.power(2.0, -age / half_life)


def sampling_probs(
    adv: np.ndarray, staleness: np.ndarray, uniform_mix: float, prioritized: bool = True
) -> np.ndarray:
    """Replay sampling distribution: staleness times ``(1 - u) * |A| / mean|A| + u``, normalized.

    High-advantage transitions (in either direction) carry most of the policy-gradient signal; the
    uniform share keeps every transition reachable.
    """
    staleness = np.asarray(staleness, dtype=np.float64)
    if prioritized:
        mag = np.abs(np.asarray(adv, dtype=np.float64))
        mean_mag = mag.mean() if mag.size else 0.0
        prio = (1.0 - uniform_mix) * mag / mean_mag + uniform_mix if mean_mag > 0 else np.ones_like(mag)
    else:
        prio = np.ones_like(staleness)
    p = staleness * prio
    total = p.sum()
    return p / total if total > 0 else p
