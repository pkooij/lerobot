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
import numpy as np
import pytest

from lerobot.rl.flex import advantages as adv
from lerobot.rl.flex.select_demos import select_demo_episodes


def test_value_targets_discount_towards_the_end():
    y = adv.value_targets(4, True, 0.5)
    np.testing.assert_allclose(y, [0.125, 0.25, 0.5, 1.0])
    np.testing.assert_allclose(adv.value_targets(4, False, 0.5), np.zeros(4))


def test_binary_auroc():
    assert adv.binary_auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert adv.binary_auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert adv.binary_auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5
    assert np.isnan(adv.binary_auroc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_explained_variance_and_critic_scale():
    g = np.array([0.0, 0.5, 1.0, 0.0])
    assert adv.explained_variance(g, g) == pytest.approx(1.0)
    assert adv.explained_variance(np.full(4, g.mean()), g) == pytest.approx(0.0)
    assert np.isnan(adv.explained_variance(g, np.ones(4)))
    # q_c = clip(E_c / E*, q_min, 1)
    assert adv.critic_scale(0.25, ev_target=0.5, q_min=0.1) == pytest.approx(0.5)
    assert adv.critic_scale(0.9, ev_target=0.5, q_min=0.1) == 1.0
    assert adv.critic_scale(-0.3, ev_target=0.5, q_min=0.1) == pytest.approx(0.1)
    assert adv.critic_scale(float("nan"), ev_target=0.5, q_min=0.1) == pytest.approx(0.1)


def test_per_task_standardize_uses_each_tasks_own_moments():
    a = np.array([0.0, 2.0, 100.0, 300.0])
    tasks = np.array([0, 0, 1, 1])
    out = adv.per_task_standardize(a, tasks)
    np.testing.assert_allclose(out, [-1.0, 1.0, -1.0, 1.0])
    # Moments from the fit_mask frames only, applied to every frame of the task.
    out = adv.per_task_standardize(np.array([0.0, 2.0, 4.0]), np.zeros(3), np.array([True, True, False]))
    np.testing.assert_allclose(out, [-1.0, 1.0, 3.0])
    assert np.all(adv.standardize(np.ones(5)) == 0)


def test_staleness_weights_halve_every_half_life():
    np.testing.assert_allclose(adv.staleness_weights(np.array([4, 3, 2]), 4, 1.0), [1.0, 0.5, 0.25])
    np.testing.assert_allclose(adv.staleness_weights(np.array([4, 2]), 4, 2.0), [1.0, 0.5])
    np.testing.assert_allclose(adv.staleness_weights(np.array([4, 0]), 4, 0.0), [1.0, 1.0])


def test_sampling_probs_prioritize_large_advantages_with_a_uniform_floor():
    a = np.array([0.0, -2.0, 1.0, 1.0])
    p = adv.sampling_probs(a, np.ones(4), uniform_mix=0.5)
    assert p.sum() == pytest.approx(1.0)
    assert p[1] > p[2] > p[0] > 0  # |A| ranks, zero-advantage frames stay reachable
    np.testing.assert_allclose(adv.sampling_probs(a, np.ones(4), 0.5, prioritized=False), np.full(4, 0.25))
    p = adv.sampling_probs(np.ones(2), np.array([1.0, 0.5]), uniform_mix=0.0)
    np.testing.assert_allclose(p, [2 / 3, 1 / 3])


def test_select_demo_episodes_is_seeded_and_matches_normalized_tasks():
    episode_tasks = [["Put the bowl on the plate."]] * 5 + [["open the drawer"]] * 5
    a = select_demo_episodes(episode_tasks, ["put the bowl on the plate", "Open the drawer."], 2, seed=0)
    b = select_demo_episodes(episode_tasks, ["put the bowl on the plate", "Open the drawer."], 2, seed=0)
    assert a == b
    assert len(a) == 4
    assert sum(e < 5 for e in a) == 2
    with pytest.raises(ValueError, match="No demo episodes"):
        select_demo_episodes(episode_tasks, ["stack the blocks"], 1, seed=0)
    with pytest.raises(ValueError, match="fewer than"):
        select_demo_episodes(episode_tasks, ["open the drawer"], 6, seed=0)
