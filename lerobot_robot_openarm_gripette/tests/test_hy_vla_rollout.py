import numpy as np
import torch

from lerobot.policies.hy_vla.configuration_hy_vla import HyVLAConfig
from lerobot.policies.hy_vla.processor_hy_vla import HyVLATokenizerStep, make_hy_vla_pre_post_processors
from lerobot_robot_openarm_gripette.hy_vla_rollout import load_processors, make_runtime_observation


class FakeTokenizer:
    def __call__(self, prompts, **kwargs):
        shape = (len(prompts), kwargs["max_length"])
        return {
            "input_ids": torch.zeros(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.long),
        }


def test_short_execution_reloads_reference_and_normalization_at_each_replan(tmp_path):
    config = HyVLAConfig(
        device="cpu",
        embodiment="single_arm",
        model_action_dim=11,
        relative_convention="current",
        execution_horizon=50,
    )
    mean = torch.zeros(50, 11)
    mean[:, 0] = torch.arange(50) * 0.01
    mean[:, 1] = 0.02
    mean[:, 3] = mean[:, 7] = 1
    mean[:, 9:] = torch.tensor([0.2, 0.5])
    stats = {
        "qpos_mean": torch.zeros(2),
        "qpos_std": torch.ones(2),
        "action_mean": mean,
        "action_std": torch.ones(50, 11),
    }
    pre, post = make_hy_vla_pre_post_processors(config, norm_stats=stats)
    pre.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    post.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")
    config.pretrained_path = str(tmp_path)
    config.execution_horizon = 10
    pre, post = load_processors(config, "cpu")
    next(step for step in pre.steps if isinstance(step, HyVLATokenizerStep))._tokenizer = FakeTokenizer()
    assert config.n_action_steps == config.chunk_size == 50
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    for index in range(22):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation
        pose[:3, 3] = [index, 2, 3]
        observation = make_runtime_observation(
            np.zeros((8, 8, 3), dtype=np.uint8),
            [0.2, 0.5],
            pose,
            "Grabette demonstration",
        )
        batch = pre(observation)
        assert batch["observation.state"].shape == (1, 32)
        actual = post(torch.zeros(1, 11))[0].numpy()
        offset = index % 10
        captured_position = np.array([(index // 10) * 10, 2, 3])
        expected_position = captured_position + rotation @ np.array([offset * 0.01, 0.02, 0])
        np.testing.assert_allclose(actual[:3], expected_position, atol=1e-6)
        np.testing.assert_allclose(actual[3:9].reshape(2, 3), rotation[:2], atol=1e-6)
        np.testing.assert_allclose(actual[9:], [0.2, 0.5], atol=1e-6)
