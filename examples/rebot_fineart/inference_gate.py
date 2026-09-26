"""Exercise both prepared policies' real inference paths before any new training."""

import gc
import json
import os
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi052.configuration_pi052 import PI052Config
from lerobot.policies.pi052.modeling_pi052 import PI052Policy
from lerobot.processor.rename_processor import rename_stats
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
from lerobot.utils.constants import QUERY_KIND, QUERY_TEXT

ROOT = Path(os.environ["REBOT_EXPERIMENT_ROOT"])
RENAME = {
    "observation.images.base": "observation.images.cam_high",
    "observation.images.left_wrist": "observation.images.cam_left_wrist",
    "observation.images.right_wrist": "observation.images.cam_right_wrist",
}


def main():
    assert torch.cuda.is_available(), "Run inside the allocated GPU job"
    dataset = LeRobotDataset(
        "pepijn223/rebot_diverse_picking_100_annotated",
        root=ROOT / "dataset_repaired",
        revision="93c97807c46535745d0587d4296416bf2d4aa80d",
        episodes=[6],
        video_backend="pyav",
    )
    item = dataset[min(100, len(dataset) - 1)]
    obs = {k: v for k, v in item.items() if k == "observation.state" or k in RENAME}
    goal = json.loads((ROOT / "episode_goals.json").read_text())["6"]
    reports = {}
    for variant in ("subtask", "task_only"):
        checkpoint = ROOT / f"init_{variant}"
        cfg = PI052Config.from_pretrained(checkpoint)
        cfg.device = "cuda"
        policy = PI052Policy.from_pretrained(checkpoint, config=cfg).to("cuda").eval()
        pre, post = make_pre_post_processors(
            policy_cfg=cfg, dataset_stats=rename_stats(dataset.meta.stats, RENAME)
        )
        instruction = "Reach toward the blue block."
        if variant == "subtask":
            query = _preprocess_dataset_batch(
                {**obs, QUERY_KIND: "next_subtask", QUERY_TEXT: goal},
                dataset.meta.camera_keys,
                RENAME,
                pre,
            )
            with torch.inference_mode():
                instruction = policy.generate_text(query)
            assert isinstance(instruction, str) and instruction.strip()
        else:
            instruction = goal
        batch = _preprocess_dataset_batch({**obs, "task": instruction}, dataset.meta.camera_keys, RENAME, pre)
        assert policy.model.precompute_denoise_times is True
        policy.reset()
        with torch.inference_mode():
            action = post(policy.select_action(batch))
        assert action.shape == (1, 14) and torch.isfinite(action).all()
        outputs = []
        for enabled in (False, True):
            policy.model.precompute_denoise_times = enabled
            torch.manual_seed(20260926)
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(batch)
            assert chunk.shape == (1, 50, 14) and torch.isfinite(chunk).all()
            outputs.append(chunk.cpu())
        policy.model.precompute_denoise_times = True
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        reports[variant] = {
            "select_action_shape": list(action.shape),
            "chunk_shape": list(outputs[0].shape),
            "finite": True,
            "precomputed_vs_legacy_bitwise_equal": True,
            "instruction": instruction,
            "scope": "GPU inference correctness; no speed benchmark, training or hardware",
        }
        print(json.dumps({variant: reports[variant]}), flush=True)
        del policy, pre, post, batch, action, chunk, outputs
        if variant == "subtask":
            del query
        gc.collect()
        torch.cuda.empty_cache()
    destination = ROOT / f"inference_gate_{os.environ['SLURM_JOB_ID']}.json"
    destination.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
