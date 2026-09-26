"""Exercise task-only stability controls on the known spike batch, without saving weights.

Run in an allocated GPU job with the updated branch on PYTHONPATH. This is a short
optimizer stress test, not a replacement training run or an inference evaluation.
"""

import argparse
import json
import random

import numpy as np
import torch
from eval_components import RENAME, ROOT
from eval_metrics import episode_indices

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.optim.grad_clip import clip_grad_norm_with_groups_
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi052.configuration_pi052 import PI052Config
from lerobot.policies.pi052.modeling_pi052 import PI052Policy
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
from lerobot.utils.collate import lerobot_collate_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    assert args.steps > 0 and torch.cuda.is_available(), "Run in a GPU allocation"
    checkpoint = ROOT / "runs/task_only_full_86031/checkpoints/010000/pretrained_model"
    config = PI052Config.from_pretrained(checkpoint)
    config.device = "cuda"
    config.conditioning_lr_scale = 0.1
    config.conditioning_grad_clip_norm = 0.1
    config.action_expert_grad_clip_norm = 1.0
    policy = PI052Policy.from_pretrained(checkpoint, config=config).to("cuda").train()
    pre, _ = make_pre_post_processors(policy_cfg=config, pretrained_path=str(checkpoint))
    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    dataset = LeRobotDataset(
        "pepijn223/rebot_diverse_picking_100_annotated",
        root=ROOT / "dataset_repaired",
        episodes=[1],
        video_backend="pyav",
        delta_timestamps={"action": [i / 30 for i in range(config.chunk_size)]},
    )
    items = [dataset[index] for index in episode_indices(len(dataset), 16)]
    report = {
        "checkpoint": str(checkpoint),
        "scope": "In-memory stress updates only; no weights saved",
        "frames": [int(item["frame_index"]) for item in items],
        "steps": [],
        "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups},
    }
    for step in range(args.steps):
        seed = 1001 + step % 2
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        optimizer.zero_grad(set_to_none=True)
        batch = _preprocess_dataset_batch(lerobot_collate_fn(items), dataset.meta.camera_keys, RENAME, pre)
        loss, metrics = policy(batch)
        loss.backward()
        raw_norm, clipping = clip_grad_norm_with_groups_(policy.parameters(), optimizer.param_groups, 1.0)
        post_norms = {
            group["name"]: torch.linalg.vector_norm(
                torch.stack(
                    [
                        torch.linalg.vector_norm(p.grad, dtype=torch.float64)
                        for p in group["params"]
                        if p.grad is not None
                    ]
                )
            ).item()
            for group in optimizer.param_groups
        }
        optimizer.step()
        row = {
            "step": step,
            "seed": seed,
            "loss": loss.item(),
            "raw_norm": raw_norm.item(),
            "post_clip_norms": post_norms,
            **metrics,
            **clipping,
        }
        report["steps"].append(row)
        (ROOT / "task_only_stability_probe.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(row), flush=True)
        del batch, loss


if __name__ == "__main__":
    main()
