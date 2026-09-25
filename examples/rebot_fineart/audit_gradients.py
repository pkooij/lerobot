"""Probe saved checkpoints on fixed training batches without optimizer updates.

Run only on a scheduled GPU after checking status.json. This is a single-rank
diagnostic, not an exact replay of four-rank training or a validation substitute.
"""

import argparse
import gc
import hashlib
import json
import random

import numpy as np
import torch
from eval_components import RENAME, ROOT
from eval_metrics import episode_indices
from gradient_metrics import gradient_summary, json_number

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi052.configuration_pi052 import PI052Config
from lerobot.policies.pi052.modeling_pi052 import PI052Policy
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
from lerobot.utils.collate import lerobot_collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="86031")
    parser.add_argument("--variant", choices=["task_only", "subtask"], default="task_only")
    parser.add_argument("--steps", type=int, nargs="+", default=[1000, 3000])
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "Use a scheduled GPU allocation"
    assert args.batch_size > 0
    split = json.loads((ROOT / "split.json").read_text())
    episodes = args.episodes or split["train"][:3]
    assert set(episodes) <= set(split["train"]), "Probe training episodes only"
    report = {
        "variant": args.variant,
        "batch_size": args.batch_size,
        "episodes": episodes,
        "torch_version": torch.__version__,
        "scope": "One-rank fresh forward/backward on fixed training frames; no optimizer, no checkpoint writes. Native clipping mutates only in-memory gradients after inspection. Not an exact DDP replay or proof of the original failure's cause.",
        "checkpoints": {},
    }
    destination = ROOT / f"gradient_probe_{args.variant}_{args.job}.json"
    for step in args.steps:
        path = ROOT / f"runs/{args.variant}_full_{args.job}/checkpoints/{step:06d}/pretrained_model"
        with (path / "model.safetensors").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        cfg = PI052Config.from_pretrained(path)
        cfg.device = "cuda"
        policy = PI052Policy.from_pretrained(path, config=cfg).to("cuda").train()
        pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(path))
        result = {"checkpoint": str(path), "sha256": digest, "batches": []}
        report["checkpoints"][str(step)] = result
        for episode in episodes:
            dataset = LeRobotDataset(
                "pepijn223/rebot_diverse_picking_100_annotated",
                root=ROOT / "dataset_repaired",
                revision="93c97807c46535745d0587d4296416bf2d4aa80d",
                episodes=[episode],
                video_backend="pyav",
                delta_timestamps={"action": [i / 30 for i in range(cfg.chunk_size)]},
            )
            indices = episode_indices(len(dataset), args.batch_size)
            assert len(indices) == args.batch_size
            items = [dataset[index] for index in indices]
            seed = 1000 + episode
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            policy.zero_grad(set_to_none=True)
            batch = _preprocess_dataset_batch(
                lerobot_collate_fn(items), dataset.meta.camera_keys, RENAME, pre
            )
            loss, metrics = policy(batch)
            loss.backward()
            before = gradient_summary(policy.named_parameters())
            native_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
            after = gradient_summary(policy.named_parameters())
            row = {
                "episode": episode,
                "frames": [int(item["frame_index"]) for item in items],
                "seed": seed,
                "loss": json_number(loss.item()),
                "metrics": {key: json_number(value) for key, value in metrics.items()},
                "before_clip": before,
                "native_clip_returned_norm": json_number(native_norm),
                "after_clip": after,
            }
            result["batches"].append(row)
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(destination)
            print(json.dumps({"step": step, "episode": episode, "norm": row["native_clip_returned_norm"]}))
            del batch, loss, items, dataset
        del policy, pre, post
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
