"""Compare saved checkpoints on fixed frames across every development episode.

Teacher-forced diagnostics use annotation/action targets, unlike reload_check.py's
causal inference. Never connect hardware or change running training source.
"""

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from eval_metrics import component_summary, episode_indices

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi052.configuration_pi052 import PI052Config
from lerobot.policies.pi052.modeling_pi052 import PI052Policy
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
from lerobot.utils.collate import lerobot_collate_fn

ROOT = Path("/fsx/pepijn/rebot-fineart-20260925")
RENAME = {
    "observation.images.base": "observation.images.cam_high",
    "observation.images.left_wrist": "observation.images.cam_left_wrist",
    "observation.images.right_wrist": "observation.images.cam_right_wrist",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="86031")
    parser.add_argument("--steps", nargs="+", type=int, default=[1000, 5000, 10000])
    parser.add_argument(
        "--variants", nargs="+", choices=["subtask", "task_only"], default=["subtask", "task_only"]
    )
    parser.add_argument("--frames-per-episode", type=int, default=50)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "Run on a scheduled GPU allocation"
    split = json.loads((ROOT / "split.json").read_text())
    assert not set(split["dev"]) & (set(split["train"]) | set(split["test"]))
    report = {
        "job": args.job,
        "steps": args.steps,
        "episodes": split["dev"],
        "frames_per_episode": args.frames_per_episode,
        "aggregation": "Per-frame component mean conditional on the recipe branch; not the native batched composite loss",
        "limitations": "Teacher-forced offline losses, not generated behavior or physical success. Compare checkpoints within each variant; action language differs between variants.",
        "checkpoints": {},
    }
    destination = ROOT / f"eval_components_{args.job}.json"
    sample_ids = None
    for variant in args.variants:
        for step in args.steps:
            checkpoint = ROOT / f"runs/{variant}_full_{args.job}/checkpoints/{step:06d}/pretrained_model"
            state = json.loads((checkpoint.parent / "training_state/training_step.json").read_text())
            assert state["step"] == step and state["batch_size"] == 16
            assert state["dp_world_size"] == 4 and state["grad_accum_steps"] == 1
            with (checkpoint / "model.safetensors").open("rb") as stream:
                sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            cfg = PI052Config.from_pretrained(checkpoint)
            cfg.device = "cuda"
            policy = PI052Policy.from_pretrained(checkpoint, config=cfg).to("cuda").eval()
            pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(checkpoint))
            rows = []
            for episode in split["dev"]:
                dataset = LeRobotDataset(
                    "pepijn223/rebot_diverse_picking_100_annotated",
                    root=ROOT / "dataset_repaired",
                    revision="93c97807c46535745d0587d4296416bf2d4aa80d",
                    episodes=[episode],
                    video_backend="pyav",
                    delta_timestamps={"action": [i / 30 for i in range(cfg.chunk_size)]},
                )
                for index in episode_indices(len(dataset), args.frames_per_episode):
                    item = dataset[index]
                    assert int(item["episode_index"]) == episode
                    frame = int(item["frame_index"])
                    seed = 1000 + episode * 100000 + frame
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    batch = _preprocess_dataset_batch(
                        lerobot_collate_fn([item]), dataset.meta.camera_keys, RENAME, pre
                    )
                    action = bool(batch["predict_actions"].any())
                    text = bool((batch["text_labels"] != -100).any())
                    assert action != text if variant == "subtask" else action and not text
                    with torch.inference_mode():
                        loss, metrics = policy(batch)
                    assert torch.isfinite(loss).all()
                    rows.append(
                        {
                            "episode": episode,
                            "frame": frame,
                            "timestamp": float(item["timestamp"]),
                            "seed": seed,
                            "branch": "action" if action else "text",
                            "losses": metrics,
                        }
                    )
                del dataset
            ids = [(row["episode"], row["frame"]) for row in rows]
            if sample_ids is None:
                sample_ids = ids
            assert ids == sample_ids, "Checkpoint comparison must use identical frames"
            assert {row["episode"] for row in rows} == set(split["dev"])
            if variant == "subtask":
                assert {row["branch"] for row in rows} == {"text", "action"}
            summary = component_summary(rows)
            report["checkpoints"][f"{variant}_{step}"] = {
                "checkpoint": str(checkpoint),
                "sha256": sha256,
                "summary": summary,
                "per_episode": {
                    str(ep): component_summary([row for row in rows if row["episode"] == ep])
                    for ep in split["dev"]
                },
                "samples": rows,
            }
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(destination)
            print(json.dumps({"variant": variant, "step": step, "summary": summary}), flush=True)
            del policy, pre, post, batch, loss
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
