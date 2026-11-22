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
"""Turn RLinf's openpi pi0.5 for RoboLab (DROID, absolute joint positions) into a LeRobot pi05 checkpoint.

    python examples/flex_robolab/build_pi05_robolab.py --out outputs/pi05_robolab

The checkpoint's weights already use openpi's PyTorch names, which `PI05Policy.from_pretrained` converts.
What it lacks is a LeRobot config and normalization stats:

- cameras: RoboLab's ``over_shoulder_left_camera`` is openpi's base camera and ``wrist_cam`` its left
  wrist camera; the right-wrist slot is pi05's placeholder camera (``empty_cameras=1``), a masked blank
  image (as RLinf does);
- state / action: 7 joint positions (rad) + gripper (0 open, 1 closed), quantile-normalized with the first 8
  dims of the checkpoint's ``norm_stats.json``;
- actions are trained relative to the state for the 7 joints (openpi ``DeltaActions``), gripper absolute;
- 15-step chunks, all executed (``n_action_steps=15``), 10 denoising steps.

The VLM is frozen (``train_expert_only``) and stored in bf16 so FLEX fits next to Isaac Sim on one 48 GB GPU.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

SOURCE = "RLinf/RLinf-Pi05-RoboLab-droid-jointpos"
CAMERAS = ["over_shoulder_left_camera", "wrist_cam"]  # base, left wrist (right wrist: empty_cameras)
JOINT_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]
DIM = 8


def build_config(device: str) -> PI05Config:
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
    return PI05Config(
        input_features={
            **{f"{OBS_IMAGES}.{cam}": image for cam in CAMERAS},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(DIM,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(DIM,))},
        empty_cameras=1,
        chunk_size=15,
        n_action_steps=15,
        num_inference_steps=10,
        tokenizer_max_length=200,
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        action_feature_names=JOINT_NAMES,
        train_expert_only=True,
        gradient_checkpointing=True,
        dtype=torch.bfloat16,
        device=device,
    )


def build_stats(revision: str | None) -> dict[str, dict[str, torch.Tensor]]:
    path = hf_hub_download(SOURCE, "assets/droid/norm_stats.json", revision=revision)
    raw = json.loads(Path(path).read_text())
    raw = raw.get("norm_stats", raw)
    stats = {}
    for ours, theirs in ((OBS_STATE, "state"), (ACTION, "actions")):
        stats[ours] = {
            k: torch.tensor(raw[theirs][k][:DIM], dtype=torch.float32) for k in ("mean", "std", "q01", "q99")
        }
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config_only", action="store_true", help="skip the weights (to check the setup)")
    args = ap.parse_args()

    config = build_config(args.device)
    stats = build_stats(args.revision)
    for key, s in stats.items():
        logging.info("%s q01 %s q99 %s", key, s["q01"].tolist(), s["q99"].tolist())
    preprocessor, postprocessor = make_pi05_pre_post_processors(config, dataset_stats=stats)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.config_only:
        config._save_pretrained(args.out)
    else:
        policy = PI05Policy.from_pretrained(SOURCE, config=config, revision=args.revision)
        policy.save_pretrained(args.out)
    preprocessor.save_pretrained(args.out)
    postprocessor.save_pretrained(args.out)
    logging.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
