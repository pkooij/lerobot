"""Exercise the actual dataset, recipe, tokenizer and split before training."""

import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi052.configuration_pi052 import PI052Config
from lerobot.processor.rename_processor import rename_stats
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
from lerobot.utils.collate import lerobot_collate_fn

ROOT = Path("/fsx/pepijn/rebot-fineart-20260925")
split = json.loads((ROOT / "split.json").read_text())
rename = {
    "observation.images.base": "observation.images.cam_high",
    "observation.images.left_wrist": "observation.images.cam_left_wrist",
    "observation.images.right_wrist": "observation.images.cam_right_wrist",
}
reports = {}
for variant in ("subtask", "task_only"):
    policy = PI052Config.from_pretrained(ROOT / f"init_{variant}")
    cfg = TrainPipelineConfig(
        policy=policy,
        dataset=DatasetConfig(
            repo_id="pepijn223/rebot_diverse_picking_100_annotated",
            revision="93c97807c46535745d0587d4296416bf2d4aa80d",
            root=str(ROOT / "dataset_repaired"),
            episodes=split["train"] + split["dev"],
            eval_split=0.05,
            video_backend="pyav",
        ),
        rename_map=rename,
    )
    dataset, dev = make_train_eval_datasets(cfg)
    assert dataset.episodes == split["train"], (dataset.episodes, split["train"])
    assert dev.episodes == split["dev"], (dev.episodes, split["dev"])
    assert not set(dataset.episodes) & set(split["test"])
    pre, _ = make_pre_post_processors(
        policy_cfg=policy, dataset_stats=rename_stats(dataset.meta.stats, rename)
    )
    counts = {"text_samples": 0, "action_samples": 0, "samples": 0}
    for indices in np.array_split(np.linspace(0, len(dataset) - 1, 64, dtype=int), 4):
        raw = lerobot_collate_fn([dataset[int(i)] for i in indices])
        batch = _preprocess_dataset_batch(raw, dataset.meta.camera_keys, rename, pre)
        assert torch.isfinite(batch["action"]).all()
        text = (batch["text_labels"] != -100).any(-1)
        actions = batch["predict_actions"].bool()
        assert torch.logical_xor(text, actions).all() if variant == "subtask" else actions.all()
        if variant == "task_only":
            assert not text.any(), "Baseline leaked subtask targets"
        counts["text_samples"] += int(text.sum())
        counts["action_samples"] += int(actions.sum())
        counts["samples"] += len(indices)
    if variant == "subtask":
        assert counts["text_samples"] > 0 and counts["action_samples"] > 0
    reports[variant] = counts
(ROOT / "processor-audit.json").write_text(json.dumps(reports, indent=2))
print("NATIVE PROCESSOR AUDIT PASSED", json.dumps(reports), flush=True)
