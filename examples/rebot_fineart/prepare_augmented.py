"""Prepare fresh, matched 20k runs with audited native task augmentation rows."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from augmentation_recipe import augment_goal, make_recipe

from lerobot.datasets.language_render import render_sample
from lerobot.datasets.recipe import TrainingRecipe

ROOT = Path(os.environ["REBOT_EXPERIMENT_ROOT"])
BASE = Path("/fsx/pepijn/rebot-fineart-20260925")
SOURCE = Path("/fsx/pepijn/rebot-pi052-sft-20260910")


def main():
    assert not (ROOT / "prepared.json").exists(), "Already prepared; do not overwrite"
    split = json.loads((BASE / "split.json").read_text())
    goals = json.loads((BASE / "episode_goals.json").read_text())
    augmentations = {ep: augment_goal(goal) for ep, goal in goals.items()}
    original = json.loads((BASE / "prepared.json").read_text())
    weights = SOURCE / "midtrain/model.safetensors"
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == original["source_sha256"], "Midtraining source changed"
    data = ROOT / "dataset_repaired"
    data.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASE / "dataset_repaired/meta", data / "meta")
    (data / "videos").symlink_to(SOURCE / "dataset/videos", target_is_directory=True)
    recipes = {v: TrainingRecipe.from_dict(make_recipe(v)) for v in ("subtask", "task_only")}
    counts = {v: Counter() for v in recipes}
    seen = {v: defaultdict(set) for v in recipes}
    source_styles = Counter()
    train_frames = 0
    for path in sorted((SOURCE / "dataset/data").rglob("*.parquet")):
        table = pq.read_table(path)
        language = []
        for row in table.select(
            ["episode_index", "index", "timestamp", "language_persistent", "language_events"]
        ).to_pylist():
            ep = row["episode_index"]
            source_styles.update(r["style"] for r in row["language_persistent"])
            # Pin the inspected source contract; new upstream annotations require review.
            assert all(r["style"] == "subtask" for r in row["language_persistent"])
            assert not row["language_events"]
            rows = row["language_persistent"] + [
                {
                    "role": "user",
                    "content": goal,
                    "style": "task_aug",
                    "timestamp": 0.0,
                    "camera": None,
                    "tool_calls": None,
                }
                for goal in augmentations[str(ep)]
            ]
            language.append(rows)
            if ep not in split["train"]:
                continue
            train_frames += 1
            for variant, recipe in recipes.items():
                rendered = render_sample(
                    recipe=recipe,
                    persistent=rows,
                    events=[],
                    t=row["timestamp"],
                    sample_idx=row["index"],
                    task="stale canonical task",
                )
                assert rendered
                prompt = rendered["messages_rendered"][0]["content"]
                assert prompt != "stale canonical task"
                is_goal = prompt in augmentations[str(ep)]
                if is_goal:
                    seen[variant][ep].add(prompt)
                kind = (
                    "text"
                    if rendered["target_message_indices"]
                    else ("goal_action" if is_goal else "subtask_action")
                )
                counts[variant][kind] += 1
        idx = table.schema.get_field_index("language_persistent")
        table = table.set_column(
            idx, "language_persistent", pa.array(language, type=table.schema.field(idx).type)
        )
        destination = data / path.relative_to(SOURCE / "dataset")
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination)
    assert train_frames == original["train_frames"] == 117379
    for variant in recipes:
        assert all(seen[variant][ep] == set(augmentations[str(ep)]) for ep in split["train"])
        destination = ROOT / f"init_{variant}"
        destination.mkdir()
        cfg = json.loads((BASE / f"init_{variant}/config.json").read_text())
        cfg.update(
            pretrained_path=str(destination),
            recipe=make_recipe(variant),
            recipe_path=None,
            scheduler_decay_steps=20000,
            scheduler_warmup_steps=1000,
            scheduler_decay_lr=5e-6,
            conditioning_lr_scale=0.1,
            conditioning_grad_clip_norm=0.1,
            action_expert_grad_clip_norm=1.0,
        )
        (destination / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (destination / "model.safetensors").symlink_to(weights)
    for filename in ("split.json", "episode_goals.json", "repair-audit.json"):
        shutil.copy2(BASE / filename, ROOT / filename)
    (ROOT / "task_augmentations.json").write_text(json.dumps(augmentations, indent=2) + "\n")
    git = shutil.which("git")
    assert git, "Git is required for source provenance"
    report = {
        "source_revision": "93c97807c46535745d0587d4296416bf2d4aa80d",
        "source_weights_sha256": digest,
        "source_language_styles": dict(source_styles),
        "initialization": "Fresh original Jade midtraining weights and fresh optimizer for both variants",
        "train_frames": train_frames,
        "steps": 20000,
        "global_batch": 64,
        "effective_epochs": 20000 * 64 / train_frames,
        "augmentation": "Six goal-preserving templates per episode; sampled deterministically by frame index",
        "coverage": {v: dict(c) for v, c in counts.items()},
        "all_training_episode_variants_sampled": True,
        "stability": "Both variants: conditioning LR x0.1, conditioning clip 0.1, expert clip 1; nonfinite gradients fail",
        "evaluation": "250 frames balanced across all five dev episodes and time; test episodes untouched",
        "code_revision": subprocess.check_output([git, "rev-parse", "HEAD"], text=True).strip(),
    }
    (ROOT / "prepared.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("audit_processors.py"))], check=True)


if __name__ == "__main__":
    main()
