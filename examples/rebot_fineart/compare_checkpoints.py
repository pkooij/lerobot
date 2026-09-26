"""Paired old/new losses on fixed observations, objectives, prompts and noise seeds.

Teacher-forced diagnostics only; no optimizer, hardware, or model uploads.
"""

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from comparison_metrics import paired_deltas, stage_indices, summarize

from lerobot.datasets.language_render import active_at
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.recipe import TrainingRecipe
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi052.configuration_pi052 import PI052Config
from lerobot.policies.pi052.modeling_pi052 import PI052Policy
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.processor.render_messages_processor import RenderTrainingMessagesStep
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
from lerobot.utils.collate import lerobot_collate_fn

RENAME = {
    "observation.images.base": "observation.images.cam_high",
    "observation.images.left_wrist": "observation.images.cam_left_wrist",
    "observation.images.right_wrist": "observation.images.cam_right_wrist",
}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluation_recipe(mode):
    if mode.startswith("predict_subtask"):
        return TrainingRecipe.from_dict(
            {
                "messages": [
                    {"role": "user", "content": "${task}", "stream": "high_level"},
                    {"role": "assistant", "content": "${subtask}", "stream": "high_level", "target": True},
                ]
            }
        )
    content = "${subtask}" if mode == "subtask_actions" else "${task}"
    return TrainingRecipe.from_dict(
        {"messages": [{"role": "user", "content": content, "stream": "low_level"}]}
    )


def manifest(root, panel, count):
    split = json.loads((root / "split.json").read_text())
    episodes = split[panel]
    assert not set(episodes) & set(split["train"])
    goals = json.loads((root / "episode_goals.json").read_text())
    annotations = {}
    lengths = {}
    frame_times = {}
    for path in sorted((root / "dataset_repaired/data").rglob("*.parquet")):
        for row in pq.read_table(
            path, columns=["episode_index", "frame_index", "timestamp", "language_persistent"]
        ).to_pylist():
            ep = row["episode_index"]
            if ep in episodes:
                annotations.setdefault(ep, row["language_persistent"])
                lengths[ep] = max(lengths.get(ep, 0), row["frame_index"] + 1)
                frame_times[ep, row["frame_index"]] = row["timestamp"]
    fps = json.loads((root / "dataset_repaired/meta/info.json").read_text())["fps"]
    rows = []
    for ep in episodes:
        transitions = [r["timestamp"] for r in annotations[ep] if r["style"] == "subtask"]
        goal = goals[str(ep)]
        # Held-out template: absent from the six training phrasings.
        paraphrase = "The goal is to " + goal[0].lower() + goal[1:]
        assert paraphrase not in [r["content"] for r in annotations[ep] if r["style"] == "task_aug"]
        for frame in stage_indices(lengths[ep], fps, transitions, count):
            subtask = active_at(frame_times[ep, frame], persistent=annotations[ep], style="subtask")[
                "content"
            ]
            rows.append(
                {
                    "episode": ep,
                    "frame": frame,
                    "goal": goal,
                    "paraphrase": paraphrase,
                    "subtask": subtask,
                    "seed": 1000 + ep * 100000 + frame,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", choices=["dev", "test"], required=True)
    parser.add_argument(
        "--checkpoint", action="append", default=[], help="Unique label=/absolute/checkpoint/path"
    )
    parser.add_argument("--frames-per-episode", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--processor-only",
        action="store_true",
        help="CPU audit of saved processors; no model loaded or losses computed",
    )
    parser.add_argument("--smoke", action="store_true", help="One frame per episode, all evaluation modes")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = manifest(args.root, args.panel, args.frames_per_episode)
    if args.smoke:
        cases = list({r["episode"]: r for r in reversed(cases)}.values())
    digest = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    write_json(args.output / "manifest.json", {"panel": args.panel, "sha256": digest, "cases": cases})
    if args.manifest_only:
        print(
            json.dumps(
                {
                    "panel": args.panel,
                    "frames": len(cases),
                    "episodes": sorted({r["episode"] for r in cases}),
                    "sha256": digest,
                }
            ),
            flush=True,
        )
        return
    assert args.checkpoint and (args.processor_only or torch.cuda.is_available())
    checkpoints = dict(entry.split("=", 1) for entry in args.checkpoint)
    assert len(checkpoints) == len(args.checkpoint), "Duplicate checkpoint labels"
    reports = {}
    normalization_hash = None
    action_names = None
    for label, path in checkpoints.items():
        checkpoint = Path(path)
        with (checkpoint / "model.safetensors").open("rb") as stream:
            weight_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        cfg = PI052Config.from_pretrained(checkpoint)
        assert cfg.chunk_size == 50 and cfg.flow_num_repeats == 5
        if action_names is None:
            action_names = cfg.action_feature_names
        assert cfg.action_feature_names == action_names, "Joint ordering differs"
        cfg.device = "cpu" if args.processor_only else "cuda"
        policy = (
            None
            if args.processor_only
            else PI052Policy.from_pretrained(checkpoint, config=cfg).to("cuda").eval()
        )
        pre, post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": cfg.device}},
        )
        renderers = [s for s in pre.steps if isinstance(s, RenderTrainingMessagesStep)]
        assert len(renderers) == 1
        # Disable language dropout in evaluation, regardless of a checkpoint's training settings.
        for step in pre.steps:
            for name in ("plan_dropout_prob", "memory_dropout_prob", "subtask_dropout_prob"):
                if hasattr(step, name):
                    setattr(step, name, 0.0)
        modes = ["goal_actions", "goal_actions_paraphrase"]
        if cfg.text_loss_weight > 0:
            modes += ["subtask_actions", "predict_subtask", "predict_subtask_paraphrase"]
        rows = []
        for ep in sorted({r["episode"] for r in cases}):
            dataset = LeRobotDataset(
                "pepijn223/rebot_diverse_picking_100_annotated",
                root=args.root / "dataset_repaired",
                revision="93c97807c46535745d0587d4296416bf2d4aa80d",
                episodes=[ep],
                video_backend="pyav",
                delta_timestamps={"action": [i / 30 for i in range(50)]},
            )
            for case in [r for r in cases if r["episode"] == ep]:
                item = dataset[case["frame"]]
                assert int(item["frame_index"]) == case["frame"] and int(item["episode_index"]) == ep
                for mode in modes:
                    renderers[0].recipe = evaluation_recipe(mode)
                    prompt = case["paraphrase"] if mode.endswith("paraphrase") else case["goal"]
                    raw = {**item, "task": prompt}
                    random.seed(case["seed"])
                    np.random.seed(case["seed"])
                    torch.manual_seed(case["seed"])
                    batch = _preprocess_dataset_batch(
                        lerobot_collate_fn([raw]), dataset.meta.camera_keys, RENAME, pre
                    )
                    text = mode.startswith("predict_subtask")
                    assert bool(batch["predict_actions"].any()) != text
                    assert bool((batch["text_labels"] != -100).any()) == text
                    if not rows:
                        normalizer = next(s for s in pre.steps if isinstance(s, NormalizerProcessorStep))
                        stats = {
                            k: {n: v.cpu().tolist() for n, v in values.items()}
                            for k, values in normalizer._tensor_stats.items()
                        }
                        stats_hash = hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()
                        if normalization_hash is None:
                            normalization_hash = stats_hash
                        assert stats_hash == normalization_hash, (
                            "Normalization differs; losses not directly comparable"
                        )
                    wanted = ["text_loss"] if text else ["flow_loss", "fast_action_loss"]
                    metrics = {}
                    loss = None
                    if not args.processor_only:
                        with torch.inference_mode():
                            loss, metrics = policy(batch)
                        assert torch.isfinite(loss).all()
                    rows.append(
                        {
                            "episode": ep,
                            "frame": case["frame"],
                            "mode": mode,
                            "seed": case["seed"],
                            "prompt": case["subtask"] if mode == "subtask_actions" else prompt,
                            "losses": {} if args.processor_only else {k: float(metrics[k]) for k in wanted},
                            "text_target_tokens": int((batch["text_labels"] != -100).sum()),
                            "predict_actions": bool(batch["predict_actions"].any()),
                        }
                    )
            del dataset
        report = {
            "checkpoint": str(checkpoint),
            "scope": "processor-only; no model loaded or losses evaluated"
            if args.processor_only
            else "teacher-forced component losses",
            "weight_sha256": weight_sha,
            "manifest_sha256": digest,
            "normalization_sha256": normalization_hash,
            "training_recipe": cfg.recipe,
            "summary": summarize(rows),
            "per_episode": {
                str(ep): summarize([r for r in rows if r["episode"] == ep])
                for ep in sorted({r["episode"] for r in rows})
            },
            "samples": rows,
            "limitations": "Teacher-forced component losses with matched inputs; not physical success. Goal-action tests are transfer tests for the old subtask-only execution model.",
        }
        write_json(args.output / f"{label}.json", report)
        reports[label] = report
        comparison = {
            name: {
                "summary": r["summary"],
                "paired_deltas_vs": {
                    old: paired_deltas(r["samples"], baseline["samples"])
                    for old, baseline in reports.items()
                    if old != name
                },
            }
            for name, r in reports.items()
        }
        write_json(args.output / "comparison.json", comparison)
        print(
            json.dumps({"checkpoint": label, "panel": args.panel, "summary": report["summary"]}), flush=True
        )
        del policy, pre, post, batch, loss, renderers, step, normalizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
