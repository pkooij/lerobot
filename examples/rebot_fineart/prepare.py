"""Prepare matched ReBot FineART runs without modifying the source dataset/weights."""

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from transformers import AutoProcessor

from lerobot.datasets.language_render import active_at, render_sample
from lerobot.datasets.recipe import TrainingRecipe
from lerobot.policies.pi052.configuration_pi052 import _pi052_default_recipe
from lerobot.policies.pi052.fit_fast_tokenizer import _normalize_actions, _validate_fast_reconstruction

ROOT = Path("/fsx/pepijn/rebot-fineart-20260925")
SOURCE = Path("/fsx/pepijn/rebot-pi052-sft-20260910")
REVISION = "93c97807c46535745d0587d4296416bf2d4aa80d"
DEV = [6, 7, 50, 72, 78]
TEST = list(range(90, 100))
TRAIN = [i for i in range(90) if i not in DEV]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT / "prepared.json").exists(), "Already prepared; inspect rather than overwrite"
    provenance = json.loads((SOURCE / "midtrain-source.json").read_text())
    assert provenance["source"] == (
        "s3://scale-hf-s3-bucket/robotics/jade/vlm-runs/"
        "pi052_subgoal_3cam_14d_balanced_taskfix2/checkpoints/300000/pretrained_model/"
    )
    weights = SOURCE / "midtrain/model.safetensors"
    source_weights = next(f for f in provenance["files"] if f["path"] == "model.safetensors")
    assert weights.stat().st_size == source_weights["bytes"]
    sha = hashlib.sha256()
    parts = []
    with weights.open("rb") as stream:
        while block := stream.read(16 * 1024**2):
            sha.update(block)
            parts.append(hashlib.md5(block, usedforsecurity=False).digest())
    etag = f'"{hashlib.md5(b"".join(parts), usedforsecurity=False).hexdigest()}-{len(parts)}"'
    assert etag == source_weights["etag"], (etag, source_weights["etag"])
    print("Midtraining weights match recorded S3 multipart ETag", flush=True)

    source_data = SOURCE / "dataset"
    files = sorted((source_data / "data").rglob("*.parquet"))
    columns = [
        "episode_index",
        "frame_index",
        "index",
        "timestamp",
        "task_index",
        "language_persistent",
        "language_events",
        "action",
        "observation.state",
    ]
    table = pa.concat_tables([pq.read_table(p, columns=columns) for p in files])
    table = table.sort_by([("index", "ascending")])
    assert table.num_rows == 131716
    train_table = table.filter(pc.is_in(table["episode_index"], value_set=pa.array(TRAIN)))
    info = json.loads((source_data / "meta/info.json").read_text())
    assert info["features"]["action"]["names"] == info["features"]["observation.state"]["names"]
    tasks = pq.read_table(source_data / "meta/tasks.parquet").to_pandas()
    print("Task table:", tasks.to_dict(), flush=True)
    assert len(tasks) == 1, "Review explicit split construction for multi-task data"
    task = str(tasks.index[0])
    recipes = {
        "subtask": _pi052_default_recipe(),
        "task_only": {"messages": [{"role": "user", "content": "${task}", "stream": "low_level"}]},
    }
    recipe_objects = {k: TrainingRecipe.from_dict(v) for k, v in recipes.items()}
    coverage = Counter()
    examples = {}
    for row in train_table.to_pylist():
        subtask = active_at(row["timestamp"], persistent=row["language_persistent"], style="subtask")
        assert subtask and subtask.get("content"), (row["episode_index"], row["frame_index"])
        for variant, recipe in recipe_objects.items():
            rendered = render_sample(
                recipe=recipe,
                persistent=row["language_persistent"],
                events=row["language_events"],
                t=row["timestamp"],
                sample_idx=row["index"],
                task=task,
            )
            assert rendered, (variant, row["index"])
            coverage[variant] += 1
            examples.setdefault(variant, rendered)

    # Private metadata copy: data/videos stay read-only symlinks; normalization uses TRAIN only.
    data = ROOT / "dataset"
    data.mkdir(exist_ok=True)
    if not (data / "meta").exists():
        shutil.copytree(source_data / "meta", data / "meta")
    for name in ("data", "videos"):
        if not (data / name).exists():
            (data / name).symlink_to(source_data / name, target_is_directory=True)
    stats = json.loads((source_data / "meta/stats.json").read_text())
    for key in ("action", "observation.state"):
        values = np.asarray(train_table[key].to_pylist(), dtype=np.float32)
        assert np.isfinite(values).all()
        stats[key] = {
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "min": values.min(0).tolist(),
            "max": values.max(0).tolist(),
            "count": [len(values)],
            **{
                f"q{int(q * 100):02d}": np.quantile(values, q, axis=0).tolist()
                for q in (0.01, 0.10, 0.50, 0.90, 0.99)
            },
        }
    dump(data / "meta/stats.json", stats)
    dump(ROOT / "split.json", {"train": TRAIN, "dev": DEV, "test": TEST})

    # Preserve the pretrained FAST vocabulary and offset; changing IDs discards learned semantics.
    tokenizer_path = SOURCE / "midtrain/action_tokenizer"
    tokenizer = AutoProcessor.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    chunks = []
    for episode in TRAIN:
        episode_table = train_table.filter(pc.equal(train_table["episode_index"], episode))
        actions = np.asarray(episode_table["action"].to_pylist(), dtype=np.float32)
        for start in np.linspace(0, len(actions) - 1, 8, dtype=int):
            chunks.append(actions[np.minimum(np.arange(start, start + 50), len(actions) - 1)])
    normalized = _normalize_actions(np.stack(chunks), "QUANTILES", stats["action"])
    fast_report, _ = _validate_fast_reconstruction(tokenizer, normalized, 0.10, 0.20)
    lengths = [len(ids) for ids in tokenizer(normalized)]
    # Worst case is one code per DCT coefficient. Budget never silently truncates any chunk.
    max_action_tokens = 50 * 14 + 16
    assert max(lengths) + 16 <= max_action_tokens
    fast_report.update(
        max_codes=max(lengths),
        max_action_tokens=max_action_tokens,
        tokenizer_source=str(tokenizer_path),
        fast_skip_tokens=128,
    )
    print("FAST audit:", json.dumps(fast_report), flush=True)

    template = json.loads((SOURCE / "restarts/3298e15a9/init_subtask/config.json").read_text())
    for variant in recipes:
        destination = ROOT / f"init_{variant}"
        destination.mkdir(exist_ok=True)
        cfg = dict(template)
        cfg.update(
            pretrained_path=str(destination),
            device="cpu",
            recipe_path=None,
            recipe=recipes[variant],
            text_loss_weight=1.0 if variant == "subtask" else 0.0,
            knowledge_insulation=True,
            enable_fast_action_loss=True,
            action_tokenizer_name=str(tokenizer_path),
            auto_fit_fast_tokenizer=False,
            max_action_tokens=max_action_tokens,
            fast_skip_tokens=128,
            action_feature_names=info["features"]["action"]["names"],
            scheduler_warmup_steps=1000,
            scheduler_decay_steps=10000,
            scheduler_decay_lr=5e-6,
            use_compiled_text_ce=False,
        )
        dump(destination / "config.json", cfg)
        if not (destination / "model.safetensors").exists():
            (destination / "model.safetensors").symlink_to(weights)
    dump(
        ROOT / "prepared.json",
        {
            "source": provenance["source"],
            "source_sha256": sha.hexdigest(),
            "source_etag_verified": etag,
            "code_revision": "3298e15a9e897b09b582b4ec38786ecdbedb1630",
            "dataset": "pepijn223/rebot_diverse_picking_100_annotated",
            "dataset_revision": REVISION,
            "train_frames": train_table.num_rows,
            "train_episodes": TRAIN,
            "dev_episodes": DEV,
            "test_episodes": TEST,
            "normalization": "training episodes only",
            "coverage": dict(coverage),
            "task": task,
            "rendered_examples": examples,
            "fast_audit": fast_report,
            "batch_per_gpu": 16,
            "gpus": 4,
            "global_batch": 64,
            "optimizer_steps_per_variant": 10000,
            "paper": "FineART___Arxiv.pdf, sections 4.1, 4.3, Appendix A/Table 5",
            "adaptations": [
                "ReBot features/stats",
                "10000-step transfer instead of paper's 5000",
                "1000-step warmup for transfer; decay to 5e-6",
                "4x16 instead of 8x8; same global 64",
                "reserved dev/test episodes",
                "full trajectories retained, including pauses",
                "uncompiled stable PR attention",
            ],
        },
    )
    print("PREPARATION PASSED", flush=True)


if __name__ == "__main__":
    main()
