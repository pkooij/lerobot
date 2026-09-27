"""Upload the user-selected final20k models and verify their Hub contents."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path("/fsx/pepijn/rebot-fineart-augmented-20260926")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true", help="Requires explicit user authorization")
    args = parser.parse_args()
    api = HfApi()
    assert api.whoami()["name"] == "pepijn223"
    results = {}
    summary = json.loads((ROOT / "comparison/final_verified_summary.json").read_text())
    for variant in ("subtask", "task_only"):
        slug = variant.replace("_", "-")
        repo = f"pepijn223/rebot-fineart-pi052-{slug}-augmented-20k-20260926"
        source = ROOT / f"runs/{variant}_full_86277/checkpoints/020000/pretrained_model"
        state = json.loads((source.parent / "training_state/training_step.json").read_text())
        assert state["step"] == 20000 and state["batch_size"] == 16 and state["dp_world_size"] == 4
        assert json.loads((ROOT / f"verified_full_{variant}_86277.json").read_text())[variant]["passed"]
        stage = ROOT / "hub_exports_20k" / variant
        stage.mkdir(parents=True, exist_ok=True)
        for src in source.rglob("*"):
            if not src.is_file():
                continue
            dest = stage / src.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.name == "model.safetensors":
                if not dest.exists():
                    os.link(src, dest)
                assert os.path.samefile(src, dest)
            else:
                shutil.copy2(src, dest)
        sha = digest(source / "model.safetensors")
        expected = summary["evaluation"]["test"]["checkpoints"][f"new_{variant}_20000"]["weight_sha256"]
        assert sha == expected
        provenance = {
            "variant": variant,
            "model_sha256": sha,
            "training_state": state,
            "training_code": "9ede7f09a7ba6c0cda7ef9333fc08c88c409e73b",
            "dataset": "pepijn223/rebot_diverse_picking_100_annotated",
            "dataset_revision": "93c97807c46535745d0587d4296416bf2d4aa80d",
            "base": "jade/pi052_subgoal_3cam_14d_balanced_taskfix2/checkpoints/300000",
            "base_sha256": "9e384c7351e5e42008efa34c5e3485bf071ca0d5c00490aa6bb63167022dd197",
            "physical_success_measured": False,
            "final_model_forward_evaluation_passed": True,
            "pretraining_inference_gate_passed": True,
            "packaging": "Unmodified saved weights, config, processors and tokenizer; documentation added",
        }
        (stage / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        for name in ("split.json", "prepared.json", "processor-audit.json", "task_augmentations.json"):
            shutil.copy2(ROOT / name, stage / name)
        shutil.copy2(ROOT / "comparison/final_verified_summary.json", stage / "evaluation_summary.json")
        mixture = (
            "20% goal-to-subtask text; 50% subtask-to-actions; 30% goal-to-actions. "
            "Supports direct goal actions and native subtask generation (/autosteer)."
            if variant == "subtask"
            else "100% high-level-goal-to-actions. No trained subtask-generation objective; do not use native autosteer."
        )
        card = f"""---
library_name: lerobot
tags: [robotics, pi052, fineart, rebot]
datasets: [pepijn223/rebot_diverse_picking_100_annotated]
---
# ReBot FineART / PI052 — {slug}, augmented20k

Experimental checkpoint uploaded at the owner's request for physical evaluation.
{mixture}

## Training

Fresh original Jade step300000 midtraining weights and optimizer. 20,000 updates,
four H100s, batch16/GPU, global64, about10.90 frame epochs. 85 training episodes,
117,379 frames; five development episodes and ten reserved test episodes.
Six conservative goal phrasings per episode, native task_aug sampling. Full BF16
fine-tuning with native FP32 components; AdamW peak LR2.5e-5, warmup1000,
cosine decay to5e-6. Conditioning LR multiplier0.1, conditioning clipping0.1,
expert clipping1 and global clipping1; nonfinite gradients reject updates.
Flow loss weight10, FAST1, text{1 if variant == "subtask" else 0}; knowledge insulation enabled.

Three RGB views (cam_high, cam_left_wrist, cam_right_wrist), current14-dimensional
joint/gripper state, 50-step absolute14-dimensional action chunks, ten denoising
steps, internal padding32. No visual/proprioceptive history, RTC training,
pointing, bounding-box or FK supervision in this experiment.

## Validation and limitations

Training and final-checkpoint forward evaluations passed with finite values.
Gradient spikes persisted despite clipping. The initialization models passed actual
GPU action-sampling checks before training; this does not establish physical success
of the final checkpoints. Paired evaluations cover500 dev frames and1000 reserved-test
frames with fixed observations, prompts and noise. See evaluation_summary.json.
Development action losses favored the saved5k models over20k; longer training showed
overfitting. These20k finals are uploaded because the user explicitly selected them.
No real-world success rate has been measured for these weights.

## Loading

Use the pinned experiment branch codex/rebot-pi052-matched-sft-20260925,
at the training revision recorded in provenance.json or a compatible descendant. Load the complete repository and
saved processors (make_pre_post_processors with pretrained_path), including bundled
action_tokenizer files and normalization tensors. The original saved configs retain
cluster paths for provenance; do not regenerate processors from those paths.
The external google/paligemma-3b-pt-224 tokenizer needs access/cache.
Optimizer state is not uploaded. Weight SHA256: {sha}.
"""
        (stage / "README.md").write_text(card)
        if api.repo_exists(repo):
            existing = api.model_info(repo, files_metadata=True)
            for file in existing.siblings:
                if file.rfilename == "model.safetensors":
                    assert file.lfs.sha256 == sha, "Refusing to overwrite different weights"
        api.create_repo(repo, private=not args.public, exist_ok=True)
        if args.public:
            api.update_repo_settings(repo, private=False)
        print(f"Uploading {repo}", flush=True)
        commit = api.upload_folder(
            repo_id=repo, folder_path=stage, commit_message="Publish verified augmented20k final checkpoint"
        )
        info = api.model_info(repo, revision=commit.oid, files_metadata=True)
        weight = next(f for f in info.siblings if f.rfilename == "model.safetensors")
        assert weight.lfs.sha256 == sha and weight.size == (source / "model.safetensors").stat().st_size
        for local in stage.rglob("*"):
            if local.is_file() and local.name != "model.safetensors":
                remote = Path(hf_hub_download(repo, str(local.relative_to(stage)), revision=commit.oid))
                assert digest(local) == digest(remote)
        results[variant] = {
            "repo_id": repo,
            "revision": commit.oid,
            "private": info.private,
            "weight_sha256": sha,
            "hub_integrity_checked": True,
        }
        (ROOT / "uploaded_20k.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results[variant]), flush=True)


if __name__ == "__main__":
    main()
