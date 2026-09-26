"""Publish the two requested final checkpoints, preserving weights and processor files."""

import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path("/fsx/pepijn/rebot-fineart-20260925")
CODE = "3298e15a9e897b09b582b4ec38786ecdbedb1630"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    api = HfApi()
    identity = api.whoami()["name"]
    assert identity == "pepijn223", f"Unexpected Hub account: {identity}"
    completion = json.loads((ROOT / "training_completion_86031.json").read_text())
    report = {}
    for variant in ("subtask", "task_only"):
        slug = variant.replace("_", "-")
        repo = f"pepijn223/rebot-fineart-pi052-{slug}-10k-20260925"
        source = ROOT / f"runs/{variant}_full_86031/checkpoints/010000/pretrained_model"
        stage = ROOT / "hub_exports" / slug
        state = json.loads((source.parent / "training_state/training_step.json").read_text())
        assert state["step"] == 10000 and state["batch_size"] == 16 and state["dp_world_size"] == 4
        stage.mkdir(parents=True, exist_ok=True)
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            dest = stage / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if path.name == "model.safetensors":
                if not dest.exists():
                    os.link(path, dest)
                assert os.path.samefile(path, dest)
            else:
                shutil.copy2(path, dest)
        weight_sha = digest(source / "model.safetensors")
        provenance = {
            "variant": variant,
            "model_sha256": weight_sha,
            "training_state": state,
            "training_code": CODE,
            "base": "jade/pi052_subgoal_3cam_14d_balanced_taskfix2/checkpoints/300000",
            "base_sha256": "9e384c7351e5e42008efa34c5e3485bf071ca0d5c00490aa6bb63167022dd197",
            "dataset": "pepijn223/rebot_diverse_picking_100_annotated",
            "dataset_revision": "93c97807c46535745d0587d4296416bf2d4aa80d",
            "validation": completion["variants"][variant],
            "inference_validated": False,
            "physical_success_measured": False,
            "packaging": "Original saved model/config/processors/tokenizer, plus documentation and provenance",
        }
        (stage / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        for name in (
            "split.json",
            "repair-audit.json",
            "processor-audit.json",
            "training_completion_86031.json",
        ):
            shutil.copy2(ROOT / name, stage / name)
        mode = (
            "30% task-to-subtask text prediction; 70% semantic-subtask-to-action training. "
            "At inference, request a next_subtask and feed the generated instruction back to action inference."
            if variant == "subtask"
            else "100% high-level-task-to-action training; text loss disabled. "
            "This variant has no trained subtask-generation objective; do not use native /autosteer."
        )
        warning = (
            "All 500 logged gradient windows were finite; actual checkpoint reload/inference remains unverified."
            if variant == "subtask"
            else "67 of 500 logged gradient windows were nonfinite. The post-training finite-gradient check FAILED. "
            "Scalar losses remained finite, but numerical reliability and action quality are UNKNOWN."
        )
        # Markdown text only; this string is never used as a database query.
        card = f"""---
library_name: lerobot
tags:
- robotics
- pi052
- fineart
- rebot
datasets:
- pepijn223/rebot_diverse_picking_100_annotated
---
# ReBot FineART / PI052 — {slug}, 10,000 updates

Experimental checkpoint uploaded at the owner's request. **Not deployment-validated.**

{warning}

## What it is

{mode}

Three current RGB camera views and 14 joint/gripper state values produce chunks
of 50 absolute 14-dimensional joint/gripper actions. Internal padding is 32 dimensions.
Camera inputs: cam_high, cam_left_wrist, cam_right_wrist. Images are resized to 224x224.
No visual/proprioceptive memory. Flow inference uses 10 denoising steps.
This experiment does not add pointing, bounding-box, FK-direction, or gripper-trace supervision.

## Training

Both variants start independently from the same original Jade 300k midtraining checkpoint,
with fresh optimizers, not from earlier ReBot fine-tunes. Training code is LeRobot PR 4184
revision `{CODE}` plus the recorded experiment preparation.

10,000 updates each; 4 H100 GPUs; batch 16 per GPU; global batch 64; accumulation 1.
85 training episodes / 117,379 frames; 5 development episodes; final 10 episodes untouched.
Normalization uses training frames only. See split.json for exact episode identities.
The source's incorrect constant global task is overridden by episode goals derived
from existing annotations; see repair-audit.json. Full trajectories, including pauses, remain.

Full fine-tuning, BF16 with native FP32 components and gradient checkpointing.
AdamW LR 2.5e-5, betas (0.9, 0.95), eps 1e-8, weight decay 1e-10; gradient clipping 1.
Warmup 1,000 updates, cosine decay to 5e-6 at update 10,000.
Flow loss weight 10; FAST loss weight 1; text loss weight {1 if variant == "subtask" else 0}.
Knowledge insulation enabled. FAST tokenizer repair appends two missing byte symbols,
preserving original IDs/merges; tokenizer files are bundled. Text budget 256, action-token budget 128.

## Loading and limitations

Use the PR 4184 PI052 implementation and **saved processors** with
`make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint)`.
Download the full repository, including action_tokenizer/ and normalizer tensors.
The original config/train_config retain cluster training paths for provenance;
do not regenerate processors from those paths. Set policy.path to this repository
or its downloaded snapshot for deployment. The PaliGemma text tokenizer remains
an external dependency: google/paligemma-3b-pt-224 requires appropriate access/cache.

Native logged eval is a mixed teacher-forced scalar over the first 256 frames of
one development episode, due to its single task_index grouping. It is not a balanced
development metric or physical success rate. No checkpoint has been selected by
balanced evaluation. No real-world trial of this checkpoint has been measured.
Check numerical inference and action continuity before physical deployment.

Weight SHA256: `{weight_sha}`. No optimizer state is uploaded.
"""  # nosec B608
        (stage / "README.md").write_text(card)
        if api.repo_exists(repo):
            previous = json.loads(Path(hf_hub_download(repo, "provenance.json")).read_text())
            assert previous["model_sha256"] == weight_sha, "Refusing to overwrite different model weights"
        api.create_repo(repo, private=True, exist_ok=True)
        print(f"Uploading {repo}", flush=True)
        commit = api.upload_folder(
            repo_id=repo,
            folder_path=stage,
            commit_message="Upload final 10000-step ReBot checkpoint with explicit validation status",
        )
        info = api.model_info(repo, revision=commit.oid, files_metadata=True)
        remote = next(s for s in info.siblings if s.rfilename == "model.safetensors")
        assert (
            remote.lfs.sha256 == weight_sha and remote.size == (source / "model.safetensors").stat().st_size
        )
        for name in (
            "config.json",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "provenance.json",
        ):
            downloaded = Path(hf_hub_download(repo, name, revision=commit.oid))
            assert digest(downloaded) == digest(stage / name)
        report[variant] = {
            "repo_id": repo,
            "revision": commit.oid,
            "url": f"https://huggingface.co/{repo}",
            "weight_sha256": weight_sha,
            "private": info.private,
            "hub_integrity_checked": True,
            "inference_validated": False,
        }
        (ROOT / "uploaded_86031.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report[variant]), flush=True)


if __name__ == "__main__":
    main()
