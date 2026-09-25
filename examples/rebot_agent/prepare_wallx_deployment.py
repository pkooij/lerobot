# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Prepare a local WALL-X checkpoint and check processors without importing robot drivers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_checkpoint(source: Path, assets: Path, output: Path, expected_sha256: str) -> dict:
    source, assets, output = source.resolve(), assets.resolve(), output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Use a fresh runtime directory: {output}")
    if output.resolve().is_relative_to(source) or output.resolve().is_relative_to(assets):
        raise ValueError("Runtime output must be outside the source checkpoint and base assets")
    if not source.is_dir() or not assets.is_dir():
        raise ValueError("Checkpoint and local base assets must already exist")
    required = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]
    if any(not (source / name).is_file() for name in required):
        raise ValueError("Checkpoint lacks weights or saved processor configuration")
    if any(
        not (assets / name).is_file()
        for name in ["config.json", "tokenizer_config.json", "preprocessor_config.json"]
    ):
        raise ValueError("Local base configuration, tokenizer, and image processor assets are required")
    original = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    if original["model.safetensors"] != expected_sha256:
        raise ValueError("Model checksum does not match the selected checkpoint")
    config = json.loads((source / "config.json").read_text())
    preprocessor = json.loads((source / "policy_preprocessor.json").read_text())
    if config.get("type") != "wall_x":
        raise ValueError("This deployment helper supports WALL-X checkpoints only")
    tokenizers = [s for s in preprocessor["steps"] if s.get("registry_name") == "wall_x_tokenizer"]
    if len(tokenizers) != 1:
        raise ValueError("Expected exactly one saved WALL-X tokenizer step")
    tokenizer = tokenizers[0]["config"]
    if tokenizer["processor_name"] != config["pretrained_name_or_path"]:
        raise ValueError("Policy and saved tokenizer refer to different base assets")
    if tokenizer.get("steering_coordinate_format", "original_pixels") != config.get(
        "steering_coordinate_format", "original_pixels"
    ):
        raise ValueError("Policy and saved tokenizer disagree on the trained coordinate format")
    source_base = {"name": config["pretrained_name_or_path"], "revision": config.get("base_model_revision")}
    if tokenizer.get("processor_revision") != source_base["revision"]:
        raise ValueError("Policy and saved tokenizer refer to different base revisions")
    # Only location metadata changes. The trained coordinate format and recipes remain intact.
    config["pretrained_name_or_path"] = str(assets)
    tokenizer.update(processor_name=str(assets), processor_revision=None)
    assets_hashes = {str(p.relative_to(assets)): digest(p) for p in assets.rglob("*") if p.is_file()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wallx-deployment-", dir=output.parent) as temporary:
        stage = Path(temporary) / "checkpoint"
        stage.mkdir()
        for name in original:
            path, target = source / name, stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".safetensors":
                target.symlink_to(path)
            else:
                shutil.copy2(path, target)
        (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (stage / "policy_preprocessor.json").write_text(json.dumps(preprocessor, indent=2) + "\n")
        if {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()} != original:
            raise RuntimeError("Source checkpoint changed during preparation")
        receipt = {
            "source": str(source),
            "runtime_checkpoint": str(output),
            "base_assets": str(assets),
            "source_base": source_base,
            "model_sha256": expected_sha256,
            "source_files_sha256": original,
            "base_assets_sha256": assets_hashes,
            "runtime_metadata_sha256": {
                name: digest(stage / name) for name in ["config.json", "policy_preprocessor.json"]
            },
            "source_unchanged": True,
            "weights_and_normalizers_linked": True,
            "robot_commands": 0,
            "policy_weights_loaded": False,
            "processor_validation": "not_run",
        }
        (stage / "deployment_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Runtime directory appeared during preparation: {output}")
        stage.rename(output)
    return receipt


def validate_processors(checkpoint: Path, observations: list[Path], tasks: list[str]) -> dict:
    # Set before importing Transformers/HF; no robot or model constructors are used.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.wall_x.configuration_wall_x import WallXConfig

    config = WallXConfig.from_pretrained(checkpoint)
    config.device = "cpu"
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    rows = []
    for observation in observations:
        with np.load(observation, allow_pickle=False) as arrays:
            sample = {}
            for key, feature in config.input_features.items():
                array = arrays[key]
                if list(array.shape) != list(feature.shape) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid recorded observation feature {key}")
                tensor = torch.from_numpy(array.copy())
                if feature.type.value == "VISUAL":
                    if array.dtype != np.uint8:
                        raise ValueError("Recorded images must be uint8 CHW arrays")
                    tensor = tensor.float() / 255
                sample[key] = tensor
        for task in tasks:
            pre.reset()
            processed = pre({**sample, "task": task})
            if not all(torch.isfinite(v).all() for v in processed.values() if isinstance(v, torch.Tensor)):
                raise ValueError("Processor produced nonfinite input")
            rows.append(
                {
                    "observation": str(observation),
                    "sha256": digest(observation),
                    "task": task,
                    "tensor_shapes": {
                        key: list(value.shape)
                        for key, value in processed.items()
                        if isinstance(value, torch.Tensor)
                    },
                }
            )
    action_dim = config.output_features["action"].shape[0]
    action = post(torch.zeros((1, action_dim)))
    if action.shape != (1, action_dim) or not torch.isfinite(action).all():
        raise ValueError("Invalid postprocessed action shape or values")
    return {
        "status": "passed_offline_processor_restore",
        "observations": rows,
        "action_dim": action_dim,
        "coordinate_format": config.steering_coordinate_format,
        "robot_commands": 0,
        "policy_weights_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--observation", type=Path, action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    args = parser.parse_args()
    if bool(args.observation) != bool(args.task):
        parser.error("Supply both --observation and --task for offline processor validation")
    receipt = prepare_checkpoint(args.checkpoint, args.base_assets, args.output, args.expected_sha256)
    if args.observation:
        validation = validate_processors(args.output, args.observation, args.task)
        (args.output / "processor_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        receipt["processor_validation"] = validation["status"]
        (args.output / "deployment_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
