# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Filesystem invariants for preparing a robot-free WALL-X deployment copy."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest


@pytest.fixture
def deployment(tmp_path):
    module = runpy.run_path(
        str(Path(__file__).parents[1] / "examples/rebot_agent/prepare_wallx_deployment.py")
    )
    source, assets, output = (tmp_path / name for name in ["source", "assets", "runtime"])
    source.mkdir()
    assets.mkdir()
    config = {
        "type": "wall_x",
        "pretrained_name_or_path": "test/base",
        "base_model_revision": "pinned",
        "steering_coordinate_format": "native_points_v1",
        "recipe": {"messages": []},
    }
    processor = {
        "steps": [
            {
                "registry_name": "wall_x_tokenizer",
                "config": {
                    "processor_name": "test/base",
                    "processor_revision": "pinned",
                    "steering_coordinate_format": "native_points_v1",
                },
            }
        ]
    }
    for name, value in [
        ("config.json", config),
        ("policy_preprocessor.json", processor),
        ("policy_postprocessor.json", {"steps": []}),
    ]:
        (source / name).write_text(json.dumps(value))
    (source / "model.safetensors").write_bytes(b"test-only-weight-bytes")
    (source / "normalizer.safetensors").write_bytes(b"test-only-normalizer-bytes")
    for name in ["config.json", "tokenizer_config.json", "preprocessor_config.json"]:
        (assets / name).write_text("{}")
    expected = hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest()
    return module["prepare_checkpoint"], source, assets, output, expected


def test_preparation_keeps_source_and_trained_format_intact(deployment):
    prepare, source, assets, output, expected = deployment
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    receipt = prepare(source, assets, output, expected)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original
    assert (output / "model.safetensors").is_symlink()
    assert (output / "normalizer.safetensors").is_symlink()
    assert (output / "model.safetensors").resolve() == source / "model.safetensors"
    assert (output / "policy_postprocessor.json").read_bytes() == original["policy_postprocessor.json"]
    config = json.loads((output / "config.json").read_text())
    processor = json.loads((output / "policy_preprocessor.json").read_text())
    assert config["pretrained_name_or_path"] == str(assets)
    assert config["steering_coordinate_format"] == "native_points_v1"
    assert config["recipe"] == {"messages": []}
    assert processor["steps"][0]["config"] == {
        "processor_name": str(assets),
        "processor_revision": None,
        "steering_coordinate_format": "native_points_v1",
    }
    assert receipt["model_sha256"] == expected and receipt["source_unchanged"]
    assert receipt["processor_validation"] == "not_run"
    with pytest.raises(FileExistsError):
        prepare(source, assets, output, expected)


def test_wrong_checkpoint_hash_and_nested_output_do_not_write(deployment):
    prepare, source, assets, output, expected = deployment
    with pytest.raises(ValueError, match="checksum"):
        prepare(source, assets, output, "0" * 64)
    assert not output.exists()
    for parent in [source, assets]:
        with pytest.raises(ValueError, match="outside"):
            prepare(source, assets, parent / "runtime", expected)
        assert not (parent / "runtime").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("processor_name", "other/base"),
        ("processor_revision", "other"),
        ("steering_coordinate_format", "original_pixels"),
    ],
)
def test_mismatched_processor_is_rejected_before_writing(deployment, key, value):
    prepare, source, assets, output, expected = deployment
    path = source / "policy_preprocessor.json"
    processor = json.loads(path.read_text())
    processor["steps"][0]["config"][key] = value
    path.write_text(json.dumps(processor))
    with pytest.raises(ValueError, match="different|disagree"):
        prepare(source, assets, output, expected)
    assert not output.exists()
