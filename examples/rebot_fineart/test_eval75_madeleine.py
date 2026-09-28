import json
import sys
from types import SimpleNamespace

import pytest
from eval75_madeleine import campaign_config, main


def test_campaign_preserves_object_goal_and_rejects_changed_settings(tmp_path, monkeypatch):
    answers = iter(["blue cube", "red cube", "cap", "marker", "tool"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    path = tmp_path / "campaign.json"
    config = campaign_config(path, {"subtask": "model"}, 120)
    assert config["goal"] == "Place blue cube, red cube, cap, marker and tool into the black bin."
    assert campaign_config(path, config["models"], 120) == config
    with pytest.raises(ValueError, match="different models/duration"):
        campaign_config(path, config["models"], 240)


@pytest.mark.parametrize(
    "condition,variant",
    [
        ("task_only_direct", "task_only"),
        ("subtask_direct", "subtask"),
        ("subtask_autosteer", "subtask"),
    ],
)
def test_single_scene_selects_pinned_model_and_resumes_without_motion(
    tmp_path, monkeypatch, condition, variant
):
    import eval75_madeleine as launcher

    monkeypatch.setattr(launcher.Path, "home", lambda: tmp_path)
    hardware = tmp_path / "hardware.json"
    hardware.write_text("{}")
    models = {
        name: {"repo_id": f"owner/{name}", "revision": "a" * 40, "hub_integrity_checked": True}
        for name in ("subtask", "task_only")
    }
    model_path = tmp_path / "models.json"
    model_path.write_text(json.dumps(models))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--condition",
            condition,
            "--trial",
            "3",
            "--models",
            str(model_path),
            "--hardware",
            str(hardware),
        ],
    )
    answers = iter(["one", "two", "three", "four", "five", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    downloads = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda repo, revision: downloads.append((repo, revision))),
    )
    plans = []
    monkeypatch.setattr(launcher, "remote", lambda plan: plans.append(plan) or 0)
    monkeypatch.setattr(
        launcher, "score", lambda: {"success": False, "objects_in_bin": 2, "intervention": False}
    )
    assert main() == 0
    assert len(plans) == 1 and plans[0]["trial"] == 3
    assert plans[0]["model"] == models[variant]
    assert downloads[0] == (models[variant]["repo_id"], "a" * 40)
    assert main() == 0
    assert len(plans) == 1
