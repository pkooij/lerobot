# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import json
import runpy
import threading
from pathlib import Path
from unittest.mock import Mock

import draccus
import numpy as np
import pytest

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.language_task import task_from_recipe
from lerobot.datasets.recipe import TrainingRecipe
from lerobot.policies.wall_x.configuration_wall_x import WallXConfig  # noqa: F401
from lerobot.rollout.inference.sync import SyncInferenceEngine
from lerobot.rollout.planner import PlannerConfig, VisionLanguagePlanner
from tests.test_interactive_rollout import _FakeEngine


@pytest.mark.parametrize("credential", [None, "", " \t\n"])
def test_missing_planner_credential_fails_before_policy_or_hardware(monkeypatch, credential):
    import lerobot.rollout.context as rollout_context
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.rollout import RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig

    key_name = "LEROBOT_TEST_PLANNER_KEY"
    if credential is None:
        monkeypatch.delenv(key_name, raising=False)
    else:
        monkeypatch.setenv(key_name, credential)
    # A default-endpoint key must not satisfy a separately configured credential.
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-default")
    cfg = RolloutConfig(
        robot=MockRobotConfig(),
        policy=ACTConfig(device="cpu"),
        device="cpu",
        interactive=True,
        planner=PlannerConfig(enabled=True, api_key_env=key_name),
    )
    load_policy, make_robot, post = Mock(), Mock(), Mock()
    monkeypatch.setattr(rollout_context, "_load_pretrained_policy", load_policy)
    monkeypatch.setattr(rollout_context, "make_robot_from_config", make_robot)
    monkeypatch.setattr("lerobot.rollout.planner.requests.post", post)
    with pytest.raises(ValueError, match=f"Set {key_name}"):
        rollout_context.build_rollout_context(cfg, threading.Event())
    load_policy.assert_not_called()
    make_robot.assert_not_called()
    post.assert_not_called()


def test_external_planning_reuses_task_switch_and_holds_on_failure():
    engine = _FakeEngine()
    engine.supports_text_queries = False
    planner = Mock(return_value="reach for the tape")
    engine.set_language_planner(planner)
    assert engine.supports_planning and not engine.supports_text_queries
    assert engine.planner_halted
    engine.start_autosteer("put tape in bin", 0)
    assert engine.pump_query({"base": "image"})
    assert engine.task == "reach for the tape"
    assert engine._take_task()[1]
    assert not engine.planner_halted
    engine.pump_query({"base": "later image"})
    assert engine._take_task()[1]  # identical command still flushes pre-API queued actions
    planner.side_effect = TimeoutError("no answer")
    engine.pump_query({"base": "next image"})
    assert engine.planner_halted
    assert engine.autosteer_goal is None
    # A real sync engine must not select an action while the planner is halted.
    assert SyncInferenceEngine.get_action(engine, {"observation.state": object()}) is None
    engine.set_task("open the left gripper")
    assert not engine.planner_halted


def test_manual_language_override_discards_in_flight_plan():
    engine = _FakeEngine()

    def planner(*args):
        engine.stop_autosteer()
        engine.set_task("operator instruction")
        return "stale plan"

    engine.set_language_planner(planner)
    engine.start_autosteer("goal", 0)
    engine.pump_query({"base": "image"})
    assert engine.task == "operator instruction"
    assert not engine.planner_halted


def test_planner_audit_times_returns_holds_and_failures_without_claiming_execution(monkeypatch, tmp_path):
    decision = {
        "command": "reach for tape",
        "camera": None,
        "points": [],
        "point_mode": None,
        "style": "subtask",
        "assessment": "tape visible",
        "status": "continue",
    }
    post = Mock(
        side_effect=lambda *a, **kw: Mock(
            json=lambda: {
                "status": "completed",
                "id": "test-response",
                "output": [{"content": [{"type": "output_text", "text": json.dumps(decision)}]}],
            }
        )
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    monkeypatch.setattr("lerobot.rollout.planner.requests.post", post)
    log = tmp_path / "planner.jsonl"
    planner = VisionLanguagePlanner(PlannerConfig(camera_keys=["base"], log_path=str(log)))
    obs = {"base": np.zeros((48, 64, 3), dtype=np.uint8)}
    planner(obs, "goal", 1)
    decision["status"] = "uncertain"
    with pytest.raises(ValueError, match="Planner stopped"):
        planner(obs, "goal", 1)
    post.side_effect = TimeoutError("test-only-secret must not enter the journal")
    with pytest.raises(TimeoutError):
        planner(obs, "goal", 1)
    events = [json.loads(line) for line in log.read_text().splitlines()]
    terminal = [e for e in events if "elapsed_s" in e]
    assert [e["event"] for e in terminal] == ["planner_returned", "planner_hold", "planner_error"]
    assert all(e["elapsed_s"] >= 0 and e["started_at"] <= e["ended_at"] for e in terminal)
    assert len({e["request_id"] for e in terminal}) == 3
    assert terminal[0]["execution_verified"] is False
    assert "test-only-secret" not in log.read_text()


def test_planner_sends_named_images_and_bounded_history_without_action_tools(monkeypatch):
    calls = []
    decision = {
        "command": "reach for tape",
        "camera": None,
        "points": [],
        "point_mode": None,
        "style": "subtask",
        "assessment": "tape visible",
        "status": "continue",
    }

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return Mock(
            json=lambda: {
                "status": "completed",
                "output": [{"content": [{"type": "output_text", "text": json.dumps(decision)}]}],
            }
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr("lerobot.rollout.planner.requests.post", post)
    planner = VisionLanguagePlanner(PlannerConfig(camera_keys=["base"], history_turns=1))
    obs = {"base": np.zeros((48, 64, 3), dtype=np.uint8)}
    for _ in range(3):
        assert planner(obs, "goal", 1) == "reach for tape"
    assert [len(c["input"]) for c in calls] == [1, 3, 3]
    assert "tools" not in calls[0]
    assert calls[0]["store"] is False
    assert "64x48" in calls[0]["input"][0]["content"][1]["text"]
    planner(obs, "goal", 2)
    assert len(calls[-1]["input"]) == 1
    decision["status"] = "complete"
    with pytest.raises(ValueError, match="Planner stopped"):
        planner(obs, "goal", 2)
    planner.config.styles.append("point")
    planner.config.grounding_camera_keys.append("base")
    decision.update(status="continue", style="point", camera="base", points=[[32, 24]], point_mode="targets")
    assert planner(obs, "goal", 2) == "In base view (64x48 pixels), reach for tape: [32, 24]."
    decision["points"] = [[64, 24]]
    with pytest.raises(ValueError, match="outside"):
        planner(obs, "goal", 2)
    planner.config.styles.append("combination")
    decision.update(style="combination", points=[[32, 24], [33, 25]], point_mode="path")
    with pytest.raises(ValueError, match="trace steering"):
        planner(obs, "goal", 2)


def test_chat_planner_preserves_images_schema_history_and_grounding(monkeypatch, tmp_path):
    decision = {
        "command": "pick at the first point and place at the second",
        "camera": "base",
        "points": [[12, 20], [40, 10]],
        "point_mode": "targets",
        "style": "point",
        "assessment": "object and bin visible",
        "status": "continue",
    }
    post = Mock(
        side_effect=lambda *a, **kw: Mock(
            json=lambda: {
                "id": "hf-response",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(decision), "reasoning_content": "not a command"},
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("HF_TOKEN", "test-only-hf-secret")
    monkeypatch.setattr("lerobot.rollout.planner.requests.post", post)
    cfg = PlannerConfig(
        api_format="chat_completions",
        api_base="https://router.huggingface.co/v1",
        api_key_env="HF_TOKEN",
        model="Qwen/Qwen3.8-Flash-Next:featherless-ai",
        reasoning_effort="low",
        enable_thinking=False,
        camera_keys=["base", "left_wrist"],
        grounding_camera_keys=["base"],
        target_point_count=2,
        styles=["point"],
        history_turns=1,
        log_path=str(tmp_path / "planner.jsonl"),
    )
    cfg = draccus.decode(PlannerConfig, draccus.encode(cfg))
    planner = VisionLanguagePlanner(cfg)
    obs = {key: np.zeros((48, 64, 3), dtype=np.uint8) for key in cfg.camera_keys}
    for _ in range(3):
        assert planner(obs, "put object in bin", 1).endswith("[12, 20], [40, 10].")
    bodies = [call.kwargs["json"] for call in post.call_args_list]
    assert [len(body["messages"]) for body in bodies] == [2, 4, 4]
    first = post.call_args_list[0]
    assert first.args[0] == "https://router.huggingface.co/v1/chat/completions"
    assert first.kwargs["headers"] == {"Authorization": "Bearer test-only-hf-secret"}
    assert first.kwargs["timeout"] == cfg.timeout_s
    body = bodies[0]
    assert body["model"] == cfg.model and body["reasoning_effort"] == "low"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == cfg.max_output_tokens and body["stream"] is False
    assert "tools" not in body and "reasoning_content" not in json.dumps(bodies)
    assert body["messages"][0]["role"] == "system"
    parts = body["messages"][1]["content"]
    assert "Camera base: 64x48" in parts[1]["text"]
    assert "Camera left_wrist: 64x48" in parts[3]["text"]
    assert all(parts[i]["image_url"]["url"].startswith("data:image/jpeg;base64,") for i in [2, 4])
    schema = body["response_format"]["json_schema"]
    assert schema["strict"] and schema["schema"]["properties"]["camera"]["enum"] == ["base", None]
    assert "test-only-hf-secret" not in Path(cfg.log_path).read_text()
    planner(obs, "put object in bin", 2)
    assert len(post.call_args.kwargs["json"]["messages"]) == 2
    decision["camera"] = "left_wrist"
    with pytest.raises(ValueError, match="without trained coordinate grounding"):
        planner(obs, "put object in bin", 2)
    decision.update(camera="base", points=[[12, 20]])
    with pytest.raises(ValueError, match="Target point count"):
        planner(obs, "put object in bin", 2)


@pytest.mark.parametrize(
    "choice",
    [
        None,
        {"finish_reason": "length", "message": {"content": "{}"}},
        {"finish_reason": "content_filter", "message": {"content": "{}"}},
        {"finish_reason": "stop", "message": {"content": None}},
        {"finish_reason": "stop", "message": {"content": " \n"}},
        {"finish_reason": "stop", "message": {"content": "{}", "refusal": "cannot comply"}},
        {"finish_reason": "stop", "message": {"content": "{}", "tool_calls": [{"name": "move"}]}},
    ],
)
def test_incomplete_or_nontext_chat_plans_hold_action_production(monkeypatch, choice):
    monkeypatch.setenv("HF_TOKEN", "test-only")
    monkeypatch.setattr(
        "lerobot.rollout.planner.requests.post",
        Mock(return_value=Mock(json=lambda: {"choices": [choice] if choice else []})),
    )
    planner = VisionLanguagePlanner(
        PlannerConfig(api_format="chat_completions", api_key_env="HF_TOKEN", camera_keys=["base"])
    )
    engine = _FakeEngine()
    engine.set_language_planner(planner)
    engine.start_autosteer("pick object", 0)
    engine.pump_query({"base": np.zeros((48, 64, 3), dtype=np.uint8)})
    assert engine.planner_halted and engine.autosteer_goal is None
    assert not planner._history
    assert SyncInferenceEngine.get_action(engine, {"observation.state": object()}) is None


def test_planner_rejects_unknown_transport_and_invalid_output_budget():
    with pytest.raises(ValueError, match="api_format"):
        PlannerConfig(api_format="unknown")
    with pytest.raises(ValueError, match="max_output_tokens"):
        PlannerConfig(max_output_tokens=0)
    with pytest.raises(ValueError, match="enable_thinking"):
        PlannerConfig(enable_thinking=False)
    for count in [True, 0, -1, 1.5]:
        with pytest.raises(ValueError, match="target_point_count"):
            PlannerConfig(target_point_count=count)


@pytest.mark.parametrize("api_format", ["responses", "chat_completions"])
def test_semantic_planner_commands_cannot_include_coordinates(monkeypatch, api_format):
    decision = {
        "style": "task",
        "command": "pick the block and put it in the bin",
        "camera": "base",
        "points": [[10, 20], [30, 10]],
        "point_mode": "targets",
        "assessment": "block and bin visible",
        "status": "continue",
    }
    text = json.dumps(decision)
    result = (
        {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
        if api_format == "chat_completions"
        else {"status": "completed", "output": [{"content": [{"type": "output_text", "text": text}]}]}
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr("lerobot.rollout.planner.requests.post", Mock(return_value=Mock(json=lambda: result)))
    planner = VisionLanguagePlanner(
        PlannerConfig(
            api_format=api_format,
            camera_keys=["base"],
            grounding_camera_keys=["base"],
            styles=["task", "point"],
            target_point_count=2,
        )
    )
    with pytest.raises(ValueError, match="Coordinate commands require a visual style"):
        planner({"base": np.zeros((48, 64, 3), dtype=np.uint8)}, "goal", 0)
    assert not planner._history


@pytest.mark.parametrize("coordinate_format", ["original_pixels", "native_points_v1"])
def test_four_gpu_training_config_uses_main_parser(tmp_path, coordinate_format):
    module = runpy.run_path(str(Path(__file__).parents[1] / "examples/rebot_agent/train_wall_oss_flow.py"))
    config, argv = module["prepare_run"](tmp_path, 4, 1, True, coordinate_format=coordinate_format)
    parsed = draccus.decode(TrainPipelineConfig, config)
    assert parsed.policy.type == "wall_x"
    assert parsed.policy.steering_coordinate_format == coordinate_format
    assert parsed.policy.base_model_revision == "44e827683819957d8c574e8b746a1a97e77f518a"
    assert parsed.policy.recipe["messages"][0]["stream"] == "low_level"
    assert parsed.steps == parsed.eval_steps == parsed.save_freq == 10
    assert parsed.max_eval_samples == 20
    assert "--nproc-per-node=4" in argv
    with pytest.raises(ValueError):
        module["prepare_run"](tmp_path, 5, 1, True)


def test_local_annotation_dataset_source_is_checked_before_training(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1] / "examples/rebot_agent/train_wall_oss_flow.py"))
    default, _ = module["prepare_run"](tmp_path / "run", 1, 1, True)
    root = tmp_path / "annotations"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}")
    source = {k: default["dataset"][k] for k in ("repo_id", "revision")}
    (root / "source.json").write_text(json.dumps(source))
    config, _ = module["prepare_run"](tmp_path / "run", 1, 1, True, dataset_root=root)
    assert Path(draccus.decode(TrainPipelineConfig, config).dataset.root) == root.resolve()
    assert config["dataset"]["revision"] == source["revision"]
    (root / "source.json").write_text(json.dumps({**source, "revision": "different-recording"}))
    with pytest.raises(ValueError, match="Local dataset source"):
        module["prepare_run"](tmp_path / "run", 1, 1, True, dataset_root=root)


def test_smoke_reload_requires_saved_checkpoint_and_preserves_topology(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1] / "examples/rebot_agent/train_wall_oss_flow.py"))
    config, argv = module["prepare_run"](tmp_path, 4, 1, True)
    with pytest.raises(FileNotFoundError, match="did not save"):
        module["reload_command"](argv, tmp_path, config["steps"])
    checkpoint = tmp_path / "training/checkpoints/last/pretrained_model/train_config.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps(config))
    reload_argv = module["reload_command"](argv, tmp_path, config["steps"])
    assert "--nproc-per-node=4" in reload_argv
    assert [arg for arg in reload_argv if arg.startswith("--config_path=")] == [f"--config_path={checkpoint}"]
    assert "--resume=true" in reload_argv
    assert "--steps=11" in reload_argv
    assert "--eval_steps=11" in reload_argv
    assert "--save_checkpoint=false" in reload_argv


def test_rebot_task_branch_corrects_source_task_in_both_training_conditions(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1] / "examples/rebot_agent/train_wall_oss_flow.py"))
    config, _ = module["prepare_run"](tmp_path, 1, 1, True)
    recipe = TrainingRecipe.from_dict(config["dataset"]["task_recipe"])
    sample = {
        "task": "Pick up all blocks on the table and place them into the green bin.",
        "timestamp": 0,
        "index": 0,
    }
    expected = "Pick up objects from the table and place them into the bin."
    assert task_from_recipe(sample, recipe.blend["high_level_task"])["task"] == expected
    # Empty coverage will fail in the dataset loader; it suffices to inspect launch conditioning here.
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source": {k: config["dataset"][k] for k in ("repo_id", "revision")},
                "segments": [],
            }
        )
    )
    rich, _ = module["prepare_run"](tmp_path, 1, 1, True, path)
    assert rich["dataset"]["steering_skip_uncovered"] is False
    partial, _ = module["prepare_run"](tmp_path, 1, 1, True, path, skip_uncovered=True)
    assert partial["dataset"]["steering_skip_uncovered"] is True
    with pytest.raises(ValueError, match="requires --steering-manifest"):
        module["prepare_run"](tmp_path, 1, 1, True, skip_uncovered=True)
    assert set(rich["dataset"]["steering_required_styles"]) == {
        "subtask",
        "motion",
        "point",
        "trace",
        "combination",
    }
    assert (
        task_from_recipe(sample, TrainingRecipe.from_dict(rich["dataset"]["task_recipe"]))["task"] == expected
    )


def test_grounding_camera_configuration_requires_explicit_observed_views():
    with pytest.raises(ValueError, match="explicit trained"):
        PlannerConfig(styles=["point"])
    with pytest.raises(ValueError, match="included"):
        PlannerConfig(camera_keys=["base"], grounding_camera_keys=["left_wrist"])


def test_motion_style_requires_the_checkpoint_command_vocabulary():
    with pytest.raises(ValueError, match="explicit trained motion_commands"):
        PlannerConfig(styles=["motion"])
    for commands in [[""], [" open the left gripper"], ["close", "close"], [None]]:
        with pytest.raises(ValueError, match="distinct nonempty trimmed"):
            PlannerConfig(styles=["motion"], motion_commands=commands)


def test_planner_rejects_untrained_motion_without_issuing_it(monkeypatch):
    calls = []
    config = PlannerConfig(camera_keys=["base"], styles=["motion"], motion_commands=["open the left gripper"])
    planner = VisionLanguagePlanner(config)
    decision = {
        "command": "move the left gripper upward",
        "style": "motion",
        "camera": None,
        "points": [],
        "point_mode": None,
        "assessment": "gripper visible",
        "status": "continue",
    }

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return Mock(
            json=lambda: {
                "status": "completed",
                "output": [{"content": [{"type": "output_text", "text": json.dumps(decision)}]}],
            }
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr("lerobot.rollout.planner.requests.post", post)
    observation = {"base": np.zeros((48, 64, 3), dtype=np.uint8)}
    with pytest.raises(ValueError, match="trained vocabulary"):
        planner(observation, "goal", 1)
    assert planner._history == []
    assert '"open the left gripper"' in calls[0]["instructions"]
    decision["command"] = "open the left gripper"
    assert planner(observation, "goal", 1) == "open the left gripper"
    assert len(planner._history) == 2


def test_planner_observes_all_views_but_limits_coordinate_commands(monkeypatch):
    config = PlannerConfig(
        camera_keys=["base", "left_wrist"], grounding_camera_keys=["base"], styles=["point", "combination"]
    )
    planner = VisionLanguagePlanner(config)
    observation = {key: np.zeros((48, 64, 3), dtype=np.uint8) for key in config.camera_keys}
    decision = {
        "command": "move the object at the first point to the second point",
        "style": "point",
        "camera": "base",
        "points": [[10, 20], [30, 40]],
        "point_mode": "targets",
        "assessment": "both targets visible",
        "status": "continue",
    }
    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return Mock(
            json=lambda: {
                "status": "completed",
                "output": [{"content": [{"type": "output_text", "text": json.dumps(decision)}]}],
            }
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr("lerobot.rollout.planner.requests.post", post)
    assert "[10, 20], [30, 40]" in planner(observation, "goal", 1)
    payload = requests[0]
    assert sum(c["type"] == "input_image" for c in payload["input"][0]["content"]) == 2
    assert payload["text"]["format"]["schema"]["properties"]["camera"]["enum"] == ["base", None]
    decision["camera"] = "left_wrist"
    with pytest.raises(ValueError, match="without trained coordinate"):
        planner(observation, "goal", 1)
    decision.update(camera="base", style="combination", point_mode="path")
    with pytest.raises(ValueError, match="trace steering"):
        planner(observation, "goal", 1)
    assert len(planner._history) == 2  # Rejected proposals never become issued-command history.


@pytest.mark.parametrize("api_format", ["responses", "chat_completions"])
@pytest.mark.parametrize("channels_first", [False, True])
def test_normalized_planner_points_use_selected_camera_and_preserve_history(
    monkeypatch, tmp_path, api_format, channels_first
):
    decision = {
        "style": "point",
        "command": "pick at the first point and place at the second point",
        "camera": "base",
        "points": [[0, 1000], [500, 500]],
        "point_mode": "targets",
        "assessment": "two visible targets",
        "status": "continue",
    }
    wire_text = json.dumps(decision)
    result = (
        {"choices": [{"finish_reason": "stop", "message": {"content": wire_text}}]}
        if api_format == "chat_completions"
        else {"status": "completed", "output": [{"content": [{"type": "output_text", "text": wire_text}]}]}
    )
    post = Mock(return_value=Mock(json=lambda: result))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr("lerobot.rollout.planner.requests.post", post)
    log = tmp_path / "decisions.jsonl"
    planner = VisionLanguagePlanner(
        PlannerConfig(
            api_format=api_format,
            camera_keys=["wrist", "base"],
            grounding_camera_keys=["base"],
            styles=["point"],
            target_point_count=2,
            output_coordinate_format="normalized_1000",
            log_path=str(log),
        )
    )
    base = np.zeros((480, 640, 3), dtype=np.uint8)
    observation = {
        "base": base.transpose(2, 0, 1) if channels_first else base,
        "wrist": np.zeros((120, 160, 3), dtype=np.uint8),
    }
    command = planner(observation, "goal", 0)
    assert "base view (640x480 pixels)" in command
    assert "[0, 479], [320, 240]" in command
    assert planner._history[-1]["content"] == wire_text
    events = [json.loads(line) for line in log.read_text().splitlines()]
    proposal = next(event for event in events if event["event"] == "planner_proposal")
    returned = next(event for event in events if event["event"] == "planner_returned")
    assert proposal["points"] == [[0, 1000], [500, 500]]
    assert proposal["output_coordinate_format"] == "normalized_1000"
    assert returned["rendered_points"] == [[0, 479], [320, 240]]
    assert returned["rendered_image_size"] == [640, 480]
    payload = post.call_args.kwargs["json"]
    schema = (
        payload["response_format"]["json_schema"]["schema"]
        if api_format == "chat_completions"
        else payload["text"]["format"]["schema"]
    )
    assert schema["properties"]["points"]["items"]["items"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 1000,
    }
    instructions = (
        payload["messages"][0]["content"] if api_format == "chat_completions" else payload["instructions"]
    )
    assert "[1000,1000] is the bottom-right pixel center" in instructions


@pytest.mark.parametrize("bad_point", [[1001, 500], [-1, 500], [True, 500], [1.5, 500], [1], "500,500"])
def test_normalized_planner_rejects_invalid_coordinates_without_history(monkeypatch, bad_point):
    planner = VisionLanguagePlanner(
        PlannerConfig(
            camera_keys=["base"],
            grounding_camera_keys=["base"],
            styles=["point"],
            output_coordinate_format="normalized_1000",
            target_point_count=2,
        )
    )
    decision = {
        "style": "point",
        "command": "pick and place",
        "camera": "base",
        "points": [bad_point, [500, 500]],
        "point_mode": "targets",
        "assessment": "targets visible",
        "status": "continue",
    }
    monkeypatch.setattr(planner, "_request", lambda payload: (json.dumps(decision), None))
    with pytest.raises(ValueError, match="Normalized points must be integer pairs"):
        planner({"base": np.zeros((480, 640, 3), dtype=np.uint8)}, "goal", 0)
    assert planner._history == []


def test_planner_coordinate_format_is_explicit():
    assert PlannerConfig().output_coordinate_format == "original_pixels"
    with pytest.raises(ValueError, match="output_coordinate_format"):
        PlannerConfig(output_coordinate_format="auto")
