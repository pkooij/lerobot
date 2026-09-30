import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from examples.yam_experiments.campaign import CONDITIONS, Journal, make_manifest, summarize
from examples.yam_experiments.run import launcher_arguments, prepare_arguments
from examples.yam_experiments.session import ExperimentSession
from lerobot.rollout.controller import RolloutEvent
from lerobot.rollout.interactive import InteractiveCommand


def test_fifty_trials_are_matched_and_order_balanced():
    plan = make_manifest(["red", "blue", "yellow"])
    assert len(plan["trials"]) == len({t["id"] for t in plan["trials"]}) == 50
    for condition in CONDITIONS:
        trials = [t for t in plan["trials"] if t["condition"] == condition]
        assert [t["layout"] for t in trials] == list(range(1, 11))
        positions = [i % 5 for i, t in enumerate(plan["trials"]) if t["condition"] == condition]
        assert all(positions.count(p) == 2 for p in range(5))
    assert plan == make_manifest(["red", "blue", "yellow"])


def test_launcher_is_parsed_never_executed(tmp_path):
    launcher = tmp_path / "launch.sh"
    launcher.write_text(
        'cd /ignored\ngit switch ignored\nlerobot-rollout --robot.type=bi_yam_follower \\\n --task="a task"\n'
    )
    assert launcher_arguments(launcher) == ["--robot.type=bi_yam_follower", "--task=a task"]
    launcher.write_text("lerobot-rollout --robot.type=x ; touch /tmp/not-allowed")
    with pytest.raises(ValueError):
        launcher_arguments(launcher)


def test_run_keeps_hardware_flags_and_enforces_recording(tmp_path):
    argv = prepare_arguments(
        ["--robot.type=bi_yam_follower", "--robot.left_arm.gripper_kd=0.05"], [], tmp_path, "task", 0
    )
    assert "--robot.left_arm.gripper_kd=0.05" in argv
    assert "--duration=0" in argv
    assert "--robot.defer_torque_enable=true" in argv
    assert "--strategy.type=sentry" in argv
    assert "--dataset.push_to_hub=false" in argv
    assert "--dataset.repo_id=local/rollout_yam_experiment" in argv
    with pytest.raises(ValueError, match="credentials"):
        prepare_arguments([], ["--planner.api_key=example"], tmp_path, "task", 0)


def test_report_excludes_pilots_and_rejects_duplicate_trials(tmp_path):
    journal = Journal(tmp_path / "sessions" / "one" / "events.jsonl")
    journal.write("verdict", condition="human", pilot=True, trial_id="warmup", outcome="success")
    journal.write("verdict", condition="human", pilot=False, trial_id="human-L01", outcome="failure")
    assert summarize(tmp_path)["human"]["success_rate"] == 0
    assert summarize(tmp_path)["astra"]["success_rate"] is None
    journal.write("verdict", condition="human", pilot=False, trial_id="human-L01", outcome="success")
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(tmp_path)


def session(tmp_path):
    # Keep controller behavior out of these bookkeeping tests; controller and
    # robot lifecycle have separate integration coverage in test_interactive_rollout.
    obj = object.__new__(ExperimentSession)
    import threading

    obj.journal = Journal(tmp_path / "events.jsonl")
    obj.condition = "human"
    obj.pilot = False
    obj.manifest = make_manifest(["red", "blue"])
    obj.manifest["timeout_s"] = 300
    obj.root = tmp_path
    obj.trial = None
    obj._trial_lock = threading.RLock()
    obj._home_ready = True
    obj._awaiting_verdict = False
    obj._started_at = None
    obj._first_episode = None
    obj.ctx = SimpleNamespace(data=SimpleNamespace(dataset=SimpleNamespace(num_episodes=0)))
    obj.controller = MagicMock()
    obj.controller.running = False
    obj._print = MagicMock()
    return obj


def test_cannot_start_unarmed_or_reuse_attempt(tmp_path):
    obj = session(tmp_path)
    obj._cmd_start(InteractiveCommand("start"))
    obj.controller.start.assert_not_called()
    obj._cmd_trial(InteractiveCommand("trial", "human-L01"))
    obj._cmd_start(InteractiveCommand("start"))
    obj.controller.start.assert_called_once()
    obj.trial = None
    obj._cmd_trial(InteractiveCommand("trial", "human-L01"))
    assert obj.trial is None


def test_score_requires_home_and_all_cubes_for_success(tmp_path):
    obj = session(tmp_path)
    obj._cmd_trial(InteractiveCommand("trial", "human-L01"))
    obj._awaiting_verdict = True
    obj._home_ready = False
    obj._cmd_score(InteractiveCommand("score", "success 2"))
    assert obj.trial is not None
    obj._home_ready = True
    obj._cmd_score(InteractiveCommand("score", "success 1"))
    assert obj.trial is not None
    obj.ctx.data.dataset.num_episodes = 2
    obj._cmd_score(InteractiveCommand("score", "success 2 both released"))
    assert obj.trial is None
    result = json.loads(obj.journal.path.read_text().splitlines()[-1])
    assert result["dataset_episode_start"] == 0
    assert result["dataset_episode_end_exclusive"] == 2


def test_live_planner_error_requests_reset(tmp_path, monkeypatch):
    obj = session(tmp_path)
    obj.condition = "planner"
    obj.trial = "planner-L01"
    obj.controller.task = "pick red"
    monkeypatch.setattr("lerobot.rollout.interactive.InteractiveSession._on_event", lambda *args: None)
    obj._on_event(RolloutEvent.QUERY_ANSWERED, SimpleNamespace(ok=False, error="planner unavailable"))
    obj.controller.reset.assert_called_once()
    assert obj._awaiting_verdict


def test_unscored_fault_is_visible_and_blocks_success_percentage(tmp_path):
    journal = Journal(tmp_path / "sessions" / "one" / "events.jsonl")
    journal.write("trial_armed", condition="human", pilot=False, trial_id="human-L01")
    journal.write("verdict", condition="human", pilot=False, trial_id="human-L01", outcome="success")
    journal.write("trial_armed", condition="human", pilot=False, trial_id="human-L02")
    result = summarize(tmp_path)["human"]
    assert result["attempted"] == 2
    assert result["unscored_attempts"] == ["human-L02"]
    assert result["success_rate"] is None


def test_stop_in_hold_mode_finishes_trial_before_scoring(tmp_path):
    obj = session(tmp_path)
    obj._stop_returns_home = True
    obj._cmd_trial(InteractiveCommand("trial", "human-L01"))
    obj._started_at = 1.0
    obj._handle_line("/stop")
    obj.controller.reset.assert_called_once()
    obj.controller.stop.assert_not_called()
    assert obj._awaiting_verdict
    assert not obj._home_ready


def test_skip_human_preserves_ten_matched_layouts_and_report(tmp_path):
    active = tuple(c for c in CONDITIONS if c != "human")
    manifest = make_manifest(["red"], conditions=active)
    assert len(manifest["trials"]) == 40
    assert manifest["excluded_conditions"] == ["human"]
    for condition in active:
        assert sorted(t["layout"] for t in manifest["trials"] if t["condition"] == condition) == list(
            range(1, 11)
        )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report = summarize(tmp_path)
    assert report["human"]["planned"] == 0
    assert report["human"]["excluded"]
    assert report["planner"]["planned"] == 10


def test_hosted_planner_flags_allow_token_budget_but_not_secret(tmp_path):
    flags = prepare_arguments(
        ["--robot.type=bi_yam_follower"],
        [
            "--planner.model_id=Qwen/Qwen3.8-27B:novita",
            "--planner.api_key_env=HF_TOKEN",
            "--planner.max_new_tokens=512",
            "--planner.log_path=/ignored",
        ],
        tmp_path,
        "goal",
        0,
    )
    assert "--planner.api_key_env=HF_TOKEN" in flags
    assert "--planner.max_new_tokens=512" in flags
    assert f"--planner.log_path={tmp_path / 'planner.jsonl'}" in flags
