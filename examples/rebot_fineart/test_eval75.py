import copy
import json
import signal
from types import SimpleNamespace

import pytest
from eval75 import assert_no_existing_rollout, make_config, remote, run_rollout_process


def hardware():
    return {
        "robot": {
            "type": "bi_rebot_b601_follower",
            "left_arm_config": {
                "port": "/dev/ttyACM0",
                "max_relative_target": None,
                "max_target_velocity_deg_s": 240,
                "max_target_step_deg": 20,
            },
            "right_arm_config": {"port": "/dev/ttyACM1", "joint_limits": {}},
            "cameras": {"base": {"index_or_path": "/dev/video2"}},
        },
        "interpolation_multiplier": 2,
        "rename_map": {"observation.images.base": "observation.images.cam_high"},
    }


@pytest.mark.parametrize("condition", ["subtask_direct", "task_only_direct", "subtask_autosteer"])
def test_config_preserves_confirmed_hardware_and_uses_fresh_local_recording(tmp_path, condition):
    source = hardware()
    before = copy.deepcopy(source)
    config = make_config(
        source,
        {"duration": 120, "goal": "Place five objects in bin.", "condition": condition},
        tmp_path,
    )
    assert source == before and config["robot"] == before["robot"]
    assert config["rename_map"] == before["rename_map"]
    assert config["inference"] == {"type": "sync"}
    assert config["interactive"] and not config["return_to_initial_position"]
    assert config["interpolation_multiplier"] == 2
    assert not config["dataset"]["push_to_hub"]
    assert config["dataset"]["repo_id"] == f"pepijn223/rollout_eval_rebot_20k_{condition}"
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize(
    "condition,starts,steering,frames,expected",
    [
        ("subtask_direct", 1, False, 100, 0),
        ("task_only_direct", 1, False, 100, 0),
        ("subtask_autosteer", 1, True, 100, 0),
        ("subtask_autosteer", 1, False, 100, 3),
        ("subtask_direct", 1, True, 100, 3),
        ("subtask_direct", 2, False, 100, 3),
        ("subtask_direct", 1, False, 0, 3),
    ],
)
def test_remote_counts_only_one_recorded_run_in_correct_mode(
    monkeypatch, tmp_path, condition, starts, steering, frames, expected
):
    monkeypatch.setattr("eval75.Path.home", lambda: tmp_path)
    source = tmp_path / "hardware.json"
    source.write_text(json.dumps(hardware()))
    plan = {
        "checkout": str(tmp_path),
        "hardware": str(source),
        "campaign": "test",
        "condition": condition,
        "attempt": "01_test",
        "trial": 1,
        "duration": 120,
        "goal": "Place all five objects in the bin.",
        "model": {"repo_id": "owner/model", "revision": "a" * 40},
    }
    monkeypatch.setattr("eval75.subprocess.check_output", lambda *a, **k: "b" * 40)
    monkeypatch.setattr("eval75.shutil.which", lambda name: "/usr/bin/" + name)

    def fake_rollout(*args, **kwargs):
        output = tmp_path / "rebot-eval75/test" / condition / "01_test"
        (output / "dataset/meta").mkdir(parents=True)
        (output / "dataset/meta/info.json").write_text(
            json.dumps({"total_frames": frames, "total_episodes": 1})
        )
        (output / "terminal.log").write_text(
            "Rollout running — task\n" * starts + ("Autosteer on — goal\n" if steering else "")
        )
        return 0

    monkeypatch.setattr("eval75.run_rollout_process", fake_rollout)
    assert remote(plan) == expected


def test_existing_rollout_blocks_new_session_but_unrelated_server_does_not(tmp_path):
    proc = tmp_path / "123"
    proc.mkdir()
    command = proc / "cmdline"
    command.write_bytes(b"python\0server.py\0")
    assert_no_existing_rollout(tmp_path)
    command.write_bytes(b"python\0-m\0lerobot.scripts.lerobot_rollout\0")
    with pytest.raises(RuntimeError, match="PID 123"):
        assert_no_existing_rollout(tmp_path)


def test_interrupt_signals_owned_group_and_reaps_parent(monkeypatch, tmp_path):
    signals = []
    polls = []

    def interrupted():
        raise KeyboardInterrupt

    def signal_group(pid, sig):
        assert pid == 12345
        if sig == 0:
            raise ProcessLookupError
        signals.append(sig)

    def spawn(command, *, cwd, start_new_session):
        assert start_new_session and cwd == tmp_path
        return SimpleNamespace(pid=12345, wait=interrupted, poll=lambda: polls.append(True))

    monkeypatch.setattr("eval75.subprocess.Popen", spawn)
    monkeypatch.setattr("eval75.os.killpg", signal_group)
    with pytest.raises(KeyboardInterrupt):
        run_rollout_process(["test"], tmp_path)
    assert signals == [signal.SIGINT] and polls
