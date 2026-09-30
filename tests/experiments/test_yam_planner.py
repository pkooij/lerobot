import json
from unittest.mock import MagicMock

import pytest

from examples.yam_experiments.campaign import CUBE_INSTRUCTIONS, TASK
from examples.yam_experiments.planner import CubePlanner, PlannerFinishedError
from lerobot.rollout.inference import PolicyQuery, QueryKind
from lerobot.rollout.planner import PlannerConfig


def planner(tmp_path, reply):
    client = MagicMock()
    client.generate_json.return_value = [reply]
    return CubePlanner(
        PlannerConfig(instructions=CUBE_INSTRUCTIONS, log_path=str(tmp_path / "planner.jsonl")),
        "bi_yam_follower",
        client=client,
    )


def test_rejects_observed_broad_goal_and_keeps_reply_in_evidence(tmp_path, capsys):
    reply = {"scene": "Three cubes outside bin", "previous_command": "none", "instruction": TASK}
    p = planner(tmp_path, reply)
    with pytest.raises(ValueError, match="color-specific"):
        p({}, PolicyQuery(QueryKind.NEXT_SUBTASK, TASK), TASK)
    record = json.loads((tmp_path / "planner.jsonl").read_text())
    assert record["reply"] == reply
    assert record["returned"] is None
    assert "Qwen reply" in capsys.readouterr().out


def test_first_subtask_replaces_goal_and_prints_under_muted_logs(tmp_path, capsys):
    from lerobot.rollout.interactive import _mute_system_output

    reply = {"scene": "Red cube outside bin", "previous_command": "none", "instruction": CUBE_INSTRUCTIONS[0]}
    p = planner(tmp_path, reply)
    query = PolicyQuery(QueryKind.NEXT_SUBTASK, TASK)
    assert "Current instruction: None:" in p.request_text(query, TASK)
    with _mute_system_output():
        assert p({}, query, TASK) == CUBE_INSTRUCTIONS[0]
    output = capsys.readouterr().out
    assert "Red cube outside bin" in output
    assert "Planner selected:" in output


def test_first_reply_cannot_hold_unissued_goal(tmp_path):
    reply = {
        "scene": "Red cube outside bin",
        "previous_command": "in progress",
        "instruction": CUBE_INSTRUCTIONS[0],
    }
    p = planner(tmp_path, reply)
    with pytest.raises(ValueError, match="First planner reply"):
        p({}, PolicyQuery(QueryKind.NEXT_SUBTASK, TASK), TASK)


def test_hold_single_cube_then_advance_then_done(tmp_path):
    red, blue = CUBE_INSTRUCTIONS[0], CUBE_INSTRUCTIONS[4]
    reply = {"scene": "Red cube held", "previous_command": "in progress", "instruction": blue}
    p = planner(tmp_path, reply)
    query = PolicyQuery(QueryKind.NEXT_SUBTASK, TASK, history=(({}, red),))
    assert p({}, query, red) == red
    reply.update(scene="Red released in bin", previous_command="completed")
    assert p({}, query, red) == blue
    reply.update(instruction="done")
    with pytest.raises(PlannerFinishedError):
        p({}, query, blue)
    assert (
        json.loads((tmp_path / "planner.jsonl").read_text().splitlines()[-1])["reply"]["instruction"]
        == "done"
    )
