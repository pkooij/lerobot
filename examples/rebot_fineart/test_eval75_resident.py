import json
import sys
from contextlib import nullcontext
from enum import Enum
from types import SimpleNamespace

import pytest
from eval75_resident import run_block, validate_trial


@pytest.mark.parametrize(
    "after,starts,steered,condition",
    [
        (0, 1, False, "task_only_direct"),
        (2, 2, False, "task_only_direct"),
        (1, 1, True, "task_only_direct"),
        (1, 1, False, "subtask_autosteer"),
    ],
)
def test_recording_boundaries_reject_missing_multiple_or_wrong_mode(after, starts, steered, condition):
    with pytest.raises(ValueError):
        validate_trial(0, after, starts, steered, condition)


def test_two_trials_reuse_context_and_reset_goal_and_actions(tmp_path, monkeypatch):
    class Events(Enum):
        SEGMENT_STARTED = "start"

    class StopEvent:
        parent = SimpleNamespace(is_set=lambda: False)

        def clear(self):
            pass

    calls = []
    engine = SimpleNamespace(
        pause=lambda: calls.append("pause"),
        stop_autosteer=lambda: calls.append("stop_autosteer"),
        drop_pending_query=lambda: calls.append("drop_query"),
        drop_ready_subtask_answers=lambda: calls.append("drop_answers"),
        set_task=lambda text: calls.append(("goal", text)),
        reset=lambda: calls.append("reset"),
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(shutdown_event=StopEvent(), action_trace=None),
        policy=SimpleNamespace(inference=engine, policy=object()),
        data=SimpleNamespace(dataset=SimpleNamespace(num_episodes=0)),
    )
    identities = []

    class BaseSession:
        def __init__(self, strategy, context):
            self._commands = {}
            self.ctx = context
            self.controller = SimpleNamespace(failed=False)

        def _cmd_stop(self, cmd):
            pass

        def _on_event(self, event, payload=None):
            pass

        def run(self):
            self._on_event(Events.SEGMENT_STARTED)
            self.ctx.data.dataset.num_episodes += 1
            identities.append(id(self.ctx.policy.policy))

    strategy = SimpleNamespace(
        setup=lambda c: calls.append("setup"), teardown=lambda c: calls.append("teardown")
    )

    def build(*args):
        calls.append("build")
        return ctx

    runtime = SimpleNamespace(
        InteractiveSession=BaseSession,
        ProcessSignalHandler=lambda **k: SimpleNamespace(shutdown_event=StopEvent.parent),
        LinkedEvent=lambda event: StopEvent(),
        build_rollout_context=build,
        create_strategy=lambda cfg: strategy,
        init_logging=lambda: None,
    )
    monkeypatch.setitem(
        sys.modules, "lerobot.rollout.action_trace", SimpleNamespace(ActionTrace=lambda p: nullcontext())
    )
    monkeypatch.setitem(sys.modules, "lerobot.rollout.controller", SimpleNamespace(RolloutEvent=Events))
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(
        "eval75_resident.score", lambda: {"intervention": False, "success": True, "objects_in_bin": 5}
    )
    root, block = tmp_path / "session", tmp_path / "condition"
    root.mkdir()
    block.mkdir()
    assert (
        run_block(
            runtime,
            SimpleNamespace(strategy=None),
            root,
            block,
            [1, 2],
            {"goal": "Place five objects.", "condition": "task_only_direct"},
        )
        == 0
    )
    assert calls.count("build") == calls.count("setup") == calls.count("teardown") == 1
    assert calls.count("reset") == calls.count("stop_autosteer") == 2
    assert calls.count(("goal", "Place five objects.")) == 2
    assert len(set(identities)) == 1 and len(identities) == 2
    first = json.loads((block / "trial_01.json").read_text())
    second = json.loads((block / "trial_02.json").read_text())
    assert (first["episode_index"], second["episode_index"]) == (0, 1)
    assert first["robot_recording"] == second["robot_recording"]
    assert json.loads((block / "summary.json").read_text())["successes"] == 2
