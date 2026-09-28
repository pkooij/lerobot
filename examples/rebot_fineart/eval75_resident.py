"""Keep one policy and robot connection alive across scored ReBot evaluation trials."""

import argparse
import contextlib
import fcntl
import json
import logging
import re
import resource
import shutil
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from eval75 import CONDITIONS, assert_no_existing_rollout, make_config, score, write_json
from eval75_madeleine import campaign_config


def prepare_trial(ctx, goal):
    if ctx.runtime.shutdown_event.parent.is_set():
        raise KeyboardInterrupt
    ctx.runtime.shutdown_event.clear()
    engine = ctx.policy.inference
    engine.pause()
    engine.stop_autosteer()
    engine.drop_pending_query()
    engine.drop_ready_subtask_answers()
    engine.set_task(goal)
    engine.reset()


def validate_trial(episodes_before, episodes_after, starts, steered, condition):
    if episodes_after - episodes_before != 1 or starts != 1:
        raise ValueError("Expected one recorded episode and one /start; attempt preserved, session stopped")
    if steered != (condition == "subtask_autosteer"):
        raise ValueError("Unexpected steering mode; attempt preserved, session stopped")


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.original.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.log.flush()


def session_class(base, events):
    class TrialSession(base):
        @staticmethod
        def _print(message):
            base._print(message.replace("/stop to shut down", "/stop to finish and score this trial"))

        def __init__(self, strategy, ctx):
            super().__init__(strategy, ctx)
            self.starts = 0
            self.steered = False
            self.intervened = False
            self.quit_requested = False
            self._commands["stop"] = (self._cmd_stop, "", "finish this trial and score it; keep model loaded")
            self._commands["quit"] = (self._quit, "", "finish this trial and exit the evaluation")

        def _on_event(self, event, payload=None):
            if event is events.SEGMENT_STARTED:
                self.starts += 1
            super()._on_event(event, payload)

        def _cmd_autosteer(self, cmd):
            super()._cmd_autosteer(cmd)
            self.steered |= self.controller.autosteer_goal is not None

        def _cmd_subtask(self, cmd):
            self.intervened |= self.controller.running
            super()._cmd_subtask(cmd)

        def _cmd_reset(self, cmd):
            self.intervened |= self.controller.running
            super()._cmd_reset(cmd)

        def _quit(self, cmd):
            self.quit_requested = True
            self._cmd_stop(cmd)

        def _handle_eof(self):
            self.quit_requested = True
            super()._handle_eof()

    return TrialSession


def parse_runtime(runtime, config_path, model):
    @runtime.parser.wrap()
    def parse(cfg: runtime.RolloutConfig):
        return cfg

    previous = sys.argv
    try:
        sys.argv = [
            "resident-eval",
            f"--config_path={config_path}",
            f"--policy.path={model['repo_id']}",
            f"--policy.pretrained_revision={model['revision']}",
        ]
        return parse()
    finally:
        sys.argv = previous


def run_block(runtime, cfg, root, block, pending, common, check_config=False):
    if check_config:
        print("CONFIG_OK", cfg.policy.pretrained_path, cfg.policy.pretrained_revision, cfg.dataset.repo_id)
        return 0
    from lerobot.rollout.action_trace import ActionTrace
    from lerobot.rollout.controller import RolloutEvent

    session_type = session_class(runtime.InteractiveSession, RolloutEvent)
    saved_signals = {
        s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)
    }
    handler = runtime.ProcessSignalHandler(use_threads=True)
    ctx = None
    strategy = None
    try:
        print("Loading model once; robot/cameras remain connected between trials. No automatic reset motion.")
        ctx = runtime.build_rollout_context(cfg, runtime.LinkedEvent(handler.shutdown_event))
        strategy = runtime.create_strategy(cfg.strategy)
        strategy.setup(ctx)
        for trial in pending:
            prepare_trial(ctx, common["goal"])
            input(
                f"Scene {trial}: restore objects, start external filming, then Enter (model stays loaded): "
            )
            if handler.shutdown_event.is_set():
                break
            attempt = root / f"trial_{trial:02d}"
            attempt.mkdir()
            plan = {
                **common,
                "trial": trial,
                "attempt": str(attempt),
                "resident": True,
                "robot_recording": str(root / "dataset"),
                "episode_index": ctx.data.dataset.num_episodes,
            }
            write_json(attempt / "trial.json", plan)
            before = ctx.data.dataset.num_episodes
            session = session_type(strategy, ctx)
            print("/start once; /stop scores this trial; /quit exits. /reset moves to the initial pose.")
            if common["condition"] == "subtask_autosteer":
                print(f"After /start, enter: /autosteer {common['goal']}")
            with (attempt / "terminal.log").open("x", buffering=1) as log:
                old_handlers = logging.getLogger().handlers[:]
                try:
                    with (
                        contextlib.redirect_stdout(Tee(sys.stdout, log)),
                        contextlib.redirect_stderr(Tee(sys.stderr, log)),
                    ):
                        runtime.init_logging()
                        with ActionTrace(str(attempt / "actions.jsonl")) as trace:
                            ctx.runtime.action_trace = trace
                            session.run()
                finally:
                    ctx.runtime.action_trace = None
                    logging.getLogger().handlers = old_handlers
            if session.controller.failed:
                raise RuntimeError(session.controller.failure_traceback)
            if handler.shutdown_event.is_set():
                print("Interrupted attempt preserved; ending evaluation.")
                break
            if session.starts == 0 and session.quit_requested:
                break
            validate_trial(
                before, ctx.data.dataset.num_episodes, session.starts, session.steered, common["condition"]
            )
            result = score()
            result["intervention"] |= session.intervened
            result["success"] &= not result["intervention"]
            write_json(
                block / f"trial_{trial:02d}.json",
                {
                    **plan,
                    **result,
                    "external_video": "Recorded separately on laptop",
                    "finished_utc": datetime.now(UTC).isoformat(),
                },
            )
            results = [json.loads(p.read_text()) for p in sorted(block.glob("trial_*.json"))]
            write_json(
                block / "summary.json",
                {
                    "trials": len(results),
                    "successes": sum(r["success"] for r in results),
                    "objects_in_bin": sum(r["objects_in_bin"] for r in results),
                    "objects_total": len(results) * 5,
                },
            )
            print(f"Scored scene {trial}; model remains on GPU. {len(results)}/25 complete.")
            if session.quit_requested:
                break
    finally:
        try:
            if strategy is not None:
                strategy.teardown(ctx)
        finally:
            for sig, previous in saved_signals.items():
                signal.signal(sig, previous)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--trial", type=int, choices=range(1, 26))
    parser.add_argument(
        "--num-trials",
        type=int,
        choices=range(1, 26),
        default=25,
        help="Run scene IDs 1 through N, skipping already scored scenes",
    )
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--campaign", default="rebot-eval75-madeleine-20k")
    parser.add_argument("--hardware", type=Path, default=Path.home() / "rebot-eval75/hardware.json")
    parser.add_argument("--models", type=Path, default=Path(__file__).with_name("models_20k.json"))
    parser.add_argument(
        "--check-config", action="store_true", help="Validate config only; no model or hardware loading"
    )
    args = parser.parse_args()
    if args.duration <= 0 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.campaign):
        parser.error("Invalid duration or campaign name")
    models = json.loads(args.models.read_text())
    for model in models.values():
        if not model.get("hub_integrity_checked") or not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            parser.error("Expected verified, pinned models")
    campaign = Path.home() / "rebot-eval75" / args.campaign
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign.parent / "rollout.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_no_existing_rollout()
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        config = campaign_config(campaign / "campaign.json", models, args.duration)
        block = campaign / args.condition
        block.mkdir(exist_ok=True)
        pending = [
            t
            for t in ([args.trial] if args.trial else range(1, args.num_trials + 1))
            if not (block / f"trial_{t:02d}.json").exists()
        ]
        if not pending:
            print("Selected trials are already scored.")
            return 0
        model = models["task_only" if args.condition == "task_only_direct" else "subtask"]
        common = {
            "condition": args.condition,
            "model": model,
            "goal": config["goal"],
            "duration": args.duration,
        }
        root = block / ("resident_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
        root.mkdir()
        path = root / "rollout.json"
        write_json(path, make_config(json.loads(args.hardware.read_text()), common, root))
        # Import after argument validation; registers all robot and policy config types.
        from lerobot.scripts import lerobot_rollout as runtime

        runtime.register_third_party_plugins()
        runtime.init_logging()
        checkout = Path(runtime.__file__).resolve().parents[3]
        git = shutil.which("git")
        if git is None:
            raise RuntimeError("Git is required to record the runtime revision")
        common["code_revision"] = subprocess.check_output(
            [git, "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        write_json(root / "session.json", common)
        cfg = parse_runtime(runtime, path, model)
        return run_block(runtime, cfg, root, block, pending, common, args.check_config)


if __name__ == "__main__":
    raise SystemExit(main())
