"""Attended trials over LeRobot's existing interactive controller and recording."""

import json
import threading
import time
from pathlib import Path

from lerobot.rollout.controller import RolloutEvent
from lerobot.rollout.interactive import InteractiveSession, parse_command

from .campaign import Journal


class ExperimentSession(InteractiveSession):
    def __init__(
        self, strategy, ctx, *, journal: Journal, condition: str, pilot: bool, manifest: dict, root: Path
    ):
        super().__init__(strategy, ctx)
        self.journal = journal
        self.condition = condition
        self.pilot = pilot
        self.manifest = manifest
        self.root = root
        self.ctx = ctx
        self.trial = None
        self._trial_lock = threading.RLock()
        self._home_ready = True
        self._awaiting_verdict = False
        self._started_at = None
        self._first_episode = None
        self._commands.update(
            {
                "trial": (self._cmd_trial, " <id>", "arm a named trial after arranging its layout"),
                "finish": (self._cmd_finish, "", "end this attempt and return home before scoring"),
                "score": (
                    self._cmd_score,
                    " <success|partial|failure|interrupted> <count> <notes>",
                    "save operator verdict",
                ),
            }
        )
        self.journal.write("session", condition=condition, pilot=pilot, task=ctx.runtime.cfg.task)

    def _cmd_trial(self, cmd):
        with self._trial_lock:
            if self.trial is not None or not self._home_ready or self.controller.running:
                self._print("Finish, wait for home, and score the previous trial first.")
                return
            trial_id = cmd.args.strip()
            if not trial_id or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in trial_id
            ):
                self._print("Use a simple trial ID, e.g. pilot-red or human-L01.")
                return
            if not self.pilot:
                planned = next((t for t in self.manifest["trials"] if t["id"] == trial_id), None)
                if planned is None or planned["condition"] != self.condition:
                    self._print("Trial ID does not match this condition's manifest.")
                    return
                if not self.manifest["cube_labels"] or not self.manifest["timeout_s"]:
                    self._print(
                        "Freeze cube labels and the common trial budget in manifest.json after pilots."
                    )
                    return
            # A reservation survives crashes. Aborted attempts remain visible and cannot be overwritten.
            reservation = self.root / "reservations" / f"{'pilot-' if self.pilot else ''}{trial_id}.json"
            reservation.parent.mkdir(parents=True, exist_ok=True)
            try:
                with reservation.open("x") as stream:
                    json.dump({"session": str(self.journal.path), "condition": self.condition}, stream)
            except FileExistsError:
                self._print(
                    "This ID was already attempted. Keep that evidence; use a new pilot ID or resolve the interrupted formal trial."
                )
                return
            self.trial = trial_id
            self._awaiting_verdict = False
            self._started_at = None
            self._first_episode = self.ctx.data.dataset.num_episodes
            self.journal.write("trial_armed", trial_id=trial_id, condition=self.condition, pilot=self.pilot)
            self._print(f"Armed {trial_id}. Set /subtask if needed, then /start.")

    def _cmd_start(self, cmd):
        with self._trial_lock:
            if self.trial is None or self._awaiting_verdict or not self._home_ready:
                self._print("Use /trial <id> first; each trial gets one /start.")
                return
            super()._cmd_start(cmd)

    def _cmd_finish(self, cmd):
        with self._trial_lock:
            if self.trial is None or self._started_at is None:
                self._print("No running trial to finish.")
                return
            self._awaiting_verdict = True
            self._home_ready = False
            self.journal.write(
                "trial_end_requested", trial_id=self.trial, elapsed_s=time.monotonic() - self._started_at
            )
            self.controller.reset()

    def _cmd_score(self, cmd):
        with self._trial_lock:
            if (
                self.trial is None
                or not self._awaiting_verdict
                or self.controller.running
                or not self._home_ready
            ):
                self._print("Use /finish and wait for a successful home return before scoring.")
                return
            parts = cmd.args.split(maxsplit=2)
            try:
                outcome, count = parts[0], int(parts[1])
                if outcome not in {"success", "partial", "failure", "interrupted"} or count < 0:
                    raise ValueError
                total = len(self.manifest["cube_labels"])
                if total and (count > total or (outcome == "success" and count != total)) and not self.pilot:
                    raise ValueError
            except (IndexError, ValueError):
                self._print(
                    "Use /score success|partial|failure|interrupted <cube count> <notes>. Success requires every cube."
                )
                return
            self.journal.write(
                "verdict",
                trial_id=self.trial,
                condition=self.condition,
                pilot=self.pilot,
                outcome=outcome,
                cubes_in_bin=count,
                notes=parts[2] if len(parts) > 2 else "",
                dataset_episode_start=self._first_episode,
                dataset_episode_end_exclusive=self.ctx.data.dataset.num_episodes,
            )
            self._print(
                f"Saved {self.trial}: {outcome}, {count} cubes. Arrange the next layout, then /trial."
            )
            self.trial = None

    def _handle_line(self, line):
        with self._trial_lock:
            self.journal.write("operator_command", trial_id=self.trial, command=line)
            cmd = parse_command(line)
            if (
                cmd
                and (cmd.name == "reset" or (cmd.name == "stop" and self._stop_returns_home))
                and self.trial
                and self._started_at is not None
            ):
                self._cmd_finish(cmd)
                return
            if cmd and cmd.name in {"subtask", "vqa", "autosteer"} and self.condition != "human":
                if not self.pilot:
                    self._print(
                        "Manual steering changes the experimental condition. Use /finish and score an interruption instead."
                    )
                    return
                self.journal.write("human_intervention", trial_id=self.trial, command=line)
            super()._handle_line(line)

    def _on_event(self, event, payload=None):
        with self._trial_lock:
            self.journal.write(
                "controller_event",
                trial_id=self.trial,
                name=event.name,
                task=self.controller.task,
                payload=vars(payload) if payload else None,
            )
            if event is RolloutEvent.SEGMENT_STARTED:
                if self.condition == "planner":
                    self.controller.autosteer(self.manifest["task"])
                self._started_at = time.monotonic()
                self._home_ready = False
            elif event is RolloutEvent.SEGMENT_ENDED:
                self._awaiting_verdict = True
                self.controller.reset()
            elif (
                event is RolloutEvent.QUERY_ANSWERED
                and payload is not None
                and not payload.ok
                and self.condition == "planner"
            ):
                self.journal.write("planner_trial_stop", trial_id=self.trial, reason=payload.error)
                self._awaiting_verdict = True
                self.controller.reset()
            elif event is RolloutEvent.RESET_DONE:
                self._home_ready = True
            elif event in {
                RolloutEvent.RESET_FAILED,
                RolloutEvent.STRATEGY_FAILED,
                RolloutEvent.ENGINE_FAILED,
            }:
                self._home_ready = False
            super()._on_event(event, payload)

    def run(self):
        try:
            super().run()
        finally:
            if self.trial is not None:
                self.journal.write(
                    "unscored_attempt",
                    trial_id=self.trial,
                    condition=self.condition,
                    pilot=self.pilot,
                    reason="session ended before verdict; retain and adjudicate this attempt",
                )
