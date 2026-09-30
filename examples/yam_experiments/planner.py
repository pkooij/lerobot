"""Cube-by-cube planning and visible diagnostics for the YAM experiment."""

from lerobot.rollout.inference import QueryKind
from lerobot.rollout.inference.base import PlannerCompleted
from lerobot.rollout.planner import VlmPlanner, normalize_instruction


class PlannerFinishedError(PlannerCompleted):
    """End the attempt for operator scoring when the planner reports completion."""


class CubePlanner(VlmPlanner):
    def request_text(self, query, task):
        if query.kind is not QueryKind.NEXT_SUBTASK:
            return super().request_text(query, task)
        # Before the first accepted plan, the loaded policy task is the overall goal,
        # not a previously dispatched cube instruction that must be kept in progress.
        current = task if query.history else "None: choose the first single-cube subtask."
        return super().request_text(query, current) + (
            "\nDecompose the overall request into ONE visible cube per instruction. "
            "Choose a color-specific pick-and-place instruction from the allowed list. "
            "The list is a vocabulary, not an inventory: only choose a cube visible in the images "
            "that has not been released inside the bin. Never repeat the overall multi-cube goal. "
            "Assess previous_command for the last SINGLE-CUBE instruction, not the overall request. "
            "Once that cube is released in the bin, mark completed and choose another remaining cube. "
            "If a cube is already held, finish placing that same cube before choosing another. "
            "On the first call previous_command must be none. Use done only when all visible target "
            "cubes are released inside the bin; do not infer completion from command text alone."
        )

    def parse_reply(self, reply, query, task):
        if query.kind is QueryKind.NEXT_SUBTASK:
            if not isinstance(reply, dict):
                raise ValueError("Planner must return scene, previous_command, and instruction")
            instruction = reply.get("instruction", "")
            if not isinstance(instruction, str):
                raise ValueError("Planner instruction must be text")
            if instruction.strip().lower() == "done":
                raise PlannerFinishedError("Planner reports done; operator must score the trial")
            allowed = {normalize_instruction(s) for s in self.config.instructions}
            if normalize_instruction(instruction) not in allowed:
                raise ValueError(f"Expected one color-specific cube instruction, got {instruction!r}")
            if not query.history and reply.get("previous_command") != "none":
                raise ValueError("First planner reply must have previous_command='none'")
            if reply.get("previous_command") == "in progress" and normalize_instruction(task) not in allowed:
                raise ValueError("Cannot keep a broad goal in progress; choose a single-cube subtask")
        return super().parse_reply(reply, query, task)

    def _log_exchange(self, query, task, started, *, reply, returned, error):
        import json
        import time

        super()._log_exchange(query, task, started, reply=reply, returned=returned, error=error)
        # Interactive mode mutes INFO logging, but direct console output remains visible.
        print(
            f"\nQwen reply ({time.perf_counter() - started:.1f}s): {json.dumps(reply, ensure_ascii=True)}",
            flush=True,
        )
        if error:
            print(f"Planner decision: {error}", flush=True)
        else:
            print(f"Planner selected: {returned}", flush=True)
