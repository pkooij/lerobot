"""Trial manifest and append-only evidence for the YAM steering comparison."""

import argparse
import json
import random
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

CONDITIONS = ("human", "planner", "hybrid", "astra", "astra_icl")
TASK = "Put all colored cubes in the green bin."


def make_manifest(colors: list[str], seed: int = 42) -> dict:
    if len(colors) != len(set(colors)) or any(not color.strip() for color in colors):
        raise ValueError("Use unique nonempty cube labels, e.g. red-1 red-2 blue-1")
    order = list(CONDITIONS)
    random.Random(seed).shuffle(order)
    trials = []
    for layout in range(1, 11):
        # Ten matched layouts, balanced position within each block (twice/condition).
        rotation = (layout - 1) % len(order)
        for condition in order[rotation:] + order[:rotation]:
            trials.append({"id": f"{condition}-L{layout:02}", "condition": condition, "layout": layout})
    return {
        "schema_version": 1,
        "task": TASK,
        "cube_labels": colors,
        "seed": seed,
        "trials": trials,
        "success_rule": "Every listed cube released fully inside the green bin and at rest for 2 seconds.",
        "physical_intervention_rule": "End the trial as interrupted; do not resume it as an unassisted success.",
        "timeout_s": None,
        "notes": "Freeze prompts, controller settings, API model and a common trial budget after pilots.",
    }


class Journal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, **fields) -> None:
        record = {
            "utc": datetime.now(UTC).isoformat(),
            "monotonic_s": time.monotonic(),
            "event": event,
            **fields,
        }
        with self._lock, self.path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False, default=str) + "\n")
            stream.flush()


def summarize(root: Path) -> dict:
    rows = []
    attempts = []
    for path in sorted(root.glob("sessions/*/events.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "trial_armed":
                attempts.append(event)
            if event["event"] == "verdict":
                rows.append(event)
    result = {}
    for condition in CONDITIONS:
        formal = [r for r in rows if r["condition"] == condition and not r["pilot"]]
        ids = [r["trial_id"] for r in formal]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate scored trial IDs in {condition}; resolve before comparing")
        attempted = {r["trial_id"] for r in attempts if r["condition"] == condition and not r["pilot"]} | set(
            ids
        )
        unscored = sorted(attempted - set(ids))
        result[condition] = {
            "attempted": len(attempted),
            "unscored_attempts": unscored,
            "scored": len(formal),
            "planned": 10,
            "successes": sum(r["outcome"] == "success" for r in formal),
            "interrupted": sum(r["outcome"] == "interrupted" for r in formal),
            "success_rate": sum(r["outcome"] == "success" for r in formal) / len(formal)
            if formal and not unscored
            else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "report"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--cubes", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "init":
        args.root.mkdir(parents=True, exist_ok=True)
        with (args.root / "manifest.json").open("x") as stream:
            json.dump(make_manifest(args.cubes, args.seed), stream, indent=2)
        print(f"Created 50 planned trials in {args.root / 'manifest.json'}. No robot connection.")
    else:
        print(json.dumps(summarize(args.root), indent=2))


if __name__ == "__main__":
    main()
