"""Run the 75-trial evaluation directly on Madeleine; film separately on the Mac."""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from eval75 import CONDITIONS, remote, score, write_json


def campaign_config(path, models, duration):
    if path.exists():
        config = json.loads(path.read_text())
        if config["models"] != models or config["duration"] != duration:
            raise ValueError("Existing campaign has different models/duration; use a new --campaign")
        return config
    objects = [input(f"Object {i + 1}/5 description: ").strip() for i in range(5)]
    if not all(objects):
        raise ValueError("All five object descriptions are required")
    config = {
        "models": models,
        "duration": duration,
        "objects": objects,
        "goal": "Place " + ", ".join(objects[:-1]) + " and " + objects[-1] + " into the black bin.",
    }
    write_json(path, config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--trial", type=int, choices=range(1, 26), help="Run one matched scene, 1–25")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--campaign", default="rebot-eval75-madeleine-20k")
    parser.add_argument("--hardware", type=Path, default=Path.home() / "rebot-eval75/hardware.json")
    parser.add_argument("--checkout", type=Path, default=Path.home() / "lerobot-fineart")
    parser.add_argument("--models", type=Path, default=Path(__file__).with_name("models_20k.json"))
    args = parser.parse_args()
    if args.duration <= 0 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.campaign):
        parser.error("Positive duration and a campaign name containing letters/digits/_/- required")
    models = json.loads(args.models.expanduser().read_text())
    for variant in ("subtask", "task_only"):
        model = models[variant]
        if not model.get("hub_integrity_checked") or not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            parser.error("Expected verified checkpoints pinned to full revisions")
    if not args.hardware.expanduser().is_file():
        parser.error(f"Missing confirmed hardware config: {args.hardware}")
    root = Path.home() / "rebot-eval75" / args.campaign
    root.mkdir(parents=True, exist_ok=True)
    config = campaign_config(root / "campaign.json", models, args.duration)
    model = models["task_only" if args.condition == "task_only_direct" else "subtask"]
    # Complete downloads before any robot hardware is opened.
    from huggingface_hub import snapshot_download

    snapshot_download(model["repo_id"], revision=model["revision"])
    block = root / args.condition
    block.mkdir(exist_ok=True)
    print("Robot cameras/actions are recorded. Start the separate laptop webcam recording before proceeding.")
    for trial in [args.trial] if args.trial else range(1, 26):
        result_file = block / f"trial_{trial:02d}.json"
        if result_file.exists():
            print(f"Skipping already scored scene {trial}")
            continue
        input(
            f"Scene {trial}, {args.condition}: restore layout/start pose and start external filming. Enter: "
        )
        plan = {
            "campaign": args.campaign,
            "condition": args.condition,
            "trial": trial,
            "attempt": f"{trial:02d}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}",
            "goal": config["goal"],
            "duration": args.duration,
            "model": model,
            "checkout": str(args.checkout.expanduser()),
            "hardware": str(args.hardware.expanduser()),
        }
        status = remote(plan)
        if status:
            print(
                "Session/recording check failed. Preserve this attempt; score any robot attempt before retrying."
            )
            return status
        write_json(
            result_file,
            {
                **plan,
                **score(),
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
    print(f"{len(results)}/25 scored: {block / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
