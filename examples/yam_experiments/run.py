"""Launch an attended YAM pilot or scored session; defaults to a no-hardware preview."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from .campaign import CONDITIONS, TASK, Journal


def launcher_arguments(path: Path) -> list[str]:
    text = path.read_text()
    if text.count("lerobot-rollout") != 1:
        raise ValueError("Expected one lerobot-rollout invocation in the setup launcher")
    # Read only the CLI arguments. Never execute the launcher's shell preamble.
    tokens = shlex.split(text.split("lerobot-rollout", 1)[1].replace("\\\n", " "))
    if any(not token.startswith("--") or "=" not in token for token in tokens):
        raise ValueError("Launcher must use --key=value arguments, without shell expressions or commands")
    return tokens


def prepare_arguments(
    base: list[str], extra: list[str], session_dir: Path, task: str, duration: float
) -> list[str]:
    flags = {}
    for token in base + extra:
        if not token.startswith("--") or "=" not in token:
            raise ValueError("Pass rollout overrides as --key=value")
        key, value = token[2:].split("=", 1)
        if key.rsplit(".", 1)[-1].lower() in {"api_key", "token", "access_token", "hf_token"}:
            raise ValueError("Do not put credentials in launcher flags or experiment logs")
        flags[key] = value
    flags.update(
        {
            "interactive": "true",
            "duration": str(duration),
            "task": task,
            "robot.defer_torque_enable": "true",
            "return_to_initial_position": "true",
            "strategy.type": "sentry",
            "dataset.repo_id": "local/rollout_yam_experiment",
            "dataset.root": str(session_dir / "dataset"),
            "dataset.single_task": task,
            "dataset.push_to_hub": "false",
            "dataset.no_stamp": "true",
            "dataset.streaming_encoding": "true",
            "dataset.rgb_encoder.vcodec": "h264",
            "dataset.encoder_threads": "1",
            "resume": "false",
        }
    )
    if "planner.model_id" in flags:
        from .campaign import CUBE_INSTRUCTIONS

        flags["planner.instructions"] = json.dumps(CUBE_INSTRUCTIONS)
        flags["planner.log_path"] = str(session_dir / "planner.jsonl")
    if flags.get("robot.type") != "bi_yam_follower":
        raise ValueError("This harness requires the tested bi_yam_follower adapter")
    return [f"--{key}={value}" for key, value in flags.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="human")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--execute", action="store_true", help="Connect hardware and wait for /trial then /start"
    )
    args, extra = parser.parse_known_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    if args.condition not in {"human", "planner"}:
        parser.error(
            "Direct/hybrid Astra motion is not commissioned yet: configure the endpoint and validate its action adapter first."
        )
    if args.condition in manifest.get("excluded_conditions", []):
        parser.error(f"Condition {args.condition} is excluded from this campaign")
    if not args.pilot and (not manifest["cube_labels"] or not manifest["timeout_s"]):
        parser.error("After pilots, freeze cube labels and a common positive timeout_s in manifest.json")
    if args.condition == "human" and any(x.startswith("--planner.") for x in extra):
        parser.error("The human condition must not have an external planner")
    if args.condition == "planner" and not any(x.startswith("--planner.model_id=") for x in extra):
        parser.error("Planner condition requires explicit --planner.model_id and endpoint settings")
    session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + args.condition
    session_dir = args.root.resolve() / "sessions" / session_id
    argv = prepare_arguments(
        launcher_arguments(args.launcher),
        extra,
        session_dir,
        TASK,
        0 if args.pilot else manifest["timeout_s"],
    )
    if not args.execute:
        print(
            json.dumps(
                {"preview": True, "condition": args.condition, "pilot": args.pilot, "argv": argv}, indent=2
            )
        )
        return
    git = shutil.which("git")
    if git is None:
        parser.error("git is required for experiment provenance")
    branch = subprocess.check_output([git, "branch", "--show-current"], text=True).strip()
    if branch != "codex/yam-steering-experiments":
        parser.error("Switch to codex/yam-steering-experiments before connecting")
    session_dir.mkdir(parents=True, exist_ok=False)
    journal = Journal(session_dir / "events.jsonl")
    head = subprocess.check_output([git, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output([git, "status", "--porcelain"], text=True).strip()
    if dirty and not args.pilot:
        parser.error("Scored trials require a clean frozen revision")
    journal.write(
        "provenance",
        git_sha=head,
        dirty=bool(dirty),
        argv=argv,
        launcher_sha256=hashlib.sha256(args.launcher.read_bytes()).hexdigest(),
        manifest=manifest,
    )
    if "--planner.api_base=https://router.huggingface.co/v1" in argv:
        from huggingface_hub import get_token

        if "--planner.api_key_env=HF_TOKEN" not in argv:
            parser.error("HF Router requires --planner.api_key_env=HF_TOKEN")
        token = get_token()
        if not token:
            parser.error("Log in with hf auth login using an Inference Providers token")
        os.environ["HF_TOKEN"] = token
    # Delayed imports keep manifest generation and preview independent of robotics dependencies.
    from lerobot.rollout import context
    from lerobot.scripts import lerobot_rollout

    from .planner import CubePlanner
    from .session import ExperimentSession

    original_planner = context.VlmPlanner
    original_session = lerobot_rollout.InteractiveSession
    try:
        context.VlmPlanner = CubePlanner
        lerobot_rollout.InteractiveSession = partial(
            ExperimentSession,
            journal=journal,
            condition=args.condition,
            pilot=args.pilot,
            manifest=manifest,
            root=args.root.resolve(),
        )
        sys.argv = ["yam-experiment", *argv]
        lerobot_rollout.main()
    except BaseException as exc:
        journal.write("session_error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        context.VlmPlanner = original_planner
        lerobot_rollout.InteractiveSession = original_session
        journal.write("session_closed")
    print(f"Evidence: {session_dir}")


if __name__ == "__main__":
    main()
