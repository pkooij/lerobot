"""Prepare native rollout configs using an existing confirmed robot configuration.

This script only reads/writes JSON; it never connects cameras or motors.
"""

import argparse
import copy
import json
import math
from pathlib import Path


def latest_hardware_config(root):
    """Find an existing manual-trial robot config; never invent hardware settings."""
    candidates = []
    for directory in root.glob("manual_*"):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.json"):
            try:
                source = json.loads(path.read_text())
            except (OSError, UnicodeError, ValueError):
                continue
            if not isinstance(source, dict):
                continue
            robot = source.get("robot", source)
            if not isinstance(robot, dict) or robot.get("type") != "bi_rebot_b601_follower":
                continue
            if all(
                isinstance(robot.get(arm), dict) and robot[arm].get("port")
                for arm in ("left_arm_config", "right_arm_config")
            ):
                candidates.append(path)
    if not candidates:
        raise ValueError(
            f"No saved manual ReBot config found under {root}; supply --hardware-config explicitly"
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware-config", type=Path, required=True, help="Existing rollout JSON or robot JSON"
    )
    parser.add_argument("--hardware-root", type=Path, default=Path.home() / "rebot-steerable-artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument(
        "--task", default="Use the left arm to pick up the blue block and place it into the black bin."
    )
    args = parser.parse_args()
    if not math.isfinite(args.max_relative_target) or not 0 < args.max_relative_target <= 5:
        parser.error("--max-relative-target must be finite and in (0, 5] degrees")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if args.hardware_config == Path("auto"):
        try:
            args.hardware_config = latest_hardware_config(args.hardware_root)
        except ValueError as exc:
            parser.error(str(exc))
    source = json.loads(args.hardware_config.read_text())
    robot = copy.deepcopy(source.get("robot", source))
    if robot.get("type") != "bi_rebot_b601_follower":
        parser.error("Expected the previously confirmed bi_rebot_b601_follower robot config")
    for arm in ("left_arm_config", "right_arm_config"):
        if not robot.get(arm, {}).get("port"):
            parser.error(f"Missing {arm}.port")
        robot[arm]["max_relative_target"] = args.max_relative_target
    camera_names = set(robot.get("cameras", {}))
    for side in ("left", "right"):
        camera_names.update(f"{side}_{name}" for name in robot[f"{side}_arm_config"].get("cameras", {}))
    expected = {"base", "left_wrist", "right_wrist"}
    policy_cameras = {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    if camera_names == expected:
        rename = {
            "observation.images.base": "observation.images.cam_high",
            "observation.images.left_wrist": "observation.images.cam_left_wrist",
            "observation.images.right_wrist": "observation.images.cam_right_wrist",
        }
    elif camera_names == policy_cameras:
        rename = {}
    else:
        parser.error(
            f"Cannot infer camera mapping for {sorted(camera_names)}; preserve the confirmed three views"
        )
    config = {
        "robot": robot,
        "strategy": {"type": "base"},
        "inference": {"type": "sync"},
        "device": "cuda",
        "fps": 30,
        "duration": args.duration,
        "interactive": True,
        "autosteer_interval_s": 5.0,
        "task": args.task,
        "rename_map": rename,
        "play_sounds": False,
        "return_to_initial_position": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    print(f"Prepared {args.output} from {args.hardware_config}; no hardware connected.")


if __name__ == "__main__":
    main()
