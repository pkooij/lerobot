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
    relative_limit = parser.add_mutually_exclusive_group()
    relative_limit.add_argument("--max-relative-target", type=float, default=5.0)
    relative_limit.add_argument(
        "--disable-relative-target",
        action="store_true",
        help="Remove the measured-position error cap; requires an explicit --max-target-velocity",
    )
    parser.add_argument(
        "--disable-joint-limits",
        action="store_true",
        help="Remove configurable software joint-angle clamps; requires --max-target-velocity",
    )
    parser.add_argument(
        "--max-target-velocity",
        type=float,
        help="Optional target slew rate in deg/s (up to 120); default send cap is rate/30 degrees",
    )
    parser.add_argument(
        "--max-target-step",
        type=float,
        help="Optional per-send cap in degrees (up to 10); requires a target rate",
    )
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument(
        "--record", action="store_true", help="Save cameras/state/actions and a command trace locally"
    )
    parser.add_argument(
        "--task", default="Use the left arm to pick up the blue block and place it into the black bin."
    )
    args = parser.parse_args()
    if (args.disable_relative_target or args.disable_joint_limits) and args.max_target_velocity is None:
        parser.error("Disabling position limits requires --max-target-velocity")
    if args.max_target_step is not None:
        if args.max_target_velocity is None:
            parser.error("--max-target-step requires --max-target-velocity")
        if not math.isfinite(args.max_target_step) or not 0 < args.max_target_step <= 10:
            parser.error("--max-target-step must be finite and in (0, 10] degrees")
    if not math.isfinite(args.max_relative_target) or not 0 < args.max_relative_target <= 5:
        parser.error("--max-relative-target must be finite and in (0, 5] degrees")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if args.max_target_velocity is not None and (
        not math.isfinite(args.max_target_velocity) or not 0 < args.max_target_velocity <= 120
    ):
        parser.error("--max-target-velocity must be finite and in (0, 120] degrees/second")
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
        robot[arm]["max_relative_target"] = None if args.disable_relative_target else args.max_relative_target
        if args.disable_joint_limits:
            robot[arm]["joint_limits"] = {}
        if args.max_target_velocity is not None:
            robot[arm]["max_target_velocity_deg_s"] = args.max_target_velocity
            robot[arm]["max_target_step_deg"] = (
                args.max_target_step if args.max_target_step is not None else args.max_target_velocity / 30
            )
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
    if args.record:
        dataset_root = args.output.resolve().parent / "dataset"
        trace_path = args.output.resolve().parent / "actions.jsonl"
        if dataset_root.exists() or trace_path.exists():
            parser.error("Recording needs a fresh directory: dataset or actions.jsonl already exists")
        config["strategy"] = {"type": "sentry"}
        config["dataset"] = {
            "repo_id": "pepijn223/rollout_rebot_fineart_diagnostic",
            "root": str(dataset_root),
            "push_to_hub": False,
            "private": True,
            "streaming_encoding": True,
            "fps": 30,
        }
        config["action_trace_path"] = str(trace_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    print(f"Prepared {args.output} from {args.hardware_config}; no hardware connected.")


if __name__ == "__main__":
    main()
