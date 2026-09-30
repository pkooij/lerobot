"""YAM hybrid pilot configuration and read-only Sol preflight."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from .campaign import CUBE_INSTRUCTIONS, TASK
from .run import launcher_arguments


def hybrid_config():
    from lerobot.robots.bi_yam_follower.config_bi_yam_follower import JOINT_LIMITS, JOINT_NAMES
    from lerobot.rollout.hybrid import HybridConfig, InterventionLimit

    limits = {}
    for side in ("left", "right"):
        for name, (lower, upper) in zip(JOINT_NAMES, JOINT_LIMITS, strict=True):
            limits[f"{side}_{name}.pos"] = InterventionLimit(
                lower,
                upper,
                0.0,
                0.1,
                0.04,
                "Measured absolute joint angle in radians. Direct correction disabled (max_delta=0); use the VLA.",
            )
        limits[f"{side}_gripper.pos"] = InterventionLimit(
            0,
            1,
            1,
            2,
            0.08,
            f"{side} gripper measured normalized opening, 0 closed and 1 fully open. Continuous linear calibrated mapping.",
        )
    return HybridConfig(limits=limits, policy_window_s=5, review_timeout_s=60)


def planner_config(log_path):
    from lerobot.rollout.planner import PlannerConfig

    return PlannerConfig(
        model_id="gpt-6.1-sol",
        api_mode="responses",
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        auto_serve=False,
        reasoning_effort="low",
        max_new_tokens=2048,
        request_timeout_s=45,
        request_max_retries=0,
        history=2,
        instructions=CUBE_INSTRUCTIONS,
        log_path=str(log_path),
    )


def capture(launcher: Path, destination: Path):
    import draccus
    import numpy as np
    from PIL import Image

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.robots.bi_yam_follower import BiYamFollower, BiYamFollowerConfig

    robot_fields = {}
    for token in launcher_arguments(launcher):
        key, value = token[2:].split("=", 1)
        if not key.startswith("robot.") or key == "robot.type":
            continue
        parent = robot_fields
        parts = key.split(".")[1:]
        for part in parts[:-1]:
            parent = parent.setdefault(part, {})
        try:
            parent[parts[-1]] = json.loads(value)
        except json.JSONDecodeError:
            parent[parts[-1]] = value
    robot_fields["read_only"] = True
    robot_fields["defer_torque_enable"] = True
    config = draccus.decode(BiYamFollowerConfig, robot_fields)
    robot = BiYamFollower(config)
    try:
        robot.connect()
        observation = robot.get_observation()
        destination.mkdir(parents=True, exist_ok=True)
        pose = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                Image.fromarray(value).save(destination / f"current_{key}.png")
            else:
                pose[key] = float(value)
        (destination / "pose.json").write_text(
            json.dumps({"captured_at": time.time(), "pose": pose}, indent=2)
        )
        print(f"Read-only capture saved to {destination}; no torque enabled or motion requested.")
    finally:
        if robot.is_connected:
            robot.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--flags", action="store_true")
    args = parser.parse_args()
    config = hybrid_config()
    if args.flags:
        # One complete JSON argument for draccus; contains no credentials.
        print(json.dumps(asdict(config)))
        return
    if args.snapshot is None:
        parser.error("--snapshot is required")
    if args.capture_only:
        if args.launcher is None:
            parser.error("--launcher is required for capture")
        capture(args.launcher, args.snapshot)
        return
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in this terminal; do not put it in CLI flags")
    import numpy as np
    from PIL import Image

    from lerobot.rollout.hybrid import HybridPlanner
    from lerobot.rollout.inference import PolicyQuery, QueryKind

    saved = json.loads((args.snapshot / "pose.json").read_text())
    obs = dict(saved["pose"])
    for name in ("top", "left", "right"):
        obs[name] = np.array(Image.open(args.snapshot / f"current_{name}.png").convert("RGB"))
    planner = HybridPlanner(
        planner_config(args.snapshot / "sol-preflight.jsonl"), "bi_yam_follower", hybrid=config
    )
    started = time.monotonic()
    proposal = planner(obs, PolicyQuery(QueryKind.NEXT_SUBTASK, TASK), TASK)
    if proposal.mode == "intervention":
        proposal.validate_motion(config, saved["pose"])
    print(
        json.dumps(
            {
                "model": "gpt-6.1-sol",
                "decision": asdict(proposal),
                "latency_s": round(time.monotonic() - started, 2),
                "snapshot_age_s": round(time.time() - saved["captured_at"], 2),
                "executed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
