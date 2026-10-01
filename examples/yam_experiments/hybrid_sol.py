"""YAM hybrid pilot configuration and read-only Sol preflight."""

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

from .campaign import CUBE_INSTRUCTIONS, TASK
from .run import launcher_arguments


def hybrid_config():
    from lerobot.robots.bi_yam_follower import bi_yam_follower
    from lerobot.robots.bi_yam_follower.config_bi_yam_follower import JOINT_LIMITS, JOINT_NAMES
    from lerobot.rollout.end_effector import EndEffectorConfig
    from lerobot.rollout.hybrid import HybridConfig, InterventionLimit

    limits = {}
    end_effectors = {}
    model_path = str(Path(bi_yam_follower.__file__).parent / "assets/yam_linear.xml")
    for side in ("left", "right"):
        for name, (lower, upper) in zip(JOINT_NAMES, JOINT_LIMITS, strict=True):
            limits[f"{side}_{name}.pos"] = InterventionLimit(
                lower,
                upper,
                0.25,
                0.2,
                0.04,
                "Canonical YAM joint radians. Raw joint corrections forbidden; use end_effector mode or the VLA.",
            )
        limits[f"{side}_gripper.pos"] = InterventionLimit(
            0,
            1,
            1,
            2,
            0.08,
            f"{side} gripper measured normalized opening, 0 closed and 1 fully open. Continuous linear calibrated mapping.",
        )
        end_effectors[side] = EndEffectorConfig(
            model_path=model_path,
            site="grasp_site",
            joint_names=[f"joint{i + 1}" for i in range(6)],
            action_keys=[f"{side}_{name}.pos" for name in JOINT_NAMES],
            frame_description=(
                f"{side} YAM v1 arm's own fixed base/model frame; metres, quaternion wxyz. "
                "Base +Z is up for the upright mounting. X/Y are native model axes, NOT image right/left. "
                "No camera-to-base or inter-arm transform is calibrated. The I2RT linear_4310 grasp_site "
                "is the midpoint of the fingertips, independent of opening, 0.14465 m along gripper-body -Z. "
                "Its local +Z points from the wrist toward the fingertips. Use measured FK as the reference; "
                "do not guess table coordinates from pixels."
            ),
            position_tolerance_m=0.003,
            rotation_tolerance_rad=0.025,
        )
    return HybridConfig(limits=limits, end_effectors=end_effectors, policy_window_s=5, review_timeout_s=60)


def planner_config(log_path):
    from lerobot.rollout.planner import PlannerConfig

    return PlannerConfig(
        model_id="gpt-6.1-sol",
        api_mode="responses",
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        auto_serve=False,
        reasoning_effort="low",
        service_tier=os.environ.get("SOL_SERVICE_TIER", "fast"),
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


def checked_pose(config, pose):
    result = {}
    for key, limit in config.limits.items():
        value = float(pose[key])
        if (
            not math.isfinite(value)
            or not limit.minimum - limit.tolerance <= value <= limit.maximum + limit.tolerance
        ):
            raise ValueError(f"Measured position outside feedback tolerance: {key}")
        result[key] = max(limit.minimum, min(limit.maximum, value))
    return result


def ik_check(config, pose):
    """No robot calls: verify local FK/IK round trips near the measured state."""
    import numpy as np

    from lerobot.rollout.end_effector import EndEffectorKinematics, parse_pose
    from lerobot.rollout.hybrid import PlannerDecision

    pose = checked_pose(config, pose)
    result = {}
    for name, ee in config.end_effectors.items():
        solver = EndEffectorKinematics(ee)
        current = solver.forward(pose)
        checks = []
        for key in ee.action_keys:
            changed = dict(pose)
            changed[key] += 0.01 if pose[key] + 0.01 <= config.limits[key].maximum else -0.01
            target = solver.forward(changed)
            decision = PlannerDecision(
                "end_effector", "Local IK check", "No execution", "", {}, 2, {name: target}
            )
            started = time.monotonic()
            solved = decision.resolve_motion(config, pose, {name: solver})
            elapsed_ms = (time.monotonic() - started) * 1000
            desired_p, desired_r = parse_pose(target)
            solved_p, solved_r = parse_pose(solver.forward(solved))
            checks.append(
                {
                    "joint_probe": key,
                    "position_error_m": float(np.linalg.norm(desired_p - solved_p)),
                    "rotation_error_rad": float((desired_r * solved_r.inv()).magnitude()),
                    "solve_ms": round(elapsed_ms, 2),
                }
            )
        result[name] = {"measured_fk": current, "round_trips": checks}
    return {"end_effectors": result, "executed": False, "physical_frame_calibration_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--flags", action="store_true")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--ik-only", action="store_true")
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
        try:
            capture(args.launcher, args.snapshot)
        except Exception as exc:
            print(f"Read-only hardware check failed: {exc}")
            raise SystemExit(1) from None
        return
    if args.ik_only:
        saved = json.loads((args.snapshot / "pose.json").read_text())
        report = ik_check(config, saved["pose"])
        (args.snapshot / "ik-check.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in this terminal; do not put it in CLI flags")
    import numpy as np
    from PIL import Image

    from lerobot.rollout.hybrid import HybridPlanner
    from lerobot.rollout.inference import PolicyQuery, QueryKind

    if args.api_only:
        from lerobot.rollout.planner import VlmPlanner

        obs = {
            name: np.array(Image.open(args.snapshot / f"current_{name}.png").convert("RGB"))
            for name in ("top", "left", "right")
        }
        planner = VlmPlanner(planner_config(args.snapshot / "sol-api-check.jsonl"), "bi_yam_follower")
        answer = planner(
            obs,
            PolicyQuery(
                QueryKind.VQA,
                "These are archived camera frames, not a live robot. Briefly describe visible cubes and the bin in each view. Reply as JSON with an answer field.",
            ),
            TASK,
        )
        print(
            json.dumps(
                {
                    "model": "gpt-6.1-sol",
                    "answer": answer,
                    "archived_frames": str(args.snapshot),
                    "robot_connected": False,
                    "executed": False,
                },
                indent=2,
            )
        )
        return
    saved = json.loads((args.snapshot / "pose.json").read_text())
    obs = checked_pose(config, saved["pose"])
    for name in ("top", "left", "right"):
        obs[name] = np.array(Image.open(args.snapshot / f"current_{name}.png").convert("RGB"))
    planner = HybridPlanner(
        planner_config(args.snapshot / "sol-preflight.jsonl"), "bi_yam_follower", hybrid=config
    )
    started = time.monotonic()
    proposal = planner(obs, PolicyQuery(QueryKind.NEXT_SUBTASK, TASK), TASK)
    resolved = None
    if proposal.mode in {"intervention", "end_effector"}:
        resolved = proposal.resolve_motion(config, checked_pose(config, saved["pose"]), planner.kinematics)
    print(
        json.dumps(
            {
                "model": "gpt-6.1-sol",
                "decision": asdict(proposal),
                "resolved_joint_targets": resolved,
                "latency_s": round(time.monotonic() - started, 2),
                "snapshot_age_s": round(time.time() - saved["captured_at"], 2),
                "executed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
