"""Single-arm Hy-VLA inference using the saved gripper-centre pose convention."""

import json
import logging
import time

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.lerobot_types import TransitionKey
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.hy_vla.modeling_hy_vla import HyVLAPolicy
from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
from lerobot.utils.device_utils import auto_select_torch_device

from .config_openarm_gripette_follower import OpenArmGripetteFollowerConfig
from .grabette_processors import (
    GRIPPER_NAMES,
    POSITION_NAMES,
    ROBOT_JOINTS,
    ROTATION_NAMES,
    GrabetteRelativePoseToJoints,
)
from .gripette_compat import DatasetGraspProjection, resolve_openarm_model
from .manual_start import prepare_manual_start
from .openarm_gripette_follower import OpenArmGripetteFollower
from .prediction_logger import PredictionLogger

logger = logging.getLogger(__name__)
ACTION_NAMES = POSITION_NAMES + ROTATION_NAMES + GRIPPER_NAMES


def make_runtime_observation(image, state, pose, task):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 HWC camera frame")
    state = np.asarray(state, dtype=np.float32)
    pose = np.asarray(pose, dtype=np.float32)
    if state.shape != (2,) or pose.shape != (4, 4):
        raise ValueError("Expected measured [strategy, closure] and a 4x4 gripper-centre pose")
    if not np.isfinite(state).all() or not np.isfinite(pose).all():
        raise ValueError("Measured state and gripper pose must be finite")
    return {
        "observation.images.grabette": torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255,
        "observation.state": torch.from_numpy(state.copy()),
        "observation.gripper_pose": torch.from_numpy(pose.copy()),
        "task": task,
    }


def load_runtime(config, device):
    config.device = device
    policy = HyVLAPolicy.from_pretrained(config.pretrained_path, config=config, strict=True)
    pre, post = load_processors(config, device)
    return policy, pre, post


def load_processors(config, device):
    return make_pre_post_processors(
        config,
        pretrained_path=config.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"hy_vla_decode_v1": {"execution_horizon": config.execution_horizon}},
    )


def make_kinematics(args, joint_names):
    kinematics = RobotKinematics(
        str(resolve_openarm_model(args.urdf)),
        target_frame_name="gripper_center",
        joint_names=joint_names,
    )
    for name in ("proximal", "distal", "r_wrist_roll_mimic"):
        if name in kinematics.robot.joint_names():
            kinematics.solver.mask_dof(name)
    kinematics.solver.enable_joint_limits(True)
    kinematics.solver.add_regularization_task(1e-4)
    return kinematics


def offline_check(args, config, policy, pre, post, kinematics, device):
    samples = np.load(args.offline_observation, allow_pickle=False)
    images, states, poses = samples["images"], samples["state"], samples["gripper_pose"]
    actions, durations = [], []
    torch.manual_seed(4196)
    with torch.inference_mode():
        for index in range(len(states)):
            observation = make_runtime_observation(
                images[index],
                states[index],
                poses[index],
                args.task,
            )
            start = time.perf_counter()
            command = post(policy.select_action(pre(observation))).cpu().float().numpy()[0]
            durations.append(time.perf_counter() - start)
            if command.shape != (11,) or not np.isfinite(command).all():
                raise RuntimeError("Invalid decoded model action")
            actions.append(command)
    q = np.zeros(7)
    pose = kinematics.forward_kinematics(q).copy()
    solved = kinematics.inverse_kinematics(q, pose, max_iters=30)
    actual = kinematics.forward_kinematics(solved)
    error = float(np.abs(actual - pose).max())
    if not np.isfinite(solved).all() or error > 1e-3:
        raise RuntimeError(f"URDF gripper-centre FK/IK roundtrip failed: {error}")
    report = {
        "status": "passed",
        "device": device,
        "actions": len(actions),
        "execution_horizon": config.execution_horizon,
        "physical_horizon": config.physical_action_horizon,
        "replanning_calls": durations[:: config.execution_horizon],
        "max_queued_action_s": max(
            (t for i, t in enumerate(durations) if i % config.execution_horizon), default=0.0
        ),
        "fk_ik_max_matrix_error": error,
        "target_frame": "gripper_center",
        "hardware_connected": False,
        "limits": "Recorded camera/state inference plus a synthetic URDF FK/IK roundtrip. No physical task-success or calibrated joint-zero validation.",
    }
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "offline-check.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.log_dir / "offline-actions.npz", actions=np.stack(actions), durations=durations)
    print(json.dumps(report, indent=2), flush=True)


def run_hy_vla_rollout(args, config, joint_names):
    if args.fps <= 0 or args.duration <= 0 or args.reset_time < 0 or args.num_episodes < 1:
        raise ValueError("Require positive fps, duration and episode count, and nonnegative reset time")
    if args.interpolation_multiplier != 1:
        raise ValueError("Hy-VLA currently requires --interpolation_multiplier 1")
    if args.fps != 50:
        logger.warning("This checkpoint was trained at 50 Hz; %.1f Hz changes playback timing", args.fps)
    device = args.device or auto_select_torch_device().type
    policy, pre, post = load_runtime(config, device)
    kinematics = make_kinematics(args, joint_names)
    if args.offline_observation:
        offline_check(args, config, policy, pre, post, kinematics, device)
        return
    if args.display_data:
        raise ValueError(
            "Hy-VLA writes JSONL predictions; Rerun display is currently available in the ACT path"
        )
    image_key, image_feature = next(iter(config.image_features.items()))
    camera = image_key.removeprefix("observation.images.")
    if image_key != "observation.images.grabette":
        raise ValueError("Expected the trained observation.images.grabette camera")
    robot = OpenArmGripetteFollower(
        OpenArmGripetteFollowerConfig(
            id=args.id,
            passive=args.dry_run,
            arm_config=OpenArmFollowerConfigBase(
                port=args.port, side="right", max_relative_target=8.0, use_can_fd=False
            ),
            gripette_addr=args.gripette,
            gripper_camera_name=camera,
            gripper_camera_height=image_feature.shape[1],
            gripper_camera_width=image_feature.shape[2],
        )
    )
    projection = DatasetGraspProjection()
    signs = np.asarray(args.joint_signs)
    action_step = GrabetteRelativePoseToJoints(
        kinematics=kinematics,
        joint_signs=tuple(signs),
        max_joint_step=args.max_joint_step,
        pose_mode="absolute",
        projection=projection,
        resync_on_chunk_start=args.dry_run,
    )
    predictions = PredictionLogger(args.log_dir)
    dataset = None
    robot.connect()
    try:
        if args.record_repo_id:
            features = {
                "observation.state": {"dtype": "float32", "shape": (2,), "names": GRIPPER_NAMES},
                "observation.gripper_pose": {"dtype": "float32", "shape": (4, 4), "names": None},
                image_key: {
                    "dtype": "video",
                    "shape": (image_feature.shape[1], image_feature.shape[2], 3),
                    "names": ["height", "width", "channels"],
                },
                "action": {"dtype": "float32", "shape": (11,), "names": ACTION_NAMES},
                "action.commanded_joints": {
                    "dtype": "float32",
                    "shape": (9,),
                    "names": ROBOT_JOINTS + ["proximal", "distal"],
                },
                "action.was_sent": {"dtype": "bool", "shape": (1,), "names": None},
            }
            record_root = (
                args.record_root
                or args.log_dir / f"{args.record_repo_id.split('/')[-1]}-{predictions.path.stem}"
            )
            dataset = LeRobotDataset.create(
                args.record_repo_id,
                fps=int(args.fps),
                features=features,
                root=record_root,
                robot_type="openarm_gripette_follower",
                video_backend="pyav",
                image_writer_threads=2,
            )
        for episode in range(args.num_episodes):
            if not args.dry_run:
                prepare_manual_start(robot, signs)
                if args.initial_gripper_strategy is not None or args.initial_gripper_closure is not None:
                    measured = robot.get_observation()
                    strategy, closure = projection.encode(measured["proximal.pos"], measured["distal.pos"])
                    strategy = (
                        strategy if args.initial_gripper_strategy is None else args.initial_gripper_strategy
                    )
                    closure = (
                        closure if args.initial_gripper_closure is None else args.initial_gripper_closure
                    )
                    proximal, distal = projection.decode(strategy, closure)
                    robot.send_action({"proximal.pos": proximal, "distal.pos": distal})
            input(
                f"Episode {episode + 1}: press ENTER for {'passive dry run' if args.dry_run else 'policy execution'}..."
            )
            policy.reset()
            pre.reset()
            post.reset()
            action_step.reset()
            started = time.monotonic()
            tick = 0
            while time.monotonic() - started < args.duration:
                begin = time.monotonic()
                raw = robot.get_observation()
                q = signs * np.array([raw[f"{joint}.pos"] for joint in ROBOT_JOINTS])
                pose = kinematics.forward_kinematics(q).copy()
                state = np.array(projection.encode(raw["proximal.pos"], raw["distal.pos"]), dtype=np.float32)
                observation = make_runtime_observation(raw[camera], state, pose, args.task)
                with torch.inference_mode():
                    native = post(policy.select_action(pre(observation))).cpu().float().numpy()[0]
                fields = dict(zip(ACTION_NAMES, native.tolist(), strict=True))
                command = action_step({TransitionKey.ACTION: fields, TransitionKey.OBSERVATION: raw})[
                    TransitionKey.ACTION
                ]
                if not args.dry_run:
                    command = robot.send_action(command)
                elapsed = time.monotonic() - begin
                predictions.write(
                    {
                        "episode": episode,
                        "tick": tick,
                        "absolute_gripper_pose_action": native,
                        "motor_targets": command,
                        "sent": not args.dry_run,
                        "step_s": elapsed,
                    }
                )
                if dataset is not None:
                    dataset.add_frame(
                        {
                            "observation.state": state,
                            "observation.gripper_pose": pose.astype(np.float32),
                            image_key: raw[camera],
                            "action": native,
                            "action.commanded_joints": np.array(
                                [command[f"{name}.pos"] for name in ROBOT_JOINTS + ["proximal", "distal"]],
                                dtype=np.float32,
                            ),
                            "action.was_sent": np.array([not args.dry_run], dtype=bool),
                            "task": args.task,
                        }
                    )
                if elapsed > 1 / args.fps:
                    logger.warning(
                        "Policy/IK step took %.3fs; requested period is %.3fs", elapsed, 1 / args.fps
                    )
                time.sleep(max(0, 1 / args.fps - elapsed))
                tick += 1
            if dataset is not None:
                dataset.save_episode()
            if episode + 1 < args.num_episodes:
                input("Support the arm and press ENTER to begin the reset interval...")
                time.sleep(args.reset_time)
        if dataset is not None and args.push_to_hub:
            dataset.finalize()
            dataset.push_to_hub()
    finally:
        try:
            predictions.close()
            if dataset is not None:
                dataset.finalize()
        finally:
            try:
                if not args.dry_run:
                    input("Support the arm, then press ENTER to disable torque and disconnect...")
            finally:
                robot.disconnect()
