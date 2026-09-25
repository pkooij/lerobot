"""Reload saved policies and run causal offline inference; never connect hardware."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from motion_metrics import chunk_motion, replan_motion

from lerobot.datasets.language_render import active_at
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi052.configuration_pi052 import PI052Config
from lerobot.policies.pi052.modeling_pi052 import PI052Policy
from lerobot.utils.constants import QUERY_KIND, QUERY_TEXT

ROOT = Path("/fsx/pepijn/rebot-fineart-20260925")
RENAME = {
    "observation.images.base": "observation.images.cam_high",
    "observation.images.left_wrist": "observation.images.cam_left_wrist",
    "observation.images.right_wrist": "observation.images.cam_right_wrist",
}


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="86031")
    parser.add_argument("--phase", choices=["smoke", "full"], default="full")
    parser.add_argument("--step", type=int, help="Saved update to evaluate; defaults to 20/10000 by phase")
    parser.add_argument(
        "--variants", nargs="+", choices=["subtask", "task_only"], default=["subtask", "task_only"]
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument(
        "--replan-after", type=int, help="Also predict again this many frames after each anchor"
    )
    args = parser.parse_args()
    if args.step is not None and args.step <= 0:
        parser.error("--step must be positive")
    if args.replan_after is not None and (args.replan_after <= 0 or args.load_only):
        parser.error("--replan-after must be positive and requires inference")
    assert args.device == "cuda" or args.load_only, "CPU gate checks reload only"
    split = json.loads((ROOT / "split.json").read_text())
    goals = json.loads((ROOT / "episode_goals.json").read_text())
    stats = json.loads((ROOT / "dataset_repaired/meta/stats.json").read_text())
    action_std = np.maximum(np.asarray(stats["action"]["std"]), 1e-6)
    steps = args.step if args.step is not None else (20 if args.phase == "smoke" else 10000)
    selection = (
        f"_{steps}_{'-'.join(args.variants)}" if args.step is not None or len(args.variants) != 2 else ""
    )
    if args.replan_after is not None:
        selection += f"_replan{args.replan_after}"
    destination = ROOT / f"reload_{args.phase}_{args.device}_{args.job}{selection}.json"
    reports = {}
    for variant in args.variants:
        checkpoint = ROOT / f"runs/{variant}_{args.phase}_{args.job}/checkpoints/{steps:06d}/pretrained_model"
        state = json.loads((checkpoint.parent / "training_state/training_step.json").read_text())
        assert state["step"] == steps and state["batch_size"] == 16
        assert state["dp_world_size"] == 4 and state["grad_accum_steps"] == 1
        digest = hashlib.sha256()
        with (checkpoint / "model.safetensors").open("rb") as stream:
            while block := stream.read(16 * 1024**2):
                digest.update(block)
        cfg = PI052Config.from_pretrained(checkpoint)
        if args.replan_after is not None and args.replan_after >= cfg.chunk_size:
            parser.error("--replan-after must be smaller than the saved action chunk size")
        cfg.device = args.device
        policy = PI052Policy.from_pretrained(checkpoint, config=cfg).to(args.device).eval()
        assert policy.supports_text_generation() == (variant == "subtask")
        pre, post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": args.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        samples = []
        for ep in split["dev"][:1] if args.load_only else split["dev"]:
            dataset = LeRobotDataset(
                "pepijn223/rebot_diverse_picking_100_annotated",
                root=ROOT / "dataset_repaired",
                revision="93c97807c46535745d0587d4296416bf2d4aa80d",
                episodes=[ep],
                video_backend="pyav",
                delta_timestamps={"action": [i / 30 for i in range(cfg.chunk_size)]},
            )
            anchors = [
                min(int(len(dataset) * fraction), len(dataset) - 1)
                for fraction in ([0.5] if args.load_only else [0.1, 0.5, 0.85])
            ]
            partners = {}
            if args.replan_after is not None:
                partners = {
                    index + args.replan_after: index
                    for index in anchors
                    if index + args.replan_after < len(dataset)
                }
            chunks = {}
            for index in sorted(set(anchors) | partners.keys()):
                item = dataset[index]
                # Only the anchor's images/state and the given episode goal enter the policy.
                # Annotation rows, future actions, timestamps and labels never enter inference.
                obs = {
                    RENAME.get(key, key): value.clone()
                    for key, value in item.items()
                    if key == "observation.state" or key in RENAME
                }
                assert set(obs) == {"observation.state", *RENAME.values()}
                goal = goals[str(ep)]
                instruction = goal
                row = {"episode": ep, "frame": int(item["frame_index"]), "goal": goal}
                seed = 1000 + ep * 100000 + int(item["frame_index"])
                torch.manual_seed(seed)
                policy.reset()
                row["seed"] = seed
                if variant == "subtask":
                    query = pre({**obs, QUERY_KIND: "next_subtask", QUERY_TEXT: goal})
                    assert "action" not in query and "language_persistent" not in query
                    if not args.load_only:
                        synchronize(args.device)
                        start = time.perf_counter()
                        with torch.inference_mode():
                            instruction = policy.generate_text(query)
                        synchronize(args.device)
                        row["planning_seconds"] = time.perf_counter() - start
                        assert isinstance(instruction, str) and instruction.strip(), "Empty generated subtask"
                        row["generated_subtask"] = instruction
                batch = pre({**obs, "task": instruction})
                assert "action" not in batch and "language_persistent" not in batch
                if not args.load_only:
                    torch.manual_seed(seed)
                    policy.reset()
                    synchronize(args.device)
                    start = time.perf_counter()
                    with torch.inference_mode():
                        predicted = post(policy.predict_action_chunk(batch))
                    synchronize(args.device)
                    row["action_seconds"] = time.perf_counter() - start
                    assert predicted.shape == (1, cfg.chunk_size, 14), predicted.shape
                    assert torch.isfinite(predicted).all()
                    predicted = predicted.squeeze(0).cpu().numpy()
                    target = item["action"].cpu().numpy()
                    valid = ~item["action_is_pad"].cpu().numpy().astype(bool)
                    assert valid.any()
                    row["motion"] = chunk_motion(
                        predicted, target, item["observation.state"].cpu().numpy(), valid
                    )
                    chunks[index] = {"predicted": predicted, "target": target, "valid": valid}
                    if index in partners:
                        row["replan_from_frame"] = partners[index]
                        row["replan_motion"] = replan_motion(
                            chunks[partners[index]], chunks[index], args.replan_after
                        )
                    row["action_normalized_mse"] = float(
                        np.square((predicted[valid] - target[valid]) / action_std).mean()
                    )
                    row["action_rmse_per_joint"] = np.sqrt(
                        np.square(predicted[valid] - target[valid]).mean(0)
                    ).tolist()
                    reference = active_at(
                        float(item["timestamp"]), persistent=item["language_persistent"], style="subtask"
                    )
                    row["reference_subtask_for_review_only"] = reference["content"] if reference else None
                    artifact = ROOT / "reload_checks" / f"{variant}_{args.job}_{steps}_{ep}_{index}.npz"
                    artifact.parent.mkdir(exist_ok=True)
                    np.savez_compressed(artifact, predicted=predicted, target=target, valid=valid)
                    row["prediction_artifact"] = str(artifact)
                samples.append(row)
                print(json.dumps({"variant": variant, **row}), flush=True)
            del dataset
        reports[variant] = {
            "checkpoint": str(checkpoint),
            "sha256": digest.hexdigest(),
            "steps": steps,
            "reload_passed": True,
            "load_only": args.load_only,
            "samples": samples,
        }
        del policy, pre, post, batch
        if variant == "subtask":
            del query
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    result = {
        "variants": reports,
        "inference_inputs": "current images/state and supplied task only",
        "episodes": split["dev"],
        "test_episodes_unused": split["test"],
        "motion_units": "Per-joint native dataset units; no clipping, filtering or unit conversion",
        "replan_after_frames": args.replan_after,
        "replan_limitation": (
            "Paired predictions use subsequent recorded observations, not states caused by predicted actions. "
            "Boundary metrics splice raw chunks at the requested offset; independently seeded action sampling "
            "and any generated subtask change both contribute. They do not measure closed-loop smoothness."
        ),
        "limitation": "Offline replay and reload checks; no physical success measurement.",
    }
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("SELECTED CHECKPOINT RELOAD CHECKS PASSED", str(destination), flush=True)


if __name__ == "__main__":
    main()
