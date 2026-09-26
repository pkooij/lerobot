"""Compare recorded ReBot joints and offline subtask-policy targets; no robot I/O."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.datasets.language_render import active_at
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi052.configuration_pi052 import PI052Config
from lerobot.policies.pi052.modeling_pi052 import PI052Policy

ROOT = Path("/fsx/pepijn/rebot-fineart-20260925")
RENAME = {
    "observation.images.base": "observation.images.cam_high",
    "observation.images.left_wrist": "observation.images.cam_left_wrist",
    "observation.images.right_wrist": "observation.images.cam_right_wrist",
}


def summarize(values):
    return {
        "min": values.min(0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "max": values.max(0).tolist(),
    }


def main():
    root = ROOT / "dataset_repaired"
    split = json.loads((ROOT / "split.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    names = info["features"]["action"]["names"]
    assert names == info["features"]["observation.state"]["names"]
    left = [i for i, name in enumerate(names) if name.startswith("left_") and "gripper" not in name]
    right = [i for i, name in enumerate(names) if name.startswith("right_") and "gripper" not in name]
    columns = [
        "episode_index",
        "frame_index",
        "timestamp",
        "observation.state",
        "action",
        "language_persistent",
    ]
    table = pa.concat_tables(
        [pq.read_table(p, columns=columns) for p in sorted((root / "data").rglob("*.parquet"))]
    )
    table = table.sort_by([("episode_index", "ascending"), ("frame_index", "ascending")])
    # Do not use the final test split in exploratory diagnosis.
    table = table.filter(
        pa.compute.is_in(table["episode_index"], value_set=pa.array(split["train"] + split["dev"]))
    )
    episodes = np.asarray(table["episode_index"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    report = {
        "joint_names": names,
        "units": "degrees, including the ReBot gripper motor",
        "test_episodes_unused": split["test"],
    }
    report["data"] = {}
    for part in ("train", "dev"):
        mask = np.isin(episodes, split[part])
        report["data"][part] = {
            "frames": int(mask.sum()),
            "state": summarize(states[mask]),
            "action": summarize(actions[mask]),
            "action_minus_state": summarize(actions[mask] - states[mask]),
        }
    report["episode_activity"] = []
    for ep in sorted(set(episodes.tolist())):
        mask = episodes == ep
        delta = np.diff(states[mask], axis=0)
        span = np.ptp(states[mask], axis=0)
        subtable = table.filter(pa.compute.equal(table["episode_index"], int(ep)))
        mentions = {"left": 0, "right": 0, "both": 0, "neither": 0}
        for row in subtable.select(["timestamp", "language_persistent"]).to_pylist():
            active = active_at(row["timestamp"], persistent=row["language_persistent"], style="subtask")
            text = (active or {}).get("content", "").lower()
            mentions_left, mentions_right = "left" in text, "right" in text
            mention = (
                "both"
                if mentions_left and mentions_right
                else "left"
                if mentions_left
                else "right"
                if mentions_right
                else "neither"
            )
            mentions[mention] += 1
        report["episode_activity"].append(
            {
                "episode": ep,
                "split": "train" if ep in split["train"] else "dev",
                "left_mean_joint_span_deg": float(span[left].mean()),
                "right_mean_joint_span_deg": float(span[right].mean()),
                "left_mean_step_deg": float(abs(delta[:, left]).mean()),
                "right_mean_step_deg": float(abs(delta[:, right]).mean()),
                "subtask_frame_mentions": mentions,
            }
        )
    output = ROOT / "right_shoulder_joint_audit.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Dataset audit saved", flush=True)

    checkpoint = ROOT / "runs/subtask_full_86031/checkpoints/010000/pretrained_model"
    cfg = PI052Config.from_pretrained(checkpoint)
    assert cfg.action_feature_names == names
    cfg.device = "cuda"
    policy = PI052Policy.from_pretrained(checkpoint, config=cfg).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    report["checkpoint"] = str(checkpoint)
    report["samples"] = []
    goals = json.loads((ROOT / "episode_goals.json").read_text())
    for ep in [0, 1, 2, *split["dev"]]:
        dataset = LeRobotDataset(
            "pepijn223/rebot_diverse_picking_100_annotated",
            root=root,
            episodes=[ep],
            video_backend="pyav",
            delta_timestamps={"action": [i / 30 for i in range(cfg.chunk_size)]},
        )
        for fraction in [0.1, 0.5, 0.85]:
            index = min(int(len(dataset) * fraction), len(dataset) - 1)
            item = dataset[index]
            obs = {
                RENAME.get(k, k): v.clone()
                for k, v in item.items()
                if k == "observation.state" or k in RENAME
            }
            reference = active_at(
                float(item["timestamp"]), persistent=item["language_persistent"], style="subtask"
            )["content"]
            state = item["observation.state"].numpy()
            target = item["action"].numpy()
            valid = ~item["action_is_pad"].numpy().astype(bool)
            # Paired language conditions share noise and observation. Future actions are metrics only.
            for mode, instruction in [("annotated_subtask", reference), ("high_level_task", goals[str(ep)])]:
                torch.manual_seed(1000 + ep * 100000 + index)
                policy.reset()
                with torch.inference_mode():
                    predicted = (
                        post(policy.predict_action_chunk(pre({**obs, "task": instruction})))
                        .squeeze(0)
                        .numpy()
                    )
                assert np.isfinite(predicted).all()
                row = {
                    "episode": ep,
                    "split": "train" if ep in split["train"] else "dev",
                    "frame": int(item["frame_index"]),
                    "mode": mode,
                    "instruction": instruction,
                    "state": state.tolist(),
                    "target_first": target[0].tolist(),
                    "predicted_first": predicted[0].tolist(),
                    "rmse_per_joint": np.sqrt(np.square(predicted[valid] - target[valid]).mean(0)).tolist(),
                    "hold_baseline_rmse_per_joint": np.sqrt(
                        np.square(state - target[valid]).mean(0)
                    ).tolist(),
                }
                artifact = ROOT / "joint_audit" / f"ep{ep}_frame{index}_{mode}.npz"
                artifact.parent.mkdir(exist_ok=True)
                np.savez_compressed(artifact, state=state, target=target, predicted=predicted, valid=valid)
                row["artifact"] = str(artifact)
                report["samples"].append(row)
                print(json.dumps(row), flush=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["complete"] = True
    report["limitation"] = (
        "Sparse offline imitation check on recorded observations; no live calibration, physical tracking or task success evaluated."
    )
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
