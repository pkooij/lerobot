"""Offline absolute-action discontinuities, in each joint's native dataset units."""

import numpy as np


def chunk_motion(predicted, target, state, valid):
    predicted, target, state = (np.asarray(x, dtype=np.float64) for x in (predicted, target, state))
    valid = np.asarray(valid, dtype=bool)
    if predicted.shape != target.shape or predicted.ndim != 2 or state.shape != predicted.shape[1:]:
        raise ValueError("Expected matching time-by-joint chunks and a joint state")
    if valid.shape != predicted.shape[:1] or not valid[0]:
        raise ValueError("Expected an unpadded anchor and one validity flag per timestep")
    if not all(np.isfinite(x).all() for x in (predicted, target, state)):
        raise ValueError("Nonfinite action or state")
    pairs = valid[1:] & valid[:-1]
    result = {}
    for name, actions in (("predicted", predicted), ("reference", target)):
        differences = np.diff(actions, axis=0)[pairs]
        result[name] = {
            "first_action_minus_state_per_joint": (actions[0] - state).tolist(),
            "adjacent_pairs": int(pairs.sum()),
            "max_abs_step_per_joint": np.abs(differences).max(0).tolist() if pairs.any() else None,
            "rms_step_per_joint": np.sqrt(np.square(differences).mean(0)).tolist() if pairs.any() else None,
        }
    return result


def replan_motion(previous, current, offset):
    """Compare two recorded-observation predictions, not a simulated robot rollout."""
    old = np.asarray(previous["predicted"], dtype=np.float64)
    new = np.asarray(current["predicted"], dtype=np.float64)
    old_valid, new_valid = (np.asarray(row["valid"], dtype=bool) for row in (previous, current))
    if old.shape != new.shape or not 0 < offset < len(old):
        raise ValueError("Replanning offset must lie inside matching action chunks")
    if not old_valid[offset - 1] or not new_valid[0]:
        raise ValueError("Boundary cannot use padded actions")
    overlap = old_valid[offset:] & new_valid[: len(old) - offset]
    result = {"offset_frames": offset, "overlap_frames": int(overlap.sum())}
    for name, key in (("predicted", "predicted"), ("reference", "target")):
        before, after = (np.asarray(row[key], dtype=np.float64) for row in (previous, current))
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError("Nonfinite action")
        result[f"{name}_boundary_delta_per_joint"] = (after[0] - before[offset - 1]).tolist()
        drift = (after[: len(before) - offset] - before[offset:])[overlap]
        result[f"{name}_overlap_rms_per_joint"] = (
            np.sqrt(np.square(drift).mean(0)).tolist() if overlap.any() else None
        )
    return result
