"""Pure helpers for balanced development diagnostics."""

import math


def episode_indices(length: int, count: int) -> list[int]:
    """Choose unique bin midpoints across an episode, including short episodes."""
    if length <= 0 or count <= 0:
        raise ValueError("Episode length and sample count must be positive")
    count = min(length, count)
    return [(2 * i + 1) * length // (2 * count) for i in range(count)]


def component_summary(rows: list[dict]) -> dict:
    """Average per-frame losses only where the corresponding objective is active."""
    result = {}
    for name, branch in (("flow_loss", "action"), ("fast_action_loss", "action"), ("text_loss", "text")):
        active = [row for row in rows if row["branch"] == branch]
        values = [row["losses"][name] for row in active]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"Nonfinite {name}")
        result[name] = {
            "frames": len(values),
            "mean_per_frame": sum(values) / len(values) if values else None,
        }
    return result
