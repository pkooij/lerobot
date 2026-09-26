"""Deterministic temporal coverage and paired checkpoint comparisons."""

import math
from collections import defaultdict


def stage_indices(length: int, fps: float, timestamps: list[float], count: int) -> list[int]:
    """Spread frames across annotated stages, including both sides of transitions."""
    if length <= 0 or fps <= 0 or count <= 0:
        raise ValueError("Positive episode length, fps and count required")
    count = min(count, length)
    boundaries = sorted({0, length, *(min(length, max(0, round(t * fps))) for t in timestamps)})
    spans = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    selected = {i for start, end in spans for i in (start, end - 1)}
    if len(selected) > count:
        ordered = sorted(selected)
        return [ordered[(2 * i + 1) * len(ordered) // (2 * count)] for i in range(count)]
    # Equal opportunity for short and long stages, rather than only long pauses.
    per_stage = 1
    while len(selected) < count:
        for start, end in spans:
            for i in range(per_stage):
                selected.add(start + (2 * i + 1) * (end - start) // (2 * per_stage))
                if len(selected) == count:
                    return sorted(selected)
        per_stage += 1
    return sorted(selected)


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        for metric, value in row["losses"].items():
            if metric == "loss":
                continue  # Mixed totals have different supervision weights.
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite {metric}")
            groups[(row["mode"], metric)].append(value)
    return {
        f"{mode}/{metric}": {"n": len(values), "mean": sum(values) / len(values)}
        for (mode, metric), values in sorted(groups.items())
    }


def paired_deltas(current: list[dict], baseline: list[dict]) -> dict:
    """Paired means; negative means lower loss. Require identical evaluated cases."""

    def key(row):
        return row["episode"], row["frame"], row["mode"]

    previous = {key(row): row for row in baseline}
    shared_modes = {r["mode"] for r in current} & {r["mode"] for r in baseline}
    assert {key(r) for r in current if r["mode"] in shared_modes} == {
        key(r) for r in baseline if r["mode"] in shared_modes
    }, "Paired comparisons require identical frame/mode coverage"
    deltas = []
    for row in current:
        if row["mode"] not in shared_modes:
            continue
        old = previous[key(row)]
        assert row["seed"] == old["seed"] and row["prompt"] == old["prompt"]
        deltas.append(
            {
                **row,
                "losses": {k: v - old["losses"][k] for k, v in row["losses"].items() if k in old["losses"]},
            }
        )
    return summarize(deltas)
