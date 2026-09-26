"""Opt-in diagnostic trace of requested and hardware-returned action targets."""

from __future__ import annotations

import json
import time
from pathlib import Path


class ActionTrace:
    """Line-flushed JSONL; an existing trace is never overwritten.

    Hardware-returned targets describe commands, not measured joint feedback or
    acknowledgement from the motor. Observations belong to the pre-send tick;
    compare subsequent observations to assess tracking.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._stream = None
        self._sequence = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", buffering=1)
        return self

    def __exit__(self, *_exc):
        self._stream.close()

    def write(self, event: str, **fields) -> int:
        sequence = self._sequence
        self._stream.write(
            json.dumps(
                {
                    "sequence": sequence,
                    "event": event,
                    "monotonic_ns": time.monotonic_ns(),
                    "wall_time_ns": time.time_ns(),
                    **fields,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self._sequence += 1
        return sequence


def joint_values(values: dict) -> dict[str, float]:
    """Exclude camera frames and copy scalars before a processor can mutate them."""
    return {key: float(value) for key, value in values.items() if key.endswith(".pos")}
