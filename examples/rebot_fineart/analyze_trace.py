"""Summarize a local rollout command trace without loading models or hardware."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def rms(values):
    return math.sqrt(mean(value * value for value in values)) if values else None


def analyze(rows):
    observations = {row["sequence"]: row for row in rows if row["event"] == "observation"}
    attempts = {row["sequence"]: row for row in rows if row["event"] == "send_attempt"}
    results = [row for row in rows if row["event"] == "send_result" and row["sent"] is not None]
    ordered_obs = list(observations.values())
    next_observation = {}
    cursor = 0
    for result in results:
        while cursor < len(ordered_obs) and ordered_obs[cursor]["monotonic_ns"] <= result["monotonic_ns"]:
            cursor += 1
        if cursor < len(ordered_obs):
            next_observation[result["sequence"]] = ordered_obs[cursor]
    joints = sorted({key for obs in ordered_obs for key in obs["measured"]})
    report = {
        "observations": len(observations),
        "send_attempts": len(attempts),
        "successful_returns": len(results),
        "send_errors": sum(row["event"] == "send_error" for row in rows),
        "units": "Native joint units (degrees for ReBot)",
        "limitation": "Sent targets are driver-returned commands, not motor acknowledgement. Next-observation errors include tracking lag and variable observation intervals; they cannot identify PID or calibration faults alone.",
        "joints": {},
    }
    for joint in joints:
        measured = [obs["measured"][joint] for obs in ordered_obs if joint in obs["measured"]]
        requested_delta, sent_delta, clipped, tracking, delays = [], [], [], [], []
        for result in results:
            attempt = attempts[result["attempt_sequence"]]
            obs = observations[attempt["observation_sequence"]]
            if (
                joint not in result["sent"]
                or joint not in obs["measured"]
                or joint not in attempt["requested"]
            ):
                continue
            sent = result["sent"][joint]
            requested_delta.append(attempt["requested"][joint] - obs["measured"][joint])
            sent_delta.append(sent - obs["measured"][joint])
            clipped.append(abs(sent - attempt["processed"][joint]) > 1e-5)
            following = next_observation.get(result["sequence"])
            if following is not None and joint in following["measured"]:
                tracking.append(following["measured"][joint] - sent)
                delays.append((following["monotonic_ns"] - result["monotonic_ns"]) / 1e9)
        report["joints"][joint] = {
            "start": measured[0],
            "end": measured[-1],
            "min": min(measured),
            "max": max(measured),
            "requested_minus_measured_mean": mean(requested_delta) if requested_delta else None,
            "sent_minus_measured_mean": mean(sent_delta) if sent_delta else None,
            "driver_clipped_fraction": mean(clipped) if clipped else None,
            "next_observation_minus_sent_rmse": rms(tracking),
            "next_observation_delay_mean_s": mean(delays) if delays else None,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    report = analyze(rows)
    output = args.trace.with_name("action_summary.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "right_shoulder_lift.pos": report["joints"].get("right_shoulder_lift.pos"),
                "report": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
