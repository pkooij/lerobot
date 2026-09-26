# ReBot target rate limit for the manual trial

`max_relative_target` runs on every `send_action`, not once per model chunk.
It limits the target relative to the **measured joint position**. If the joint
stays at 0°, a 1° setting continues to send at most 1°; it does not accumulate
targets 1°, 2°, 3°.

The optional new settings use the previous successfully sent target instead:

- `max_target_velocity_deg_s: 30.0`: at most 30°/s of target-reference motion.
- `max_target_step_deg: 1.0`: at most 1° in any one send, including after pauses.
- `max_relative_target: 5.0`: an independent bound on target minus measured position.

At 30 Hz, this allows nominally 1° per tick. Faster ticks get proportionally
smaller steps. Slow inference never accumulates more than the one-send limit.
The first command starts at the measured pose. A stationary joint can therefore
receive targets 0°, 1°, 2°, 3°, 4°, 5°, but the tracking-error bound prevents the
reference continuing toward 100° while the joint is stuck.

All three constraints (target rate, tracking error, soft joint limits) are applied
together. If feedback moves so far that they cannot all be satisfied, dispatch
raises before sending that arm's targets. Missing/nonfinite feedback also fails.
A transport exception requires reconnecting before further rate-limited commands,
because some motors may have received a partial command. A successful return is
still not an acknowledgement from the motor, and cached feedback freshness is not
verified by this limiter.

These are limits on the **commanded reference**, not guarantees of physical joint
speed, torque, table clearance or task success. More position error permits more
MIT controller effort than the previous 1° feedback-relative setting. Gains,
control modes and calibration are unchanged. Keep the first trial short and stop
on unexpected motion.

Prepare a new recording on Madeleine (preparation itself does not connect hardware):

```bash
cd ~/lerobot-fineart
RUN_DIR=$(mktemp -d "$HOME/rebot-fineart-rate30.XXXXXXXX")
~/.local/bin/uv run --no-sync python examples/rebot_fineart/prepare_rollout.py \
  --hardware-config /home/madeleine/rebot-camera-check.ZrFQCWyD/hardware_verified.json \
  --output "$RUN_DIR/rollout.json" --record --duration 10 \
  --max-target-velocity 30 --max-relative-target 5 \
  --task "Use the left arm to reach toward the blue block."
~/.local/bin/uv run --no-sync python -m lerobot.scripts.lerobot_rollout \
  --config_path="$RUN_DIR/rollout.json" \
  --policy.path=pepijn223/rebot-fineart-pi052-subtask-10k-20260925 \
  --policy.pretrained_revision=f3a585e670100c56bee954fce61c2db43bbb6f07
```

Use `/start` followed by Enter, then `/stop` followed by Enter to finish. Automatic
return motion is disabled. The same recorded-action trace captures the final
rate-limited, tracking-limited targets. Each retry needs a fresh recording directory.

`--max-target-velocity 60` prepares the alternative 2°-per-send / 60°/s target
setting, while preserving the 5° tracking-error bound. Start with the 30°/s setup;
the higher setting is not needed just to distinguish rate limiting from the old
measurement-relative clamp.
