# Record the right shoulder-lift issue

Run these commands yourself on Madeleine. Configuration preparation only reads
files; the rollout command connects and configures the hardware. `/start` begins
policy actions. Because the right shoulder has moved toward the table, start with
clearance and stop at the first unexpected motion. The 1° limit is a per-command
target offset, not a collision limit or a 1° total-motion limit. Do not increase it
to diagnose this issue.

```bash
set -e
cd "$HOME/lerobot-fineart"
git switch codex/rebot-pi052-matched-sft-20260925
git pull --ff-only
RUN_DIR=$(mktemp -d "$HOME/rebot-fineart-diagnostic.XXXXXXXX")
uv run python examples/rebot_fineart/prepare_rollout.py \
  --hardware-config auto --output "$RUN_DIR/rollout.json" \
  --record --max-relative-target 1 --duration 10 \
  --task "Use the left arm to reach toward the blue block."
uv run python -m lerobot.scripts.lerobot_rollout \
  --config_path="$RUN_DIR/rollout.json" \
  --policy.path=pepijn223/rebot-fineart-pi052-subtask-10k-20260925 \
  --policy.pretrained_revision=f3a585e670100c56bee954fce61c2db43bbb6f07
```

The helper prints which saved hardware config it copied. Ports, cameras, motor
IDs, calibration paths and gains remain those of that file. Inspect the generated
`rollout.json` before starting if the selected file is not the one you used.
Do not create the `dataset` directory yourself: LeRobot creates it.

At the interactive prompt, use `/start`, then `/stop` to finish. The run is limited
to ten seconds; after it ends, use `/stop` to finalize and disconnect. Do not use
`/reset` for this diagnostic: that command requests a return motion. Keep a fixed
instruction for this first recording so planner changes do not confound it.
If a different instruction caused the drift, supply that exact text with `--task`.
Stopping/disconnecting is not a guarantee of gravity support; support the arm as
appropriate for your setup.

Then summarize the trace locally:

```bash
uv run python examples/rebot_fineart/analyze_trace.py "$RUN_DIR/actions.jsonl"
printf '%s\n' "$RUN_DIR"
```

The directory contains:

- `dataset/`: native LeRobot state, requested-action and instruction rows, plus
  the three camera videos. Hub uploading is disabled.
- `actions.jsonl`: pre-send measured positions, requested targets, processed
  targets, the driver's returned targets after its clipping, and monotonic/wall
  timestamps. Failed sends are separate events. Observations also appear on ticks
  with no action. Trace files are exclusive-created and line-flushed.
- `action_summary.json`: per-joint motion, clipping and subsequent tracking errors.
- `rollout.json`: the exact saved deployment configuration.

The native dataset `action` column retains its existing requested-target meaning.
Use `send_result.sent` in the sidecar for targets returned by the ReBot driver.
These are **not** motor acknowledgements or measured achieved positions. The
driver reads feedback again when limiting relative targets, so its clipping basis
can be slightly newer than the recorded pre-send observation. Trace observation
timestamps mark availability to dispatch, not camera exposure or motor bus time.
JSONL writes add diagnostic overhead; timing is not a performance benchmark.

Interpretation:

- Requested targets drive downward and later measurements follow: inspect policy
  inputs, instruction, camera mapping and training distribution first.
- Returned targets differ greatly from requested ones: inspect clipping, soft
  limits and robot processors.
- Targets remain steady but measurements drift: investigate controller tracking,
  load, feedback quality and calibration. This alone does not identify a gain bug.
- Comparable physical poses have different joint coordinates in demonstrations
  and live data: investigate zeros, signs and physical arm assignments before tuning.

The ReBot MIT implementation sends position and velocity gains with zero
feedforward torque. That makes load-related tracking error plausible, but does
not establish that it caused this issue. Missing motor feedback is also currently
represented as zero by the driver; repeated exact zeros need scrutiny.

`joint_audit.py` is the separate GPU-only offline comparison against train/dev
demonstrations. It evaluates the uploaded subtask checkpoint with annotated
subtasks and episode goals, at three anchors per selected episode. Future actions
are used only for scoring. It does not connect a robot, tune gains, change zeros,
or evaluate physical success. The final test episodes remain unused.
