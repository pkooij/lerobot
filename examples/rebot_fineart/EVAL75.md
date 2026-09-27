# Five-object evaluation of the augmented20k checkpoints

The user selected both new20k final checkpoints. The three conditions are25
trials each: subtask model with the overall goal, task-only model with the same
goal, and the subtask model with native autosteer. This is75 trials and375 object
opportunities. The direct subtask-model condition tests its30% goal-to-action
training branch. Autosteer uses its trained text head, with no external Qwen/API.

Each trial has a120-second motion budget. Native sync inference, no RTC. The
helper reads the last successful configuration at
`~/rebot-fineart-sync240-7yq7_oyr/rollout.json` on Madeleine and preserves its
robot/camera mapping, controller gains, joint settings and interpolation multiplier.
It does not increase movement limits. Use `--hardware PATH` if that confirmed
configuration is elsewhere. Every attempt gets a fresh dataset directory.

## Setup

These commands are for the user to execute. The agent has not run a physical
trial or opened a camera. Madeleine's tunnel was closed during preparation.

On the Mac, leave this tunnel running in a separate terminal:

```bash
ssh -N -L 127.0.0.1:2222:127.0.0.1:22 madeleine@172.18.130.211
```

Update the experiment checkout on Madeleine while no rollout is active:

```bash
ssh -p 2222 madeleine@127.0.0.1 \
  'cd ~/lerobot-fineart && git pull --ff-only'
```

If Git reports local conflicts, preserve those changes instead of resetting the
checkout. Use the experiment branch `codex/rebot-pi052-matched-sft-20260925` for
these PI052 checkpoints and helper; the feature branch uses the FineART rename.
The existing Madeleine environment and Hub login are reused. The helper downloads
the selected checkpoint at its pinned revision before starting the webcam.

On the Mac:

```bash
cd /Users/pepijnkooijmans/Documents/GitHub_local/lerobot-rebot-pi052-matched-sft
uv run --no-project python examples/rebot_fineart/eval75.py --list-cameras
```

Pick the **USB webcam's video device index**, not the built-in camera or an audio
index. The helper uses FFmpeg AVFoundation. Grant Terminal camera access if macOS
asks. FFmpeg is installed on this laptop. Video defaults to1280x720 at30fps;
use `--resolution 640x480` if the webcam does not support that mode. No microphone
audio is recorded.

Set the index below to the one just listed:

```bash
CAMERA=1
```

## Three blocks of25

Run from that same Mac terminal. The first invocation asks for the five object
descriptions; the goal and checkpoint revisions are saved and reused by all three
conditions. Destination defaults to the black bin. Optional explicit setup:
`--objects "the blue block" "the red block" "the green cap" "the screwdriver" "the white marker"`.
These are example names; use the objects actually on your table.

```bash
uv run --no-project python examples/rebot_fineart/eval75.py \
  --camera "$CAMERA" --condition subtask_direct

uv run --no-project python examples/rebot_fineart/eval75.py \
  --camera "$CAMERA" --condition task_only_direct

uv run --no-project python examples/rebot_fineart/eval75.py \
  --camera "$CAMERA" --condition subtask_autosteer
```

Before each trial, restore the same five objects, matching layout and arm starting
pose for that trial number across all three conditions. The helper waits for Enter,
then starts a webcam recording and verifies it has produced frames before launching
the robot session over SSH. The robot stays idle until you type `/start`.

For either direct condition:

```text
/start
```

For the autosteer condition, type `/start`, then immediately paste the exact
`/autosteer <goal>` command printed by the launcher. Autosteer is enabled after
the session starts because this runtime requires a live observation. Generation
is synchronous and its pauses are included in the same120-second budget.
Typing `/autosteer` is part of the planned condition, not an intervention.

When five objects are in the bin or the120-second segment ends:

```text
/stop
```

Use `/start` only once per trial. `/stop` ends the session and finalizes robot
recording. `/reset` moves the arms; it is not required by this launcher. Prepare
the next scene using your usual reset procedure between trials. If stopping early
for unexpected behavior, count the observed outcome and record it in notes.

After each trial, the laptop asks for0–5 objects in the bin, whether any manual
or extra language intervention occurred, and optional notes. Full success means
5/5 objects with no intervention. No dataset annotation is requested. Each block
writes its25-trial success count and object count. The helper rejects attempts
with no robot frames, multiple run segments, missing expected autosteer, unexpected
autosteer in a direct block, or a failed webcam recording. Failed attempts remain
on disk; rerun the same command to retry the unfinished trial. Scored trials are
skipped on resume.

## Recordings

- Mac: `~/Movies/rebot-eval75-20k-20260927/<condition>/` contains one `webcam.mkv`
  per attempt, camera log, timestamps and trial results. `summary.json` holds block
  success/object counts. MKV tolerates interrupted recording better than ordinary
  MP4; play with VLC or remux after recording with `ffmpeg -i webcam.mkv -c copy webcam.mp4`.
- Madeleine: `~/rebot-eval75/rebot-eval75-20k-20260927/<condition>/<attempt>/`
  contains three policy-camera videos in the native LeRobot dataset, state/actions,
  `actions.jsonl`, the exact rollout config, checkpoint revision, code revision,
  terminal output and recording checks.

Robot recordings stay local; no rollout dataset is automatically published. The
webcam and robot records share condition/trial/attempt IDs and UTC timestamps, but
their frame clocks are not hardware-synchronized. The laptop video includes model
loading, idle time and the trial; the recorded control segment is at most120s.

The helper's configuration preservation and trial-accounting checks are tested
without hardware. Live webcam compatibility and this75-trial launcher have not
been physically tested on Madeleine. Keep the laptop awake while filming.
