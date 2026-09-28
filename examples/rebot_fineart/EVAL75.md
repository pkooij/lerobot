# Five-object evaluation of the augmented20k checkpoints

Evaluate both new **20k final checkpoints** on ReBot: **75 rollouts total**.

| Condition | Model           | Control                            | Trials |
| --------- | --------------- | ---------------------------------- | -----: |
| A         | Task-only       | High-level goal → actions          |     25 |
| B         | Subtask-trained | High-level goal → actions          |     25 |
| C         | Subtask-trained | Native FineART autosteer → actions |     25 |

- Use the same five objects, black bin, and instruction across conditions.
- Prepare **25 photographed layouts**. Repeat each for A/B/C, rotating order ABC, BCA, CAB.
- Keep starting poses and control settings identical: no RTC; default 120 seconds, including planning. Fix timeout and autosteer cadence after excluded pilots.
- Film every attempt with the laptop USB webcam; save robot videos, actions, generated subtasks, and model/code revisions.
- Score **all-five success**, objects placed (0–5), stage progress (grasp/lift → transport → release), completion time, and failures. Human assistance counts as failure; retain every attempt.
- Compare **C vs A** primarily; B vs A tests the training recipe, C vs B tests autosteer. Report paired scene results and confidence intervals; objects are not independent trials.

**Claim:** subtask-aware adaptation and control on a new embodiment. Both models share subtask-trained midtraining, so this does not isolate midtraining benefits or prove language steerability.

**Checkpoints:** both 20k finals are public and integrity-verified; exact revisions are pinned in `models_20k.json`.

**Before scoring:** freeze settings using excluded pilots. Use `--trial N` to follow the matched scene schedule; omitting it runs a full condition block. Stage progress and completion time still need video scoring. Physical trials are operator-run.

## Commands directly on Madeleine

Run this setup once **in your Madeleine terminal**. It installs only evaluation
helpers; it does not update the robot checkout or open hardware. Your checked
`~/rebot-eval75/hardware.json` supplies the exact arm/camera paths and controller settings.

```bash
mkdir -p ~/rebot-eval75
EVAL_SOURCE=https://raw.githubusercontent.com/pkooij/lerobot/c213eb01f/examples/rebot_fineart
for FILE in eval75.py eval75_madeleine.py eval75_resident.py models_20k.json; do
  curl -fL "$EVAL_SOURCE/$FILE" -o "$HOME/rebot-eval75/$FILE" || break
done
cd ~/lerobot-fineart
```

**A — task-only, 25 trials:**

```bash
~/.local/bin/uv run --no-sync python ~/rebot-eval75/eval75_resident.py \
  --hardware /home/madeleine/rebot-eval75/hardware.json \
  --duration 120 --condition task_only_direct
```

**B — subtask-trained, direct, 25 trials:**

```bash
~/.local/bin/uv run --no-sync python ~/rebot-eval75/eval75_resident.py \
  --hardware /home/madeleine/rebot-eval75/hardware.json \
  --duration 120 --condition subtask_direct
```

For the current **10-trial comparison**, append `--num-trials 10` to B and C.
This runs scene IDs 1–10, matching the ten completed task-only trials, for 30
episodes total across conditions. Already scored scenes are skipped on resume.

**C — subtask-trained, native autosteer, 25 trials:**

```bash
~/.local/bin/uv run --no-sync python ~/rebot-eval75/eval75_resident.py \
  --hardware /home/madeleine/rebot-eval75/hardware.json \
  --duration 120 --condition subtask_autosteer
```

For the paper schedule, append `--trial 1` to each command in ABC order, then
`--trial 2` in BCA order, `--trial 3` in CAB order, and repeat through 25.
The first invocation asks for five object names; all conditions reuse that goal.
The model loads **once per condition**, and robot/cameras stay connected between trials.
Type `/start` once per trial. For C, immediately paste the printed `/autosteer …`
command. `/stop` ends and scores the current trial while keeping the model loaded;
`/quit` ends the evaluation. No automatic reset motion: `/reset` explicitly returns
the arms to the initial pose before the next `/start`.
Already scored trials are skipped, including results from the previous launcher.
Each new trial is a separate episode in the resident session dataset, with its
episode index saved alongside the score and individual action trace. Recordings/results are in
`~/rebot-eval75/rebot-eval75-madeleine-20k/`.

## Laptop webcam (separate Mac terminal)

Madeleine records the robot cameras, but cannot access the webcam plugged into
your Mac. Start this **on the Mac** before the trials; press `q` when finished.
UGREEN was video device **0** on September 28; run
`ffmpeg -f avfoundation -list_devices true -i ""` if it has been reconnected.

```bash
mkdir -p ~/Movies/rebot-eval75
ffmpeg -n -f avfoundation -framerate 30 -video_size 1280x720 -i '0:none' \
  -c:v libx264 -preset veryfast -crf 20 \
  "$HOME/Movies/rebot-eval75/session-$(date -u +%Y%m%dT%H%M%SZ).mkv"
```

Show a card with scene ID and condition to the webcam before each trial. Keep the
Mac awake; this continuous external video is not automatically synchronized to
robot recordings. Native autosteer uses FineART, not Qwen. No RTC.
