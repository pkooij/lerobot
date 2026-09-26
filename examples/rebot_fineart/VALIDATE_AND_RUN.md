# ReBot FineART checkpoints: validation and manual rollout

Both final models completed 10,000 updates, independently from the same original
Jade 300k midtraining weights. Each used 4 H100s, batch 16/GPU (global 64), no
gradient accumulation, and fresh optimizer state. The two jobs ran sequentially.

| Model                                                  | Training language                               | Runtime use                                                                                     |
| ------------------------------------------------------ | ----------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `pepijn223/rebot-fineart-pi052-subtask-10k-20260925`   | 30% task → subtask text, 70% subtask → actions  | Generate its own subtasks with native `/autosteer`, or supply semantic commands with `/subtask` |
| `pepijn223/rebot-fineart-pi052-task-only-10k-20260925` | 100% overall task → actions; text loss disabled | Give the overall task; do not use native `/autosteer`                                           |

These are private, experimental repositories. The task-only run recorded 67/500
nonfinite gradient-norm windows and failed its post-training log check. The subtask
run recorded 0/500 nonfinite windows. Neither result establishes valid inference
or physical success. Upload integrity and inference validation are separate checks.

Both learn 50-step absolute action chunks for 14 joints/grippers from three
current cameras and current state. Internal state/action padding is 32 dimensions;
images resize to 224×224. Action generation uses 10 flow steps. Neither was trained
on pointing, boxes, gripper traces, FK directions, or visual history in this experiment.

Dataset: `pepijn223/rebot_diverse_picking_100_annotated`, revision
`93c97807c46535745d0587d4296416bf2d4aa80d`. 85 train episodes / 117,379 frames,
dev episodes 6, 7, 50, 72, 78, and untouched test episodes 90–99. Normalization uses
training frames only. Episode-global tasks were repaired from existing annotations;
original subtasks remain. Full trajectories include pauses. Both use full BF16
fine-tuning with native FP32 components, AdamW peak LR 2.5e-5, 1k warmup, cosine
decay to 5e-6, clipping 1, flow weight 10 and FAST weight 1. Text weight is 1/0.
Bundled FAST tokenizer appends two missing byte symbols while preserving existing IDs.

## Validate on the science cluster without hardware

From your laptop, open a cluster login shell:

```bash
sft ssh hpc-cluster-science-login-81-129
```

Submit the prepared one-GPU check from that shell:

```bash
ROOT=/fsx/pepijn/rebot-fineart-20260925
sbatch --output="$ROOT/logs/hub-eval-%j.log" \
  "$ROOT/posttrain/evaluate_job.sh" reload --hub --replan-after 10
```

This downloads both uploaded snapshots pinned by `uploaded_86031.json`, verifies
their weight hashes, loads saved processors, and predicts from current images/state
and the supplied task only. The subtask model generates its own subtask first.
It checks finite `(1, 50, 14)` action chunks, records latency, errors against the
demonstration, first-command offsets, adjacent action differences, and raw action
chunk replacement jumps at ten frames. This ten-frame offset is a diagnostic;
the saved runtime default executes 50 actions per chunk. Use `--replan-after 49`
to inspect a near-full-chunk boundary (overlap requires an offset below 50).
Replay uses recorded observations, not states caused by predicted actions.

Read the submitted job ID with `squeue -j JOB_ID`, and inspect its log:

```bash
tail -f "$ROOT/logs/hub-eval-JOB_ID.log"
cat "$ROOT/reload_full_cuda_86031_replan10_hub.json"
```

The report lists each generated subtask and links each prediction `.npz` artifact.
No such report is a physical success measurement. This evaluator has not yet been
run on these uploaded checkpoints; its metric helpers are tested.

Additional diagnostics, each using one GPU:

```bash
sbatch --output="$ROOT/logs/components-%j.log" \
  "$ROOT/posttrain/evaluate_job.sh" components --steps 1000 5000 10000
sbatch --output="$ROOT/logs/gradients-%j.log" \
  "$ROOT/posttrain/evaluate_job.sh" gradients
```

The first separates loss components on balanced samples from all five dev episodes.
The second probes the task-only 1k/3k checkpoints' gradients and clipping without
optimizer updates. These use the retained cluster checkpoints. At most three GPUs
are requested if all three commands run concurrently. The original training-log
eval sampled only the first 256 frames of one episode and is not suitable for
checkpoint selection by itself. Preserve test episodes for a final selected model.

## Prepare a separate checkout on Madeleine

The old WALL-OSS launcher is not a FineART launcher. Use this branch in a separate
directory so the previous robot setup remains available:

```bash
git clone --branch codex/rebot-pi052-matched-sft-20260925 \
  https://github.com/pkooij/lerobot.git "$HOME/lerobot-fineart"
cd "$HOME/lerobot-fineart"
uv sync --python 3.12 --extra pi --extra rebot --extra dataset
uv run hf auth login
```

Use an account with access to the private checkpoints and the PaliGemma tokenizer
(`google/paligemma-3b-pt-224`). Loading requires compatible CUDA/PyTorch on Madeleine;
this new environment has not been installed or tested there.

Set `HARDWARE_CONFIG` to the JSON used for your previous confirmed ReBot rollout
(either the complete rollout config or its `robot` object). To locate candidates:

```bash
find "$HOME/rebot-steerable-artifacts" -type f -name '*.json' | sort
```

The hardware path is intentionally not guessed: Madeleine was unreachable while
these instructions were prepared. Preserve its actual arm ports, camera paths,
arm control parameters and left/right assignments.

```bash
HARDWARE_CONFIG=/absolute/path/to/your/previous/rollout.json
uv run python examples/rebot_fineart/prepare_rollout.py \
  --hardware-config "$HARDWARE_CONFIG" \
  --output "$HOME/rebot-fineart-rollout.json" \
  --max-relative-target 5 --duration 120
```

This only creates JSON. It retains the robot settings, sets both arms' relative
target limit to 5°, maps the three camera names, uses native sync inference and
interactive control, and avoids an automatic return-to-start on shutdown.
It uses the base strategy without recording for the initial trial. The limit is
per command relative to measured position, not a five-degree total travel limit.

## Run manually after checking offline outputs

Running either command below connects the robot and cameras. You control when
movement starts using `/start`; no command here has been executed on hardware.

Subtask model:

```bash
cd "$HOME/lerobot-fineart"
uv run python -m lerobot.scripts.lerobot_rollout \
  --config_path="$HOME/rebot-fineart-rollout.json" \
  --policy.path=pepijn223/rebot-fineart-pi052-subtask-10k-20260925
```

In the interactive prompt, set a first semantic instruction before starting,
then enable the model's own planner:

```text
/subtask Use the left arm to reach toward the blue block.
/start
/autosteer Use the left arm to pick up the blue block and place it into the black bin.
```

`/autosteer` here uses FineART's own text head, not Qwen or an external API.
Use `/subtask ...` to take over the language instruction; `/stop` ends the session.
`/reset` intentionally moves back toward the startup pose. Planning is synchronous,
so inspect latency as well as action continuity; language generation may pause action production.

Task-only model, in a separate session after stopping the first:

```bash
uv run python -m lerobot.scripts.lerobot_rollout \
  --config_path="$HOME/rebot-fineart-rollout.json" \
  --policy.path=pepijn223/rebot-fineart-pi052-task-only-10k-20260925
```

```text
/start
```

The config supplies the complete blue-block-to-black-bin task. Do not use
`/autosteer` for this model. Investigate its gradient issue before physical use.
Both commands retain the saved 50-action execution horizon; changing that horizon
is a separate runtime experiment and does not alter the learned checkpoint.
