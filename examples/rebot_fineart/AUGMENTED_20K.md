# Augmented 20k comparison

Both runs start from the original Jade step-300000 midtraining weights, with
fresh optimizers. This is a matched retraining experiment, not continuation of
the unstable task-only optimizer. The source weights are SHA-256 verified.

Each variant trains for 20,000 steps on four H100s, 16 frames per GPU, global
batch 64. With 117,379 training frames this is approximately 10.91 frame epochs;
the previous 10,000-step runs were 5.45 epochs. Augmentation does not create new
physical demonstrations. Runs are sequential and use at most four GPUs.

The user-confirmed language mix is:

| Variant         | Goal to subtask text | Subtask to actions | Goal to actions |
| --------------- | -------------------- | ------------------ | --------------- |
| Subtask-capable | 20%                  | 50%                | 30%             |
| Task-only       | 0%                   | 0%                 | 100%            |

The pinned source dataset revision `93c97807c46535745d0587d4296416bf2d4aa80d`
contains only `subtask` annotations, empty language events, and one stale generic
green-bin task. A full scan on 2026-09-26 confirmed there were no original
`task_aug` rows. The prior repair code would replace that style if present, but
did not actually delete augmentation variety from this snapshot.

`prepare_augmented.py` copies the source parquet into a separate experiment
directory and preserves the native subtask annotations. It adds six conservative
goal-preserving templates per episode as native `language_persistent` rows with
style `task_aug`. These vary wording while retaining the original derived object
list and destinations; they do not invent objects, arm assignments, or evidence
that every visible object belongs in a bin. Videos are reused read-only. Bounding
box, pointing and FK supervision are not part of this FineART experiment.

The explicit recipe binding `task: sample_task()` samples stored augmentations
deterministically by frame index, overriding the dataset's stale canonical task.
The default resolver still honors runtime task overrides. Every training frame
is rendered during preparation, auditing all three branches and confirming that
every augmentation is sampled in every training episode. The native processor
then checks tokenization and text/action labels on real batches. Frame-based
sampling repeats for a given frame across epochs; it is not fresh randomness on
every revisit. `task_augmentations.json` and `prepared.json` preserve the audit.

Both models retain flow loss x10, FAST x1, knowledge insulation, and the existing
train-only normalization and 85/5/10 episode split. Both use conditioning LR x0.1,
conditioning gradient clip 0.1, action-expert clip 1.0, and global clip 1.0. These
settings contain spikes rather than proving their underlying cause is fixed.
Nonfinite gradients fail before an optimizer update. Peak LR remains 2.5e-5,
warmup is 1,000 steps, and cosine decay reaches 5e-6 at 20,000 steps.

Development loss runs every 1,000 steps on 250 frames balanced across all five
development episodes and their timelines. This replaces the old selection of
the first 256 frames of one episode. Compare trends within each variant: their
composite objectives differ. Checkpoints are saved at 5k, 10k, 15k and 20k; more
training is not automatically better, and physical success still needs evaluation.
The final ten test episodes are excluded from preparation audits and optimization.

`augmented_prepare_job.sh` runs the CPU data/processor audit.
`augmented_job.sh` should depend on that job succeeding. It runs 20-step distributed
smoke checks for both variants before either full run. Those smoke weights are
discarded; full runs start fresh from midtraining. No robot operation or Hub
upload is performed by these jobs. Set `REBOT_EXPERIMENT_ROOT` to the new root,
with a pinned checkout at `$REBOT_EXPERIMENT_ROOT/lerobot` and a `logs` directory.

Before the smoke training runs, `inference_gate.py` now loads both prepared
policies on GPU and exercises their actual `select_action` and chunk-prediction
paths, including generated language for the subtask model. It requires finite
14-joint outputs and bitwise agreement between legacy and precomputed timestep
schedules for fixed noise. This follows the earlier `embed_prefix` signature fix
(`states` / `state_masks`) and restoration of the previously unused
`precompute_denoise_times` switch in the shared Euler integrator. The schedule is
allocated on the device once per chunk, rather than once per denoising step.
These gates establish compatibility and numerical agreement, not a measured
end-to-end speedup. The first launch was stopped during preliminary smoke checks
so that the full runs cannot begin before this inference gate passes.

## Paired old/new evaluation

`compare_checkpoints.py` evaluates identical frames, explicit objective recipes,
prompts, and per-frame noise seeds for every checkpoint. It reloads each saved
processor, verifies identical normalization and joint ordering, disables language
dropout, and reports flow, FAST, and subtask text losses separately. Different
training mixtures are never compared through their weighted total loss.

Development uses 100 frames per episode across episodes 6, 7, 50, 72 and 78:
500 frames total. Sampling balances annotated stages and includes frames on both
sides of subtask boundaries. Each frame tests the canonical goal and a held-out
paraphrase template, plus annotated-subtask execution and goal-to-subtask text
prediction for the subtask-capable models. Task-only models are compared on the
shared goal-conditioned action cases. Goal-conditioned action evaluation is a
transfer test for the old subtask model, which did not train that execution mode.

The job compares both old 10k models with both new models at 5k/10k/15k/20k.
Afterward, the predeclared final 20k models and old 10k models are evaluated on
100 frames from each of the ten reserved test episodes: 1,000 frames total.
Test losses do not select checkpoints or tune training. The split holds out
trajectories, not necessarily every object identity or instruction type.

Reports retain per-frame and per-episode metrics, paired loss differences,
manifest hashes and weight hashes. Negative paired differences mean lower loss.
These are teacher-forced diagnostics, not autonomous robot success rates.
`--processor-only --smoke` audits real saved processors on CPU without loading
model weights or computing losses. The GPU job first runs all final checkpoints
and modes on a small development smoke panel before the full comparison.

Run `comparison_job.sh` from a separate pinned checkout with an `afterok`
dependency on the four-GPU training job and the CPU processor check. It uses one
GPU after training has released its four GPUs. Never modify the running training
checkout to add evaluation code. See the experiment's `status.json` for job IDs.
