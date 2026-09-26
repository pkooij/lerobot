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
