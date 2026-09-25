# ReBot FineART matched fine-tuning

User-requested experiment, 2026-09-25. Policy/trainer source is PR #4184 at
`3298e15a9e897b09b582b4ec38786ecdbedb1630`; experiment scripts live on
`codex/rebot-pi052-matched-sft-20260925`. No PR creation or merge.

Both variants start fresh from the identical Jade midtraining step 300000:
`s3://scale-hf-s3-bucket/robotics/jade/vlm-runs/pi052_subgoal_3cam_14d_balanced_taskfix2/checkpoints/300000/pretrained_model/`.
Cached source weights are checked against the recorded multipart S3 ETag and
SHA-256 is recorded. No optimizer state or earlier ReBot fine-tune is resumed.

| Setting              | Subtask variant                                        | Task-only variant                    |
| -------------------- | ------------------------------------------------------ | ------------------------------------ |
| Language             | 30% task → subtask CE; 70% subtask-conditioned actions | High-level task input, no subtask CE |
| Knowledge insulation | Enabled                                                | Enabled                              |
| Action losses        | Flow ×10 + FAST ×1                                     | Flow ×10 + FAST ×1                   |
| Subtask CE weight    | 1                                                      | 0                                    |
| Training             | 10,000 updates, 16/GPU ×4 H100, accumulation 1         | Same                                 |
| Initialization       | Jade step 300000                                       | Identical                            |

The baseline disables subtask supervision and subtask inputs during ReBot
fine-tuning; the shared initialization already received subtask midtraining.
It is therefore a fine-tuning/inference ablation, not an ablation of the original
midtraining dataset or knowledge.

## Data and normalization

Dataset `pepijn223/rebot_diverse_picking_100_annotated`, pinned revision
`93c97807c46535745d0587d4296416bf2d4aa80d`. All complete trajectories in the
85 training episodes are used. Development episodes: 6, 7, 50, 72, 78.
Test episodes: 90–99. The split is identical for both variants. Development
loss uses up to 256 samples every 1,000 steps; physical success remains untested.
An audit found that the native trainer selects the first frames per canonical
`task_index`. Since this dataset has one canonical task index, the logged loss
covers only the first 256 frames (0–8.5 seconds) of episode 6. It is not a
representative metric across all five development episodes.

`eval_components.py` is a separate, post-training diagnostic for checkpoints
1,000, 5,000, and 10,000. It samples 50 fixed bin midpoints through each dev
episode, repeats the same random seed per frame, and reports flow, text, and
FAST losses separately, including per-episode results and branch counts.
Component means are conditional on the active objective and averaged per frame;
they are not directly comparable to the native batched composite scalar.
These are teacher-forced diagnostics using labels. `reload_check.py` separately
tests causal inference with self-generated subtasks and no future targets in
the policy input. Neither diagnostic measures physical success. Consult
`status.json` before scheduling: cancelled validation jobs may require clarification.

The experiment copies metadata and computes state/action normalization from
117,379 training frames only. Original data, videos, and checkpoint weights
remain unchanged. ReBot's actual ordered joint names replace the ALOHA names in the
checkpoint configuration. The policy uses absolute joint actions, as in its
source checkpoint. Three cameras map to `cam_high`, `cam_left_wrist`, and
`cam_right_wrist`.

Native language annotations provide object-level picking/placing subtasks.
The upstream high-level task table incorrectly assigns every episode the same
green-bin instruction. `repair.py` derives episode goals from annotated
object/destination pairs, sorted independently of demonstration order, and stores
them as native `language_persistent` rows with style `task_aug`. Explicit recipe
bindings use these goals in both variants, overriding the stale table string.
The source task table remains the single grouping key for the exact episode
split. An attempt annotation contributes pickup intent without inventing a
destination or asserting that it succeeded. The repaired dataset is a separate
copy at `dataset_repaired`; it is not published over the source dataset.
No new low-level, point, bounding-box, or FK annotations are introduced here.
Preparation checks every training frame has an active subtask. The native
processor audit checks actual text labels, action routing, and exact splits.

The initial 680-chunk FAST check exposed a malformed decode despite low average
RMSE: the source tokenizer lacks two byte-alphabet symbols. The repair appends
only those symbols, preserving all existing 1,024 IDs, BPE merges and token
offset. It checks every training action chunk, including padded episode ends,
for exactly 700 decoded DCT coefficients and reconstruction error, and sizes the
FAST token budget from the observed maximum plus formatting headroom. No silent
truncation is allowed. Text context increases to 256 tokens to accommodate
episode goals and joint state.

## Paper relationship

Source: user-supplied `FineART___Arxiv.pdf`, §4.1, §4.3 and Appendix A/Table 5.
The paper uses 300k midtraining steps, global batch 64 on 8 H100s, and a 5k-step
YAM transfer experiment. These runs use the requested 10k transfer steps with
the same global batch on four GPUs. AdamW LR 2.5e-5, betas (0.9, 0.95),
weight decay 1e-10, global gradient clipping 1, text z-loss 1e-4, native BF16
with the branch's FP32 components and gradient checkpointing are retained.
For transfer, warmup is 1,000 steps and cosine decay reaches 5e-6 at 10k.
This is an explicit transfer schedule, not the paper's 2k/300k midtraining
schedule. Full ReBot trajectories, including pauses, are retained; the paper's
midtraining idle-frame stripping is not replicated.

## Cluster execution

Root: `/fsx/pepijn/rebot-fineart-20260925` on the science cluster.
Existing environment/cache: `/fsx/pepijn/rebot-pi052-sft-20260910`.

`prepare.sh` runs CPU checks. `job.sh` requests exactly four GPUs and runs both
20-step smoke checks before either full run. Full runs are sequential, so this
experiment never requests eight GPUs. Every full run starts from midtraining,
not from smoke weights. It saves every 1,000 updates and verifies the final
checkpoint at 10,000. A failed smoke or verification stops the job.

Job logs and audit JSONs under the cluster root are authoritative. Initial CPU
submission 86016 failed because the default home UV cache exceeded quota;
preparation 86018 passed after moving UV/XDG caches to FSX. Repair 86029 stopped
on an unhandled adjustment annotation; the parser was extended explicitly after
inspecting all unmatched commands. Four-GPU job 86017 depends on the corrected
repair/audit job. Consult `status.json` and scheduler state for current IDs before
rerunning anything. Do not start robot deployment from these scripts.
