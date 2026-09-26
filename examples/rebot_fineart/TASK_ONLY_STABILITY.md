# Task-only training stability

The original 10k task-only checkpoint remains experimental. The changes here
contain gradient spikes during a new training run; they do not repair saved
weights or establish physical performance. The subtask model and all inference
computations remain unchanged.

## Evidence

Fixed batch 16 from training episode 1, seed 1001, produced pre-clip norms of
11.18 at 1k steps, 103.19 at 3k, and 4.13 million at 10k. Other fixed batches at
10k were around 14. The subtask 10k control was 15–18 on all three batches.
The spike's largest parameter gradients were in `time_mlp_in`, `time_mlp_out`,
and the first expert block's adaptive normalization projections.

An instrumented follow-up localized the dominant sample to timestep 0.052.
Its early action-expert blocks amplify backward gradients strongly. Activation
RMS values were not close to the normalization epsilon. Increasing epsilon to
1e-4 and using FP32 residuals did not resolve the spike. The exact cause of the
learned sensitivity remains unresolved; timestep layers accumulate gradients
from the expert and are not proven to be its sole origin. The earlier infinite
training norms were not reproduced by these fixed-batch probes.

## Enable for a fresh task-only run

Add these arguments to the task-only training command **using this branch**:

```bash
--policy.conditioning_lr_scale=0.1 \
--policy.conditioning_grad_clip_norm=0.1 \
--policy.action_expert_grad_clip_norm=1.0
```

Start from the original midtraining weights / prepared `init_task_only` and a
fresh optimizer and output directory. Do not resume the unstable run's optimizer:
the new configuration changes its parameter-group layout. The original
`train.sh` remains an unchanged recipe for reproducing the old matched runs;
its frozen cluster source needs to be replaced by this branch for these flags.

The conditioning group contains both time MLPs and all expert adaptive
scale/shift/gate projections, including their biases. Its learning rate is
`optimizer_lr * action_expert_lr_scale * conditioning_lr_scale`. At the original
peak LR of 2.5e-5 and expert multiplier 1, this is 2.5e-6.

After backward and DDP synchronization, the trainer measures norms in float64,
checks every group for nonfinite values before any mutation, applies separate
limits of 0.1 to conditioning and 1 to the remaining action expert, then retains
the existing global norm limit of 1. This prevents an expert spike from reducing
the backbone/FAST gradients by a factor of millions. Logs retain original raw
norms and add `grad_norm_<group>` and `grad_scale_<group>`; a large raw norm is
still visible and must not be interpreted as cured merely because clipping works.

An actual NaN/Inf gradient stops the run before optimizer/scheduler updates.
Float32 norm-reduction overflow on otherwise finite gradients is avoided with
float64 reductions. The opt-in path supports CPU/CUDA single-device and ordinary
DDP training without a GradScaler. It rejects FSDP, DeepSpeed and FP16 GradScaler
training rather than treating partial or scaled gradients as complete ones.
Default configurations continue using Accelerate's existing clipping path.

## Verification

CPU tests cover a 1e30 finite spike, protection of healthy gradients, global and
group budgets, duplicate/missing groups, rejection of NaN/Inf before mutation,
optimizer/scheduler protection, and exact gradient agreement across two DDP
replicas. Optimizer tests check conditioning membership and multiplied LRs.

`stability_probe.py` loads the problematic 10k checkpoint and performs eight
in-memory optimizer updates on the fixed training batch with alternating seeds
1001/1002. It saves only scalar diagnostics, never weights. This intentionally
stresses already unstable weights; it is not the proposed fresh training run.
A full training rerun and held-out evaluation are still needed before claiming
that the task-only model has recovered.

The GPU stress run (science-cluster job 86247) completed all eight updates with
finite losses. Raw norms on the difficult seed remained 3.62–6.34 million; the
applied total norms were 0.998–1.003 (BF16 rounding around the budget of 1).
On the first spike, the backbone's applied norm was 0.9765, compared with about
0.000002 under the old global-only limit. This validates containment and continued
backbone updates, not elimination of the old checkpoint's raw spikes. The
normalization/precision ablation was job 86245. Reports are retained under
`/fsx/pepijn/rebot-fineart-20260925/{conditioning_diagnosis,task_only_stability_probe}.json`.
