#!/usr/bin/env bash
#SBATCH --job-name=rebot-paired-eval
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --time=12:00:00
set -euo pipefail
ROOT=${REBOT_EXPERIMENT_ROOT:?}
CODE=${REBOT_COMPARISON_SOURCE:?Use an isolated pinned checkout}
TRAIN_JOB=${REBOT_TRAIN_JOB:?}
BASE=/fsx/pepijn/rebot-fineart-20260925
SOURCE=/fsx/pepijn/rebot-pi052-sft-20260910
export PYTHONPATH="$CODE/src" HF_HOME="$SOURCE/hf-cache"
export HF_TOKEN_PATH=/admin/home/pepijn/.cache/huggingface/token
export UV_CACHE_DIR="$SOURCE/uv-cache" XDG_CACHE_HOME="$SOURCE/cache"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "$CODE"
git rev-parse HEAD > "$ROOT/comparison_source_revision.txt"
UV="$SOURCE/tools/bin/uv"
PYTHON="$SOURCE/venv/bin/python"
SCRIPT="$CODE/examples/rebot_fineart/compare_checkpoints.py"
FINAL=()
for VARIANT in subtask task_only; do
  FINAL+=(--checkpoint "old_${VARIANT}_10000=$BASE/runs/${VARIANT}_full_86031/checkpoints/010000/pretrained_model")
  FINAL+=(--checkpoint "new_${VARIANT}_20000=$ROOT/runs/${VARIANT}_full_${TRAIN_JOB}/checkpoints/020000/pretrained_model")
done
# First verify every final checkpoint and evaluation objective on real dev inputs.
"$UV" run --no-project --python "$PYTHON" "$SCRIPT" --root "$ROOT" --panel dev --smoke \
  --output "$ROOT/comparison/smoke_${SLURM_JOB_ID}" "${FINAL[@]}"
DEV=()
for VARIANT in subtask task_only; do
  DEV+=(--checkpoint "old_${VARIANT}_10000=$BASE/runs/${VARIANT}_full_86031/checkpoints/010000/pretrained_model")
  for STEP in 5000 10000 15000 20000; do
    printf -v PADDED '%06d' "$STEP"
    DEV+=(--checkpoint "new_${VARIANT}_${STEP}=$ROOT/runs/${VARIANT}_full_${TRAIN_JOB}/checkpoints/${PADDED}/pretrained_model")
  done
done
"$UV" run --no-project --python "$PYTHON" "$SCRIPT" --root "$ROOT" --panel dev \
  --output "$ROOT/comparison/dev_${SLURM_JOB_ID}" "${DEV[@]}"
# Predeclared final-step comparison; do not select or tune using this test panel.
"$UV" run --no-project --python "$PYTHON" "$SCRIPT" --root "$ROOT" --panel test \
  --output "$ROOT/comparison/test_${SLURM_JOB_ID}" "${FINAL[@]}"
echo 'PAIRED DEVELOPMENT AND RESERVED-TEST LOSS COMPARISONS COMPLETE'
