#!/usr/bin/env bash
#SBATCH --job-name=rebot-fineart-aug20k
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
set -euo pipefail
ROOT=${REBOT_EXPERIMENT_ROOT:?Set a fresh experiment directory}
SOURCE=/fsx/pepijn/rebot-pi052-sft-20260910
export REBOT_TRAIN_STEPS=20000 REBOT_EVAL_SAMPLES=250 REBOT_SAVE_FREQ=5000
export REBOT_TRAIN_NOTES='Fresh Jade; six goal-preserving paraphrases; 20% goal-to-subtask, 50% subtask actions, 30% goal actions vs 100% goal actions; matched stabilization; balanced dev frames.'
export PYTHONPATH="$ROOT/lerobot/src" HF_HOME="$SOURCE/hf-cache"
export HF_TOKEN_PATH=/admin/home/pepijn/.cache/huggingface/token
export UV_CACHE_DIR="$SOURCE/uv-cache" XDG_CACHE_HOME="$SOURCE/cache"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#GPU_NAMES[@]} -eq 4 ]] || exit 1
for GPU_NAME in "${GPU_NAMES[@]}"; do [[ "$GPU_NAME" == *H100* ]] || exit 1; done
test -f "$ROOT/processor-audit.json"
for VARIANT in subtask task_only; do
  bash "$ROOT/lerobot/examples/rebot_fineart/train.sh" "$VARIANT" smoke 2>&1 | tee "$ROOT/logs/${VARIANT}_smoke_${SLURM_JOB_ID}.log"
done
"$SOURCE/venv/bin/python" "$ROOT/lerobot/examples/rebot_fineart/verify.py" "$SLURM_JOB_ID" smoke
for VARIANT in subtask task_only; do
  bash "$ROOT/lerobot/examples/rebot_fineart/train.sh" "$VARIANT" full 2>&1 | tee "$ROOT/logs/${VARIANT}_full_${SLURM_JOB_ID}.log"
  "$SOURCE/venv/bin/python" "$ROOT/lerobot/examples/rebot_fineart/verify.py" "$SLURM_JOB_ID" full "$VARIANT"
done
echo 'BOTH AUGMENTED 20000-STEP RUNS COMPLETED AND VERIFIED'
