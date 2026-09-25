#!/usr/bin/env bash
#SBATCH --job-name=rebot-fineart-matched
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
set -euo pipefail
ROOT=/fsx/pepijn/rebot-fineart-20260925
SOURCE=/fsx/pepijn/rebot-pi052-sft-20260910
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#GPU_NAMES[@]} -eq 4 ]] || { echo 'Expected four GPUs'; exit 1; }
for GPU_NAME in "${GPU_NAMES[@]}"; do [[ "$GPU_NAME" == *H100* ]] || exit 1; done
export PYTHONPATH="$ROOT/lerobot/src" HF_HOME="$SOURCE/hf-cache"
export HF_TOKEN_PATH=/admin/home/pepijn/.cache/huggingface/token
export UV_CACHE_DIR="$SOURCE/uv-cache" XDG_CACHE_HOME="$SOURCE/cache"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
"$SOURCE/tools/bin/uv" run --no-project --python "$SOURCE/venv/bin/python" \
  "$ROOT/lerobot/examples/rebot_fineart/audit_processors.py"
for VARIANT in subtask task_only; do
  bash "$ROOT/lerobot/examples/rebot_fineart/train.sh" "$VARIANT" smoke 2>&1 | tee "$ROOT/logs/${VARIANT}_smoke_${SLURM_JOB_ID}.log"
done
"$SOURCE/venv/bin/python" "$ROOT/lerobot/examples/rebot_fineart/verify.py" "$SLURM_JOB_ID" smoke
for VARIANT in subtask task_only; do
  bash "$ROOT/lerobot/examples/rebot_fineart/train.sh" "$VARIANT" full 2>&1 | tee "$ROOT/logs/${VARIANT}_full_${SLURM_JOB_ID}.log"
  "$SOURCE/venv/bin/python" "$ROOT/lerobot/examples/rebot_fineart/verify.py" "$SLURM_JOB_ID" full "$VARIANT"
done
echo 'BOTH MATCHED 10000-STEP RUNS COMPLETED AND VERIFIED'
