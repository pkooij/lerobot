#!/usr/bin/env bash
#SBATCH --partition=hopper-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=rebot-aug-prepare
set -euo pipefail
ROOT=${REBOT_EXPERIMENT_ROOT:?Set a fresh experiment directory}
SOURCE=/fsx/pepijn/rebot-pi052-sft-20260910
export PYTHONPATH="$ROOT/lerobot/src" HF_HOME="$SOURCE/hf-cache"
export HF_TOKEN_PATH=/admin/home/pepijn/.cache/huggingface/token
export UV_CACHE_DIR="$SOURCE/uv-cache" XDG_CACHE_HOME="$SOURCE/cache"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "$ROOT/lerobot"
exec "$SOURCE/tools/bin/uv" run --no-project --python "$SOURCE/venv/bin/python" \
  examples/rebot_fineart/prepare_augmented.py
