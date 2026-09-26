#!/usr/bin/env bash
#SBATCH --partition=hopper-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=01:00:00
#SBATCH --job-name=rebot-fineart-upload
set -euo pipefail
ROOT=/fsx/pepijn/rebot-fineart-20260925
SOURCE=/fsx/pepijn/rebot-pi052-sft-20260910
export HF_HOME="$SOURCE/hf-cache"
export HF_TOKEN_PATH=/admin/home/pepijn/.cache/huggingface/token
export UV_CACHE_DIR="$SOURCE/uv-cache" XDG_CACHE_HOME="$SOURCE/cache"
exec "$SOURCE/tools/bin/uv" run --no-project --python "$SOURCE/venv/bin/python" \
  "$ROOT/posttrain/upload_final.py"
