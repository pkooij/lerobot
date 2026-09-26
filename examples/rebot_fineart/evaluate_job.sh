#!/usr/bin/env bash
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000M
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=rebot-fineart-evaluate
set -euo pipefail
ROOT=/fsx/pepijn/rebot-fineart-20260925
SOURCE=/fsx/pepijn/rebot-pi052-sft-20260910
case "${1:-}" in
  reload) SCRIPT=reload_check.py ;;
  components) SCRIPT=eval_components.py ;;
  gradients) SCRIPT=audit_gradients.py ;;
  *) echo 'Usage: evaluate_job.sh reload|components|gradients [options]' >&2; exit 2 ;;
esac
shift
export PYTHONPATH="$ROOT/lerobot/src" HF_HOME="$SOURCE/hf-cache"
export HF_TOKEN_PATH=/admin/home/pepijn/.cache/huggingface/token
export UV_CACHE_DIR="$SOURCE/uv-cache" XDG_CACHE_HOME="$SOURCE/cache"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
exec "$SOURCE/tools/bin/uv" run --no-project --python "$SOURCE/venv/bin/python" \
  "$ROOT/posttrain/$SCRIPT" "$@"
