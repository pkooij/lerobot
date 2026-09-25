#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/pepijn/rebot-fineart-20260925
SOURCE=/fsx/pepijn/rebot-pi052-sft-20260910
VARIANT=${1:?subtask or task_only}
PHASE=${2:?smoke or full}
case "$VARIANT" in subtask|task_only) ;; *) exit 2 ;; esac
case "$PHASE" in smoke) STEPS=20; LOG_FREQ=1; WANDB=false; EVAL_STEPS=0 ;; full) STEPS=10000; LOG_FREQ=20; WANDB=true; EVAL_STEPS=1000 ;; *) exit 2 ;; esac
export PATH="$SOURCE/venv/bin:$PATH" PYTHONPATH="$ROOT/lerobot/src"
export HF_HOME="$SOURCE/hf-cache" HF_TOKEN_PATH=/admin/home/pepijn/.cache/huggingface/token
export XDG_CACHE_HOME="$SOURCE/cache" UV_CACHE_DIR="$SOURCE/uv-cache"
export WANDB_DIR="$ROOT/wandb" WANDB_CACHE_DIR="$ROOT/wandb-cache"
export WANDB_CONFIG_DIR="$SOURCE/wandb-config" WANDB_DATA_DIR="$ROOT/wandb-data"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
cd "$ROOT/lerobot"
EPISODES=$("$SOURCE/venv/bin/python" -c 'import json; s=json.load(open("/fsx/pepijn/rebot-fineart-20260925/split.json")); print(json.dumps(s["train"]+s["dev"]))')
test -f "$ROOT/prepared.json"
test -f "$ROOT/repair-audit.json"
if [[ ! -f "$ROOT/processor-audit.json" ]]; then
  "$SOURCE/tools/bin/uv" run --no-project --python "$SOURCE/venv/bin/python" \
    "$ROOT/lerobot/examples/rebot_fineart/audit_processors.py"
fi
exec "$SOURCE/tools/bin/uv" run --no-project --python "$SOURCE/venv/bin/python" \
  torchrun --standalone --nnodes=1 --nproc_per_node=4 --module lerobot.scripts.lerobot_train \
  --policy.path="$ROOT/init_$VARIANT" --policy.device=cuda \
  --dataset.repo_id=pepijn223/rebot_diverse_picking_100_annotated \
  --dataset.revision=93c97807c46535745d0587d4296416bf2d4aa80d \
  --dataset.root="$ROOT/dataset_repaired" --dataset.video_backend=pyav \
  --dataset.episodes="$EPISODES" --dataset.eval_split=0.05 \
  --rename_map='{"observation.images.base":"observation.images.cam_high","observation.images.left_wrist":"observation.images.cam_left_wrist","observation.images.right_wrist":"observation.images.cam_right_wrist"}' \
  --batch_size=16 --steps="$STEPS" --num_workers=8 --prefetch_factor=2 \
  --parallelism.dp_replicate=4 --accelerator.gradient_accumulation.steps=1 \
  --seed=1000 --env_eval_freq=0 --eval_steps="$EVAL_STEPS" --max_eval_samples=256 \
  --save_freq=1000 --log_freq="$LOG_FREQ" \
  --output_dir="$ROOT/runs/${VARIANT}_${PHASE}_${SLURM_JOB_ID:?}" \
  --job_name="fineart_rebot_${VARIANT}_10k_b64_${PHASE}" \
  --wandb.enable="$WANDB" --wandb.project=rebot-fineart-matched-10k \
  --wandb.entity=pepijn1999kooijmans-open-universiteit --wandb.disable_artifact=true --wandb.mode=online \
  --wandb.notes='Fresh Jade midtrain 300000; PR4184 3298e15a9; KI+FAST; source FAST vocabulary; per-GPU16/global64; 85 train/5 dev/10 test; train-only normalization; 10k optimizer updates.'
