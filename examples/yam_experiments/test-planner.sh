#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/lerobot-yam-experiments"
export UV_PROJECT_ENVIRONMENT="$HOME/lerobot-yam/.venv"
export PYTHONPATH="$PWD/src:$PWD"
uv_bin="$HOME/.local/bin/uv"
planner_model='Qwen/Qwen3.8-27B:novita'
mode="${1:-pilot}"
case "$mode" in
  pilot|scored|preflight) ;;
  *) echo 'Usage: test-planner.sh [pilot|scored|preflight]' >&2; exit 2 ;;
esac
if [[ "$(git branch --show-current)" != 'codex/yam-steering-experiments' ]]; then
  echo 'Switch the experiment checkout to codex/yam-steering-experiments first.' >&2
  exit 1
fi
# Authentication + three-image parsing test before model loading or hardware access.
"$uv_bin" run --no-sync python -m examples.yam_experiments.preflight \
  --images "$HOME/yam-setup/camera-recheck-20260930" \
  --model "$planner_model" --log "$HOME/yam-experiments/preflight.jsonl"
[[ "$mode" == preflight ]] && exit 0
pilot_flags=()
[[ "$mode" == pilot ]] && pilot_flags+=(--pilot)
exec "$uv_bin" run --no-sync python -m examples.yam_experiments.run \
  --root "$HOME/yam-experiments" --launcher "$HOME/yam-setup/rollout-molmoact2.sh" \
  --condition planner "${pilot_flags[@]}" --execute \
  --planner.model_id="$planner_model" \
  --planner.api_base=https://router.huggingface.co/v1 \
  --planner.api_key_env=HF_TOKEN --planner.auto_serve=false \
  --planner.request_timeout_s=30 --planner.request_max_retries=0 \
  --planner.max_new_tokens=512 --planner.history=2 \
  '--planner.chat_template_kwargs={"enable_thinking":false}' \
  --autosteer_interval_s=10
