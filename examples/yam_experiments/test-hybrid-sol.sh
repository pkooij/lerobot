#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/lerobot-yam-hybrid"
export UV_PROJECT_ENVIRONMENT="$HOME/lerobot-yam/.venv"
export PYTHONPATH="$PWD/src:$PWD"
uv_bin="$HOME/.local/bin/uv"
mode="${1:-pilot}"
case "$mode" in
  pilot|preflight|capture|preview|api) ;;
  *) echo 'Usage: test-hybrid-sol.sh [pilot|preflight|capture|preview|api]' >&2; exit 2 ;;
esac
if [[ "$(git branch --show-current)" != 'codex/yam-hybrid-sol' ]]; then
  echo 'Expected codex/yam-hybrid-sol; refusing to use another branch.' >&2
  exit 1
fi
if pgrep -f '[e]xamples.yam_experiments.run|[l]erobot-rollout' >/dev/null; then
  echo 'An existing rollout is running. Finish it and /quit before using this checkout.' >&2
  exit 1
fi
snapshot="$HOME/yam-setup/hybrid-sol-snapshot"
if [[ "$mode" != capture && "$mode" != preview && -z "${OPENAI_API_KEY:-}" ]]; then
  read -rsp 'OpenAI API key: ' OPENAI_API_KEY
  echo
  export OPENAI_API_KEY
fi
if [[ "$mode" == api ]]; then
  exec "$uv_bin" run --no-sync python -m examples.yam_experiments.hybrid_sol \
    --api-only --snapshot "$HOME/yam-setup/camera-recheck-20260930"
fi
if [[ "$mode" != preview ]]; then
  "$uv_bin" run --no-sync python -m examples.yam_experiments.hybrid_sol \
    --capture-only --launcher "$HOME/yam-setup/rollout-molmoact2.sh" --snapshot "$snapshot"
fi
[[ "$mode" == capture ]] && exit 0
if [[ "$mode" != preview ]]; then
  "$uv_bin" run --no-sync python -m examples.yam_experiments.hybrid_sol --snapshot "$snapshot"
fi
[[ "$mode" == preflight ]] && exit 0
hybrid_config="$("$uv_bin" run --no-sync python -m examples.yam_experiments.hybrid_sol --flags)"
execute_flags=()
[[ "$mode" == pilot ]] && execute_flags+=(--execute)
exec "$uv_bin" run --no-sync python -m examples.yam_experiments.run \
  --root "$HOME/yam-experiments" --launcher "$HOME/yam-setup/rollout-molmoact2.sh" \
  --condition hybrid --pilot "${execute_flags[@]}" \
  --planner.model_id=gpt-6.1-sol --planner.api_mode=responses \
  --planner.api_base=https://api.openai.com/v1 --planner.api_key_env=OPENAI_API_KEY \
  --planner.auto_serve=false --planner.reasoning_effort=low \
  --planner.max_new_tokens=2048 --planner.request_timeout_s=45 --planner.request_max_retries=0 \
  --planner.history=2 "--hybrid=$hybrid_config"
