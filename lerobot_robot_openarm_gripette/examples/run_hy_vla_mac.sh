#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
engineering_root="$HOME/Engineering"
export UV_CACHE_DIR="$engineering_root/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$repo_root/.venv"
export HF_HOME="$engineering_root/cache/huggingface"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MOTORBRIDGE_LIB="$engineering_root/motorbridge-hy-vla-mac/target/release/libmotor_abi.dylib"
export PYTHONPATH="$repo_root/lerobot_robot_openarm_gripette:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$HOME/.local/bin/uv" run --no-sync --project "$repo_root" python \
    "$repo_root/lerobot_robot_openarm_gripette/examples/run_grabette_rollout.py" \
    --policy "$engineering_root/models/hy-vla-umi-grabette-5epochs" \
    --device mps --port peakusb:can0@1000000 --gripette 172.18.7.107:50051 \
    --task "Grabette demonstration" --fps 10 --n_action_steps 10 \
    --max_joint_step 7 --initial_gripper_strategy 0.0 --initial_gripper_closure 0.48 \
    "$@"
