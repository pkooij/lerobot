#!/usr/bin/env bash
# FLEX on NVIDIA RoboLab in one HF Job (an L40S: Isaac Sim needs an RTX GPU).
#
# Two Python environments in one container, because Isaac Lab pins transformers 4.57 / numpy 2.3:
#   /opt/isaac - Isaac Sim 6.0.1 + Isaac Lab 3.0b2 + RoboLab, serving the sim with `lerobot-env-server`;
#   /opt/lr    - LeRobot with pi05, the policy side (lerobot-eval or the FLEX loop), on tcp/127.0.0.1:7448.
#
# Env vars:
#   MODE        pilot (eval the starting policy on TASKS) | flex (FLEX on the first of TASKS)
#   TASKS       comma-separated RoboLab task names
#   EPISODES    pilot: episodes per task (default 20)
#   FLEX_ARGS   extra train_flex arguments
#   HUB_REPO    private Hub repo the results go to (default pepijn223/flex-robolab)
#   RUN_NAME    folder in HUB_REPO (default: $MODE-<date>)
#   LEROBOT_REPO / LEROBOT_REF  code to run (default the FLEX branch of pkooij/lerobot)
#
#   HF_TOKEN=... hf jobs run --flavor l40sx1 --timeout 6h --secrets HF_TOKEN \
#     -e NVIDIA_DRIVER_CAPABILITIES=all -e MODE=pilot -e TASKS=BlocksInBinTask ubuntu:24.04 \
#     bash -c 'apt-get update -qq && apt-get install -y -qq curl >/dev/null && \
#       curl -fsSL https://raw.githubusercontent.com/pkooij/lerobot/feat/flex-libero/examples/flex_robolab/hf_job.sh | bash'
set -euo pipefail

MODE=${MODE:-pilot}
TASKS=${TASKS:?set TASKS}
EPISODES=${EPISODES:-20}
HUB_REPO=${HUB_REPO:-pepijn223/flex-robolab}
RUN_NAME=${RUN_NAME:-${MODE}-$(date -u +%Y%m%d-%H%M%S)}
LEROBOT_REPO=${LEROBOT_REPO:-https://github.com/pkooij/lerobot.git}
LEROBOT_REF=${LEROBOT_REF:-feat/flex-libero}
ROBOLAB_SHA=9cce00cadd5dc575fd2b8ea3836f5bdcde574229
WORK=/work
OUT=${WORK}/out/${RUN_NAME}
ENDPOINT=tcp/127.0.0.1:7448
# RoboLab renders 720x1280; pi05 resizes with padding to 224, so 180x320 (same aspect) loses nothing.
OBS_H=180
OBS_W=320
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/job.log") 2>&1
t0=$(date +%s)
stamp() { echo "[$(( $(date +%s) - t0 ))s] $*"; }

stamp "system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends python3 python3-dev python3-venv git git-lfs curl ca-certificates \
  build-essential cmake ffmpeg libegl1 libgl1 libglx0 libgles2 libglib2.0-0 libvulkan1 libxrender1 libsm6 \
  libxt6 libopengl0 libglu1-mesa vulkan-tools >/dev/null
curl -LsSf https://astral.sh/uv/0.11.30/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600

stamp "GPU and graphics driver files"
nvidia-smi
# Isaac Sim renders through Vulkan: the NVIDIA ICD must be visible (NVIDIA_DRIVER_CAPABILITIES=all mounts it).
if [[ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]] && ldconfig -p | grep -q libGLX_nvidia.so.0; then
  mkdir -p /usr/share/vulkan/icd.d
  echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}' \
    >/usr/share/vulkan/icd.d/nvidia_icd.json
  echo "wrote the NVIDIA Vulkan ICD file"
fi
if [[ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]] && ldconfig -p | grep -q libEGL_nvidia.so.0; then
  mkdir -p /usr/share/glvnd/egl_vendor.d
  echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    >/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  echo "wrote the NVIDIA EGL vendor file"
fi
ldconfig -p | grep -E "libGLX_nvidia|libEGL_nvidia|libnvidia-glcore" || echo "WARNING: NVIDIA graphics libraries missing"
vulkaninfo --summary 2>&1 | grep -E "deviceName|driverName|apiVersion|ERROR" | head -8 || true

stamp "code"
git clone -q --depth 1 -b "${LEROBOT_REF}" "${LEROBOT_REPO}" ${WORK}/lerobot
git -C ${WORK}/lerobot log -1 --format="lerobot %H %s"
GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/NVlabs/RoboLab.git ${WORK}/robolab
git -C ${WORK}/robolab checkout -q ${ROBOLAB_SHA}

stamp "Isaac env (background) and LeRobot env"
(
  uv venv -q --python 3.12 /opt/isaac
  VIRTUAL_ENV=/opt/isaac uv pip install -q --index-strategy unsafe-best-match --extra-index-url https://pypi.nvidia.com \
    "isaaclab[isaacsim]==3.0.0b2.post1" "tinyobjloader==2.0.0rc13"
  git -C ${WORK}/robolab lfs install --local >/dev/null
  git -C ${WORK}/robolab lfs pull
  VIRTUAL_ENV=/opt/isaac uv pip install -q -e ${WORK}/robolab
  VIRTUAL_ENV=/opt/isaac uv pip install -q --no-deps -e ${WORK}/lerobot
  VIRTUAL_ENV=/opt/isaac uv pip install -q draccus termcolor "eclipse-zenoh==1.10.1" msgpack pyyaml
  # Isaac Lab pins huggingface-hub<1.0; LeRobot's utils import download_bucket_files (hub >= 1.x).
  VIRTUAL_ENV=/opt/isaac uv pip install -q "huggingface-hub==1.33.0"
  /opt/isaac/bin/python -c "import lerobot.env_server.server, lerobot.sim.integrations.robolab"
) >"${OUT}/isaac_install.log" 2>&1 &
ISAAC_INSTALL=$!
uv venv -q --python 3.12 /opt/lr
VIRTUAL_ENV=/opt/lr uv pip install -q -e "${WORK}/lerobot[pi,dataset,remote,evaluation]"
wait ${ISAAC_INSTALL} || { tail -50 "${OUT}/isaac_install.log"; exit 1; }
stamp "envs ready"
LR=/opt/lr/bin

stamp "pi05 checkpoint"
POLICY=${WORK}/pi05_robolab
${LR}/python ${WORK}/lerobot/examples/flex_robolab/build_pi05_robolab.py --out ${POLICY}

stamp "RoboLab server"
cat >${WORK}/robolab_server.yaml <<EOF
deployment: robolab
sim:
  type: robolab
  task: "${TASKS%%,*}"
  kwargs:
    cameras: WRIST_LEFT
    instruction_type: default
    observation_height: ${OBS_H}
    observation_width: ${OBS_W}
EOF
(cd ${WORK}/robolab && OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y /opt/isaac/bin/lerobot-env-server \
  --config_path=${WORK}/robolab_server.yaml) >"${OUT}/server.log" 2>&1 &
SERVER=$!
trap 'kill ${SERVER} 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  grep -q "Serving robolab" "${OUT}/server.log" && break
  kill -0 ${SERVER} 2>/dev/null || { tail -80 "${OUT}/server.log"; exit 1; }
  sleep 5
done
grep -q "Serving robolab" "${OUT}/server.log" || { tail -80 "${OUT}/server.log"; exit 1; }
stamp "server up"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

ENV_ARGS=(--env.type=robolab --env.server.endpoint=${ENDPOINT} --env.observation_height=${OBS_H}
          --env.observation_width=${OBS_W})
upload() {
  ${LR}/hf upload "${HUB_REPO}" "${OUT}" "${RUN_NAME}" --repo-type model --private \
    --exclude "*/practice/*/data/*" --exclude "*/practice/*/images/*" --exclude "*.safetensors" \
    --commit-message "${RUN_NAME}" >/dev/null || echo "upload failed"
}

if [[ ${MODE} == pilot ]]; then
  IFS=, read -ra TASK_LIST <<<"${TASKS}"
  for task in "${TASK_LIST[@]}"; do
    stamp "eval ${task}"
    ${LR}/lerobot-eval --policy.path=${POLICY} "${ENV_ARGS[@]}" --env.task="${task}" \
      --eval.batch_size=10 --eval.n_episodes="${EPISODES}" --seed=1000 \
      --output_dir="${OUT}/eval_${task}" >"${OUT}/eval_${task}.log" 2>&1 || echo "eval ${task} failed"
    grep -h -E "pc_success|Overall" "${OUT}/eval_${task}.log" | tail -3 || true
    upload
  done
elif [[ ${MODE} == flex ]]; then
  stamp "FLEX on ${TASKS%%,*}"
  # Writes metrics.jsonl to the Hub every round (hub_sync_repo); the final checkpoint at the end.
  ${LR}/python -m lerobot.rl.flex.train_flex --policy.path=${POLICY} "${ENV_ARGS[@]}" --env.task="${TASKS%%,*}" \
    --demo_repo_id=null --mode=flex --beta=0 --record_chunks=true --pg_train_mode=true \
    --eval_reuse_practice_envs=true --loss_reduction=mean --hub_sync_repo="${HUB_REPO}" \
    --output_dir="${WORK}/flex/${RUN_NAME}" ${FLEX_ARGS:-} || echo "FLEX failed"
  cp ${WORK}/flex/${RUN_NAME}/metrics.jsonl "${OUT}/" 2>/dev/null || true
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
upload
stamp "done"
