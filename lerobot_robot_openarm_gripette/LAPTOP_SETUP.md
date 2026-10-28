# Run the Grabette Hy-VLA checkpoint on another laptop

These instructions are for an **Apple silicon Mac**, a right OpenArm and the
Grabette/Gripette wrist camera and gripper. They use the same endpoint
`172.18.7.107:50051` and classic CAN at **1 Mbit/s**. The native PEAK backend is
for PCAN-USB FD / Pro FD / X6 adapters supported by Motorbridge PR #28.

Share this branch: <https://github.com/pkooij/lerobot/tree/codex/hy-vla-openarm-mac>.
It includes the requested `feat/openarms-motorbridge` branch and the Hy-VLA fixes
from PR #4196. The public model is
[pepijn223/hy-vla-umi-grabette-5epochs](https://huggingface.co/pepijn223/hy-vla-umi-grabette-5epochs).

The model has about 4.5B parameters and a 9.05 GB checkpoint. Full loading and
inference have been tested on a 24 GB Mac; other memory sizes have not been
validated. Allow at least 25 GB of free disk for weights, the environment,
download caches, native build and working space. Keep this new environment
separate from the colleague's working teleoperation environment.

## 1. Install the pinned sources and environment

Have Git, [Git LFS](https://git-lfs.com), Xcode Command Line Tools,
[uv](https://docs.astral.sh/uv/getting-started/installation/)
and [Rust/Cargo](https://rust-lang.org/tools/install/) available. Check with
`git --version`, `git lfs version`, `xcode-select -p`, `uv --version`, and `cargo --version`.
The Motorbridge build below vendors libusb; it does not require installing the
older PCBUSB driver. Do not change an already-working robot/server installation.

Run these commands in a fresh Terminal. The clone commands assume the target
folders do not already exist. All new sources, caches and outputs stay under
`~/Engineering`, outside iCloud-backed folders.

```bash
export HY_VLA_ENGINEERING_ROOT="$HOME/Engineering"
export UV_CACHE_DIR="$HY_VLA_ENGINEERING_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$HY_VLA_ENGINEERING_ROOT/runtimes"
export HF_HOME="$HY_VLA_ENGINEERING_ROOT/cache/huggingface"
export CARGO_HOME="$HY_VLA_ENGINEERING_ROOT/cache/cargo"
mkdir -p "$HY_VLA_ENGINEERING_ROOT"

GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch codex/hy-vla-openarm-mac \
  https://github.com/pkooij/lerobot.git \
  "$HY_VLA_ENGINEERING_ROOT/lerobot-hy-vla-openarm-mac"

git clone --single-branch --branch feat/pcan-usb-fd-native \
  https://github.com/CarolinePascal/motorbridge.git \
  "$HY_VLA_ENGINEERING_ROOT/motorbridge-hy-vla-mac"
git -C "$HY_VLA_ENGINEERING_ROOT/motorbridge-hy-vla-mac" \
  checkout --detach e0667b5b09bbb99e987fa8809c4f4a460d989e50

GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch develop \
  https://github.com/pollen-robotics/grabette.git \
  "$HY_VLA_ENGINEERING_ROOT/grabette-openarm-mac"
GIT_LFS_SKIP_SMUDGE=1 git -C "$HY_VLA_ENGINEERING_ROOT/grabette-openarm-mac" \
  checkout --detach 3eda3cdcab8eb6d1007e344756b7e9079c702572
git -C "$HY_VLA_ENGINEERING_ROOT/grabette-openarm-mac" lfs install --local
git -C "$HY_VLA_ENGINEERING_ROOT/grabette-openarm-mac" lfs pull \
  --include='packages/gripette/urdf/openarm_gripette_right/**' --exclude=''

cd "$HY_VLA_ENGINEERING_ROOT/lerobot-hy-vla-openarm-mac"
git -C "$HY_VLA_ENGINEERING_ROOT/motorbridge-hy-vla-mac" apply \
  "$PWD/lerobot_robot_openarm_gripette/patches/motorbridge-rust-185.patch"
CARGO_TARGET_DIR="$HY_VLA_ENGINEERING_ROOT/motorbridge-hy-vla-mac/target" \
  cargo build --locked --release \
  --manifest-path "$HY_VLA_ENGINEERING_ROOT/motorbridge-hy-vla-mac/Cargo.toml" \
  -p motor_abi -p motor_cli --features motor_core/pcan-usb-fd

uv venv --python 3.12.11 .venv
uv pip install --python .venv/bin/python \
  --constraint lerobot_robot_openarm_gripette/examples/hy_vla_mac_constraints.txt \
  -e '.[hy_vla,openarms,kinematics]' \
  -e "$HY_VLA_ENGINEERING_ROOT/motorbridge-hy-vla-mac/bindings/python" \
  -e "$HY_VLA_ENGINEERING_ROOT/grabette-openarm-mac/packages/gripette" \
  'protobuf>=6.33.5,<8'

# Install the third-party robot plugin from its own folder.
cd lerobot_robot_openarm_gripette
uv pip install --python ../.venv/bin/python --no-deps -e '.[hy-vla]'
cd ..
uv pip check --python .venv/bin/python
```

The included one-line Motorbridge patch is the equivalent modulo expression
needed by Rust 1.85.1. It works on newer Rust too. The Gripette checkout is
editable so its bundled `urdf/openarm_gripette_right` model is available.
`--no-deps` on the last install preserves the explicitly selected local sources;
the first install supplies all dependencies of the plugin's `hy-vla` extra.

## 2. Download the final checkpoint once

Still in the LeRobot checkout:

```bash
HF_HUB_OFFLINE=0 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 uv run --no-sync python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    "pepijn223/hy-vla-umi-grabette-5epochs",
    revision="51242f93221e9f1d83584a70ea17bd3ac56e71b4",
    local_dir=Path(os.environ["HY_VLA_ENGINEERING_ROOT"]) / "models/hy-vla-umi-grabette-5epochs",
    max_workers=1,
    token=False,
)
PY
```

The model is public, so no Hugging Face login is needed. This is the final
five-epoch / 6,640-step checkpoint, initialized from `lerobot/hy_vla_umi`.
The weights are stored at the model repository root: **omit `--checkpoint`**.
The launcher uses this local folder and offline Hub mode for subsequent runs.

## 3. Check the installation without opening hardware

```bash
export MOTORBRIDGE_LIB="$HY_VLA_ENGINEERING_ROOT/motorbridge-hy-vla-mac/target/release/libmotor_abi.dylib"
PYTHONPATH="$PWD/lerobot_robot_openarm_gripette:$PWD/src" uv run --no-sync python - <<'PY'
import torch
import motorbridge
from lerobot_robot_openarm_gripette.gripette_compat import resolve_openarm_model

assert torch.backends.mps.is_available(), "Apple MPS is unavailable; use a native Apple silicon Python"
print("MPS:", torch.backends.mps.is_available())
print("Motorbridge ABI:", motorbridge.abi_version())
print("URDF:", resolve_openarm_model())
PY
```

Expect MPS `True`, Motorbridge ABI `0.5.6`, and a model folder containing
`robot.urdf`. This checks dependencies and the native library without opening
CAN or the gripper connection. It does not load the full policy yet.

## 4. Run inference on the connected robot without sending targets

Keep the Gripette server running. Support the arm and stop other programs that
own the CAN adapter before opening this reader. From the LeRobot checkout:

```bash
bash lerobot_robot_openarm_gripette/examples/run_hy_vla_mac.sh \
  --port peakusb:can0@1000000 \
  --gripette 172.18.7.107:50051 \
  --dry_run --duration 30
```

Loading the model happens first. When prompted, press ENTER to start the
episode. This reads the real camera and measured joints/gripper state, runs
Hy-VLA and IK, and logs the predicted poses and joint targets. Passive mode
skips motor configuration, calibration, torque changes and target sends,
including during disconnect. An IK/joint-step error stops the run.

Predictions are saved as timestamped JSONL files in
`outputs/openarm_gripette_policy`. Each record contains the predicted absolute
gripper-centre action, motor targets and `sent: false`.

If her arm is on the adapter's other channel, use `--port peakusb:can1@1000000`.
Keep the bitrate matched to the existing setup. `--gripette HOST:PORT` overrides
the server address. A custom model folder can be selected with `--urdf FOLDER`;
it must contain `robot.urdf` with the `gripper_center` frame and matching joints.
Use her calibrated `--joint_signs` if they differ from the default seven +1s.

## 5. Physical execution after validating the frame and joint mapping

A working camera/CAN connection does not establish matching joint zeros, signs
or the gripper mounting geometry. Compare the included right-arm URDF's
`gripper_center` FK against her actual robot before enabling this policy.
Then inspect the passive predictions and keep the existing motion limits.

Once those checks pass, a short physical trial is:

```bash
bash lerobot_robot_openarm_gripette/examples/run_hy_vla_mac.sh \
  --port peakusb:can0@1000000 \
  --gripette 172.18.7.107:50051 \
  --duration 10
```

This command sends targets. The manual-start prompt disables arm torque so an
operator can support and place the arm, then enables torque and holds the
measured pose. The gripper is initialized to strategy 0 and closure 0.48 before
the episode's second ENTER prompt. At shutdown, support the arm before
confirming torque-off/disconnect. Ctrl-C also enters this cleanup sequence.

To record ten episodes locally, add
`--record_repo_id YOUR_HF_USERNAME/rollout_grabette_hy_vla_eval --num_episodes 10 --reset_time 20`.
Replace `YOUR_HF_USERNAME`; the repository name must begin with `rollout_`.
Recordings include RGB, measured state/pose, policy actions, commanded joints
and whether commands were sent. Uploading requires the separate
`--push_to_hub` flag, `HF_HUB_OFFLINE=0` before the launcher command, and a
Hugging Face login in the launcher's `HF_HOME`.

## Action convention and measured performance

Keep task text **`Grabette demonstration`** for this first run. The launcher
already supplies it, MPS, 10 Hz playback, ten executed actions per plan and
the calibrated strategy/closure mapping. Command-line arguments can override
these defaults. `UV_BIN`, `HY_VLA_ENGINEERING_ROOT`, `HY_VLA_MODEL_PATH` and
`MOTORBRIDGE_LIB` allow different executable, install, model and native-library paths.

The saved preprocessor uses current gripper-centre FK, measured
`[strategy, closure]` and wrist RGB. Hy-VLA predicts current-relative XYZ and
the first two rotation-matrix rows. Its saved postprocessor decodes those into
absolute poses in the robot control frame, which go directly to IK. Do not add
the ACT runner's camera/chunk anchoring. Strategy/closure are absolute and clipped
to [0, 1], then mapped to proximal/distal radians using the training dataset's
93.5-degree proximal and 102-degree distal limits.

`--n_action_steps 10` changes only the execution horizon and saved decoder's
reference cadence. The model's 50-step flow horizon is preserved.

On the tested 24 GB Mac, warm plans took **3.88–4.16 seconds** and queued actions
took at most 7.57 ms. The current synchronous runner pauses during replanning;
it does not sustain uninterrupted 10–50 Hz control. The dataset was recorded at
50 Hz, so the launcher's 10 Hz playback also stretches action timing fivefold.
Use the passive run to measure this laptop's latency before physical trials.

The local evidence covers 22 real recorded observations across three plans,
27 plugin/OpenArm tests, and 31 Motorbridge core tests. No moving-robot task
success has been validated for this checkpoint. The local OpenArm branch's
full mypy check has the same 18 errors in four unchanged files as its baseline;
the separate Hy-VLA PR passed its applicable hooks and 104 selected tests.
