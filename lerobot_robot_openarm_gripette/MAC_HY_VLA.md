# Grabette Hy-VLA on this Mac

For installation on a colleague's laptop, use [LAPTOP_SETUP.md](LAPTOP_SETUP.md).
The notes below record the original local validation setup.

Public checkpoint: [pepijn223/hy-vla-umi-grabette-5epochs](https://huggingface.co/pepijn223/hy-vla-umi-grabette-5epochs),
revision `088c73834a10ec22117458355de47ab6d488c3ff`. This is the validated final
five-epoch checkpoint, with unchanged model tensors. It stores the checkpoint at
the repository root, so omit `--checkpoint`.

The local copy is `/Users/pepijnkooijmans/Engineering/models/hy-vla-umi-grabette-5epochs`.
The environment is `.venv` in this checkout. The OpenArm/Gripette package is installed
editable from its folder, alongside the local LeRobot checkout and Motorbridge
PR #28 (`e0667b5b09bbb99e987fa8809c4f4a460d989e50`). Its native PEAK backend is built.
One equivalent modulo expression replaces a newer Rust integer helper so the PR
builds with this Mac's Rust 1.85.1. The native driver has passed its 31 core tests.
That compatibility patch is committed locally as `f54bfc7` in the Motorbridge checkout.

This checkout descends from `feat/openarms-motorbridge` at `bea80ed2407998dcb7e6ff0817ae1c28b7b711fc`
and includes the Hy-VLA integration and Mac loading fixes. The Gripette client and
URDF are installed from `/Users/pepijnkooijmans/Engineering/grabette-openarm-mac`,
develop commit `3eda3cdcab8eb6d1007e344756b7e9079c702572`.

## Verified locally

- Full checkpoint loading and inference on Apple MPS with the standard Hy-VLA API.
- 22 recorded RGB/state observations, ten executed actions per plan, across three plans.
- Independent observation/decoder test of reference recapture and per-offset
  normalization after every ten actions, while retaining all 50 model flow tokens.
- A separate URDF FK/IK roundtrip at `gripper_center` (maximum matrix error about 1.65e-16).
- Passive connection tests that prohibit torque changes and motor targets.

The result is saved in
`/Users/pepijnkooijmans/Engineering/lerobot-hy-vla-validation/grabette/mac-inference/offline-check.json`.
Hardware has not been connected or moved during these checks.

The Hy-VLA PR's 104 selected tests and applicable hooks passed. This local
OpenArm integration passed 27 tests and lint checks. Full-branch mypy reports
18 errors in four unchanged files; the unmodified OpenArm branch reports the
exact same errors. See `mac-branch-baseline-mypy.log` in the validation folder.

Warm chunk generation measured **3.88–4.16 seconds** on this 24 GB Mac; the first
chunk took 5.99 seconds. Queued actions took at most 7.57 ms. This synchronous
setup therefore does not sustain a fresh ten-action plan at 10 or 50 Hz.
The model was trained with 50 Hz action timing. The example's 10 Hz playback
stretches those action intervals fivefold, with further pauses while planning.

## Run the recorded-data check again

```bash
cd /Users/pepijnkooijmans/Engineering/lerobot-hy-vla-openarm-mac
bash lerobot_robot_openarm_gripette/examples/run_hy_vla_mac.sh \
  --offline_observation /Users/pepijnkooijmans/Engineering/lerobot-hy-vla-validation/grabette/artifacts/mac-observations.npz \
  --log_dir /Users/pepijnkooijmans/Engineering/lerobot-hy-vla-validation/grabette/mac-inference
```

This reads the local checkpoint and observations and never opens the robot.

## Passive live test

With the arm supported and the camera/server available:

```bash
cd /Users/pepijnkooijmans/Engineering/lerobot-hy-vla-openarm-mac
bash lerobot_robot_openarm_gripette/examples/run_hy_vla_mac.sh \
  --dry_run --duration 60
```

The confirmed settings are `peakusb:can0@1000000` (classic CAN at 1 Mbit/s) and
Gripette `172.18.7.107:50051`. Passive mode reads telemetry and computes targets;
it skips motor configuration, calibration, torque changes and target sends.
An IK/joint-step error terminates the test and is recorded in the terminal.

## Physical execution and recording

The URDF's actual mounting frame, motor zeros and joint signs still need physical
validation. `--joint_signs` defaults to seven +1 values. The model consumes measured
`[strategy, closure]`, wrist RGB, current `gripper_center` FK and task text.
Saved policy processors decode current-relative XYZ/row-rotation6D into absolute
poses. Those absolute poses go directly to gripper-centre IK; no camera-centred
chunk anchoring is applied. Strategy/closure are clipped to [0, 1] using the
training dataset's 93.5-degree proximal and 102-degree distal calibration.

After that calibration and the passive replay are validated, the physical command
is the same launcher without `--dry_run`. It uses the branch's manual supported
arm placement/hold prompts. To record ten episodes locally:

```bash
bash lerobot_robot_openarm_gripette/examples/run_hy_vla_mac.sh \
  --duration 60 --record_repo_id pepijn223/rollout_grabette_hy_vla_eval \
  --num_episodes 10 --reset_time 20
```

Each episode requires an operator start/reset. Recordings and timestamped prediction
logs go under this checkout's `outputs/openarm_gripette_policy`, with separate
folders per run. The recording includes raw policy Cartesian/gripper commands,
measured state/pose, RGB, clamped commanded joint targets and whether each command
was sent. Uploading a rollout
requires the explicit `--push_to_hub` flag.
