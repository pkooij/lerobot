"""User-operated75-trial evaluation with per-attempt Mac webcam recordings.

No hardware is opened until the user runs a condition. Uses only stdlib on Mac.
"""

import argparse
import base64
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

CONDITIONS = ("subtask_direct", "task_only_direct", "subtask_autosteer")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_config(source, plan, output):
    """Preserve confirmed robot settings; every attempt gets a fresh dataset root."""
    robot = source["robot"]
    if robot["type"] != "bi_rebot_b601_follower":
        raise ValueError("Expected the previously confirmed bimanual ReBot configuration")
    for side in ("left", "right"):
        if not robot[f"{side}_arm_config"]["port"]:
            raise ValueError(f"Missing {side} port")
    return {
        "robot": robot,
        "rename_map": source.get("rename_map", {}),
        "strategy": {"type": "sentry"},
        "inference": {"type": "sync"},
        "device": "cuda",
        "fps": 30,
        "duration": plan["duration"],
        "interpolation_multiplier": source.get("interpolation_multiplier", 2),
        "interactive": True,
        "autosteer_interval_s": 5.0,
        "task": plan["goal"],
        "play_sounds": False,
        "return_to_initial_position": False,
        "dataset": {
            "repo_id": f"pepijn223/rollout_eval_rebot_20k_{plan['condition']}",
            "root": str(output / "dataset"),
            "push_to_hub": False,
            "private": True,
            "streaming_encoding": True,
            "fps": 30,
        },
        "action_trace_path": str(output / "actions.jsonl"),
    }


def remote(plan):
    # Hold this across the complete session, including hardware shutdown.
    import fcntl

    lock = Path.home() / "rebot-eval75/rollout.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another evaluation is running; stop it before starting a new one") from exc
        assert_no_existing_rollout()
        return _remote(plan)


def assert_no_existing_rollout(proc_root=Path("/proc")):
    """Also catch abandoned sessions started by older launcher versions."""
    for path in proc_root.glob("[0-9]*/cmdline"):
        try:
            if path.stat().st_uid != os.getuid():
                continue
            argv = path.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError):
            continue
        if b"lerobot.scripts.lerobot_rollout" in argv:
            raise RuntimeError(
                f"Rollout PID {path.parent.name} is still running or shutting down. "
                "Stop that session and wait for it to exit before retrying."
            )


def run_rollout_process(command, checkout):
    """Own the entire uv/python/tee process group so Ctrl+C cannot orphan it."""
    process = subprocess.Popen(command, cwd=checkout, start_new_session=True)
    try:
        return process.wait()
    finally:
        for sig, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                process.poll()  # Reap the parent so its zombie does not keep the group alive.
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                continue
            break
        process.poll()


def _remote(plan):
    checkout = Path(plan["checkout"]).expanduser()
    hardware = Path(plan["hardware"]).expanduser()
    output = Path.home() / "rebot-eval75" / plan["campaign"] / plan["condition"] / plan["attempt"]
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads(hardware.read_text())
    config = make_config(source, plan, output)
    write_json(output / "rollout.json", config)
    write_json(output / "trial.json", plan)
    git = shutil.which("git")
    if not git:
        raise RuntimeError("Git is required to record the checkout revision")
    code = subprocess.check_output([git, "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    (output / "code_revision.txt").write_text(code + "\n")
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    model = plan["model"]
    command = [
        uv,
        "run",
        "--no-sync",
        "python",
        "-m",
        "lerobot.scripts.lerobot_rollout",
        f"--config_path={output / 'rollout.json'}",
        f"--policy.path={model['repo_id']}",
        f"--policy.pretrained_revision={model['revision']}",
    ]
    print(f"\nTrial {plan['trial']}/25: {plan['condition']}\nGoal: {plan['goal']}", flush=True)
    print(
        "Use /start once. After completion or timeout, use /stop. Do not start a second segment.", flush=True
    )
    if plan["condition"] == "subtask_autosteer":
        print(f"Immediately after /start, enter:\n/autosteer {plan['goal']}", flush=True)
    print(f"Recording directory: {output}", flush=True)
    bash = shutil.which("bash")
    tee = shutil.which("tee")
    if not bash or not tee:
        raise RuntimeError("bash and tee are required")
    returncode = run_rollout_process(
        [
            bash,
            "-o",
            "pipefail",
            "-c",
            "ulimit -c 0; "
            + shlex.join(command)
            + " 2>&1 | "
            + shlex.join([tee, str(output / "terminal.log")]),
        ],
        checkout,
    )
    if returncode:
        return returncode
    info_path = output / "dataset/meta/info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    log = (output / "terminal.log").read_text(errors="replace")
    autosteer = "Autosteer on — goal" in log
    starts = log.count("Rollout running — task")
    valid = (
        info.get("total_frames", 0) > 0
        and info.get("total_episodes") == 1
        and starts == 1
        and autosteer == (plan["condition"] == "subtask_autosteer")
    )
    write_json(
        output / "recording_check.json",
        {
            "valid": valid,
            "starts": starts,
            "autosteer_enabled": autosteer,
            "frames": info.get("total_frames", 0),
            "episodes": info.get("total_episodes", 0),
        },
    )
    if not valid:
        print("Recording check failed: expected one nonempty episode and the requested steering mode.")
        return 3
    return 0


def ssh_args(args):
    return [
        shutil.which("ssh"),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-p",
        str(args.port),
        args.host,
    ]


def score():
    while True:
        value = input("Objects successfully in bin (0–5): ").strip()
        if value in {"0", "1", "2", "3", "4", "5"}:
            break
    while True:
        help_used = input("Any manual/language intervention during motion? [y/n]: ").strip().lower()
        if help_used in {"y", "n"}:
            break
    return {
        "objects_in_bin": int(value),
        "intervention": help_used == "y",
        "success": value == "5" and help_used == "n",
        "notes": input("Notes (optional): "),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-plan", help=argparse.SUPPRESS)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--camera", help="AVFoundation video device index from --list-cameras")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--session", type=Path, default=Path.home() / "Movies/rebot-eval75-20k-20260927")
    parser.add_argument("--models", type=Path, default=Path(__file__).with_name("models_20k.json"))
    parser.add_argument(
        "--objects", nargs=5, help="Five quoted object descriptions, reused across all conditions"
    )
    parser.add_argument("--destination", default="the black bin")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--host", default="madeleine@127.0.0.1")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--checkout", default="~/lerobot-fineart")
    parser.add_argument("--hardware", default="~/rebot-fineart-sync240-7yq7_oyr/rollout.json")
    args = parser.parse_args()
    if args.remote_plan:
        return remote(json.loads(base64.b64decode(args.remote_plan)))
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or sys.platform != "darwin":
        parser.error("Run on the Mac laptop with ffmpeg installed")
    if args.list_cameras:
        subprocess.run(
            [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""], check=False
        )
        return 0
    if not args.condition or args.camera is None or not args.camera.isdigit():
        parser.error("--condition and a numeric --camera index are required")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    models = json.loads(args.models.read_text())
    for model in models.values():
        if not model.get("hub_integrity_checked") or not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            parser.error("Expected verified model repositories pinned to full commit revisions")
    args.session = args.session.expanduser().resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.session.name):
        parser.error("Session directory name must use letters, digits, underscores or hyphens")
    args.session.mkdir(parents=True, exist_ok=True)
    manifest = args.session / "campaign.json"
    if manifest.exists():
        campaign = json.loads(manifest.read_text())
        if campaign["models"] != models or campaign["duration"] != args.duration:
            parser.error("Existing campaign has different models/duration; choose a new --session")
        if args.objects and args.objects != campaign["objects"]:
            parser.error("Existing campaign uses different objects")
    else:
        objects = args.objects or [input(f"Object {i + 1}/5 description: ").strip() for i in range(5)]
        if not all(objects):
            parser.error("All five object descriptions are required")
        goal = "Place " + ", ".join(objects[:-1]) + " and " + objects[-1] + " into " + args.destination + "."
        campaign = {"objects": objects, "goal": goal, "models": models, "duration": args.duration}
        write_json(manifest, campaign)
    ssh = ssh_args(args)
    if not ssh[0]:
        parser.error("ssh is required")
    # Copy this helper outside the checkout; this does not start cameras or motors.
    install = 'mkdir -p "$HOME/rebot-eval75" && cat > "$HOME/rebot-eval75/eval75.py"'
    subprocess.run(ssh + [install], input=Path(__file__).read_bytes(), check=True)
    check = "test -f " + shlex.quote(args.hardware.replace("~/", "/home/madeleine/", 1))
    subprocess.run(ssh + [check], check=True)
    model = models["task_only" if args.condition == "task_only_direct" else "subtask"]
    download = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({model['repo_id']!r}, revision={model['revision']!r})"
    )
    # Download before starting the webcam or connecting hardware.
    prefetch = (
        "cd "
        + shlex.quote(args.checkout.replace("~/", "/home/madeleine/", 1))
        + ' && "$HOME/.local/bin/uv" run --no-sync python -c '
        + shlex.quote(download)
    )
    subprocess.run(ssh + [prefetch], check=True)
    block = args.session / args.condition
    block.mkdir(exist_ok=True)
    for trial in range(1, 26):
        result_file = block / f"trial_{trial:02d}.json"
        if result_file.exists():
            continue
        input(
            f"\n{args.condition}: trial {trial}/25. Reset the same five objects/layout and arm start pose. Enter when ready: "
        )
        attempt = f"{trial:02d}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
        directory = block / attempt
        directory.mkdir()
        progress = directory / "camera_progress.txt"
        video = directory / "webcam.mkv"
        plan = {
            "campaign": args.session.name,
            "condition": args.condition,
            "trial": trial,
            "attempt": attempt,
            "goal": campaign["goal"],
            "duration": args.duration,
            "model": models["task_only" if args.condition == "task_only_direct" else "subtask"],
            "checkout": args.checkout,
            "hardware": args.hardware,
            "started_utc": datetime.now(UTC).isoformat(),
        }
        write_json(directory / "trial.json", plan)
        with (directory / "camera.log").open("w") as log:
            camera = subprocess.Popen(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-f",
                    "avfoundation",
                    "-framerate",
                    "30",
                    "-video_size",
                    args.resolution,
                    "-i",
                    f"{args.camera}:none",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-progress",
                    str(progress),
                    "-n",
                    str(video),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    text = progress.read_text() if progress.exists() else ""
                    if re.search(r"frame=[1-9][0-9]*", text):
                        break
                    if camera.poll() is not None:
                        raise RuntimeError(
                            f"Webcam failed before robot launch: see {directory / 'camera.log'}"
                        )
                    time.sleep(0.2)
                else:
                    raise RuntimeError("Webcam produced no frames within20s; robot was not launched")
                encoded = base64.b64encode(json.dumps(plan).encode()).decode()
                command = 'python3 "$HOME/rebot-eval75/eval75.py" --remote-plan ' + shlex.quote(encoded)
                code = subprocess.call(ssh[:-1] + ["-tt", ssh[-1], command])
                camera_alive = camera.poll() is None
            finally:
                if camera.poll() is None:
                    camera.send_signal(signal.SIGINT)
                    try:
                        camera.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        camera.kill()
                        camera.wait()
        if code != 0 or not camera_alive or camera.returncode not in (0, 255) or not video.exists():
            raise RuntimeError(
                f"Incomplete attempt retained at {directory}; rerun this condition to retry trial{trial}"
            )
        result = {
            **plan,
            **score(),
            "webcam": str(video),
            "robot_recording": f"~/rebot-eval75/{args.session.name}/{args.condition}/{attempt}",
            "finished_utc": datetime.now(UTC).isoformat(),
        }
        write_json(result_file, result)
    results = [json.loads(p.read_text()) for p in sorted(block.glob("trial_*.json"))]
    write_json(
        block / "summary.json",
        {
            "trials": len(results),
            "successes": sum(r["success"] for r in results),
            "objects_in_bin": sum(r["objects_in_bin"] for r in results),
            "objects_total": 5 * len(results),
        },
    )
    print(f"Completed25 trials: {block / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
