"""Fail the job when required losses/configurations/checkpoints are missing."""

import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("REBOT_EXPERIMENT_ROOT", "/fsx/pepijn/rebot-fineart-20260925"))
job, phase, *requested = sys.argv[1:]
variants = requested or ["subtask", "task_only"]
steps = 20 if phase == "smoke" else int(os.environ.get("REBOT_TRAIN_STEPS", "10000"))
reports = {}
for variant in variants:
    run = ROOT / f"runs/{variant}_{phase}_{job}"
    checkpoint = run / f"checkpoints/{steps:06d}/pretrained_model"
    assert (checkpoint / "model.safetensors").stat().st_size > 1_000_000_000
    train = json.loads((checkpoint / "train_config.json").read_text())
    state = json.loads((checkpoint.parent / "training_state/training_step.json").read_text())
    assert state["step"] == steps and state["batch_size"] == 16
    assert state["dp_world_size"] == 4 and state["grad_accum_steps"] == 1
    policy = train["policy"]
    assert train["steps"] == steps and train["batch_size"] == 16
    assert train["parallelism"]["dp_replicate"] == 4
    assert train["accelerator"]["gradient_accumulation"]["steps"] == 1
    assert policy["knowledge_insulation"] and policy["enable_fast_action_loss"]
    assert policy["text_loss_weight"] == (1 if variant == "subtask" else 0)
    if variant == "task_only":
        assert "subtask" not in json.dumps(policy["recipe"])
    log = (ROOT / f"logs/{variant}_{phase}_{job}.log").read_text()
    assert "16 x 4 dp workers x 1 grad accum = 64" in log
    assert "85 train, 5 eval" in log
    required = ["loss", "flow_loss", "fast_action_loss", "grdn"]
    if variant == "subtask":
        required.append("text_loss")
    metrics = {}
    step_lines = "\n".join(line for line in log.splitlines() if "step:" in line and "loss:" in line)
    for name in required:
        values = [float(x) for x in re.findall(rf"\b{name}:([-+\w.]+)", step_lines)]
        assert values and all(math.isfinite(v) for v in values), (variant, name, values[-5:])
        assert max(values) > 0, (variant, name, "No positive training signal")
        metrics[name] = {"first": values[0], "last": values[-1], "max": max(values)}
    if phase == "full":
        eval_values = [float(x) for x in re.findall(r"eval_loss=([-+\w.]+)", log)]
        assert eval_values and all(math.isfinite(x) for x in eval_values)
        metrics["dev_loss"] = eval_values
    reports[variant] = {"steps": steps, "checkpoint": str(checkpoint), "metrics": metrics, "passed": True}
(ROOT / f"verified_{phase}_{'_'.join(variants)}_{job}.json").write_text(json.dumps(reports, indent=2))
print(json.dumps(reports, indent=2))
