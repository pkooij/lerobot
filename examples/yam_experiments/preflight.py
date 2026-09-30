"""Check hosted planning on three saved camera frames without connecting a robot."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from huggingface_hub import get_token
from PIL import Image

from lerobot.rollout.inference import PolicyQuery, QueryKind
from lerobot.rollout.planner import PlannerConfig, VlmPlanner

from .campaign import TASK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B:novita")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    token = get_token()
    if not token:
        parser.error("Run hf auth login with an Inference Providers token first")
    os.environ["HF_TOKEN"] = token
    obs = {
        name: np.array(Image.open(args.images / f"current_{name}.png").convert("RGB"))
        for name in ("top", "left", "right")
    }
    config = PlannerConfig(
        model_id=args.model,
        api_base="https://router.huggingface.co/v1",
        api_key_env="HF_TOKEN",
        auto_serve=False,
        request_timeout_s=30,
        request_max_retries=0,
        max_new_tokens=512,
        chat_template_kwargs={"enable_thinking": False},
        history=2,
        log_path=str(args.log),
    )
    started = time.monotonic()
    try:
        planner = VlmPlanner(config, "bi_yam_follower")
        instruction = planner(obs, PolicyQuery(QueryKind.NEXT_SUBTASK, TASK), TASK)
    except Exception as exc:
        # Avoid traceback/config dumps in the authentication failure path.
        print(f"Planner preflight failed: {type(exc).__name__}: {str(exc).replace(token, '[REDACTED]')}")
        raise SystemExit(1) from None
    print(
        json.dumps(
            {
                "ok": True,
                "model": args.model,
                "instruction": instruction,
                "elapsed_s": round(time.monotonic() - started, 2),
                "images": str(args.images),
                "robot_connected": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
