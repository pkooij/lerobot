"""Repair episode goals and complete FAST's byte alphabet without remapping old IDs."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.fft import dct
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoProcessor

from lerobot.policies.pi052.fit_fast_tokenizer import _normalize_actions

ROOT = Path("/fsx/pepijn/rebot-fineart-20260925")
SOURCE = Path("/fsx/pepijn/rebot-pi052-sft-20260910")


def episode_goal(rows):
    groups = {}
    for row in rows:
        if row["style"] != "subtask":
            continue
        command = row["content"].strip().rstrip(".")
        if command.lower() == "return to home position":
            continue
        attempt = re.fullmatch(r"Attempt to pick up (.+)", command, re.I)
        if attempt:
            groups.setdefault(("pick", ""), set()).add(attempt.group(1))
            continue
        match = re.fullmatch(r"Pick up (.+?) and (?:place|put) it (into|in|on) (.+)", command, re.I)
        if match is None:
            match = re.fullmatch(r"(?:Put|Place|Adjust) (.+?) (into|in|on) (.+)", command, re.I)
        assert match, f"Cannot safely derive object/destination: {command!r}"
        obj, preposition, destination = match.groups()
        obj = re.sub(r" with the (?:left|right) arm", "", obj, flags=re.I)
        preposition = "into" if preposition.lower() == "in" else preposition.lower()
        groups.setdefault((preposition, destination), set()).add(obj)
    assert groups
    # Sorting makes this an unordered episode goal, not the demonstrated action sequence.
    clauses = []
    for (preposition, destination), objects in sorted(groups.items()):
        objects = sorted(objects)
        names = objects[0] if len(objects) == 1 else ", ".join(objects[:-1]) + " and " + objects[-1]
        clauses.append(
            f"Pick up {names}" if preposition == "pick" else f"Place {names} {preposition} {destination}"
        )
    return "; ".join(clauses) + "."


def main():
    assert not (ROOT / "repair-audit.json").exists(), "Already repaired"
    split = json.loads((ROOT / "split.json").read_text())
    data = ROOT / "dataset_repaired"
    data.mkdir(exist_ok=True)
    if not (data / "meta").exists():
        shutil.copytree(ROOT / "dataset/meta", data / "meta")
    if not (data / "videos").exists():
        (data / "videos").symlink_to(SOURCE / "dataset/videos", target_is_directory=True)
    goals = {}
    tables = []
    for path in sorted((SOURCE / "dataset/data").rglob("*.parquet")):
        table = pq.read_table(path)
        language = []
        for row in table.select(["episode_index", "language_persistent"]).to_pylist():
            ep = row["episode_index"]
            if ep not in goals:
                goals[ep] = episode_goal(row["language_persistent"])
            language.append(
                [r for r in row["language_persistent"] if r["style"] != "task_aug"]
                + [
                    {
                        "role": "user",
                        "content": goals[ep],
                        "style": "task_aug",
                        "timestamp": 0.0,
                        "camera": None,
                        "tool_calls": None,
                    }
                ]
            )
        idx = table.schema.get_field_index("language_persistent")
        table = table.set_column(
            idx, "language_persistent", pa.array(language, type=table.schema.field(idx).type)
        )
        target = data / path.relative_to(SOURCE / "dataset")
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)
        tables.append(table.select(["episode_index", "index", "action"]))
    assert len(goals) == 100
    (ROOT / "episode_goals.json").write_text(json.dumps(goals, indent=2))
    print("Derived 100 episode-level goals from existing object/destination annotations", flush=True)

    # The source tokenizer lacks two ByteLevel alphabet symbols, silently dropping coefficients.
    # Append only the missing symbols. Every existing token ID and merge stays exactly unchanged.
    tok_path = ROOT / "action_tokenizer"
    if not tok_path.exists():
        shutil.copytree(SOURCE / "midtrain/action_tokenizer", tok_path)
    path = tok_path / "bpe_tokenizer/tokenizer.json"
    original = json.loads((SOURCE / "midtrain/action_tokenizer/bpe_tokenizer/tokenizer.json").read_text())
    repaired = json.loads(json.dumps(original))
    vocab = repaired["model"]["vocab"]
    added = {}
    for char in sorted(set(ByteLevel.alphabet()) - set(vocab)):
        added[char] = max(vocab.values()) + 1
        vocab[char] = added[char]
    assert len(added) == 2
    assert all(vocab[token] == index for token, index in original["model"]["vocab"].items())
    path.write_text(json.dumps(repaired))
    proc_path = tok_path / "processor_config.json"
    proc = json.loads(proc_path.read_text())
    proc["vocab_size"] = len(vocab)
    proc_path.write_text(json.dumps(proc, indent=2))
    tokenizer = AutoProcessor.from_pretrained(str(tok_path), trust_remote_code=True)

    all_data = pa.concat_tables(tables).sort_by([("index", "ascending")])
    stats = json.loads((data / "meta/stats.json").read_text())["action"]
    max_codes = 0
    total_chunks = 0
    squared_error = 0.0
    max_chunk_rmse = 0.0
    for ep in split["train"]:
        rows = all_data.filter(pc.equal(all_data["episode_index"], ep))
        actions = np.asarray(rows["action"].to_pylist(), dtype=np.float32)
        for start in range(0, len(actions), 256):
            anchors = np.arange(start, min(start + 256, len(actions)))
            chunks = actions[np.minimum(anchors[:, None] + np.arange(50), len(actions) - 1)]
            normalized = _normalize_actions(chunks, "QUANTILES", stats)
            codes = tokenizer(normalized)
            decoded = tokenizer.bpe_tokenizer.batch_decode(codes, clean_up_tokenization_spaces=False)
            assert all(len(s) == 700 for s in decoded), (ep, start, [len(s) for s in decoded])
            recovered = np.asarray([[ord(c) for c in s] for s in decoded], dtype=np.float32).reshape(
                -1, 50, 14
            )
            recovered = (recovered + tokenizer.min_token) / tokenizer.scale
            errors = np.square(recovered - dct(normalized, axis=1, norm="ortho"))
            squared_error += float(errors.sum())
            max_chunk_rmse = max(max_chunk_rmse, float(np.sqrt(errors.mean((1, 2))).max()))
            total_chunks += len(anchors)
            max_codes = max(max_codes, max(map(len, codes)))
        if ep % 10 == 0:
            print(
                f"FAST full-training scan: episode={ep}, chunks={total_chunks}, max_codes={max_codes}",
                flush=True,
            )
    rmse = (squared_error / (total_chunks * 700)) ** 0.5
    assert rmse < 0.1 and max_chunk_rmse < 0.2, (rmse, max_chunk_rmse)
    # Tested every training anchor, including padded episode ends. Reserve formatting headroom.
    max_action_tokens = max(128, ((max_codes + 16 + 31) // 32) * 32)
    report = {
        "added_byte_symbols": added,
        "old_token_ids_preserved": 1024,
        "all_training_chunks_checked": total_chunks,
        "malformed_decodes": 0,
        "max_codes": max_codes,
        "max_action_tokens": max_action_tokens,
        "reconstruction_rmse": rmse,
        "max_chunk_rmse": max_chunk_rmse,
        "goal_source": "object/destination pairs from existing subtasks; unordered by object",
        "goal_storage": "language_persistent task_aug; explicit task recipe binding",
        "dataset_root": str(data),
    }
    for variant in ("subtask", "task_only"):
        path = ROOT / f"init_{variant}/config.json"
        cfg = json.loads(path.read_text())
        cfg.update(
            action_tokenizer_name=str(tok_path), max_action_tokens=max_action_tokens, tokenizer_max_length=256
        )
        components = cfg["recipe"].get("blend", {"only": cfg["recipe"]}).values()
        for component in components:
            if any("${task}" in message.get("content", "") for message in component["messages"]):
                component["bindings"] = {"task": 'active_at(style="task_aug", role="user")'}
        path.write_text(json.dumps(cfg, indent=2))
    shutil.copy2(ROOT / "prepared.json", ROOT / "prepared-before-repair.json")
    prepared = json.loads((ROOT / "prepared.json").read_text())
    prepared["repairs"] = report
    (ROOT / "prepared.json").write_text(json.dumps(prepared, indent=2))
    (ROOT / "repair-audit.json").write_text(json.dumps(report, indent=2))
    print("GOAL AND FAST REPAIRS VERIFIED", json.dumps(report), flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "lerobot/examples/rebot_fineart/audit_processors.py")], check=True
    )


if __name__ == "__main__":
    main()
