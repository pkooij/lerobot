"""Goal-preserving language variants and the matched 20/50/30 experiment recipe."""


def augment_goal(goal: str) -> list[str]:
    """Vary wording while retaining every object's description and destination.

    These are conservative templates, not visual relabeling. In particular, do
    not infer that every visible object belongs in a bin, or change arm choice.
    """
    clauses = goal.rstrip(".").split("; ")
    if any(not clause.startswith(("Place ", "Pick up ")) for clause in clauses):
        raise ValueError(f"Unsupported episode goal: {goal}")
    variants = [goal]
    for verb in ("Put", "Move", "Transfer"):
        variants.append(
            "; ".join(verb + clause[5:] if clause.startswith("Place ") else clause for clause in clauses)
            + "."
        )
    variants.append("Please " + goal[0].lower() + goal[1:])
    variants.append("Your task is to " + goal[0].lower() + goal[1:])
    return list(dict.fromkeys(variants))


def make_recipe(variant: str) -> dict:
    task_action = {
        "bindings": {"task": "sample_task()"},
        "messages": [{"role": "user", "content": "${task}", "stream": "low_level"}],
    }
    if variant == "task_only":
        return task_action
    if variant != "subtask":
        raise ValueError(variant)
    return {
        "blend": {
            "high_level_subtask": {
                "weight": 0.20,
                "bindings": {"task": "sample_task()"},
                "messages": [
                    {"role": "user", "content": "${task}", "stream": "high_level"},
                    {"role": "assistant", "content": "${subtask}", "stream": "high_level", "target": True},
                ],
            },
            "low_level_execution": {
                "weight": 0.50,
                "messages": [{"role": "user", "content": "${subtask}", "stream": "low_level"}],
            },
            "high_level_execution": {"weight": 0.30, **task_action},
        }
    }
