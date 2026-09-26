import pytest
from augmentation_recipe import augment_goal, make_recipe

from lerobot.datasets.language_render import render_sample
from lerobot.datasets.recipe import TrainingRecipe


def test_augmentations_preserve_multi_destination_goals():
    goals = augment_goal("Place the green bin into the black bin; Place the white block into the green bin.")
    assert len(goals) == 6
    for goal in goals:
        assert "the green bin into the black bin" in goal
        assert "the white block into the green bin" in goal
    with pytest.raises(ValueError):
        augment_goal("Invent an action.")


def test_explicit_augmented_binding_overrides_stale_task_and_samples_all_variants():
    goals = augment_goal("Place the blue block into the black bin.")
    rows = [{"role": "user", "content": g, "style": "task_aug", "timestamp": 0.0} for g in goals]
    rows.append(
        {"role": "user", "content": "Reach for the blue block.", "style": "subtask", "timestamp": 0.0}
    )
    for variant in ("subtask", "task_only"):
        recipe = TrainingRecipe.from_dict(make_recipe(variant))
        seen = set()
        branches = {"text": 0, "goal_action": 0, "subtask_action": 0}
        for index in range(2000):
            rendered = render_sample(
                recipe=recipe,
                persistent=rows,
                events=[],
                t=0.0,
                sample_idx=index,
                task="WRONG green bin task",
            )
            prompt = rendered["messages_rendered"][0]["content"]
            assert "WRONG" not in prompt
            if prompt in goals:
                seen.add(prompt)
            kind = (
                "text"
                if rendered["target_message_indices"]
                else ("goal_action" if prompt in goals else "subtask_action")
            )
            branches[kind] += 1
        assert seen == set(goals)
        if variant == "task_only":
            assert branches == {"text": 0, "goal_action": 2000, "subtask_action": 0}
        else:
            assert 0.17 < branches["text"] / 2000 < 0.23
            assert 0.47 < branches["subtask_action"] / 2000 < 0.53
            assert 0.27 < branches["goal_action"] / 2000 < 0.33
