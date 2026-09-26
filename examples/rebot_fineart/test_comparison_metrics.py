import pytest
from comparison_metrics import paired_deltas, stage_indices, summarize


def test_short_stages_and_both_sides_of_boundaries_are_represented():
    indices = stage_indices(3000, 30, [0, 1, 60, 99], 100)
    assert len(indices) == len(set(indices)) == 100
    assert {0, 29, 30, 1799, 1800, 2969, 2970, 2999} <= set(indices)
    assert stage_indices(3, 30, [0, 1], 100) == [0, 1, 2]
    assert len(stage_indices(3000, 30, [0, 1, 60], 2)) == 2


def test_paired_metrics_reject_misaligned_prompts_and_exclude_mixed_total():
    old = [
        {
            "episode": 6,
            "frame": 3,
            "mode": "goal",
            "seed": 42,
            "prompt": "Put it in the bin.",
            "losses": {"flow_loss": 2.0, "loss": 20.0},
        }
    ]
    new = [{**old[0], "losses": {"flow_loss": 1.0, "loss": 10.0}}]
    assert paired_deltas(new, old) == {"goal/flow_loss": {"n": 1, "mean": -1.0}}
    assert "goal/loss" not in summarize(new)
    with pytest.raises(AssertionError):
        paired_deltas([{**new[0], "prompt": "Different"}], old)
    with pytest.raises(AssertionError):
        paired_deltas([{**new[0], "frame": 4}], old)
