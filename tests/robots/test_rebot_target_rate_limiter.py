"""Command slew, tracking error and failed-send behavior without hardware."""

import math

import pytest

from lerobot.robots.rebot_b601_follower.target_rate_limiter import JointTargetRateLimiter


def command(limiter, now, desired=100.0, measured=0.0):
    result = limiter.limit({"joint": desired}, {"joint": measured}, {"joint": (-150, 150)}, 5.0, now)
    limiter.commit(result, now)
    return result["joint"]


def test_reference_advances_while_measured_joint_is_stationary_then_stops_at_error_bound():
    limiter = JointTargetRateLimiter(30, 1)
    sent = [command(limiter, tick / 30) for tick in range(10)]
    assert sent == pytest.approx([0, 1, 2, 3, 4, 5, 5, 5, 5, 5])


def test_optional_error_cap_can_be_disabled_without_disabling_rate_or_joint_limits():
    limiter = JointTargetRateLimiter(60, 2)
    sent = []
    for tick in range(10):
        now = tick / 30
        result = limiter.limit({"joint": 100.0}, {"joint": 0.0}, {"joint": (-10, 12)}, None, now)
        limiter.commit(result, now)
        sent.append(result["joint"])
    assert sent == pytest.approx([0, 2, 4, 6, 8, 10, 12, 12, 12, 12])
    # A feedback jump no longer pulls the reference to a new measured-position envelope.
    result = limiter.limit({"joint": -100.0}, {"joint": -20.0}, {"joint": (-10, 12)}, None, 10)
    assert result["joint"] == pytest.approx(10)
    with pytest.raises(ValueError, match="feedback"):
        limiter.limit({"joint": 0.0}, {"joint": math.nan}, {}, None, 10)


def test_elapsed_time_bounds_fast_ticks_and_pause_cannot_accumulate_a_large_jump():
    limiter = JointTargetRateLimiter(30, 1)
    assert command(limiter, 0) == 0
    assert command(limiter, 0.01) == pytest.approx(0.3)
    assert command(limiter, 0.01) == pytest.approx(0.3)
    assert command(limiter, 100) == pytest.approx(1.3)
    assert command(limiter, 100.01, desired=-100) == pytest.approx(1.0)


def test_two_degree_mode_is_explicit_and_reinitialization_holds_measured_pose():
    limiter = JointTargetRateLimiter(60, 2)
    assert command(limiter, 0) == 0
    assert command(limiter, 1 / 30) == pytest.approx(2)
    limiter.reset()
    assert command(limiter, 100, measured=10) == 10


def test_conflicting_constraints_fail_instead_of_violating_rate_limit():
    limiter = JointTargetRateLimiter(30, 1)
    command(limiter, 0)
    with pytest.raises(ValueError, match="Cannot satisfy"):
        command(limiter, 1 / 30, measured=20)


def test_joint_limit_is_respected_even_with_tracking_headroom():
    limiter = JointTargetRateLimiter(30, 1)
    assert command(limiter, 0, desired=200, measured=149.5) == 149.5
    assert command(limiter, 1 / 30, desired=200, measured=149.5) == 150


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_invalid_rate_is_rejected(value):
    with pytest.raises(ValueError):
        JointTargetRateLimiter(value, 1)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_nonfinite_feedback_or_target_never_produces_a_command(value):
    limiter = JointTargetRateLimiter(30, 1)
    with pytest.raises(ValueError):
        command(limiter, 0, measured=value)
    with pytest.raises(ValueError):
        command(limiter, 0, desired=value)
