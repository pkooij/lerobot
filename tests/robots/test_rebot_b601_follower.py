#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_rebot_b601_follower import BiRebotB601Follower, BiRebotB601FollowerConfig
from lerobot.robots.rebot_b601_follower import (
    RebotB601Follower,
    RebotB601FollowerConfig,
    RebotB601FollowerRobotConfig,
)

_MODULE = "lerobot.robots.rebot_b601_follower.rebot_b601_follower"


def _make_motor_mock(position_rad: float = 0.0) -> MagicMock:
    motor = MagicMock(name="MotorMock")
    state = MagicMock()
    state.pos = position_rad
    motor.get_state.return_value = state
    return motor


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="MotorBridgeControllerMock")
    # add_damiao_motor returns a fresh motor mock; position encodes the call order.
    bus._motor_count = 0

    def _add_motor(_send_id, _recv_id, _model):
        bus._motor_count += 1
        return _make_motor_mock(position_rad=math.radians(bus._motor_count))

    bus.add_damiao_motor.side_effect = _add_motor
    return bus


@pytest.fixture
def follower():
    bus_mock = _make_bus_mock()
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.MotorBridgeMode", MagicMock()),
    ):
        controller_cls.from_dm_serial.return_value = bus_mock
        cfg = RebotB601FollowerRobotConfig(port="/dev/null")
        robot = RebotB601Follower(cfg)
        robot.connect(calibrate=False)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_features_match_joints():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        robot = RebotB601Follower(RebotB601FollowerRobotConfig(port="/dev/null"))
    expected = {f"{m}.pos" for m in robot.motor_names}
    assert set(robot.action_features) == expected
    assert set(robot.observation_features) == expected
    assert "gripper.pos" in expected


def test_connect_disconnect(follower):
    assert follower.is_connected
    follower.disconnect()
    assert not follower.is_connected


def test_get_observation_converts_to_degrees(follower):
    obs = follower.get_observation()
    assert set(obs) == {f"{m}.pos" for m in follower.motor_names}
    # The bus mock seeds each motor's position with its 1-indexed creation order (radians).
    for idx, motor in enumerate(follower.motor_names, 1):
        assert obs[f"{motor}.pos"] == pytest.approx(math.degrees(math.radians(idx)))


@pytest.mark.parametrize("invalid_position", [None, float("nan"), float("inf"), -float("inf")])
def test_missing_or_nonfinite_feedback_rejects_observation(follower, invalid_position):
    motor = follower.motors["elbow_flex"]
    if invalid_position is None:
        motor.get_state.return_value = None
    else:
        motor.get_state.return_value.pos = invalid_position
    with pytest.raises(RuntimeError, match="elbow_flex feedback"):
        follower.get_observation()


def test_failed_feedback_poll_does_not_return_cached_positions(follower):
    follower.bus.poll_feedback_once.side_effect = OSError("CAN disconnected")
    with pytest.raises(RuntimeError, match="CAN feedback poll failed"):
        follower.get_observation()


def test_relative_target_check_sends_nothing_without_feedback(follower):
    follower.config.max_relative_target = 3.0
    follower.motors["elbow_flex"].get_state.return_value = None
    with pytest.raises(RuntimeError, match="elbow_flex feedback"):
        follower.send_action({"shoulder_pan.pos": 10.0, "gripper.pos": -10.0})
    for motor in follower.motors.values():
        motor.send_mit.assert_not_called()
        motor.send_pos_vel.assert_not_called()
        motor.send_force_pos.assert_not_called()


@pytest.mark.parametrize("invalid_target", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_action_rejected_before_any_motor_command(follower, invalid_target):
    with pytest.raises(ValueError, match="nonfinite joint target"):
        follower.send_action({"shoulder_pan.pos": 10.0, "gripper.pos": invalid_target})
    for motor in follower.motors.values():
        motor.send_mit.assert_not_called()
        motor.send_pos_vel.assert_not_called()
        motor.send_force_pos.assert_not_called()


def test_send_action_clips_to_joint_limits(follower):
    # shoulder_pan limit is (-150, 150); request beyond the upper bound.
    returned = follower.send_action({"shoulder_pan.pos": 999.0})
    assert returned["shoulder_pan.pos"] == 150.0
    # Default control_mode is "mit", so arm joints are driven via send_mit.
    follower.motors["shoulder_pan"].send_mit.assert_called_once()


def test_send_action_routes_gripper_to_force_pos(follower):
    follower.send_action({"gripper.pos": -10.0})
    follower.motors["gripper"].send_force_pos.assert_called_once()
    follower.motors["gripper"].send_pos_vel.assert_not_called()


def test_gripper_mit_mode_routes_to_send_mit():
    bus_mock = _make_bus_mock()
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.MotorBridgeMode", MagicMock()),
    ):
        controller_cls.from_dm_serial.return_value = bus_mock
        cfg = RebotB601FollowerRobotConfig(port="/dev/null", gripper_control_mode="mit")
        robot = RebotB601Follower(cfg)
        robot.connect(calibrate=False)
        robot.send_action({"gripper.pos": -10.0})
        robot.motors["gripper"].send_mit.assert_called_once()
        robot.motors["gripper"].send_force_pos.assert_not_called()


def test_bimanual_prefixes_features():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        cfg = BiRebotB601FollowerConfig(
            left_arm_config=RebotB601FollowerConfig(port="/dev/null0"),
            right_arm_config=RebotB601FollowerConfig(port="/dev/null1"),
        )
        robot = BiRebotB601Follower(cfg)
    assert any(k.startswith("left_") for k in robot.action_features)
    assert any(k.startswith("right_") for k in robot.action_features)
    assert "left_gripper.pos" in robot.action_features
    assert "right_gripper.pos" in robot.action_features


@pytest.mark.parametrize("arm", ["left", "right"])
def test_bimanual_invalid_target_rejected_before_either_arm_moves(arm):
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        robot = BiRebotB601Follower(
            BiRebotB601FollowerConfig(
                left_arm_config=RebotB601FollowerConfig(port="/dev/null0"),
                right_arm_config=RebotB601FollowerConfig(port="/dev/null1"),
            )
        )
    robot.left_arm = MagicMock(is_connected=True)
    robot.right_arm = MagicMock(is_connected=True)
    action = {"left_shoulder_pan.pos": 10.0, "right_shoulder_pan.pos": 10.0}
    action[f"{arm}_gripper.pos"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite joint target"):
        robot.send_action(action)
    robot.left_arm.send_action.assert_not_called()
    robot.right_arm.send_action.assert_not_called()
