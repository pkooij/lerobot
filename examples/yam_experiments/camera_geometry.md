# Gemini 305 wrist-camera estimate

This pilot supplies an **estimated**, rigid tool-to-camera transform to Sol.
It is not a completed hand-eye calibration. MolmoAct2's joint predictions,
normalization and camera inputs are unchanged.

## Evidence and assumptions

- Operator measurement: camera 75 mm behind and 70 mm above the fingertip midpoint.
- [I2RT linear_4310 D405 mount](https://github.com/i2rt-robotics/i2rt/blob/120c3c81400171174604e503943f8d1ebc891058/i2rt/robot_models/station/yambox_linear_4310_d405_decxin/README.md):
  camera 70 mm above the axis, 77 mm forward of the flange, 25 degrees downward.
  We borrow the orientation as a prior; this is a D405 mount, not verified Gemini CAD.
- [Amazon/XDOF ABC example](https://github.com/amazon-far/abc/blob/main/abc_sim/models/yam_bimanual_empty.xml)
  uses a different D405 mount with a 50 degree tilt. It is not substituted here.
- [MolmoAct2 calibration discussion](https://github.com/allenai/molmoact2/issues/2)
  gives D405/D435 intrinsics and a top-camera-to-station-midpoint transform, not a
  Gemini wrist transform. Their top-camera placement cannot calibrate this station.
- [Gemini 305](https://www.orbbec.com/gemini-305/) optical axes follow image right,
  image down, and optical forward. Factory lens intrinsics are separate from the
  camera-to-gripper mounting transform.

Unverified assumptions: the measurements refer to the active lens optical centre
rather than housing; lateral offset is zero; both mounts are identical; RGB images
are not mirrored or rotated; pitch is 25 degrees with no extra roll/yaw. The photo
supports a downward cant but cannot uniquely determine all six pose parameters.
These assumptions are included in every planner camera contract. No top-camera
extrinsics or inter-arm transform are assumed.

## Transform

Our MJCF `grasp_site` has local +Z toward the fingertips, +X down and +Y left at
zero joint angles. Camera optical coordinates map into this site using:

```text
p_tool = T_tool_from_camera @ p_camera

T_tool_from_camera =
[[ 0,  0.906307787,  0.422618262, -0.070],
 [-1,  0,            0,            0    ],
 [ 0, -0.422618262,  0.906307787, -0.075],
 [ 0,  0,            0,            1    ]]

T_base_from_camera(q) = T_base_from_tool(q) @ T_tool_from_camera
```

Translations are metres. The rotation is `Rz(-90 degrees) Rx(-25 degrees)`.
At zero joints the camera is 75 mm behind and 70 mm above the TCP in base axes,
looking forward and downward. FK recomputes its base transform at every observation;
the image-right direction must not be treated as a constant base direction.

With these offsets the tip ray is about 43 degrees below the tool's forward axis,
or 18 degrees below the estimated optical axis. That is qualitatively consistent
with the fingertips appearing below the image centre in saved wrist frames. It is
not a fitted angle or a reprojection validation with surveyed correspondences.

Sol receives the transform and its estimated status with measured FK and proposed
MolmoAct2 joint trajectories. It can reason about approximate directions when
clearance is visible. A pixel still needs matching intrinsics and depth to specify
a metric object location. Existing correction sizes, speed limits and IK checks
remain in effect; this estimate does not authorize arbitrary image-to-3D moves.
