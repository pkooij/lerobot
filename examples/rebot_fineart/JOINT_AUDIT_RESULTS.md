# Right shoulder-lift offline comparison, 2026-09-26

Cluster job 86256 completed successfully. No robot was connected.
The 10,000-update subtask checkpoint was evaluated on recorded observations:
three anchors (10%, 50%, 85%) in train episodes 0/1/2 and all five development
episodes 6/7/50/72/78. Each observation was evaluated with its annotated subtask
and with its episode goal, using matched sampling noise: 48 predictions total.
The ten final test episodes were not used. This is a sparse diagnostic, not a
physical success evaluation or exhaustive validation score.

For `right_shoulder_lift.pos`, with annotated subtasks:

| Metric                                    | 9 training anchors | 15 development anchors |
| ----------------------------------------- | -----------------: | ---------------------: |
| First target mean absolute error          |              2.01° |                  1.65° |
| Full 50-action chunk RMSE                 |              3.10° |                 10.45° |
| Hold-current-position baseline chunk RMSE |             15.95° |                 12.61° |

Errors compare model commands with demonstration commands, after checkpoint
denormalization, before hardware clipping. Full chunks span about 1.67 seconds
at 30 Hz. With high-level tasks directly as action input, development chunk RMSE
was 11.19°. This is not a native generated-subtask planner evaluation.

The largest development shoulder error in the annotated-subtask condition was
31.66° RMS at episode 6, frame 971. The first command was -108.38°, close to the
demonstrated -108.50°; the current measured joint was -128.57°. Later predictions
diverged. Episode 50, frame 1000 had 20.54° chunk RMSE. Thus plausible first targets
do not establish reliable motion for the remainder of a chunk.

The checkpoint and dataset both name this joint at index 8 in the 14-dimensional
vector, and their full joint name lists match. Default ReBot hardware declaration
order also matches. Actual live port assignments, hardware motor zeros, gains and
saved custom joint order were not verified because Madeleine was unavailable.

A separate synchronous-runtime bug was fixed: labeling checkpoint-ordered outputs
using hardware-ordered dataset columns could swap targets when those orders differ.
The default matching order is unaffected. A regression test covers reversed orders;
this is not proof that the bug occurred on Madeleine.

The training dataset's shoulder command-minus-measurement signed distribution
has a 1st percentile of -8.07° and a 99th percentile of +24.78°. Demonstrated
commands are not interchangeable with measured positions. Tracking lag, load and
teleoperation/controller conventions are possible contributors; this audit does
not identify their cause.

Next, use `RECORD_DIAGNOSTIC.md` to record measured, requested and driver-returned
targets together. Diagnose coordinate mismatch and tracking before changing motor
gains or zeros. Shortening the executed action horizon is worth testing after a
baseline recording, but requires more frequent inference and is not established
as a fix by this offline comparison.

Full results: `/fsx/pepijn/rebot-fineart-20260925/right_shoulder_joint_audit.json`.
Raw per-anchor predictions: `/fsx/pepijn/rebot-fineart-20260925/joint_audit/`.
Reproduce using `joint_audit.py` on an allocated GPU with this branch's runtime.
