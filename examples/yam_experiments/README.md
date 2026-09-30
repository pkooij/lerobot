# YAM prompting and five-condition campaign

This experiment branch is based on `codex/yam-motorbus-molmoact2` at `9413515a9`
and integrates PR #4675 at `efcc65c99`. It does not alter the working hardware PR.
All actuation continues through LeRobot's YAM adapter and Damiao MotorsBus.

## Current readiness

| Condition   | Controller                                                           | Trials | Status                                                                        |
| ----------- | -------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------- |
| `human`     | MolmoAct2 with operator `/subtask` updates                           | 10     | Pilot harness implemented; physical validation pending                        |
| `planner`   | External VLM chooses text subtasks, MolmoAct2 produces actions       | 10     | PR integrated with RTC takeover fix; endpoint and physical validation pending |
| `hybrid`    | Same VLM can choose a VLA subtask or propose a bounded direct motion | 10     | Action adapter and endpoint not commissioned; runner refuses this mode        |
| `astra`     | Astra chooses every motion; no VLA actions                           | 10     | Action adapter and endpoint not commissioned; runner refuses this mode        |
| `astra_icl` | Identical Astra controller plus one fixed demonstration episode      | 10     | Same blockers plus demonstration selection                                    |

No physical trial has been run by this harness yet. Do not interpret a generated
manifest or software test as a completed rollout. The pilot must establish usable
prompt response and stable robot control before collecting the scored comparison.

## First: a human-prompting pilot

Run on champagne from the experiment checkout using its launcher. The harness
waits for an explicit trial and `/start`; there is no pilot duration limit.

```text
/trial pilot-red-01
/subtask Pick up the red block and place it in the green bin.
/start
```

Observe the robot. If it needs a correction, use a short, concrete instruction:

```text
/subtask Open the gripper and release the red block into the green bin.
```

This is a task instruction, not a guaranteed low-level command. To stop a trial,
use `/finish` (or `/reset`), not a natural-language "stop" prompt. Check the outcome
before returning home: released cubes must be fully in the bin and at rest for two
seconds; a cube still held above the bin is not complete.

```text
/finish
# Wait for "Robot reset". Score the outcome observed before home:
/score success 1 red cube released inside bin
```

Next, arrange two distinguishable cubes. Use a new pilot ID, start with one color,
and change the instruction to the other color before grasping. Record whether it
switches targets, the delay, and whether any queued old-task motion continues.
Repeat the same reset layout once with an unchanged instruction as a control.
Then pilot the full task with one explicit color subtask at a time. Pilots do not
count toward the 50 scored trials. `/stop` returns home and keeps holding when the launcher sets
`--interactive_stop_returns_home=true`; score the attempt afterward. `/quit` ends
the session and disables torque, so support the arms before quitting.

The new gripper damping and return recovery remain subject to physical validation.
Do not tune gains between scored conditions. A feedback fault is an infrastructure
interruption, not evidence that the model ignored the prompt.

## CLI and evidence

```bash
python -m examples.yam_experiments.campaign init ~/yam-experiments --cubes red blue yellow
python -m examples.yam_experiments.run \
  --root ~/yam-experiments --launcher ~/yam-setup/rollout-molmoact2.sh \
  --condition human --pilot
```

Substitute the actual cube inventory; unique labels such as `red-1 red-2` support
duplicate colors. `init` refuses to overwrite an existing manifest. `run` defaults
to a preview; add `--execute` to connect and wait for `/trial` and `/start`. It reads
the setup launcher's flags without executing its shell preamble or branch switch.
A scored run requires `codex/yam-steering-experiments`, a clean revision, a frozen
cube inventory, and a common positive `timeout_s` in `manifest.json`. Pilots always
use duration zero. Choose the scored budget after observing pilot completion times.

Every session writes `events.jsonl` and a local LeRobot dataset containing the three
cameras, measured joint/gripper state, dispatched actions, and the task that generated
each action. `/subtask` timestamps record requested instructions; dataset task labels
show when the new instruction actually reaches action dispatch. Sentry may split a
trial across several dataset episodes; each verdict records its episode range.
No dataset is uploaded. MolmoAct2 keeps its saved checkpoint processors and statistics;
the empty recording dataset does not replace its normalization.

A reservation is written before every attempt. A fault or process crash leaves the
attempt unscored and reserved; do not delete it and rerun under the same ID. Preserve
those artifacts and adjudicate the interruption separately. `/score` requires a
completed home reset; a failed reset requires operator intervention. Scores are
operator judgments supported by recordings, not model self-reported success.

```bash
python -m examples.yam_experiments.campaign report ~/yam-experiments
```

## Scored protocol: 10 layouts × 5 conditions

1. Freeze the physical cube/bin inventory, cameras, home pose, calibration, gains,
   model/checkpoint revisions, API model, prompts, and time budget after pilots.
2. Photograph and label ten starting layouts. Each layout must be physically recreated
   for every condition; use table markers and reference images, not an RNG seed alone.
3. Follow the manifest order within each layout. It balances each condition's position
   across the ten blocks. No automatic advance moves the robot or rearranges cubes.
4. For `human`, predeclare allowed prompt interventions (for example, one color-specific
   instruction per cube plus one corrective instruction after a visibly failed attempt).
   Log every prompt. Conditions 2–5 receive no human task hints in scored trials; a
   physical intervention ends the attempt as interrupted.
5. Primary outcome: every cube released fully inside the green bin and stationary for
   two seconds before the trial ends. Also report cubes placed / total, completion time,
   prompt count, grasps/retries/drops, model calls/latency, interventions, and robot faults.
6. Keep all ten attempts per condition, including failed and interrupted attempts. Report
   interruptions separately; do not silently replace them. Ten trials per condition is
   exploratory evidence, so retain per-layout outcomes and uncertainty rather than only
   ranking percentages.

For a formal human trial, use `/trial human-L01`, `/start`, allowed `/subtask` updates,
`/finish`, and `/score success N <notes>` after the home return. Other outcomes are
`partial`, `failure`, and `interrupted`. A new trial requires a fresh layout reset.

## External planner integration and review

PR #4675's current head passed fast CI and quality checks. The combined YAM branch
passes targeted tests. The upstream PR is not merged: the experiment branch fixes an
RTC takeover issue demonstrated by
`test_external_planner_first_reply_cannot_release_preplanner_rtc_actions`. Queued and
in-flight actions from before planner takeover must be discarded, and RTC must not
prefill actions from the broad goal while waiting for its first plan.

The harness translates a planner `done` response into a stopped trial followed by
home and operator scoring. Planner errors also request a reset. The original PR
holds the current task on `done`; this is not sufficient for experiment termination.
These changes do not claim a full upstream review or successful physical planning.

The planner condition requires an explicitly configured compatible endpoint:

```bash
python -m examples.yam_experiments.run \
  --root ~/yam-experiments --launcher ~/yam-setup/rollout-molmoact2.sh \
  --condition planner --pilot \
  --planner.model_id=YOUR_MODEL --planner.api_base=YOUR_ENDPOINT
```

This remains a preview without `--execute`. Do not place API keys in CLI flags or
logs. Authenticated Astra use needs the endpoint adapter; Pablo's current client is
an OpenAI-compatible Chat Completions client designed for locally served VLMs. No
model fallback should silently substitute for Astra.

## Direct and hybrid Astra commissioning

Keep the same cameras and physical controller. The high-level action protocol should
separate `observe`, `vla_subtask` (hybrid only), bounded `move`, gripper control, `done`,
and `give_up`. Exactly one controller may own targets at a time. Switching away from
the VLA must pause inference and clear queued actions; returning must acquire a fresh
observation. Late model replies must be tied to a trial and observation generation.

The YAM interface accepts absolute joint radians and grippers in [0, 1], with the
same left/right ordering as MolmoAct2. Cartesian proposals require a validated IK
model, tool frame, arm-base transforms, and camera/workspace grounding. Do not treat
pixel coordinates or end-effector deltas as joint angles. No model-generated shell
or Python should write to the CAN bus directly. Reject invalid, stale, unreachable,
or oversized proposals before they reach the existing controller.

Before enabling conditions 3–5, complete an observation-only API probe, offline
proposal validation, FK/IK round-trip and frame tests, then an attended small-motion
pilot. Freeze the same action primitives and limits for conditions 4 and 5. Direct
Astra must not secretly call MolmoAct2 as a motion primitive.

For ICL, use one fixed successful LeRobot demonstration from a held-out layout, with
all camera views, timestamps, measured state, absolute actions, task labels, and
explicit units/gripper polarity. Select representative time-aligned keyframes from
that one episode, include action consequences, and save dataset revision, episode
index, frame selection, and content hashes. Do not use scored trials as evolving
memory. Reset model session/history between trials in both Astra conditions.

## References

- [Pablo's external steering PR](https://github.com/huggingface/lerobot/pull/4675)
- [GPT-as-Policy](https://github.com/anonymous-report-421/GPT-as-Policy): explicit action contracts, hybrid/direct comparison, per-case provenance.
- [Inspect Robots](https://github.com/robocurve/inspect-robots): swappable policy/embodiment contracts, attended resets, auditable trial records.
- [SE3 Labs comparison](https://se3labs.ai/blog/agentic-vs-code-as-policy/): distinguishes online agent decisions from frozen generated control code; the proposed conditions are online agentic control.
- [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs): schema-constrained replies still require semantic and robot-state validation.
