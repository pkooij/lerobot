# YAM external-planner campaign

This experiment branch is based on `codex/yam-motorbus-molmoact2` with the latest home-return and interpolation fixes
and integrates PR #4675 at `efcc65c99`. It does not alter the working hardware PR.
All actuation continues through LeRobot's YAM adapter and Damiao MotorsBus.

## Current readiness

| Condition   | Controller                                                           | Trials | Status                                                                                                      |
| ----------- | -------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------- |
| `human`     | MolmoAct2 with operator `/subtask` updates                           | 0      | Skipped for this campaign                                                                                   |
| `planner`   | External VLM chooses text subtasks, MolmoAct2 produces actions       | 10     | PR integrated with RTC takeover fix; HF Router configured; token permission and physical validation pending |
| `hybrid`    | Same VLM can choose a VLA subtask or propose a bounded direct motion | 10     | Action adapter and endpoint not commissioned; runner refuses this mode                                      |
| `astra`     | Astra chooses every motion; no VLA actions                           | 10     | Action adapter and endpoint not commissioned; runner refuses this mode                                      |
| `astra_icl` | Identical Astra controller plus one fixed demonstration episode      | 10     | Same blockers plus demonstration selection                                                                  |

No physical trial has been run by this harness yet. Do not interpret a generated
manifest or software test as a completed rollout. The pilot must establish usable
prompt response and stable robot control before collecting the scored comparison.

## First: a hosted-planner pilot

Experiment 1 (human prompting) is excluded. The active plan has ten trials each
for `planner`, `hybrid`, `astra`, and `astra_icl` (40 total); only `planner` is ready
for an attended pilot. The other three still need their action adapters.

On Champagne, authenticate with a Hugging Face token with **Make calls to Inference
Providers** permission. Never put the token in a command argument or an experiment
log. The stored token failed with HTTP 403 during setup.

```bash
~/lerobot-yam/.venv/bin/hf auth login
bash ~/yam-setup/test-planner.sh preflight
bash ~/yam-setup/test-planner.sh
```

The launcher selects `Qwen/Qwen3.8-27B:novita` through
`https://router.huggingface.co/v1`, disables thinking, uses a 512-token reply budget,
and allows 30 seconds per API request without transport retries. It runs a saved
three-image preflight before loading MolmoAct2 or connecting hardware. This is an
authentication/vision check, not evidence of a successful physical rollout. Novita's
multi-image behavior remains unverified until authentication succeeds. No fallback
silently changes providers. Copy `test-planner.sh` from this directory to
`~/yam-setup/` if setting up a fresh machine.

When the interactive prompt appears:

```text
/trial pilot-planner-01
/start
```

The session automatically starts external steering for **Put all colored cubes in
the green bin.** Qwen sees the three camera images plus up to two previous
observation/command pairs. It selects a concrete text subtask; local MolmoAct2
produces the joint/gripper actions. Autosteering uses a ten-second query interval;
API latency is additional. The initial action queue is held until the first plan.
Subsequent queries run asynchronously while the current subtask continues.

There is no pilot duration limit. `/stop`, `/reset`, or `/finish` ends the attempt
and requests home. After a successful return, record the observed outcome:

```text
/score partial 2 two cubes placed; third grasp failed
```

Use a new pilot ID for another attempt. A planner `done` reply or API error also
ends the attempt and requests home; only the operator scores success. `/quit` ends
the session and disables torque, so support the arms before quitting. Feedback
faults can prevent a completed home return; do not count that as model failure.

After the pilot, freeze the cube inventory and a common positive `timeout_s` in
`~/yam-experiments/manifest.json`, then run:

```bash
bash ~/yam-setup/test-planner.sh scored
```

Use `/trial planner-L01` through `/trial planner-L10`, restoring the named starting
layout between attempts. No automatic restart or scene rearrangement occurs.

## CLI and evidence

```bash
python -m examples.yam_experiments.campaign init ~/yam-experiments --conditions planner hybrid astra astra_icl --cubes red blue yellow
python -m examples.yam_experiments.run \
  --root ~/yam-experiments --launcher ~/yam-setup/rollout-molmoact2.sh \
  --condition planner --pilot \
  --planner.model_id=Qwen/Qwen3.8-27B:novita \
  --planner.api_base=https://router.huggingface.co/v1 --planner.api_key_env=HF_TOKEN
```

Substitute the actual cube inventory; unique labels such as `red-1 red-2` support
duplicate colors. `init` refuses to overwrite an existing manifest. `run` defaults
to a preview; add `--execute` to connect and wait for `/trial` and `/start`. It reads
the setup launcher's flags without executing its shell preamble or branch switch.
A scored run requires `codex/yam-steering-experiments`, a clean revision, a frozen
cube inventory, and a common positive `timeout_s` in `manifest.json`. Pilots always
use duration zero. Choose the scored budget after observing pilot completion times.

Every planner session writes `planner.jsonl` (request text, raw replies, latency and errors),
`events.jsonl` and a local LeRobot dataset containing the three
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

## Scored protocol: 10 layouts × 4 active conditions

1. Freeze the physical cube/bin inventory, cameras, home pose, calibration, gains,
   model/checkpoint revisions, API model, prompts, and time budget after pilots.
2. Photograph and label ten starting layouts. Each layout must be physically recreated
   for every condition; use table markers and reference images, not an RNG seed alone.
3. Follow the manifest order within each layout. It rotates condition positions
   across the ten blocks (four positions cannot be exactly balanced over ten layouts). No automatic advance moves the robot or rearranges cubes.
4. Conditions 2–5 receive no human task hints in scored trials; a physical
   intervention ends the attempt as interrupted. The human condition is excluded.
5. Primary outcome: every cube released fully inside the green bin and stationary for
   two seconds before the trial ends. Also report cubes placed / total, completion time,
   prompt count, grasps/retries/drops, model calls/latency, interventions, and robot faults.
6. Keep all ten attempts per condition, including failed and interrupted attempts. Report
   interruptions separately; do not silently replace them. Ten trials per condition is
   exploratory evidence, so retain per-layout outcomes and uncertainty rather than only
   ranking percentages.

For a formal planner trial, use `/trial planner-L01`, `/start`,
`/finish`, and `/score success N <notes>` after the home return. Other outcomes are
`partial`, `failure`, and `interrupted`. A new trial requires a fresh layout reset.

## External planner integration and review

PR #4675's current head passed fast CI and quality checks. The combined YAM branch
passes targeted tests. The upstream PR is not merged: the experiment branch fixes an
RTC takeover issue demonstrated by
`test_external_planner_first_reply_cannot_release_preplanner_rtc_actions`. Queued and
in-flight actions from before planner takeover must be discarded, and RTC must not
prefill actions from the broad goal while waiting for its first plan.

The harness translates a planner `done` response into a normal completion status, a stopped trial followed by
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
logs. This branch adds `api_key_env`, request timeout, and retry settings to the
OpenAI-compatible client. For HF Router the runner resolves the stored Hub login
into `HF_TOKEN`, without inserting the key into configuration/provenance. Authenticated
Astra use still needs its endpoint and model ID. No model fallback substitutes for Astra.

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

## Single-cube planner contract

The first pilot exposed a failure mode: Qwen repeated the entire multi-cube goal,
then marked that broad instruction as still in progress indefinitely. The cube
experiment now supplies a closed vocabulary of one-cube pick-and-place instructions
(red, orange, yellow, green, blue, purple). This is not an assertion that all six
colors are present: Qwen must choose only visible, unfinished cubes. Different
objects or colors require updating this experiment vocabulary before scoring.

Before the first plan there is explicitly no previous command. The first reply
must select one cube and report `previous_command: none`. A broad goal is rejected
before it can release the initial RTC action queue. Subsequent `in progress`
assessments hold the accepted single-cube command; `completed` permits the next
cube. Invalid replies end the attempt and request home, rather than silently
turning the planner condition into broad-goal-only MolmoAct2.

Each reply prints its scene, previous-command assessment, proposed instruction,
and latency even while normal INFO logs are muted. The selected instruction prints
separately, since an in-progress proposal can be held. Rejected and `done` replies
are retained in `planner.jsonl`. These scene assessments are model claims, not
operator-verified outcomes. The same contract runs in the saved-image preflight.

Planner completion is a successful query status, not a traceback or an automatically
successful trial. It stops autosteering, requests home, and waits for the operator
verdict. Genuine query failures still report errors.

YAM observation reads use the latest camera frame with a 200 ms freshness limit.
This allows 60 Hz interpolated commands between 30 fps camera exposures instead of
waiting for a new exposure on every tick. The motor feedback deadline is unchanged.
A stalled camera still raises an error; actual achieved cadence must be checked in
the next physical run's summary.

## Sol hybrid pilot

The separate `codex/yam-hybrid-sol` branch combines the YAM experiment harness with
[the hybrid supervisor stacked on Pablo's PR](https://github.com/huggingface/lerobot/pull/4823).
On Champagne it uses `~/lerobot-yam-hybrid`, leaving the Qwen checkout available.

```bash
bash ~/yam-setup/test-hybrid-sol.sh api  # API access only, using archived camera frames
bash ~/yam-setup/test-hybrid-sol.sh preflight
bash ~/yam-setup/test-hybrid-sol.sh
```

Set `OPENAI_API_KEY` in that terminal, or enter it at the hidden prompt. Preflight
captures the three cameras and calibrated joint state using `read_only=true`, then
asks `gpt-6.1-sol` for one decision without executing it. A successful preflight
checks endpoint/schema access, not physical performance. `preview` prints launch
arguments without API calls or hardware; `capture` takes only the read-only snapshot.

The launch remains interactive: `/trial pilot-sol-hybrid-01`, then `/start`.
Use `/finish` to return home, then `/score success|partial|failure|interrupted <count> <notes>`.
Sol completion also requests home. A fault can prevent a completed return.

MolmoAct2 uses RTC and the existing normalization/cameras/gripper calibration.
Sol reviews after five seconds of policy execution and after each correction;
the robot holds during the API call. For this first pilot only **gripper** corrections
are enabled (absolute 0 closed, 1 open, at most 2 strokes/s). Joint corrections have
`max_delta=0`; arm directions and collision clearance must be commissioned before
expanding that contract. Policy joint motion retains the existing driver limits.
The supervisor accepts `policy`, `intervention`, `hold`, or `done`; every accepted
decision appears in the terminal and is saved with the raw reply and API latency.
The robot waits for the next review before returning from a direct correction to the VLA.

This is the `hybrid` condition, distinct from the planner-only experiment. Pilot
attempts do not count toward the ten formal trials. Freeze the model, action contract,
layouts, cube inventory and common time budget before collecting scored comparisons.
