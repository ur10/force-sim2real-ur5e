# Recruiter-facing repository and project page: implementation guide

## The 30-second test

A visitor should see, without scrolling far:

1. **A single-sentence problem:** A video-derived insertion trajectory lacks force-aware recovery when the real grasp or fixture is misaligned.
2. **The technical answer:** Measurement-informed contact twin + force-conditioned DAgger / residual RL + robot-compatible inference.
3. **The credible evidence:** a real calibration result; an accurately labeled simulation benchmark; a separately labeled *preliminary* hardware RL observation and clips with numbers when available.
4. **Your ownership:** what you developed, reused, measured, fixed and evaluated.

The rewritten README uses this ordering. It is intentionally not a chronological engineering diary; `docs/PLAN.md` remains that diary.

## Highest-priority changes on GitHub

| Priority | Change | Why |
| --- | --- | --- |
| P0 | Replace root `README.md` with the rewritten version after checking factual hardware phrasing | Remove the outdated 'hardware trials pending' headline |
| P0 | Publish matched hardware rollouts: count, denominator, checkpoint hashes, offsets, force and completion time | Establish the hardware RL claim |
| P0 | Add 3 short videos to the project page; link posters in README | Show physical transfer in under 30 seconds |
| P1 | Add GitHub About description, project-page URL and topics | Make the repo interpretable from a search result |
| P1 | Move bulky run and implementation details behind `contact_sim/README.md`, `runs/` and `docs/PLAN.md` | Keep first view focused on results and contributions |
| P1 | Add a table: parameter, measurement method, value/range, measured vs provisional, code location | Make calibration traceable and portable |
| P1 | Provide full hardware logs and aggregation script (not just edited videos) | Let engineers independently evaluate the transfer claim |
| P2 | Add `configs/task.yaml` schema and a reusable `calibration/` interface | Show how the same machinery could be configured for a second task |
| P2 | Add one second insertion task or held-out fixture/part, with before/after calibration | Support the general-purpose framework claim |

**Suggested GitHub About:** `Measurement-driven sim-to-real contact learning on a UR5e: force/torque calibration, Isaac Lab contact simulation, DAgger, residual PPO and real-robot deployment.`

**Suggested topics:** `robotics`, `sim2real`, `reinforcement-learning`, `residual-rl`, `isaac-lab`, `robot-manipulation`, `force-control`, `system-identification`, `ur5e`, `imitation-learning`.

## Calibrated framework versus one calibrated task

Be precise with the terms:

- **Already measured:** payload, CoG, stationary wrist sensor noise/drift, rod position from robot kinematics, some geometry and observation/export alignment.
- **Built and simulated:** parameter-grid probe comparison, effective compliance model, wrench chain, DAgger, PPO, deployment dry-run equivalence.
- **Still to demonstrate on real hardware:** identified contact stiffness/friction/clearance from real probe data; calibrated vs uncalibrated *hardware* gain; portability to a new task.

For a *general-purpose lab framework*, create a small task config exposing geometry, `T_base_fixture`, `T_flange_tcp`, wrench reference point/frame, payload/CoG, filter parameters, controller limits, contact parameter prior/range, success condition, and nominal policy path. Each field should carry a **source tag** (`measured`, `estimated`, `sim-identified`, `assumed`) and a timestamp/session ID. This is a proposed refactor, not a description of existing interfaces.

Recommended README diagram: **Demo extraction -> Physics contact -> DAgger / residual PPO -> On-robot control -> Calibration/probes feedback**. The figure should visibly distinguish the *real measurements* entering the simulator and the *privileged teacher/critic signals* that are omitted at deployment.

## The three-video storyboard

**Hero (optional):** a 6–10-second montage with real-robot contact, release, and insertion. Keep it silent and obvious; it is a teaser, not experimental evidence.

**Video 1: controlled comparison (best evidence).** 15–30 s. Baseline left, RL right; preserve identical camera framing, align the contact event and annotate both actual start offsets. State if starts could not be made identical.

**Video 2: contact recovery (how).** 15–30 s. Real-time motion with a compact timeline `approach | contact | correction | insertion`; use `Fz` and lateral force trace aligned to video. Show how the actor changes commands when contact starts.

**Video 3: generalization or limits (credibility).** 15–30 s. Different reset, different grasp or a failed attempt. A failed trial with a clear analysis is often more persuasive to a technical recruiter than another easy success.

For each, display: `REAL UR5e`, policy checkpoint, initial offset/tilt, result, runtime, peak force, whether the run was selected from a larger trial set. Put aggregate success rates *outside* the videos with exact sample sizes.

**Web media implementation:** host compressed MP4s on the project site with `<video controls preload="metadata" playsinline poster="...">`; include accessible captions and a text summary. In GitHub README, use static poster images as links to videos/site; don't rely on raw HTML video playback everywhere.

## Research interpretation: how residual RL might win on hardware

| Hypothesis | Expected trace | Best discriminating test |
| --- | --- | --- |
| Better closed-loop recovery after off-nominal contact | Higher fraction of jammed attempts that release and retry successfully | Pair initial offsets, mark first-contact time and trace XY correction + success-after-contact |
| Expert-action imitation is brittle to latent errors | DAgger fails after wrench/pose combinations absent from simulated expert data | Cluster failure states and compare hardware states against DAgger rollout distribution |
| End-task reward handles action-label ambiguity | RL makes several successful trajectories from the same ambiguous sensed state | Compare policy actions and rewards at similar observations with different hidden grasp errors |
| RL simply pushes harder | Higher peak/integrated force explains apparent extra successes | Equalize action/force budget; report >5 N duration and 20 N stops as well as success |
| Evaluation or checkpoint mismatch | Advantage disappears on paired, randomized, independently scored test set | Same feature source, resets, force limits, cycles, checkpoint selection protocol |

Even if the first hypothesis is true, a learned policy cannot *provably* overcome unobservable in-hand displacement without informative interaction. Emphasize an empirical recovery mechanism rather than universal robustness.

## A compelling calibration ablation

Hold the policy architecture and training budget constant; train on: (a) ideal/uncalibrated observation chain, (b) measured sensor chain + provisional contact, (c) measured sensor + identified contact + bounded randomization. Run the same held-out real starts. Separately measure model prediction error for contact onset and force slope. This isolates whether calibration improves sensory alignment or policy success and whether randomization provides the extra robustness.

The present repository's measured-sensor simulation comparison showed only modest performance differences; do not retroactively call that a positive hardware calibration ablation.
