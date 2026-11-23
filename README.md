# From Simulation to Real Contact on a UR5e

**Learn contact recovery in simulation. Calibrate against the real robot. Deploy force-aware insertion.**

Generated demonstrations provide the approach; **DAgger and residual RL** learn the contact response. Real measurements of payload, wrist sensing and fixture position connect the simulator to the physical UR5e.

<p align="center">
  <a href="docs/media/pipeline_overview.mp4">
    <img src="docs/media/pipeline_overview.gif" alt="Three-stage pipeline: Isaac Lab contact simulation and DAgger/residual PPO; real UR5e payload, sensor and frame calibration feeding back into simulation; exported policy running real nut-on-rod insertion." width="100%">
  </a>
</p>

<p align="center"><strong>SIMULATE → CALIBRATE → DEPLOY</strong><br><em>Actual project evidence: simulation reference still, real calibration footage, and a real insertion excerpt. Calibration feeds back into simulation and training; the excerpts are independent, not a paired benchmark.</em></p>

[Watch the overview](docs/media/pipeline_overview.mp4) · [Static overview](docs/media/pipeline_overview.png) · [Hardware rollouts](docs/media/hardware/README.md) · [Calibration evidence](docs/CALIBRATION.md) · [Policy & reward design](#3-learn-contact-recovery-with-residual-ppo) · [Results](#results-and-what-they-do-and-do-not-establish)

| Simulate contact | Calibrate the model | Deploy on hardware |
| --- | --- | --- |
| Generated approaches become physical contact episodes with grasp and fixture uncertainty. **DAgger + residual PPO** learn corrections. | Measure **payload / CoG, wrist noise and rod pose**; use those measurements in the simulated observation chain. | Export **Numpy inference** with matching observation processing. Run at **30 Hz**, with **120 Hz** UR control. |

**Learning approach:** bounded **residual PPO** on a frozen DAgger base, a **3 × 256 actor** with a training-only **asymmetric critic**, and a reward that balances geometric seating, alignment progress, excessive wrench and correction smoothness. [Explore the architecture and design choices ↓](#3-learn-contact-recovery-with-residual-ppo)

**Evidence at a glance:** measured tool payload **1.35 kg**; post-tare cross-pose force error **1.17 → 0.70 N RMS**; held-out simulation evaluations and real rollout footage are included. Hardware policy comparisons remain **preliminary and qualitative**; full contact-parameter identification and a controlled hardware benchmark remain open.

[Precursor: Real2Render2Real](https://github.com/ur10/real2render2real-ur5e) · [Simulation implementation](contact_sim/README.md) · [Research log](docs/PLAN.md) · [Hardware evaluation protocol](docs/HARDWARE_EVALUATION_TEMPLATE.md)

## The problem: good demonstrations do not teach contact recovery

The preceding [Real2Render2Real project](https://github.com/ur10/real2render2real-ur5e) generated robot trajectories from a human demonstration, using reconstruction, retargeting and trajectory repair. Those demonstrations describe **where the manipulated object should move**—but not how it should respond when a real object jams, tilts or shifts inside the gripper.

For nut-on-rod insertion, a few millimetres of grasp or fixture error can turn an apparently perfect trajectory into a contact failure. In the kinematic demonstration generator, the nut follows a prescribed trajectory without contact forces. With no injected in-hand error, 47/48 physically replayed demonstrations entered without contacting the rod in simulation. Successful demonstrations alone therefore provided almost no examples of recovery.

**Research question:** Can a measurement-informed contact simulator turn generated demonstrations into policies that recover from misalignment on a real robot? And does **residual RL** transfer more effectively than imitating a privileged contact expert?

## At a glance

| Component | Implemented in this repository |
| --- | --- |
| **Robot / task** | UR5e, Robotiq 2F-85 fin-ray grasp, nut insertion over a fixed rod; **not** threaded fastening |
| **Simulation** | Isaac Lab 2.0.2 / Isaac Sim 4.5; contact-capable nut/rod geometry and an in-hand compliance model |
| **Hardware calibration** | Wrist F/T characterization, payload and CoG estimation, robot-frame rod registration, measured geometry |
| **Policies** | Privileged-expert DAgger student; bounded residual PPO on a frozen base policy |
| **Real deployment** | Numpy-only inference, synchronized wrist/pose observations, `ur_rtde`, 30 Hz policy and 120 Hz `servoL` |
| **Evaluation** | Held-out simulation tests with 128 attempts per reported condition; preliminary hardware comparison, quantitative release pending |

## Pipeline

```mermaid
flowchart LR
    A[Human demonstration] --> B[Reconstruct, retarget, repair]
    B --> C[Generated approach trajectories]
    C --> D[Physics-based contact episodes]
    E[Real UR5e: payload, sensor, geometry and registration] --> F[Measurement-informed sim parameters]
    F --> D
    D --> G[Privileged expert + DAgger]
    D --> H[Residual PPO]
    G --> H
    G --> I[Hardware-compatible inference]
    H --> I
    I --> J[UR5e contact rollouts]
    J -. Future probe-based model refinement .-> F
```

**What transfers from one task to another:** sensor identification, coordinate-frame verification, probe-based parameter comparison, rollout logging, policy export, safety checks and evaluation protocol. **What must change:** task geometry, grasp/contact model, reset distribution, success detector, reference behavior and task-specific reward. This is a reusable **framework architecture**, not yet a cross-task-validated general-purpose solution.

## Four technical contributions

### 1. Turn kinematic demonstrations into physically executable starts

Generated trajectories provide a nominal approach and insertion intent. Before training, physics replay revealed failures invisible in the rendering-only pipeline: **11/49** original paths grazed the upper rod because collision checks omitted the held nut and part of the fixture. After fixing clearance checks and the post-warp lift, **48/48** regenerated paths passed the collision check with a minimum reported clearance of 2.90 mm.

The contact environment then introduces grasp offset, tilt and fixture uncertainty. Unlike kinematic replay, it exposes realistic failure and recovery states to the policy.

### 2. Build the observation model from the real robot outward

The controller does not receive ideal simulated contact force. The simulated wrist observation models tool gravity and payload compensation, bias/drift, sensor noise, reporting frame and reference point, 500 Hz sampling, latency, causal low-pass filtering and a pre-attempt tare. The same feature construction is used in deployment.

**Measured on the UR5e:** See the [calibration provenance table](docs/CALIBRATION.md) for sources and provisional parameters.

| Quantity | Hardware finding | Why it matters |
| --- | --- | --- |
| Tool payload | **1.35 kg** fitted vs **1.0 kg** configured | An incorrect payload creates orientation-dependent force errors |
| Payload center of gravity | **43 mm** along the tool | Changes torque compensation |
| Post-tare compensated force variation across poses | **1.17 N → 0.70 N RMS** using the fitted payload | Less spurious force input from gravity model mismatch |
| Wrist force noise | **0.11–0.17 N per axis** at rest | Sets the useful scale for low-force contact features |
| Wrist torque noise | **~0.004 N·m per axis** | Limits observability of small contact moments |
| Sensor logging | **500 Hz** RTDE, no observed drops in the characterized session | Supports timed alignment of action and force observations |

<p align="center"><img src="docs/media/payload_calibration.gif" width="85%" alt="REAL: payload identification across multiple robot orientations"></p>
<p align="center"><em>REAL HARDWARE · The fitted tool payload is checked across multiple static wrist orientations.</em></p>

<p align="center"><img src="docs/media/payload_result.jpg" width="80%" alt="Real payload and compensation identification results"></p>

**Fixture registration:** A robot-kinematics-based measurement, from three teleoperated releases, estimates the rod axis in the UR5e base frame. This intentionally does not depend on the available camera hand–eye estimates, which were inconsistent across streams. The experiment also uses measured or image-estimated geometry and grasp length; **the bore and several contact parameters still need direct physical validation**.

<p align="center"><img src="docs/media/registration_demo.gif" width="90%" alt="REAL: logged teleoperated rod registration with a camera view and robot kinematics"></p>
<p align="center"><em>REAL HARDWARE · Rod-axis registration from robot pose logs and release locations.</em></p>

**Probe-based calibration path:** The repository contains an 18-configuration simulated library spanning clearance, effective grip stiffness and friction. A common feature extractor compares contact onset, force-vs-descent slope and seating/jamming outcomes. The ranking mechanism has been checked on **held-out simulation probes**; calibration against a complete real contact-probe dataset is not yet established. See [`robot_side/SIM_LIBRARY.md`](robot_side/SIM_LIBRARY.md).

### 3. Learn contact recovery with residual PPO

**The key choice: learn a correction to a useful policy.** A frozen DAgger student supplies the nominal action; PPO learns a bounded residual that can change lateral motion, descent and tilt when contact calls for recovery. The implementation also supports a force-limited replay base. Freezing the base preserves its behavior during RL training, while initializing the actor's mean near zero starts the mean controller close to that base.

#### Policy architecture: what runs on the robot

```mermaid
flowchart LR
    O["62-D observation: wrench history + TCP state + offset estimate"] --> B["Frozen base: DAgger or replay"]
    O --> X["Concatenate observation + 5-D base action"]
    B --> X
    X --> A["Residual actor: 67 → 256 → 256 → 256 → 5"]
    A --> S["Clip residual and scale correction"]
    B --> SUM["Add base + residual; clip final action"]
    S --> SUM
    SUM --> R["5-D TCP velocity → UR5e"]
    X -. training only .-> V["Critic: 73 → 256 → 256 → 256 → 1"]
    P["6 privileged values: true offset, depth, tilt, in-bore flag"] -. training only .-> V
```

| Component | Implemented architecture / contract |
| --- | --- |
| **Observation — 62 values** | Commanded and measured TCP position offsets (3 + 3); eight six-axis wrench samples (48); previous action (5); XY alignment estimate and availability flag (2 + 1). This is a stacked-history MLP, not a recurrent or image encoder. |
| **DAgger base** | `62 → 256 → 256 → 256 → 5`, ReLU hidden layers, tanh output; normalization fitted on aggregated training data. Frozen during PPO. |
| **Residual actor** | `67 → 256 → 256 → 256 → 5`, ELU hidden layers, tanh mean output; running input normalization. Training samples a Gaussian with learned per-action standard deviation (`log_std` initialized to −1.2); evaluation/deployment uses the mean. |
| **Asymmetric critic** | `73 → 256 → 256 → 256 → 1`, ELU hidden layers and separate running normalization. Its extra six inputs are true XY offset, depth, two tilt components and an in-bore flag. Used only during training. |
| **Action — 5 values** | Normalized TCP velocity `[vx, vy, vz, ωx, ωy]`; no learned yaw or gripper action. At default speed scale, full-action limits are ±6 mm/s laterally, ±8 mm/s vertically and ±0.1 rad/s in tilt. |

**Observation source matters:** the hardware path uses the registered rod position and nominal grasp to estimate alignment. It cannot directly see in-hand displacement. Some published simulation runs instead use an **oracle-derived noisy offset estimate**; those are explicitly separated in the results table. Privileged critic inputs never reach the exported actor, but that alone does not make an oracle-estimate experiment deployable.

For a base action `a_base`, normalized residual `u`, and elementwise scale `s`:

```text
u = clip(residual_actor(observation, a_base), −1, 1)
a = clip(a_base + s ⊙ u, −1, 1)
s = (0.6, 0.6, 0.6, 0.5, 0.5)
```

The scale bounds how much the residual can correct the base before final clipping. With the default velocity limits, that is up to **3.6 mm/s in X/Y, 4.8 mm/s in Z and 0.05 rad/s in tilt**. These bounds constrain commands; they do not guarantee a particular contact force.

Sources: [student](contact_sim/student.py), [actor–critic and PPO](contact_sim/ppo.py), [base policies / privileged vector](contact_sim/base_policies.py), [training and action composition](contact_sim/run_residual_rl.py).

#### Reward design: seat the nut without rewarding a jam

The per-tick reward combines a terminal task outcome with dense progress and time-integrated costs:

```text
r = 10 · first_success − 10 · hard_failure
    + γ Φ(next_state) − Φ(current_state)
    − Δt · (0.1 + 2 · wrench_cost + λres · ||u||² + 0.02 · ||u − u_prev||²)
```

Here `γ = 0.99`, `Δt = 1/30 s` and the default `λres = 0.1`. The following are **implemented design choices and their intent**, not independently validated optimal weights:

| Choice | Implementation | Why this design |
| --- | --- | --- |
| **Reward geometric seating** | One-time **+10** when the rod is in the bore and depth reaches within 1 mm of the configured target (15 mm by default). | A large contact force alone must not count as successful insertion. |
| **Gate depth progress by alignment** | `Φ = −alignment_error + sigmoid((clearance − XY_error) / 0.2 mm) × depth_fraction`. Alignment is scaled by 2 mm and capped at 2; depth fraction is clipped to `[0, 1]`. | Give dense guidance toward alignment while reducing the depth incentive when the nut is outside the opening. |
| **Use potential differences** | Progress is `γΦ(next) − Φ(current)`; absorbing success/failure states have zero potential. Time limits bootstrap the critic instead of becoming hard-failure terminals. | Reward changes in progress rather than repeatedly paying for occupying a favorable state; distinguish timeout from physical failure. |
| **Penalize excessive wrench** | Sum squared excess above per-axis soft thresholds, normalized by the soft-to-hard span; average over physics substeps. Uses simulated wrench shifted to the TCP. | Permit useful contact while charging for sustained or brief excessive loads that a policy-rate sample could miss. |
| **Penalize hard failures** | **−10** on a force abort or invalid penetration beyond **0.5 mm**; invalid physics cannot count as success. | Prevent the learner from benefiting from simulated interpenetration or an abort. |
| **Keep corrections small and smooth** | Squared normalized residual size plus squared change from the preceding residual; a constant time cost discourages delay. | Encourage economical corrections and discourage residual chatter. Slew penalizes the **residual**, not the complete base-plus-residual command. |

The reward's soft wrench thresholds are `(5, 5, 8) N` and `(0.3, 0.3, 0.3) N·m`; its normalization upper thresholds are `(30, 30, 40) N` and `(2, 2, 2) N·m`. These are configurable starting choices. **The upper thresholds normalize the cost; they are not per-axis termination limits.** The simulation's separate default abort is a **40 N force norm**; the deployed controller has a separate **20 N hardware force stop**. Lower reward cost is therefore not a hardware safety guarantee.

Sources: [reward terms and defaults](contact_sim/residual_reward.py), [environment success / abort logic](contact_sim/contact_env.py), [offline reward checks](contact_sim/test_contact_modules.py).

#### Training recipe and the imitation-learning baseline

**PPO:** clipped objective (`ε = 0.2`), GAE (`γ = 0.99`, `λ = 0.95`), Adam at `3e−4`, five epochs and four minibatches per update, value-loss coefficient `0.5`, no entropy bonus, and gradient-norm clipping at `1.0`. The runner defaults to 128 parallel environments and 48 policy ticks per rollout chunk; `--lr-anneal` optionally decays the learning rate to 10% over the training budget. Completed-episode samples are masked from updates, and a small-batch guard avoids unstable updates when too few valid samples remain. Exact experiment overrides live in [`runs/`](runs/).

**DAgger:** a privileged contact expert labels the states visited by the student. Four aggregation rounds reduce expert participation from **100% → 50% → 25% → 0%**, and the student minimizes action-matching MSE. The expert's decisions must be inferable from the student's inputs: a privileged alignment-gated hovering strategy produced a hovering student, while a contact-driven descend–release–shift strategy supplied more useful recovery behavior. See the [DAgger schedule](contact_sim/dagger.sh).

**Why try residual RL after imitation?** DAgger optimizes agreement with expert actions; PPO optimizes the downstream insertion outcome under its own rollouts. That makes contact recovery under imperfect grasp, registration and sensing a useful target for residual learning. The preliminary hardware advantage is consistent with this motivation, but does **not** establish the mechanism: matched starts, force budgets and failure analysis are still needed to distinguish better recovery from simply pushing harder.

### 4. Keep the training/deployment observation contract identical

The same observation dimensions, normalization, wrench transforms, causal filtering and action integration are rebuilt for deployment. The robot-side policies run in Numpy without Isaac or PyTorch. The existing offline suite includes **10 deployment tests**, checking observation equivalence to float32 precision, student/residual inference equivalence and simulated stop behavior.

Policy frequency: **30 Hz**; UR control via `servoL`: **120 Hz**. Deployment includes configured workspace and orientation/tracking boundaries, robot-mode checks, operator intervention and a **20 N hardware force stop**. These are operational safeguards, **not a formal safety guarantee**; review the [session guide](robot_side/SESSION_GUIDE.md) before any robot motion.

## Results and what they do (and do not) establish

### Preliminary hardware finding

> **Author-observed:** In an independent evaluation on the physical UR5e, residual RL performed better than the tested imitation-learning policy. A paired benchmark and supporting data—success counts, contact force, elapsed time, offsets, termination reasons and checkpoint identifiers—will be added here. Until then this is a **preliminary qualitative finding**, not a published numerical improvement.

**Why this matters:** A policy that looks comparable to imitation in simulation may behave differently once contact, compliance, sensor artifacts and reset variability come from the physical system. The next experiment is designed to identify the source of that difference.

### Reproducible simulation reference

<p align="center">
  <img src="docs/media/comparison_measured.jpg" alt="Five simulated insertion strategies evaluated on the same held-out examples" width="95%">
</p>
<p align="center"><em>SIMULATION · Same held-out insertion starts, comparing replay, DAgger, residual RL and a privileged teacher. An invalid physics episode is counted as failure.</em></p>

All results below are **SIMULATION**, measured-parameter configuration, fin-ray grasp, **0.5 mm radial clearance**, 128 held-out episodes per evaluation, one training seed per policy. See [`results/sim_evaluations.json`](results/sim_evaluations.json).

| Policy / observation assumption | Seated / 128 | Interpretation |
| --- | ---: | --- |
| Blind force-limited replay | **35** | Nominal motion without learned recovery |
| DAgger with **deployable kinematic estimate** | **78** | Practical estimator does not know in-hand displacement |
| DAgger with **oracle-derived noisy offset estimate** | **91** | Evaluation assumption not available on the current robot |
| Residual PPO on the **oracle-estimate DAgger base** | **93** | Comparable to the oracle-estimate DAgger row; overlap is substantial |

**Do not compare 93/128 with 78/128 as an RL gain:** those rows use different offset-estimation assumptions. The repo documents a training path for residual PPO on the **kinematic-estimate** DAgger base, but the same-condition held-out evaluation and hardware results should be published together before claiming improvement.

At **2.7 mm clearance** (the physical nut's approximate camera-derived fit), blind replay already succeeded in **122/128** simulated attempts. Tight clearances and deliberately varied grasp/fixture offsets are thus much more informative than nominally easy starts.

<p align="center"><img src="docs/media/residual_learning_curve.png" width="85%" alt="Residual policy held-out simulation success over training time"></p>
<p align="center"><em>SIMULATION · Learning curves across grasp models. Curves are not direct evidence of hardware transfer.</em></p>

### What the calibration results mean

Sensor identification corrected a real mismatch in the payload model. However, substituting the measured noise/drift model into the reported simulation changed policy outcomes by at most five of 128 episodes in the published conditions. **This does not establish a hardware performance gain from calibration.** The right experiment is a controlled hardware ablation: measured vs deliberately mismatched sensor/contact parameters, with identical actors, training budgets and held-out starts.

## Real-robot evaluation footage

The supplied `force_rollout` footage is now included: **three source groups, five original clips per group**. The numbered clip files provide camera views; the previews use **camera view `4`**, selected for its clear side view of the gripper, nut and rod. Recordings retain their original folder names; policy/checkpoint labels have not been supplied. File counts are not independent trial counts, and no success rates, matched comparisons or force measurements are inferred from the clips.

Click a poster to open the full-speed preview (camera view `4` from each group):

| Recording 0 · 26.53 s | Recording 1 · 22.27 s | Recording 2 · 28.43 s |
| --- | --- | --- |
| [![REAL UR5e: source group 0](docs/media/hardware/group-0.jpg)](docs/media/hardware/group-0.mp4) | [![REAL UR5e: source group 1](docs/media/hardware/group-1.jpg)](docs/media/hardware/group-1.mp4) | [![REAL UR5e: source group 2](docs/media/hardware/group-2.jpg)](docs/media/hardware/group-2.mp4) |

[Browse all 15 original clips and provenance](docs/media/hardware/README.md) · [Hardware evaluation protocol](docs/HARDWARE_EVALUATION_TEMPLATE.md)

Preview durations describe the video files, not insertion completion times. Original clips are preserved byte-for-byte; preview copies retain their frames and timing. Checkpoint IDs, outcomes, offsets and synchronized force logs remain to be supplied before a quantitative hardware comparison can be reported.

## Reusable calibration workflow

For a new contact task, the intended workflow is:

1. **Register geometry and frames:** measure the manipulated part, fixture, grasp offset, TCP and sensor reporting frame.
2. **Identify the observation channel:** fit tool mass/CoG, characterize bias/drift/noise, log timestamps and latency, verify wrench axes and moment reference point.
3. **Probe contact:** collect a small, controlled set of offset/tilt/force tests; compare onset, force slope and outcomes against simulated parameter candidates.
4. **Build a variability model:** fit effective grip compliance/friction and randomize around defensible measured ranges rather than arbitrary values.
5. **Learn corrections:** use nominal demonstrations and an imitation or residual RL policy; keep privileged state on the training side only.
6. **Validate on robot:** run paired starts, monitor limits, log failure modes and report both completion and force/cycle-time costs. Feed new identification data back into the simulator.

**Built and evidenced here:** the measurement and registration tools, contact simulator, candidate-probe comparison software, training approaches and deployable inference. **Not yet demonstrated:** a completed real-contact parameter fit followed by zero-shot transfer across multiple independent tasks or robots.

## Repository guide

| Directory | Contents |
| --- | --- |
| [`contact_sim/`](contact_sim/) | Isaac contact environment, wrench sensor, DAgger, residual PPO, training and evaluation |
| [`robot_side/`](robot_side/) | Calibration capture, probe comparison, Numpy deployment, offline tests and operator guide |
| [`generator_fix/`](generator_fix/) | Collision and trajectory export improvements found during physics replay |
| [`assets/`](assets/) | Demonstration trajectories, configuration measurements, rod registration and policy bundles |
| [`results/`](results/) | Sim evaluation JSON and measured sensor/payload results |
| [`runs/`](runs/) | Queues and experiment invocations |
| [`docs/PLAN.md`](docs/PLAN.md) | Original working plan and dated lab notebook (historical status updates are not current) |

## Reproduce the implementation

**Simulation** requires Isaac Lab 2.0.2 / Isaac Sim 4.5 and a compatible [`real2render2real-ur5e`](https://github.com/ur10/real2render2real-ur5e) checkout, which provides the UR5e / Robotiq models and task assets.

```bash
./install.sh /path/to/real2render2real
cd /path/to/real2render2real
.venv/bin/python scripts/ur_contact/test_contact_modules.py
# Training, held-out evaluation and exact example commands: contact_sim/README.md and runs/
```

Robot-side **offline checks** (do not move the robot):

```bash
robot_side/make_robot_folder.sh ~/force_deploy
cd ~/force_deploy
python deploy/test_deploy.py
python deploy/deploy_policy.py --dry-run -y --policy student
```

**Do not run live robot commands without checking `robot_side/SESSION_GUIDE.md`, establishing hardware stops and completing lab operator approval.**

## Limitations and next measurements

- Initial hardware advantage of residual RL is **not yet quantified publicly**; publish matched trials, initialization distribution, training seed, policy/checkpoint IDs, confidence intervals and force/time trade-offs.
- Most numeric policy evaluations have **one training seed**; uncertainty intervals across evaluation episodes are not training-seed variability.
- Printed part bore / grip slip / friction / dynamics need direct real-contact validation. Current elastic in-hand spring does not reproduce irreversible grasp slip.
- The full vision-conditioned approach-and-contact policy is **not implemented** here; the deployable policy currently uses a kinematic offset estimate, not live camera images.
- The framework is demonstrated on **one UR5e contact task**. Cross-task portability remains a design goal rather than a verified result.
- Higher success with RL can be accompanied by **higher peak forces**; report force, safety stops and time alongside seating rate.

## Related work

This project builds on [Real2Render2Real](https://real2render2real.com/), Factory, IndustReal, DAgger, asymmetric actor-critic methods and residual reinforcement learning. Detailed methods, decisions and references are retained in [`docs/PLAN.md`](docs/PLAN.md) and [`contact_sim/README.md`](contact_sim/README.md).

MIT license — see [`LICENSE`](LICENSE).
