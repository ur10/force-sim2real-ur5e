# From Demonstrations to Contact

**Calibration-driven sim-to-real learning for contact-rich manipulation on a UR5e**

*Isaac Lab · Force/torque sensing · Hardware calibration · DAgger · Residual PPO · UR5e deployment*

## Overview

A robot can imitate the trajectory of a successful assembly and still fail as soon as the grasp shifts or the parts meet at the wrong angle. My earlier **Real2Render2Real** pipeline generated trajectories from human video, but those trajectories carried no meaningful information about contact forces or recovery.

I extended that work into a contact-aware sim-to-real pipeline on a **UR5e with a Robotiq 2F-85 gripper**. The system starts from generated approach motions, executes the contact phase in Isaac Lab, builds a wrist force/torque observation model informed by real robot measurements, and trains two kinds of feedback policies: a **DAgger student** that imitates a privileged contact expert, and **residual reinforcement learning** that learns bounded corrections to a nominal behavior.

**Preliminary hardware result:** In a separate evaluation on the physical robot, the residual RL controller behaved better than the imitation-learning baseline. This is an early qualitative result; I will publish matched trial counts and force/time metrics alongside the supplied rollout footage. The existing numerical success results are simulation-only.

## The system in four stages

**1 — Demonstrations become physical starts.** The kinematic generator supplies the approach and nominal insertion motion. A contact-capable PhysX scene introduces in-hand offsets, tilt, fixture registration error and fin-ray grasp compliance. Running those trajectories with physics exposed collision-checking bugs in the original generated data; I corrected the planner and regenerated the trajectories.

**2 — Calibration connects the two domains.** I measured wrist noise and drift, fitted the tool payload and center of gravity, and registered the rod in the robot base frame using teleoperated release poses. These measurements enter the simulated wrist-sensor chain, with the same reference-point transforms, filtering and observation contract used during deployment. A separate probe-matching module compares simulated contact signatures across candidate stiffness, friction and clearance values. Real-contact identification remains the next step.

**3 — Policies learn to recover.** The imitation baseline learns from a privileged simulation teacher but acts only on wrench history and robot-state features. Residual PPO instead learns a bounded correction over a frozen base policy. Its critic receives extra simulator state during training, while the deployed actor uses only observable signals. Both policies operate as closed-loop task-space velocity controllers during contact.

**4 — The same observation reaches the real robot.** I implemented Numpy inference and a robot-side controller that recreates the simulator's observation processing. Ten offline tests check observation and policy equivalence before robot execution. Deployment runs the policy at 30 Hz with UR commands at 120 Hz, subject to robot-side stopping conditions and operator control.

## Why the hardware result is interesting

In the reported tight-clearance simulation, residual RL and DAgger with an oracle-derived offset estimate achieved similar success (93/128 vs 91/128 seated). The advantage seen in my preliminary hardware tests was therefore not obvious from the simulator alone.

One possible explanation is **objective mismatch**. Imitation learning rewards agreement with a teacher; reinforcement learning directly optimizes completing the insertion while balancing excessive force and action smoothness. On real hardware, unexpected grasp shifts, compliance and timing errors may create states in which repeating the teacher's nominal correction is less effective than searching and recovering from contact. This is a hypothesis—not yet a causal conclusion—and I am checking it using matched starts, force traces and failure modes.

## Calibration as a reusable capability

The most reusable part of this work is not a nut-specific policy. It is the loop connecting **measurement → simulation → training → robot execution → new measurements**. The robot-side tools handle payload identification, frame checks, contact probes, logged deployment and comparable evaluation. A different insertion task would still require new contact geometry, resets and reward/success definitions, but would reuse much of the measurement and deployment machinery.

This is a **task-portable framework in development**, currently evaluated on one contact assembly—not a claim of demonstrated zero-shot generalization to arbitrary robots or objects.

## Hardware videos

[Three hardware previews and all 15 original clips](media/hardware/README.md) are available in the repository. Preview labels preserve source groups `0`, `1` and `2`; each shows camera view `4` at its original playback rate, selected for its clear side view of the insertion. Policy/checkpoint metadata have not been supplied, so these clips are not labeled as a matched comparison, a verified success set, or a robustness benchmark. Quantitative hardware results still require the corresponding trial metadata and force logs.

## Selected implementation details

- **Robot:** UR5e + Robotiq 2F-85 fin-ray grasp; nut-on-rod insertion (not screw threading).
- **Sim:** Isaac Lab 2.0.2 / Isaac Sim 4.5; 240 Hz contact physics, 30 Hz policy.
- **Sensor:** 500 Hz wrist samples; measured force noise 0.11–0.17 N per axis at rest.
- **Payload identification:** 1.35 kg fitted vs 1.0 kg configured; after-tare cross-pose force error reduced from 1.17 N to 0.70 N RMS.
- **Learning:** DAgger / 3×256 student; bounded residual PPO with an asymmetric critic.
- **Deployment:** Numpy policy inference, `ur_rtde`, 120 Hz `servoL`, offline equivalence tests and operator stops.

**Code:** https://github.com/ur10/force-sim2real-ur5e

**Precursor project:** https://ur10.github.io/projects/real2render2real/

*Evaluation note: All numerically quoted policy success rates above are simulations. The preliminary hardware finding is author-reported pending the public controlled evaluation.*
