# contact_sim: the nut-on-rod contact simulation and the policies

Installed to `scripts/ur_contact/` of a real2render2real checkout (see the top-level README). Isaac Lab
2.0.2 / Isaac Sim 4.5 (PhysX). Every hardware number in `params.py` carries its source (rgb_fit / urdf
/ factory / placeholder / choice); measured values load from a JSON with the same nesting (`--params`).

## Physics and sensing

| File | What it does |
|---|---|
| `params.py` | All parameters: geometry, in-hand grasp model (`finray` default, `center`), contact, sensor chain, controller, solver. `load(path, overrides)`. |
| `ring_rod_sim.py` | Scene for N envs: UR5e + 2F-85 (stiff joint position control), rod as a convex frustum with a tip chamfer, nut as 32 convex sectors, a 6-DoF spring joint from the gripper to the nut (fin-ray: pivot in the pinched band, per-axis gains), visual fin-ray blades, Factory solver settings, phantom 1 kg URDF frames set to 1 g. |
| `wrench_sensor.py` | Simulated UR wrist channel: tool weight, bias + drift, noise, payload compensation with configurable mass/CoG error, reference point and frame, sample-and-hold, latency, causal first-order low-pass, a tare that averages 0.2 s. |
| `admittance.py`, `torch_fk.py` | Implicit admittance (probe controller), damped least-squares IK on the commanded configuration (servoJ-style), batched FK and Jacobian. |
| `contact_utils.py` | Pose helpers, numpy IK, nut-vs-rod metrics (depth, offset, penetration test), the fin-ray grasp geometry, the measured-grip override. |

## Probes and replays (no learning)

| File | What it does |
|---|---|
| `probe_protocol.py`, `run_contact_probe.py` | Offsets x tilts, constant-speed descent with the admittance controller or a stiff descent with a force stop (`--controller stiff`); validates the wrench readout against a known load first. |
| `run_physical_replay.py` | Executes the exported generated trajectories with physics and sampled in-hand errors (`--mode replay` / `expert`). |
| `expert.py` | Hand-designed wrench-only controller (descend, back off, step sideways). |
| `build_probe_library.py` | Sweep runs -> feature library for the offline check (`robot_side/probe_check.py`); writes sim runs in the real export format for tests. |
| `compare_probe.py`, `plot_probe.py`, `animate_probe.py` | Probe features, plots, cross-section animations (`--layout tall` for phones). |
| `summarize_speed_sweep.py` | Speed x impedance probe sweep and the expert speed sweep. |
| `real_sensor_stats.py` | Noise, drift and bias of a REAL ring_rod_probe session next to the sensor model. |

## Policies

| File | What it does |
|---|---|
| `contact_env.py` | Batched contact-phase env: entry pose from the export, per-env in-hand error (fin-ray: across the fingers, along the closing axis, rocking), approach error, TCP-velocity actions through IK at 120 Hz (30 Hz policy), deployable observation (62-d), privileged state. Offset estimate: `oracle` (true + bias + noise, a vision stand-in) or `kinematic` (registered rod - nominal-grasp prediction from the TCP; deployable). `speed_scale` for speed sweeps. |
| `privileged_expert.py` | The DAgger teacher: sees the true offset and tilt. `contact` style (descend, lift until the force releases, step toward the axis; decisions follow observable force) and `hover` style (not imitable). |
| `student.py`, `train_student.py` | MLP student (3 x 256, tanh output) with per-feature normalisation and wrench / vision masks for ablations. |
| `run_dagger_rollouts.py`, `dagger.sh` | DAgger collection (expert with probability beta, else student; every visited state labelled by the expert), training, held-out evaluation against the baselines; `--policy residual` evaluates residuals; `--video-dir` films close-ups. |
| `ppo.py` | PPO with an asymmetric actor-critic (the critic also sees privileged state), GAE, running observation normalisation, tiny-batch guard. |
| `residual_reward.py` | Potential-based progress reward, success bonus / failure cost, wrench cost with soft/hard limits averaged over physics sub-steps, residual size and slew costs; unit-tested. |
| `base_policies.py`, `run_residual_rl.py` | Frozen bases (blind force-limited replay or a DAgger student) and the residual PPO training loop (wall-clock budget, LR anneal, checkpoints). |
| `export_deploy_bundle.py`, `log_deploy_reference.py` | Export a student to the numpy bundle the robot runs; log a simulated reference run for the deployment equivalence tests. |
| `render_utils.py` | Close-up cameras, labels, video writing. |
| `test_contact_modules.py` | Offline tests (no Isaac): sensor chain, admittance, FK, metrics, rewards, ... |

## Run

```bash
.venv/bin/python scripts/ur_contact/test_contact_modules.py
# DAgger (contact teacher), measured parameters, 0.5 mm clearance, kinematic offset estimate
PARAMS=outputs/ur_nut/contact/measured.json ENV_JSON='{"vision_source": "kinematic", "vision_noise": 0.0001}' \
  bash scripts/ur_contact/dagger.sh outputs/ur_nut/dagger_kinematic '{"geometry": {"ring_bore_r": 0.0085}}'
# residual RL on a DAgger student
scripts/ur_cube/isaac_run.sh outputs/ur_nut/residual/train.log "RL DONE" scripts/ur_contact/run_residual_rl.py \
  --num-envs 128 --minutes 90 --lr-anneal --params outputs/ur_nut/contact/measured.json \
  --set '{"geometry": {"ring_bore_r": 0.0085}}' --base student --student outputs/ur_nut/dagger_kinematic/student_it4.pt \
  --env '{"vision_source": "kinematic", "vision_noise": 0.0001}' --output outputs/ur_nut/residual/kinematic_student_base
# held-out evaluation (128 episodes, seed 100)
scripts/ur_cube/isaac_run.sh outputs/ur_nut/eval.log "ROLLOUTS DONE" scripts/ur_contact/run_dagger_rollouts.py \
  --policy residual --residual outputs/ur_nut/residual/kinematic_student_base/residual.pt --num-envs 64 --rounds 2 --seed 100 \
  --eval --params outputs/ur_nut/contact/measured.json --set '{"geometry": {"ring_bore_r": 0.0085}}' \
  --env '{"vision_source": "kinematic", "vision_noise": 0.0001}' --output outputs/ur_nut/eval.npz
```

`runs/` at the repository root has the exact queues that produced the results in the top-level README.
Isaac runs on one GPU (`ISAAC_GPU`, default 0) through `scripts/ur_cube/isaac_run.sh` (from the
real2render2real-ur5e overlay), which stops the process once its done-line is printed.
