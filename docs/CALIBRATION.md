# Calibration provenance

Values describe the recorded bench setup, not universal robot defaults. Source status is kept separate from software configuration; remeasure geometry and frames for a new setup.

| Parameter | Method and value | Status / session | Source and implementation |
| --- | --- | --- | --- |
| Tool payload / CoG | Static wrist orientations; fitted 1.35 kg and 43 mm tool-axis CoG, versus 1.0 kg configured | Hardware fit, `20261122_132417_payload_demo` | [Payload fit](../results/real_payload_fit.json) |
| Wrist noise | Stationary samples; 0.11–0.17 N per force axis, approximately 0.004 N·m per torque axis | Hardware characterization, 2026-11-22 | [Sensor results](../results/real_sensor.json), [sensor model](../contact_sim/wrench_sensor.py) |
| Rod axis / tip | Three teleoperated releases and robot kinematics; base-frame XY = (−85, −810) mm, tip Z = 180 mm | Hardware estimate, author-confirmed, 2026-11-22 | [Registration with method](../assets/contact/rod_registration.json) |
| Rod radius / nut outer radius / nut height | 8 mm / 16 mm / 14 mm; camera estimates checked against external measurements | Estimated geometry, 2026-11-22; see per-field uncertainty | [Geometry sources](../assets/contact/measured.json) |
| Nut bore / radial clearance | RGB silhouette fit: 10.71 mm bore radius, approximately 2.7 mm radial clearance | Provisional, not directly measured | [Geometry notes](../assets/contact/measured.json); 0.5 mm benchmark clearance is a simulation condition |
| Nut mass / grasp length | Estimated 8 g; nut bottom 225 mm below flange | PLA volume / camera-based estimates, 2026-11-22 | [In-hand values and uncertainty](../assets/contact/measured.json) |
| Effective grip stiffness / friction | 18 candidate probe configurations | Simulation library; complete real-contact identification pending | [Probe methodology](../robot_side/SIM_LIBRARY.md), [library](../assets/probe_library/library.json) |
| Filter / controller rates | 20 Hz low-pass; 30 Hz policy, 120 Hz control | Software configuration, not a fitted physical parameter | [Deployment observation and control](../robot_side/deploy/deploy_core.py) |

The deployable offset estimator cannot observe in-hand displacement. Real-contact stiffness, friction, bore validation and irreversible grasp slip remain open measurements. The [hardware evaluation protocol](HARDWARE_EVALUATION_TEMPLATE.md) specifies the controlled comparisons needed to test whether calibration improves hardware performance.
