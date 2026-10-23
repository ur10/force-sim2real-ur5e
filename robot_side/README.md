# Robot-side collection for the force-aware plan (steps 1 and 5)

`collect_contact.py` is `memory/scripts/collect.py` adapted for the contact-probe and calibration
sessions. It keeps collect.py's GELLO teleop, cameras, replay buffer and viewer, and adds
`ring_rod_probe`'s read-only 500 Hz RTDE wrench log and its event marks. Sessions it writes are
analysed by `ring_rod_probe`'s own scripts, unchanged.

Written from a read-only look at the robot machine (`~/memory/scripts/collect.py`,
`memory/real_robot/robot_controller/gello_env.py`, `~/ring_rod_probe`). Nothing was run there.

## Install (on the robot machine)

```bash
scp scripts/ur_contact/robot_side/collect_contact.py <user>@<robot-machine>:ring_rod_probe/scripts/
scp scripts/ur_contact/robot_side/test_collect_contact.py <user>@<robot-machine>:ring_rod_probe/tests/
# on the robot machine
cd ~/ring_rod_probe && conda activate data_collect
python tests/test_collect_contact.py                         # offline, no robot
python scripts/collect_contact.py --dry-run --plan attempts -l rehearse   # rehearse keys, no robot
```

## What runs what

| Part | Source | Moves the robot? |
|---|---|---|
| teleop, cameras, replay buffer, viewer | `memory` (`GelloEnv`, `KeystrokeCounter`, `scripts/viewer.py`) | yes, exactly as collect.py: `env.calibrate()` at start (asks first), then following the GELLO |
| wrench log | `ring_rod_probe/utils/rtde_logger.py` | no: output recipes only |
| event marks, session files | `ring_rod_probe/utils/events.py`, `session_io.py` | no |
| tare | pendant + `t` to mark (default); `--allow-tare`: `ring_rod_probe/utils/tare.py`'s guarded tare | only the guarded sensor tare |

The script cannot stop the arm. Above `--alarm-n` (25 N default) it shows a red warning, rings the
terminal bell and logs a `force_alarm` event; the UR's protective limits stay the safety layer.
Contact under GELLO is position-controlled with no compliance: approach slowly, and watch the
live force line.

## Sessions (run order as in `ring_rod_probe/README.md`)

```bash
DISPLAY=:1 python scripts/collect_contact.py --plan sensor -l sensor1      # still + free-motion segments
DISPLAY=:1 python scripts/collect_contact.py --plan reg -l reg1            # rod registration captures
DISPLAY=:1 python scripts/collect_contact.py --plan attempts -l attempts_a --ring pla_0p3 \
    --registration outputs/rod_registration.json                         # attempts (+ camera episodes)
```

Each prints its checklist at start and the analysis command at the end:
`characterize_sensor.py`, `register_rod.py`, `segment_attempts.py` + `export_probe.py`.

## Keys

| | Key | Event |
|---|---|---|
| camera episodes | `v` / `b` / Backspace x2 | start / save / drop (`episode_start`, `episode_end`, `episode_dropped`) |
| sensor | `s`, `f` | still and free-motion segments |
| registration | `r`, then `1`-`6` | seated, tip, side+x, side-x, side+y, side-y |
| attempts | `a`, `c`, `o` then `1`-`3` | attempt start, contact, outcome (seated / jammed / aborted) |
| new for the plan | `p` | press-and-hold segment (5-10 N on a jam): the direction signal |
| new for the plan | `h` | note "recovered by hand" saved with the outcome |
| | `t` | tare (mark, or guarded send) |
| | `i` | coverage table: offset bin x tilt bin -> outcomes |
| | `q` (twice if anything is open) | quit and save |

With `--plan attempts` (`--video attempts`), `a` also starts a camera episode and the outcome saves
it, so every attempt has synchronised images in `<video-dir>/ring_rod_<label>` and the episode
index is in the session's events.

## Output

- `data/sessions/<time>_<label>/session.npz`, `meta.json`, `events.jsonl`, `session_info.json`:
  ring_rod_probe's layout (+ episode, hold, note and force-alarm events, + the attempt's live
  offset / tilt / height above the tip when `--registration` is given).
- `<video-dir>/ring_rod_<label>/`: collect.py's replay buffer and videos for the camera episodes.

Hand-back to the simulation machine as in `ring_rod_probe`: `outputs/export/{summary.json,traces.npz}`,
`outputs/rod_registration.json`, `outputs/sensor/sensor_report.json`, `config/measured.json`.
`scripts/ur_contact/compare_probe.py` reads the export directly.

## Things to check before the first session

- The GELLO task: `--task ring_rod` falls back to `default` (`j_init_deg [90, -90, 90, -90, -90, 0]`)
  unless you add a `ring_rod` entry with a start pose near the stand to `memory/config.yaml`.
- `config.yaml` and `ring_rod_probe/config/robot.yaml` must name the same robot IP (checked).
- `ring_rod_probe` noted on 2026-11-18 that the configured payload is 1.0 kg with CoG [0, 0, 0] and
  `actual_TCP_force` reads about -9.8 N on z at rest: the compensation is ~1 kg off. Fix the payload
  on the pendant (gripper + ring) before the sensor session, or record it as it is and let the
  sensor report quantify it; do not change it between sessions.
- The wrench frame and sign check (`ring_rod_probe` acceptance check 4) is still pending: run
  `scripts/verify_wrench_frame.py` once, by hand, with the robot still.

## Deployment (`deploy/`) and the offline probe check

- `make_robot_folder.sh <dest>` assembles everything the robot machine needs (numpy + `ur_rtde`):
  `deploy/`, the policy bundles, the probe check and its library, the session guide, config templates.
- `deploy/deploy_policy.py` runs one attempt with `--policy replay | student | residual | still`
  (`--dry-run` for the stand-in robot, `--air-mm 30` above the tip). **It moves the robot**: the
  operator runs it, following `SESSION_GUIDE.md` step E.
- `deploy/test_deploy.py`: 10 offline tests (observation equivalence with a simulated reference run,
  numpy student and residual against PyTorch, stand-in-robot episodes).
- `probe_check.py` compares a real probe export with the simulated probe library (step 7 of the guide).
