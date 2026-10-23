# Session guide: recording the ring-on-rod calibration data (for the agent on the robot machine)

You are helping the user record real UR5e data that calibrates a contact simulation on another
machine. **The user drives the robot. You guide, check outputs, and run analysis.** Read this whole
file before doing anything.

---

## ▶ DO THIS NEXT (added 2026-11-22): get ready for the first policy deployment

> [!IMPORTANT]
> **Update 2026-11-22 (evening): the deployment is here (step E).** The user decided to treat the
> provisional values as correct for a first deployment (camera-estimated geometry, the 225 mm grasp
> length, the rod-registration guess, wrench sign +1), so E can run before A-D. A-D stay the way to
> replace those values; the deployment picks up `outputs/rod_registration.json` and the newest
> `data/wrench_check/*/wrench_frame.json` automatically once they exist. **You never run
> `deploy/deploy_policy.py` without `--dry-run` yourself**: give the user the command. The rules in
> the next section apply to every step here.

**Status on 2026-11-22**

| | Item | State |
|---|---|---|
| done | Sensor session `20261122_123422_sensor1` | noise 0.11-0.17 N per axis, drift <= 0.023 N/s |
| done | Payload sessions `20261122_130724_payload`, `20261122_132417_payload_demo` | fitted 1.35 kg, CoG 43 mm along the tool (pendant still says 1.0 kg) |
| **open** | **A. Wrench frame and sign check** | the earlier recording `data/wrench_check/20261122_122644_wrench` has **1 of 5 presses**: incomplete |
| **open** | **B. Rod registration by touch** | not done; only a guess from old data: axis (-0.085, -0.810) m +/-8 mm, tip z 0.180 m +/-15 mm |
| **open** | **C. Calipers: nut bore, rod tip chamfer, grasp length** | provisional camera values only |
| **open** | **D. Hand-back** | after A-C |
| **ready** | **E. Deployment** (`deploy/`, `sim_library/deploy/`) | offline tests and dry runs pass on the simulation machine |

Why these three: the policy reads the wrench as the load on the tool in the base frame (A); its
offset input is the registered rod position minus where the TCP pose says the nut is (B, and the
grasp length from C); and whether force feedback matters at all depends on the clearance and the
chamfer (C): in simulation, blind replay already seats 122/128 at the camera-estimated 2.7 mm.

### A. Wrench frame and sign check (about 10 min, robot still)
1. Before starting, turn the wrist about 45 deg about vertical (pendant, or a short GELLO session),
   so the tool axes are not parallel to the base axes. Then stop all motion: no teleop, no freedrive.
2. The user runs, from a desktop terminal (it only reads over RTDE and sends nothing):
   ```bash
   cd ~/ring_rod_probe && conda activate data_collect
   python scripts/verify_wrench_frame.py
   ```
3. Follow its prompts. Hands off for each 1.5 s baseline, then press gently (10-20 N) for about
   3 s: towards base +x, +y, +z (upwards), then towards +x at two points on the tool axis at known
   distances from the flange face (on the gripper body, then near the fingertips). Measure those two
   distances with a ruler.
4. You: read the printed table, paste it into `README.md`'s conventions section, and report to the
   user, in one line each: the sign (positive = the load on the tool, or reversed), the frame (base or
   tool) and the torque reference point (flange or TCP). Any press that does not show up on the
   expected axis: redo the check (new recording; keep the old one).

### B. Rod registration by touch (about 15 min, GELLO)
1. The user grasps the nut **exactly as it will be held for the attempts and the deployment**, and
   keeps this grasp and wrist yaw from here on. A new grasp later means a new registration.
2. Section "3. Registration session" below has the protocol; the command is
   ```bash
   DISPLAY=:1 python scripts/collect_contact.py --plan reg -l reg1
   ```
   Captures: `r` `1` seated (x1-2, hold 2 s); `r` `2` tip (x4: from +x, -x, +y, -y); `r` `3`-`6`
   side +x, -x, +y, -y.
3. You run:
   ```bash
   python scripts/register_rod.py data/sessions/*_reg1 --measured config/measured.json -o outputs/rod_registration.json
   ```
   Read its printout to the user: axis xy, tip z, tip-z spread, residuals, cross-check, warnings.
   Ask for a redo of any flagged capture. **Compare with the guess above**: a result outside
   +/-8 mm (axis) or +/-15 mm (tip) of it probably means a frame or TCP problem; stop and report it
   instead of continuing.

### C. Calipers (about 10 min, no robot)
Ask the user to measure and read each number to you (mm; check units), then write them into
`config/measured.json` only after reading them back and getting a yes:
- nut bore diameter -> `ring.bore_d_mm` (currently null: **the most important number**)
- rod tip chamfer -> `rod.tip_chamfer_mm` (currently a guess of 1.0: the tip was hidden in the photos)
- rod diameter at the tip -> `rod.d_mm`; nut outer diameter and height -> `ring.outer_d_mm`, `ring.height_mm`
- with the grasp from B: flange face to the nut's bottom along the tool axis ->
  `grasp.ring_bottom_below_tcp_mm` (TCP = flange; currently 225 +/-15)

Move each measured key out of `_provisional`; leave the rest as they are.

### D. Hand-back to the simulation machine
Tell the user to copy (do not push anything anywhere yourself):
`outputs/rod_registration.json`, the new `data/wrench_check/<label>/wrench_frame.json` plus the
pasted table, and `config/measured.json`. The simulation side then re-checks the policies with the
measured clearance and writes the deployment script (dry-run mode and offline tests first).

### E. Deployment: a policy trained in simulation, on the real arm
**Policies.** `replay`: blind force-limited descent (5 mm/s, stops pushing above 3 N), the baseline.
`student`: the DAgger student with the kinematic offset estimate (registered rod minus where the TCP
pose says the nut is), the only trained policy that needs no in-hand tracker. Simulation, provisional
parameters: 78/128 seated at 0.5 mm clearance, 95/128 at 1.0 mm, 128/128 at 2.7 mm (80/128 at 0.5 mm
with the identified sensor). `residual`: a residual-RL policy on its base, from
`sim_library/deploy/bundle_residual_kinematic.npz` once one has been trained with the kinematic
estimate and exported (none yet; residuals trained with the oracle estimate are refused). `still`:
no motion for 2 s at the entry (sensor check).

**What one run does.** Checks the start pose (nut >= 10 mm above the tip, move <= 100 mm, wrist
<= 20 deg from the entry orientation, tool tilt <= 10 deg), prints the plan and waits for `y`;
moveL to 25 mm above the entry, then down to the entry (nut bottom 6 mm above the tip) at 5 mm/s;
holds 0.25 s and tares; runs the policy at 30 Hz (servoL at 120 Hz) for at most 8 s; stops at the
first of: kinematic depth 14 mm, timeout, tared |F| > 20 N, workspace box (8 mm sideways, 5 mm up),
orientation > 6 deg, robot not in normal mode, **Enter**. Then servoStop; it never moves again. It
asks for the observed outcome and saves `data/deploy/<time>_<label>/{episode.npz,summary.json}`.

1. **You, offline:**
   ```bash
   cd ~/ring_rod_probe && conda activate data_collect
   python deploy/test_deploy.py                                   # 9 x ok
   python deploy/deploy_policy.py --dry-run -y --policy replay     # SEATED, stand-in robot
   python deploy/deploy_policy.py --dry-run -y --policy student    # SEATED
   python deploy/deploy_policy.py --dry-run -y --policy replay --dry-start-mm 0 0 -45   # NOT MOVING (rehearses a refusal)
   ```
   Also check `python -c "import rtde_control, rtde_receive"` works in this env (the GELLO uses them).
2. **User, preparation:** robot on, in the same control mode the GELLO uses; TCP all zeros and
   payload unchanged. With the GELLO: grip the nut (same grasp as registration, if B was done),
   bring it **1-5 cm above the rod tip** (**5-7 cm** for the air runs), wrist roughly upright. Then
   **quit the GELLO script** (it holds the robot's control interface). Hand near the e-stop.
3. **User, sensor check at the entry:** `python deploy/deploy_policy.py --policy still -l still_01`.
   You: read the printout (peak |F| should stay below ~1 N) and report.
4. **User, air runs** (no contact, 30 mm above the tip):
   ```bash
   python deploy/deploy_policy.py --policy replay  --air-mm 30 -l air_replay
   python deploy/deploy_policy.py --policy student --air-mm 30 -l air_student
   ```
   Expect `SEATED (kinematic depth ...)` in air after about 2.5-4 s. You: check the **tracking**
   line: median below ~0.3 mm (the simulation's arm: 0.03 mm). Worse than 0.5 mm: stop and report
   (servo settings `--lookahead` / `--gain` to be changed on the simulation side, not by guessing).
5. **User, real attempts:** alternate the two, about 10 each to start:
   `python deploy/deploy_policy.py --policy replay -l replay_01`, then `--policy student -l student_01`,
   and so on. After each run answer the **observed outcome** prompt honestly (the script's "seated"
   is computed from the arm's pose, not seen). Then with the GELLO: open the gripper, lift the nut
   off, regrasp it the same way and bring it back above the rod. Film the attempts (phone is fine)
   if they should go on the project page; the summary's timestamp matches each clip.
6. **You, after each block:** a table from `data/deploy/*/summary.json`: policy, observed and script
   outcome, reason, peak force, tracking. Any `aborted` with reason `force`: ask the user what they
   saw before the next attempt. Hand-back: the whole `data/deploy/` folder.

| Symptom | Likely cause / fix |
|---|---|
| `NOT MOVING: ...` | do what it says with the GELLO (it prints which way to turn the wrist), quit the GELLO, rerun |
| cannot connect / `RTDEControlInterface` error | the GELLO script is still running, or the robot is not in the control mode the GELLO uses |
| `TCP offset on the pendant is ...` | the TCP was changed: set it back to all zeros (ask the user) |
| `moveL was refused` | protective stop or unreachable pose: check the pendant |
| `aborted (workspace ...)` on real attempts | the rod is further from the registration (guess: +/-8 mm) than the policy searches: do B |
| `aborted (force ...)` | report it; do not raise `--force-abort` |

---

## Rules for you (the agent)

1. **Never start anything that connects to the robot or moves it on your own.** That includes
   `scripts/collect_contact.py` (without `--dry-run`), `scripts/record_session.py`,
   `scripts/rtde_logger.py`, `scripts/verify_wrench_frame.py`, `deploy/deploy_policy.py` (it moves the
   robot), and anything in `~/memory`. Give
   the user the exact command and let them run it (`! <command>` in your session, or their own
   terminal). `--dry-run` and the offline tests are safe for you to run.
2. Never pass `--allow-tare` unless the user asks for it for that session.
3. Never change robot settings (TCP, payload, safety) and never edit `~/memory/config.yaml`
   without the user's go-ahead; suggest the change instead.
4. Never kill processes by pattern (`pkill -f`, `pgrep -f ... | xargs kill`). If something hangs,
   tell the user and use a specific PID only with their agreement.
5. Do not delete recorded sessions. Re-record into a new label instead.
6. Watch for the force alarm. If the user reports or the log shows `force_alarm`, ask them to
   back off on the GELLO before anything else. The script cannot stop the arm; the UR's protective
   stop is the real limit.

## What is on this machine

| Path | What |
|---|---|
| `~/ring_rod_probe/` | read-only RTDE probe toolkit: logger, analysis, export (see its `README.md`) |
| `~/ring_rod_probe/scripts/collect_contact.py` | **new**: GELLO teleop + cameras (from `~/memory/scripts/collect.py`) + the read-only wrench log and event marks |
| `~/ring_rod_probe/tests/test_collect_contact.py` | **new**: offline test of the script's key handling and session files |
| `~/ring_rod_probe/scripts/probe_check.py`, `probe_features.py` | **new**: offline comparison of the real export with the simulated probe library (step 7) |
| `~/ring_rod_probe/sim_library/` | **new**: the simulated probe library and one held-out simulated run (test fixture); **simulation, not data** |
| `~/ring_rod_probe/tests/test_probe_check.py` | **new**: offline test of the check |
| `~/ring_rod_probe/deploy/` | **new**: `deploy_policy.py` (MOVES THE ROBOT, user only), `deploy_core.py`, `fake_ur.py`, `test_deploy.py` |
| `~/ring_rod_probe/sim_library/deploy/` | **new**: the policy bundle, the simulation reference log for the tests, the registration guess |
| `~/ring_rod_probe/SESSION_GUIDE.md` | this file |
| `~/memory/` | the lab's GELLO collection repo (`GelloEnv`, `config.yaml`, `scripts/viewer.py`) |

Known facts (2026-11-18 to 11-22; the status table at the top is newer): UR5e (IP in `config/robot.yaml`), PolyScope 5.13; RTDE delivered 500 Hz
with no drops; configured TCP = flange (all zeros); configured payload 1.0 kg, CoG [0,0,0], and
`actual_TCP_force` reads about -9.8 N on z at rest (payload compensation ~1 kg off); wrench
frame/sign check **pending**. The gripper is a Robotiq 2F-85 with **fin-ray fingers** (blue,
tapered) that pinch the ring's outside at their tips.

## Setup (once)

```bash
cd ~/ring_rod_probe
conda activate data_collect
python tests/test_collect_contact.py                    # 3 tests, no robot
python -m pytest tests                                  # ring_rod_probe's own tests
bash scripts/check_no_motion.sh                         # PASS expected
python scripts/collect_contact.py --dry-run --plan attempts -l rehearse   # practise the keys, no robot
```

The dry run uses terminal keys and a fake wrench; walk the user through every key once (table
below), then confirm `data/sessions/*_rehearse/events.jsonl` has the marks.

## Keys (real runs read them through pynput, so the user needs an X display: `DISPLAY=:1`)

| Key | Mark |
|---|---|
| `v` / `b` / Backspace x2 | camera episode start / save / drop |
| `s`, `f` | still segment, free-motion segment (start/end) |
| `r` then `1`-`6` | registration capture: seated, tip, side+x, side-x, side+y, side-y (`r` again ends it) |
| `a`, `c`, `o` then `1`-`3` | attempt start, contact noticed, outcome: seated / jammed / aborted |
| `p` | press-and-hold segment (5-10 N on a jam) |
| `h` | note "recovered by hand" (saved with the outcome) |
| `t` | tare mark (pendant tare first) |
| `i` | coverage table: offset bin x tilt bin -> outcomes |
| `q` | quit and save (twice if anything is open) |

## Step 0: before the recording day

1. **Measurements** into `config/measured.json`, following `config/MEASURE.md`: ring bore, outer
   diameter, height, mass; rod diameter, tip chamfer, threaded, spacer; printed rings;
   `grasp.ring_bottom_below_tcp_mm`. Also write in `grasp.notes`, for the fin-ray fingers: length
   from the 2F-85 inner-finger joint to the tip, tip width and thickness, and how far up the ring
   the tips touch when gripped. Ask the user to read each number to you and check units (mm, g).
2. **Payload**: ask the user to set gripper + ring mass and CoG on the pendant, write the values
   down, and not change them for the day.
3. **Wrench frame and sign** (pending check 4): the user runs
   `python scripts/verify_wrench_frame.py` with the robot still and presses +x, +y, +z, then at two
   points along the tool. Paste the printed table into `README.md`'s conventions section.
4. Optional: a `ring_rod` task with a start pose near the stand in `~/memory/config.yaml`
   (otherwise GELLO starts from the default pose `[90, -90, 90, -90, -90, 0]` deg). Suggest it;
   the user decides.

## The recording day (about 2.5-3 h)

Every real run starts the same way: the script prints a checklist, then asks the user to type `y`
before `env.calibrate()` moves the arm to the GELLO pose. The user must have the cell clear.

### 1. Warm-up (10-15 min)
Robot powered and moving gently before recording; the sensor drifts while it warms up.

### 2. Sensor session (~20 min)
```bash
DISPLAY=:1 python scripts/collect_contact.py --plan sensor -l sensor1
```
- Tare once at the start (pendant, then `t`); no more tares in this session.
- 3-5 still segments (`s` ... `s`), 10-30 s each, at different tool orientations including 0, 3 and
  6 deg tilt and a few larger ones. GELLO held completely still, no hand on the tool.
- Several free-motion segments (`f` ... `f`): slow moves near the rod without touching it, each
  starting with ~0.5 s still.
- Then you run: `python scripts/characterize_sensor.py data/sessions/*_sensor1 -o outputs/sensor`
  and report noise, drift and bias vs orientation to the user. If the force noise std is far
  above ~1 N, stop and tell the user (the simulation side needs to know before attempts).

### 3. Registration session (~15 min)
The user grasps the ring now and **keeps this grasp and wrist yaw for the rest of the day**.
```bash
DISPLAY=:1 python scripts/collect_contact.py --plan reg -l reg1
```
- seated (x1-2): ring centred and seated on the rod; hold 2 s.
- tip (x4: from +x, -x, +y, -y): hover ~6 mm off-axis, 5 mm above the tip, open the capture,
  hold >= 0.3 s, descend ~2 mm/s until the rim lands on the tip, stop, close.
- side +x / -x / +y / -y: a few mm over the rod, centred; open, hold >= 0.3 s, move ~1 mm/s in the
  named direction until the bore touches, stop, close.
- Then you run `python scripts/register_rod.py data/sessions/*_reg1 -o outputs/rod_registration.json`
  and read its printout: tip-z spread, residuals, warnings. Ask for a redo of any flagged capture.

### 4. Attempt sessions (60-90 min per ring)
Blocks of about 25 attempts per session (`attempts_a`, `attempts_b`, ...):
```bash
DISPLAY=:1 python scripts/collect_contact.py --plan attempts -l attempts_a \
    --registration outputs/rod_registration.json            # add --ring <name> for a printed ring
```
Per attempt (remind the user while they get used to it):
1. Line up at the target condition 5-10 mm above the tip using the live offset / tilt line.
2. Still; pendant tare at this tilt; `t`.
3. `a` (also starts the camera episode).
4. Steady descent ~5 mm/s (unsteady speed is flagged).
5. `c` when contact is noticed.
6. On about a third of the jams: press to 5-10 N (watch the live |F|) and hold ~1 s inside `p` ...
   `p`; then back off and recover by hand (`h`) or stop.
7. `o` + outcome.

Coverage per ring: offsets ~0, 1, 2, 2.5, 3, 3.5, 4, 6 mm in several directions x tilts ~0, 3,
6 deg x 3 repeats (~70 attempts), including deliberate jams and some recoveries. **Interleave the
conditions** (do not do all 0 mm first). Check `i` between blocks and suggest missing cells.
Ring order: main nut first, then printed rings, tightest last. A new grasp means re-measuring
`ring_bottom_below_tcp_mm` and a new registration.

### 5. Grip stiffness (optional, ~10 min)
Resting on the tip, step down in small increments and hold each. GELLO is too coarse for 0.5 mm
steps; with a SpaceMouse or pendant jog use `scripts/record_session.py -l stiffness` instead.

### 6. After each session (you)
- Read the end-of-session report: delivered rate ~500 Hz, no gaps, no safety stops. Anything else:
  re-record under a new label.
- End of day: `python scripts/segment_attempts.py data/sessions/*_attempts* -r outputs/rod_registration.json -o outputs/attempts.json`
  then `python scripts/export_probe.py -a outputs/attempts.json -o outputs/export` (add `--ring` per
  printed ring). Summarise the flags (`unsteady_speed`, `short_rest`, `contact_mark_mismatch`,
  `tcp_differs_from_registration`, `missing_outcome`, `data_gap`) and the outcome counts per
  condition for the user.

### 7. Offline check against the simulation (you run this; no robot)
After the export, compare the real attempts with the simulated probe library (numpy only, reads
files only; `sim_library/README.md` says what the library is: **simulation**, not data):
```bash
python tests/test_probe_check.py                     # once: 3 tests on a simulated held-out run
python scripts/probe_check.py outputs/export --measured config/measured.json \
    --library sim_library/probe_library.json -o outputs/probe_check
```
Read `outputs/probe_check/report.md` with the user, section by section:
- **Sensor**: real rest noise vs the simulated 0.34 N. Much larger (> 1 N): say so; the sensor
  session numbers go to the simulation side.
- **Height**: the median onset height of rod-top landings (jams >= 3.5 mm off) relative to the
  simulation. It is the error of `grasp.ring_bottom_below_tcp_mm` (provisional +/-15 mm) or of the
  registered tip z. More than ~3 mm: suggest re-measuring the grasp length; do not edit
  `measured.json` yourself. The check removes it before matching (as a grasp-length error).
- **Registration from outcomes**: the rod-axis shift that best separates seated from jammed level
  attempts. Shift > 1.5 mm with `directions_ok`: suggest a re-registration. Fewer than 8 level
  attempts or one-sided directions: "poorly determined", ignore it.
- **Best matching simulation parameters**: nearest grid point for clearance, grip stiffness and
  friction, with the entries within 10 %. Several entries close together means the data does not
  separate them yet: suggest the attempts that would (the per-bin table shows which bins are n = 1).

## Hand-back to the simulation machine

The user copies these (tell them the exact paths; do not push anything anywhere yourself):
`outputs/export/summary.json`, `outputs/export/traces.npz`, `outputs/rod_registration.json`,
`outputs/sensor/sensor_report.json`, `config/measured.json`, `outputs/probe_check/report.json`,
and the `README.md` conventions table.
Camera episodes stay here (`~/Desktop/data/ring_rod_<label>/`) until vision needs them.

## Rules that protect the data (repeat them to the user)

- No hand on the tool or gripper while recording (it loads the wrist sensor).
- Teleop starts before recording and is not restarted mid-session (the script does this order).
- Grasp, wrist yaw, TCP and payload unchanged from registration onwards.
- Under GELLO the arm is stiff: back off as soon as the force line turns orange (10 N); the alarm
  is at 25 N.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `No devices found` / `Multiple devices found` at start | GelloEnv needs exactly one `/dev/serial/by-id/*` device: the GELLO. Unplug other serial devices. |
| keys do nothing | pynput needs the X display: run with `DISPLAY=:1` from a desktop terminal |
| `robot IP differs` | `~/memory/config.yaml` and `config/robot.yaml` name different robots |
| `LOGGER STOPPED` | the RTDE connection dropped; the session up to there is saved |
| crash mid-session | `python scripts/record_session.py --recover data/sessions/<dir>` rebuilds `session.npz` from the 10 s chunks (same logger and files) |
| live offset shows `off` | no `--registration`, or `measured.json` is missing the ring height / `ring_bottom_below_tcp_mm` |
| tare check "NOT confirmed" (only with `--allow-tare`) | robot in Local mode; use the pendant tare and `t` |
