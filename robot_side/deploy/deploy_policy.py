#!/usr/bin/env python
"""Run one insertion attempt on the UR5e with a policy trained in simulation.

THIS SCRIPT MOVES THE ROBOT (except with --dry-run). The operator runs it from a desktop terminal,
with the cell clear, the GELLO script closed (it holds the robot's control interface), the nut
already gripped and held 1-5 cm above the rod tip, and a hand near the e-stop.

    python deploy/deploy_policy.py --dry-run --policy replay                 # rehearsal, no robot
    python deploy/deploy_policy.py --policy replay --air-mm 30 -l air_replay  # in air, 30 mm above the tip
    python deploy/deploy_policy.py --policy student -l student_01            # a real attempt

Flow: load the bundle, registration, grasp length and wrench sign; check the current pose; show
the planned moves and wait for 'y'; moveL to 25 mm above the entry, then down to the entry
(nut bottom 6 mm above the tip, on the registered axis, entry orientation); hold still 0.25 s and
tare (the mean of the last 0.2 s, as in the simulation); run the policy at 30 Hz with servoL at
120 Hz for at most 8 s; stop at the first of: kinematic depth 14 mm (seated), timeout, force
or workspace limit, robot not in normal mode, Enter pressed. Then servoStop and save the log. It
never moves after the attempt: release or regrasp the nut with the GELLO afterwards.

Policies: `replay` (blind force-limited descent, the baseline), `student` (DAgger student with the
kinematic offset estimate: registered rod minus the nut position predicted from the TCP), `residual`
(a residual-RL policy on its base, from a bundle exported with export_deploy_bundle.py --residual;
only residuals trained with the kinematic estimate are accepted), `still` (no motion for 2 s:
checks the wrench channel at the entry).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deploy_core import (Grasp, Limits, ReplayPolicy, ResidualNP, SimConstants, StudentNP, T_to_pose6, WrenchChannel,  # noqa: E402
                         _tcp_from_flange, angle_between, entry_pose, load_bundle, load_grasp_length, load_rod,
                         load_wrench_sign, pick_entry_rotation, pose6_to_T, run_episode)

ROOT = HERE.parent  # ~/ring_rod_probe


class URRobot:
    """ur_rtde control + receive (the library the lab's GELLO controller uses)."""

    def __init__(self, ip, lookahead, gain, ctrl_hz):
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
        self.r = RTDEReceiveInterface(ip)
        self.c = RTDEControlInterface(ip)
        self.lookahead, self.gain, self.dt = lookahead, gain, 1.0 / ctrl_hz
        self.last_r = np.array(self.r.getActualTCPPose(), float)[3:]
        if hasattr(self.c, "getTCPOffset"):
            off = list(self.c.getTCPOffset())
            if any(abs(x) > 1e-6 for x in off):
                raise SystemExit(f"TCP offset on the pendant is {off}; this deployment assumes TCP = flange (all zeros)")

    def flange(self):
        return pose6_to_T(np.array(self.r.getActualTCPPose(), float))

    def wrench_raw(self):
        return np.array(self.r.getActualTCPForce(), float)

    def _pose6(self, T):
        # the entry orientation is a half turn (tool straight down): keep consecutive commands on the
        # same axis-angle branch so the controller never sees r and its equivalent -r(2pi-|r|)/|r| alternate
        p = np.array(T_to_pose6(T))
        r = p[3:]
        th = float(np.linalg.norm(r))
        if th > 1e-6 and self.last_r is not None:
            alt = r / th * (th - 2 * math.pi)
            if np.linalg.norm(alt - self.last_r) < np.linalg.norm(r - self.last_r):
                p[3:] = alt
        self.last_r = p[3:].copy()
        return [float(x) for x in p]

    def servo(self, T, dt):
        self.c.servoL(self._pose6(T), 0.0, 0.0, dt, self.lookahead, self.gain)

    def move_to(self, T, speed):
        ok = self.c.moveL(self._pose6(T), speed, 0.1)
        if not ok:
            raise SystemExit("moveL was refused by the controller")

    def period_start(self):
        return self.c.initPeriod() if hasattr(self.c, "initPeriod") else time.perf_counter()

    def period_wait(self, t0):
        if hasattr(self.c, "waitPeriod"):
            self.c.waitPeriod(t0)
        else:
            rest = self.dt - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)

    def stop(self):
        try:
            self.c.servoStop()
        except Exception:
            pass

    def ok(self):
        for name in ("isProtectiveStopped", "isEmergencyStopped"):
            if hasattr(self.r, name) and getattr(self.r, name)():
                return False, name
        sm = self.r.getSafetyMode()
        if sm not in (1, 2):  # 1 normal, 2 reduced
            return False, f"safety mode {sm}"
        return True, ""

    def close(self):
        for f in (getattr(self.c, "servoStop", None), getattr(self.c, "stopScript", None), getattr(self.c, "disconnect", None),
                  getattr(self.r, "disconnect", None)):
            try:
                if f:
                    f()
            except Exception:
                pass


def robot_ip(path: Path):
    for line in path.read_text().splitlines():
        s = line.split("#")[0].strip()
        if s.startswith("ip:"):
            return s.split(":", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"no robot ip in {path}")


def newest(pattern):
    c = sorted(ROOT.glob(pattern))
    return c[-1] if c else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["replay", "student", "residual", "still"], required=True)
    ap.add_argument("--bundle", type=Path, default=None,
                    help="default: sim_library/deploy/bundle_dagger_kinematic.npz (bundle_residual_kinematic.npz for --policy residual)")
    ap.add_argument("--registration", type=Path, default=ROOT / "outputs/rod_registration.json")
    ap.add_argument("--guess", type=Path, default=ROOT / "sim_library/deploy/rod_registration_guess.json",
                    help="used only if there is no registration (a warning is printed)")
    ap.add_argument("--measured", type=Path, default=ROOT / "config/measured.json")
    ap.add_argument("--wrench-frame", type=Path, default=None, help="wrench_frame.json (default: the newest under data/wrench_check)")
    ap.add_argument("--wrench-sign", type=float, default=None, help="override: +1 or -1")
    ap.add_argument("--robot-config", type=Path, default=ROOT / "config/robot.yaml")
    ap.add_argument("--air-mm", type=float, default=0.0, help="run this far above the real tip (no contact)")
    ap.add_argument("--force-abort", type=float, default=20.0, help="N, tared |F|")
    ap.add_argument("--lookahead", type=float, default=0.03, help="servoL lookahead (s): small, the policy was trained on a stiff arm")
    ap.add_argument("--gain", type=float, default=600.0, help="servoL gain")
    ap.add_argument("--approach-speed", type=float, default=0.015, help="m/s for the moves to the entry")
    ap.add_argument("--dry-run", action="store_true", help="no robot: the stand-in robot (fake_ur.py)")
    ap.add_argument("--dry-offset-mm", type=float, nargs=2, default=(0.0, 0.0), help="dry run: rod offset from the registration")
    ap.add_argument("--dry-start-mm", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="dry run: start offset from 25 mm above the entry")
    ap.add_argument("--dry-start-rot-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="dry run: start rotation (base-frame rotvec, deg)")
    ap.add_argument("-l", "--label", default="attempt")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "data/deploy")
    ap.add_argument("-y", "--yes", action="store_true", help="dry run only: skip the confirmation")
    a = ap.parse_args()

    k = SimConstants()
    if a.bundle is None:
        a.bundle = ROOT / "sim_library/deploy" / ("bundle_residual_kinematic.npz" if a.policy == "residual" else "bundle_dagger_kinematic.npz")
    if not Path(a.bundle).exists():
        hint = ("train a residual on the kinematic-estimate DAgger student (contact_sim/README.md, 'residual RL on a DAgger "
                "student') and export it: export_deploy_bundle.py --residual <run>/residual.pt --vision kinematic ... -o "
                "bundle_residual_kinematic.npz") if a.policy == "residual" else "copy the bundle from the simulation side"
        raise SystemExit(f"no policy bundle at {a.bundle}: {hint}")
    z, meta = load_bundle(a.bundle)
    if meta.get("vision") != "kinematic":
        raise SystemExit("this bundle's student was trained with the oracle offset estimate; only kinematic-estimate students can run here")
    rod = load_rod(a.registration, a.guess)
    below, measured = load_grasp_length(a.measured)
    grasp = Grasp(z["R_tcp_obj"], z["xy_tcp_obj"], float(z["ring_height"]), below)
    wf = a.wrench_frame or newest("data/wrench_check/*/wrench_frame.json")
    sign, sign_src = (a.wrench_sign, "command line") if a.wrench_sign is not None else load_wrench_sign(wf)
    lim = Limits(force_abort_n=a.force_abort)
    policy = {"replay": lambda: ReplayPolicy(k), "student": lambda: StudentNP.from_bundle(z), "residual": lambda: ResidualNP.from_bundle(z, k),
              "still": lambda: (lambda obs: np.zeros(5))}[a.policy]()
    if a.policy == "still":
        k.episode_s = 2.0
    air = a.air_mm / 1000.0

    print("=" * 78)
    print(f"policy        {a.policy}   ({meta.get('residual', '') + ' on ' if a.policy == 'residual' else ''}"
          f"{meta.get('student') if a.policy in ('student', 'residual') else 'no learning'})")
    print(f"              trained in SIMULATION: {meta.get('sim_results', '')}")
    print(f"rod           xy ({rod.xy[0]:.4f}, {rod.xy[1]:.4f}) m, tip z {rod.tip_z:.4f} m   [{rod.source}]")
    if "GUESS" in rod.source:
        print("              !! not a registration: the axis can be off by ~8 mm, the tip by ~15 mm")
    print(f"grasp         nut bottom {below * 1000:.1f} mm below the flange   [{a.measured}]")
    print(f"wrench sign   {sign:+.0f}   [{sign_src}]")
    print(f"air offset    {a.air_mm:.0f} mm" + ("   (no contact expected)" if a.air_mm > 0 else "   (REAL CONTACT)"))
    print(f"limits        |F| {lim.force_abort_n:.0f} N, box +/-{lim.box_xy_m * 1e3:.0f} mm, {k.episode_s:.0f} s; servoL lookahead {a.lookahead} s gain {a.gain:.0f}")
    print("=" * 78)

    if a.dry_run:
        from fake_ur import FakeUR
        bore = (measured.get("ring", {}) or {}).get("bore_d_mm")
        rod_d = (measured.get("rod", {}) or {}).get("d_mm") or 16.0
        clr = (bore - rod_d) / 2000.0 if bore else 0.0027
        R0 = z["R_entry"]
        T_pre = entry_pose(rod, grasp, R0, k, air + 0.025)
        T_fl0 = T_pre.copy()
        T_fl0[:3, 3] = T_pre[:3, 3] - R0 @ np.array([0, 0, k.flange_to_tcp]) + np.asarray(a.dry_start_mm) / 1000.0
        from deploy_core import exp_so3
        T_fl0[:3, :3] = exp_so3(np.radians(a.dry_start_rot_deg)) @ T_fl0[:3, :3]
        world = type(rod)(rod.xy + np.asarray(a.dry_offset_mm) / 1000.0, rod.tip_z, "dry-run world")
        robot = FakeUR(T_fl0, world, grasp, k, clearance_m=clr, sign=sign, realtime=False)
        print(f"DRY RUN: stand-in robot, clearance {clr * 1e3:.1f} mm, rod offset {a.dry_offset_mm} mm")
    else:
        robot = None

    # ---- where are we, where do we go
    if robot is None:
        ip = robot_ip(a.robot_config)
        print(f"connecting to {ip} (control + receive) ...")
        robot = URRobot(ip, a.lookahead, a.gain, k.ctrl_hz)
    try:
        T_now = robot.flange()
        R_entry = pick_entry_rotation(z["R_entry"], T_now[:3, :3])
        T_entry_tcp = entry_pose(rod, grasp, R_entry, k, air)
        T_pre_tcp = T_entry_tcp.copy()
        T_pre_tcp[2, 3] += 0.025

        def fl(Ttcp):
            T = Ttcp.copy()
            T[:3, 3] = Ttcp[:3, 3] - Ttcp[:3, :3] @ np.array([0, 0, k.flange_to_tcp])
            return T

        bottom_now = grasp.bottom(_tcp_from_flange(T_now, k))
        h_above = bottom_now[2] - (rod.tip_z + air)
        lateral = float(np.linalg.norm(bottom_now[:2] - rod.xy))
        rot = math.degrees(angle_between(T_now[:3, :3], R_entry))
        # moveL interpolates the flange; the nut hangs ~0.22 m below it, so levelling a tilted wrist
        # swings the nut down by L (1 - cos tilt): lift the first waypoint by that plus a margin, and make
        # the move long enough that the rotation stays below ~10 deg/s
        tilt = math.degrees(math.acos(float(np.clip(-T_now[2, 2], -1, 1))))
        swing = below * (1 - math.cos(math.radians(tilt)))
        T_mid = fl(T_pre_tcp)
        T_mid[2, 3] = max(T_mid[2, 3], T_now[2, 3] + swing + 0.005 * (tilt > 1))  # never lower before the lateral move
        need = a.approach_speed * rot / 10.0  # travel (m) that makes the move last rot / (10 deg/s)
        d0 = float(np.linalg.norm(T_mid[:3, 3] - T_now[:3, 3]))
        if d0 < need:
            T_mid[2, 3] += need - d0
        dist = float(np.linalg.norm(T_mid[:3, 3] - T_now[:3, 3]))
        from deploy_core import mat_to_rotvec
        yaw = math.degrees(float(mat_to_rotvec(R_entry @ T_now[:3, :3].T)[2]))
        print(f"now           nut bottom {h_above * 1e3:.0f} mm above the tip, {lateral * 1e3:.0f} mm off the axis, "
              f"{rot:.0f} deg from the entry orientation (turn the wrist {yaw:+.0f} deg about vertical, + = counter-clockwise seen from above)")
        problems = []
        if h_above < 0.010:
            problems.append("the nut must be at least 10 mm above the rod tip")
        if dist > 0.10:
            problems.append(f"the move to above the rod would be {dist * 1e3:.0f} mm (limit 100 mm)")
        if rot > 20:
            problems.append(f"the wrist is {rot:.0f} deg from the entry orientation (limit 20): turn it closer first")
        if tilt > 10:
            problems.append(f"the tool is tilted {tilt:.0f} deg from vertical (limit 10): straighten it first")
        if problems:
            print("NOT MOVING:\n  - " + "\n  - ".join(problems))
            print("With the GELLO, bring the nut 1-5 cm above the rod tip, roughly upright; close the GELLO script; run this again.")
            return 2
        print(f"plan          1) moveL {dist * 1e3:.0f} mm to above the entry (rotate {rot:.0f} deg, tilt now {tilt:.0f} deg) at {a.approach_speed * 1e3:.0f} mm/s")
        print(f"              2) moveL down to 25 mm above the entry, 3) down to the entry at 5 mm/s")
        print(f"              4) hold 0.25 s, tare, run '{a.policy}' (<= {k.episode_s:.0f} s; Enter stops it)")
        if not (a.dry_run and a.yes):
            if input("cell clear, hand on the e-stop? type y to start: ").strip().lower() != "y":
                print("not started")
                return 1
        robot.move_to(T_mid, a.approach_speed)
        robot.move_to(fl(T_pre_tcp), a.approach_speed)
        robot.move_to(fl(T_entry_tcp), 0.005)

        stop = threading.Event()
        if not a.dry_run:
            threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        wrench = WrenchChannel(sign=sign, flange_to_tcp=k.flange_to_tcp, lowpass_hz=k.lowpass_hz)
        print("running (press Enter to stop) ...")
        res = run_episode(robot, policy, k, grasp, rod, R_entry, lim, wrench, stop_flag=stop, air_m=air)
    finally:
        robot.stop()
        robot.close()

    track = np.linalg.norm(res.log["p_cmd"] - res.log["tcp"], axis=1) if len(res.log["p_cmd"]) else np.zeros(1)
    print("-" * 78)
    print(f"outcome       {res.outcome.upper()}  ({res.reason or 'no stop condition'}) after {res.t_end:.2f} s")
    print(f"              peak |F| {res.peak_force_n:.1f} N, kinematic depth {res.final_depth_mm:.1f} mm, "
          f"tracking median {np.median(track) * 1e3:.2f} mm / max {track.max() * 1e3:.2f} mm (sim: 0.03 / 0.19)")
    print("The arm is stopped and will not move again from this script. Release or regrasp with the GELLO.")
    observed, note = "", ""
    if not a.dry_run:  # the script's 'seated' is kinematic; the operator's eyes are the ground truth
        while observed not in ("s", "j", "o"):
            observed = input("observed outcome? [s]eated on the rod / [j]ammed or not on / [o]ther: ").strip().lower()[:1]
        note = input("note (optional, Enter to skip): ").strip()
    observed = {"s": "seated", "j": "jammed", "o": "other", "": None}[observed]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = a.out / f"{ts}_{a.label}{'_dry' if a.dry_run else ''}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "episode.npz", **res.log)
    (d / "summary.json").write_text(json.dumps(dict(
        note="dry run (stand-in robot)" if a.dry_run else "REAL UR5e attempt",
        policy=a.policy, bundle=str(a.bundle), bundle_meta=meta, outcome_script=res.outcome, outcome_observed=observed, operator_note=note,
        reason=res.reason, t_end_s=res.t_end,
        peak_force_n=res.peak_force_n, final_depth_mm=res.final_depth_mm, tracking_median_mm=float(np.median(track) * 1e3),
        tracking_max_mm=float(track.max() * 1e3), rod=dict(xy=rod.xy.tolist(), tip_z=rod.tip_z, source=rod.source),
        ring_bottom_below_flange_m=below, wrench_sign=sign, wrench_sign_source=sign_src, air_mm=a.air_mm,
        limits=lim.__dict__, servo=dict(lookahead=a.lookahead, gain=a.gain), entry_tcp_pose6=T_to_pose6(T_entry_tcp),
        constants=k.__dict__), indent=1))
    print(f"saved         {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
