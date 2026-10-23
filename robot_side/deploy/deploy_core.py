"""Deployment core (numpy only): rebuilds, on the real UR5e, exactly the observation the simulated
policies were trained on, runs the policy, and integrates its actions into servo targets.

Conventions (checked against the simulation, see test_deploy.py):
  * Frames: the simulation's root link `ur_base` is the UR controller's base frame (RTDE reports
    poses in it); the sim's `tool0` is the UR flange frame (TCP offset all zeros on the pendant);
    the sim's `tcp` point is 0.20 m out along tool z. Policy quantities use that tcp point.
  * Wrench: the sim channel is the load on the tool, base frame, torque about the tcp point, tared
    (mean of 0.2 s, robot still) and low-passed at 20 Hz (first order). From RTDE actual_TCP_force
    (base frame, torque about the configured TCP = flange): multiply by the sign from the wrench
    check, shift the torque to the tcp point, tare, filter.
  * Observation (62): [p_cmd - p_start (3), tcp - p_start (3), 8 x filtered wrench (48),
    previous action (5), offset estimate (2), 1]. Offset estimate, kinematic: registered rod xy
    minus the nut bottom predicted from the tcp pose with the nominal grasp.
  * Action (5) in [-1, 1]: TCP velocity x, y, z and rotation rate about base x, y, scaled by the
    training limits (6 mm/s, 6 mm/s, 8 mm/s, 0.1 rad/s), integrated at the 120 Hz controller rate,
    4 sub-steps per 30 Hz policy tick; rotation about the tcp point.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- rotations / poses


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)


def exp_so3(w):
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    K = skew(w / th)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def rotvec_to_mat(r):
    return exp_so3(r)


def mat_to_rotvec(R):
    R = np.asarray(R, float)
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if math.pi - th < 1e-6:  # near 180 deg: axis from the symmetric part
        M = (R + np.eye(3)) / 2
        k = int(np.argmax(np.diag(M)))
        ax = M[:, k] / math.sqrt(max(M[k, k], 1e-12))
        return ax / np.linalg.norm(ax) * th
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return ax * th


def angle_between(Ra, Rb):
    return float(np.linalg.norm(mat_to_rotvec(Ra.T @ Rb)))


def pose6_to_T(p6):
    T = np.eye(4)
    T[:3, :3] = rotvec_to_mat(p6[3:6])
    T[:3, 3] = p6[:3]
    return T


def T_to_pose6(T):
    return [float(x) for x in np.concatenate([T[:3, 3], mat_to_rotvec(T[:3, :3])])]


# ---------------------------------------------------------------- configuration


@dataclass
class SimConstants:
    """Values the policies were trained with (contact_env.EnvCfg, privileged_expert.ActionLimits,
    params.Sensor/Controller); exported into the bundle."""
    policy_hz: float = 30.0
    ctrl_hz: float = 120.0
    history: int = 8
    v_lat: float = 0.006
    v_z: float = 0.008
    w: float = 0.10
    lowpass_hz: float = 20.0
    tare_s: float = 0.2
    settle_s: float = 0.25
    entry_gap: float = 0.006
    target_depth: float = 0.015
    episode_s: float = 8.0
    flange_to_tcp: float = 0.20

    @property
    def lim(self):
        return np.array([self.v_lat, self.v_lat, self.v_z, self.w, self.w])

    @property
    def sub_steps(self):
        return int(round(self.ctrl_hz / self.policy_hz))


@dataclass
class Grasp:
    """Nominal grasp: the nut frame in the tcp frame (sim export, z from the measured grasp length)."""
    R_tcp_obj: np.ndarray
    xy_tcp_obj: np.ndarray
    ring_height: float
    ring_bottom_below_flange: float
    flange_to_tcp: float = 0.20

    def T_tcp_obj(self):
        T = np.eye(4)
        T[:3, :3] = self.R_tcp_obj
        T[:2, 3] = self.xy_tcp_obj
        T[2, 3] = self.ring_bottom_below_flange - self.flange_to_tcp - self.ring_height / 2
        return T

    def bottom(self, T_tcp):
        """Nut bottom centre (base frame) predicted from the tcp pose with the nominal grasp."""
        Tn = T_tcp @ self.T_tcp_obj()
        axis = Tn[:3, 2] * (1.0 if Tn[2, 2] >= 0 else -1.0)
        return Tn[:3, 3] - axis * self.ring_height / 2


@dataclass
class Rod:
    xy: np.ndarray
    tip_z: float
    source: str = ""


# ---------------------------------------------------------------- wrench channel


class WrenchChannel:
    """RTDE actual_TCP_force -> the simulated policy channel (load on the tool, base frame, torque
    about the tcp point, tared, first-order low-pass)."""

    def __init__(self, sign: float = 1.0, flange_to_tcp: float = 0.20, lowpass_hz: float = 20.0):
        self.sign, self.d, self.f = float(sign), float(flange_to_tcp), float(lowpass_hz)
        self.offset = np.zeros(6)
        self.lp = None

    def to_sim(self, w_ur, T_flange):
        w = self.sign * np.asarray(w_ur, float)
        F, tau = w[:3], w[3:]
        p_fl = T_flange[:3, 3]
        p_tcp = p_fl + T_flange[:3, :3] @ np.array([0.0, 0.0, self.d])
        return np.concatenate([F, tau + np.cross(p_fl - p_tcp, F)])

    def tare(self, samples):
        """samples: (k, 6) channel values (to_sim) from the last tare window, robot still."""
        self.offset = np.asarray(samples, float).mean(0)
        self.lp = None

    def step(self, w_sim, dt):
        x = np.asarray(w_sim, float) - self.offset
        if self.lp is None:
            self.lp = x.copy()
        else:
            a = 1.0 - math.exp(-2 * math.pi * self.f * dt) if self.f > 0 else 1.0
            self.lp = self.lp + a * (x - self.lp)
        return self.lp.copy()


# ---------------------------------------------------------------- observation / action


class ObsBuilder:
    def __init__(self, k: SimConstants):
        self.k = k
        self.hist = None
        self.prev_a = np.zeros(5)
        self.p_start = None

    def reset(self, p_start, filtered0):
        self.p_start = np.asarray(p_start, float).copy()
        self.hist = np.tile(np.asarray(filtered0, float), (self.k.history, 1))
        self.prev_a = np.zeros(5)

    def push(self, filtered, a):
        self.hist = np.vstack([self.hist[1:], np.asarray(filtered, float)[None]])
        self.prev_a = np.clip(np.asarray(a, float), -1, 1)

    def observe(self, p_cmd, tcp_pos, est):
        return np.concatenate([np.asarray(p_cmd) - self.p_start, np.asarray(tcp_pos) - self.p_start, self.hist.reshape(-1),
                               self.prev_a, np.asarray(est, float), [1.0]])


def kinematic_estimate(rod: Rod, grasp: Grasp, T_tcp):
    return rod.xy - grasp.bottom(T_tcp)[:2]


def integrate(a, p_cmd, R_cmd, k: SimConstants):
    """One controller sub-step of the simulated action integration (contact_env.ContactEnv.step)."""
    v = np.clip(np.asarray(a, float), -1, 1) * k.lim
    dtc = 1.0 / k.ctrl_hz
    p = np.asarray(p_cmd, float) + v[:3] * dtc
    R = exp_so3(np.array([v[3], v[4], 0.0]) * dtc) @ R_cmd
    return p, R


# ---------------------------------------------------------------- policies


class StudentNP:
    """The DAgger student (student.py) in numpy: masked, normalised, 3 x 256 ReLU, tanh."""

    def __init__(self, W, b, mu, sd, mask_wrench=False, mask_vision=False):
        self.W, self.b, self.mu, self.sd = W, b, mu, sd
        self.mask_wrench, self.mask_vision = bool(mask_wrench), bool(mask_vision)

    @classmethod
    def from_bundle(cls, z):
        n = int(z["n_layers"])
        return cls([z[f"W{i}"] for i in range(n)], [z[f"b{i}"] for i in range(n)], z["mu"], z["sd"],
                   bool(z["mask_wrench"]), bool(z["mask_vision"]))

    def __call__(self, obs):
        x = np.asarray(obs, float).copy()
        if self.mask_wrench:
            x[6:54] = 0
        if self.mask_vision:
            x[59:62] = 0
        x = (x - self.mu) / self.sd
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = W @ x + b
            x = np.maximum(x, 0) if i < len(self.W) - 1 else np.tanh(x)
        return x


class ResidualNP:
    """Residual RL policy (ppo.ActorCritic, deterministic mean) on a frozen base, as evaluated in
    run_dagger_rollouts.py: a = clip(base + res_scale * clip(actor(norm([obs, base])), -1, 1), -1, 1);
    actor 3 x 256 ELU, tanh output; running-mean/var normalisation clipped at +/- clip."""

    def __init__(self, base, W, b, mean, var, clip, res_scale):
        self.base, self.W, self.b = base, W, b
        self.mean, self.var, self.clip, self.res_scale = mean, var, float(clip), np.asarray(res_scale, float)

    @classmethod
    def from_bundle(cls, z, k: "SimConstants"):
        if "n_res_layers" not in z:
            raise KeyError("this bundle has no residual policy (export it with export_deploy_bundle.py --residual)")
        base = StudentNP.from_bundle(z) if str(z["res_base"]) == "student" else ReplayPolicy(k)
        n = int(z["n_res_layers"])
        return cls(base, [z[f"rW{i}"] for i in range(n)], [z[f"rb{i}"] for i in range(n)], z["r_mean"], z["r_var"],
                   float(z["r_clip"]), z["res_scale"])

    def __call__(self, obs):
        base = np.clip(self.base(obs), -1, 1)
        x = np.concatenate([np.asarray(obs, float), base])
        x = np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-6), -self.clip, self.clip)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = W @ x + b
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0))) if i < len(self.W) - 1 else np.tanh(x)
        return np.clip(base + self.res_scale * np.clip(x, -1, 1), -1, 1)


class ReplayPolicy:
    """Blind force-limited replay (run_dagger_rollouts --policy replay): straight down at 5 mm/s,
    stop pushing while the sensed axial force exceeds 3 N."""

    def __init__(self, k: SimConstants, speed=0.005, f_stop=3.0):
        self.k, self.speed, self.f_stop = k, speed, f_stop

    def __call__(self, obs):
        fz = obs[6 + 6 * (self.k.history - 1) + 2]
        a = np.zeros(5)
        a[2] = 0.0 if fz > self.f_stop else -self.speed / self.k.v_z
        return a


# ---------------------------------------------------------------- safety


@dataclass
class Limits:
    force_abort_n: float = 20.0       # |F| of the tared channel (unfiltered)
    torque_abort_nm: float = 2.0
    box_xy_m: float = 0.008           # commanded tcp within this of the entry, laterally
    box_up_m: float = 0.005           # ... and at most this far above the entry
    box_down_extra_m: float = 0.003   # ... and at most entry_gap + target_depth + this below it
    tilt_max_deg: float = 6.0         # commanded orientation from the entry orientation
    track_max_m: float = 0.003        # commanded vs measured tcp


def check(lim: Limits, k: SimConstants, p_entry, R_entry, p_cmd, R_cmd, tcp_meas, w_tared):
    d = np.asarray(p_cmd) - p_entry
    if np.hypot(d[0], d[1]) > lim.box_xy_m:
        return f"workspace: {np.hypot(d[0], d[1]) * 1e3:.1f} mm sideways from the entry"
    if d[2] > lim.box_up_m:
        return f"workspace: {d[2] * 1e3:.1f} mm above the entry"
    if -d[2] > k.entry_gap + k.target_depth + lim.box_down_extra_m:
        return f"workspace: {-d[2] * 1e3:.1f} mm below the entry"
    if math.degrees(angle_between(R_entry, R_cmd)) > lim.tilt_max_deg:
        return f"orientation: {math.degrees(angle_between(R_entry, R_cmd)):.1f} deg from the entry"
    if np.linalg.norm(np.asarray(p_cmd) - tcp_meas) > lim.track_max_m:
        return f"tracking: commanded and measured TCP {np.linalg.norm(np.asarray(p_cmd) - tcp_meas) * 1e3:.1f} mm apart"
    f = float(np.linalg.norm(w_tared[:3]))
    if f > lim.force_abort_n:
        return f"force: |F| {f:.1f} N"
    tq = float(np.linalg.norm(w_tared[3:]))
    if tq > lim.torque_abort_nm:
        return f"torque: |T| {tq:.2f} N m"
    return None


# ---------------------------------------------------------------- bundle / config files


def load_bundle(path):
    z = dict(np.load(path, allow_pickle=False))
    meta = json.loads(str(z.pop("meta_json")))
    return z, meta


def load_rod(registration: Path | None, guess: Path | None):
    for p, src in ((registration, "registration"), (guess, "GUESS (not a registration)")):
        if p is not None and Path(p).exists():
            d = json.loads(Path(p).read_text())
            xy = d.get("axis_xy") or d.get("xy")
            return Rod(np.asarray(xy, float), float(d["tip_z"]), f"{src}: {p}")
    raise FileNotFoundError("no rod registration found")


def load_grasp_length(measured: Path):
    """Nut bottom below the flange (m). Accepts the robot-side measured.json (grasp.ring_bottom_below_tcp_mm,
    mm, TCP = flange at measurement) or the simulation-side one (in_hand.ring_bottom_below_flange, m)."""
    m = json.loads(Path(measured).read_text())
    if "grasp" in m:
        g = m["grasp"]
        tcp = g.get("tcp_offset_at_measurement", [0, 0, 0, 0, 0, 0])
        if any(abs(float(x)) > 1e-9 for x in tcp):
            raise ValueError(f"grasp.ring_bottom_below_tcp_mm was measured with TCP offset {tcp}; the deployment assumes TCP = flange")
        return float(g["ring_bottom_below_tcp_mm"]) / 1000.0, m
    v = (m.get("in_hand") or {}).get("ring_bottom_below_flange")
    if v is None:
        raise KeyError(f"{measured}: neither grasp.ring_bottom_below_tcp_mm nor in_hand.ring_bottom_below_flange")
    geo = m.get("geometry") or {}
    # expose the bore / rod diameter in the robot-side keys for the dry run's clearance
    m.setdefault("ring", {}).setdefault("bore_d_mm", geo["ring_bore_r"] * 2000 if geo.get("ring_bore_r") else None)
    m.setdefault("rod", {}).setdefault("d_mm", geo["rod_r"] * 2000 if geo.get("rod_r") else None)
    return float(v), m


def load_wrench_sign(wrench_frame: Path | None, default: float = 1.0):
    """Sign from verify_wrench_frame.py's wrench_frame.json if given (several key spellings are
    accepted); otherwise the default, flagged as unchecked."""
    if wrench_frame is None or not Path(wrench_frame).exists():
        return default, "UNCHECKED (default +1: actual_TCP_force = load on the tool)"
    d = json.loads(Path(wrench_frame).read_text())
    for key in ("sign", "force_sign", "wrench_sign"):
        v = d.get(key) if isinstance(d, dict) else None
        if isinstance(v, (int, float)) and abs(abs(v) - 1) < 1e-6:
            return float(v), f"wrench check: {wrench_frame}"
        if isinstance(v, str) and v.strip() in ("+1", "1", "-1", "positive", "negative"):
            return (-1.0 if v.strip() in ("-1", "negative") else 1.0), f"wrench check: {wrench_frame}"
    return default, f"UNCHECKED ({wrench_frame} has no recognised sign field; default +1)"


# ---------------------------------------------------------------- one episode (robot-agnostic)


@dataclass
class EpisodeResult:
    outcome: str
    reason: str
    t_end: float
    peak_force_n: float
    final_depth_mm: float
    log: dict = field(default_factory=dict)


def run_episode(robot, policy, k: SimConstants, grasp: Grasp, rod: Rod, R_entry, lim: Limits, wrench: WrenchChannel,
                stop_flag=None, air_m: float = 0.0, verbose=True) -> EpisodeResult:
    """robot: object with flange() -> 4x4, wrench_raw() -> 6, servo(T_flange_target, dt), period_start(),
    period_wait(t0), stop(), ok() -> (bool, msg). The robot must already be at the entry pose and still."""
    tip_z = rod.tip_z + air_m
    T_fl = robot.flange()
    T_tcp = T_fl.copy()
    T_tcp[:3, 3] = T_fl[:3, 3] + T_fl[:3, :3] @ np.array([0, 0, k.flange_to_tcp])
    p_entry, R_cmd = T_tcp[:3, 3].copy(), R_entry.copy()
    dt_c = 1.0 / k.ctrl_hz
    # settle + tare (sim: 0.25 s still, tare = mean of the last 0.2 s)
    samples = []
    for _ in range(int(round(k.settle_s * k.ctrl_hz))):
        t0 = robot.period_start()
        samples.append(wrench.to_sim(robot.wrench_raw(), robot.flange()))
        robot.servo(_flange_target(p_entry, R_cmd, k), dt_c)
        robot.period_wait(t0)
    wrench.tare(np.array(samples[-int(round(k.tare_s * k.ctrl_hz)):]))
    filt = wrench.step(wrench.to_sim(robot.wrench_raw(), robot.flange()), dt_c)
    ob = ObsBuilder(k)
    p_cmd = p_entry.copy()
    ob.reset(p_cmd, filt)
    log = {key: [] for key in ("t", "obs", "action", "p_cmd", "R_cmd", "tcp", "flange", "w_raw", "w_tared", "filtered", "est", "depth")}
    peak, outcome, reason, t = 0.0, "timeout", "", 0.0
    n_ticks = int(round(k.episode_s * k.policy_hz))
    for tick in range(n_ticks):
        T_fl = robot.flange()
        T_tcp = _tcp_from_flange(T_fl, k)
        est = kinematic_estimate(rod, grasp, T_tcp)
        obs = ob.observe(p_cmd, T_tcp[:3, 3], est)
        a = np.clip(policy(obs), -1, 1)
        stop_reason = None
        for _ in range(k.sub_steps):
            t0 = robot.period_start()
            p_cmd, R_cmd = integrate(a, p_cmd, R_cmd, k)
            robot.servo(_flange_target(p_cmd, R_cmd, k), dt_c)
            T_fl = robot.flange()
            w_raw = robot.wrench_raw()
            w_sim = wrench.to_sim(w_raw, T_fl)
            filt = wrench.step(w_sim, dt_c)
            w_tared = w_sim - wrench.offset
            peak = max(peak, float(np.linalg.norm(w_tared[:3])))
            T_tcp = _tcp_from_flange(T_fl, k)
            ok, msg = robot.ok()
            stop_reason = (None if ok else f"robot: {msg}") or check(lim, k, p_entry, R_entry, p_cmd, R_cmd, T_tcp[:3, 3], w_tared)
            if stop_flag is not None and stop_flag.is_set():
                stop_reason = "stopped by the operator"
            robot.period_wait(t0)
            if stop_reason:
                break
        t = (tick + 1) / k.policy_hz
        ob.push(filt, a)
        depth = tip_z - grasp.bottom(T_tcp)[2]
        for key, v in (("t", t), ("obs", obs), ("action", a), ("p_cmd", p_cmd), ("R_cmd", R_cmd), ("tcp", T_tcp[:3, 3]),
                       ("flange", T_fl), ("w_raw", w_raw), ("w_tared", w_tared), ("filtered", filt), ("est", est), ("depth", depth)):
            log[key].append(np.array(v, float))
        if stop_reason:
            outcome, reason = "aborted", stop_reason
            break
        if depth >= k.target_depth - 0.001:
            outcome, reason = "seated", f"kinematic depth {depth * 1e3:.1f} mm"
            break
        if verbose and tick % 15 == 0:
            print(f"  t {t:4.2f} s  depth {depth * 1e3:5.1f} mm  Fz {filt[2]:5.2f} N  |F| {np.linalg.norm(filt[:3]):5.2f} N  "
                  f"est ({est[0] * 1e3:+.1f}, {est[1] * 1e3:+.1f}) mm  a {np.round(a, 2)}", flush=True)
    robot.stop()
    log = {key: np.array(v) for key, v in log.items()}
    return EpisodeResult(outcome, reason, t, peak, float(log["depth"][-1] * 1e3) if len(log["depth"]) else 0.0, log)


def _tcp_from_flange(T_fl, k: SimConstants):
    T = T_fl.copy()
    T[:3, 3] = T_fl[:3, 3] + T_fl[:3, :3] @ np.array([0, 0, k.flange_to_tcp])
    return T


def _flange_target(p_tcp, R, k: SimConstants):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p_tcp) - R @ np.array([0, 0, k.flange_to_tcp])
    return T


def entry_pose(rod: Rod, grasp: Grasp, R_entry, k: SimConstants, air_m: float = 0.0):
    """tcp pose with the nut bottom entry_gap above the (air-shifted) tip, axis on the rod axis."""
    T = np.eye(4)
    T[:3, :3] = R_entry
    T_obj = T @ grasp.T_tcp_obj()
    # place the nut centre above the rod: solve tcp position so the nominal nut centre is there
    nut_centre = np.array([rod.xy[0], rod.xy[1], rod.tip_z + air_m + k.entry_gap + grasp.ring_height / 2])
    T[:3, 3] = nut_centre - (T_obj[:3, 3] - T[:3, 3])
    return T


def pick_entry_rotation(R_entry_sim, R_now):
    """The fin-ray grasp is symmetric under a half turn about the tool axis: use whichever of the
    two entry yaws is closer to the current orientation."""
    R_flip = R_entry_sim @ exp_so3([0, 0, math.pi])
    return R_entry_sim if angle_between(R_entry_sim, R_now) <= angle_between(R_flip, R_now) else R_flip
