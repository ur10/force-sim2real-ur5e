"""Batched contact-phase environment for DAgger (import after the Isaac app is launched).

Each env holds the ring with its own in-hand error (offset + tilt, fixed for the process; the
controller assumes the nominal grasp). Every episode starts at the generated entry pose (ring
~6 mm above the rod tip on the nominal axis) plus a fresh approach error, and ends when the ring
is seated, the force limit is hit, or time runs out. Actions are TCP velocities in the task frame
(base axes, z up the rod), integrated into a commanded pose and executed through differential IK
on the commanded configuration, like servoJ on the real arm.

Episodes run in lockstep (all envs reset together); an env that finished holds still and its
samples are marked invalid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from admittance import dls_ik_step, exp_so3, skew_log
from contact_utils import M, P, NumpyIK, grasp_T_tcp_obj, ring_metrics
from privileged_expert import ActionLimits
from ring_rod_sim import RingRodScene
from student import OBS_DIM
from torch_fk import TorchFK
from wrench_sensor import WrenchSensor, quat_to_mat, shift_wrench


@dataclass
class EnvCfg:
    inhand_offset_max: float = 0.002   # m, disc
    inhand_tilt_max: float = 2.0       # deg, random in-plane axis (finray: rocking about the grasp line)
    inhand_closing_offset_max: float = 0.0003  # m, finray only: along the closing axis (finger asymmetry)
    approach_max: float = 0.002        # m, disc, per episode
    entry_gap: float = 0.006           # m, ring bottom above the tip at the start
    target_depth: float = 0.015        # m
    episode_s: float = 8.0
    policy_hz: float = 30.0
    vision_bias: float = 0.001         # m, per-episode bias of the vision offset estimate
    vision_noise: float = 0.0003       # m, per-step noise
    # "oracle": the true ring-vs-rod offset + bias + noise (stand-in for an in-hand ring tracker);
    # "kinematic": registered rod axis - ring position predicted from the TCP with the nominal grasp
    # (deployable without a tracker: blind to the in-hand error, carries the registration error)
    vision_source: str = "oracle"
    registration_bias: float = 0.001   # m, per-episode registration error (kinematic source), std per axis
    speed_scale: float = 1.0           # scales the action limits (speed sweeps of the privileged expert)
    abort_force: float = 40.0          # N
    history: int = 8                   # policy ticks of wrench history


def _mat(pose: torch.Tensor) -> torch.Tensor:
    T = torch.eye(4, device=pose.device).repeat(pose.shape[0], 1, 1)
    T[:, :3, :3] = quat_to_mat(pose[:, :4])
    T[:, :3, 3] = pose[:, 4:]
    return T


class ContactEnv:
    def __init__(self, sim, p, n: int, export_npz, urdf, cfg: EnvCfg, seed: int = 0, device="cuda:0"):
        self.sim, self.p, self.n, self.cfg, self.dev = sim, p, n, cfg, device
        self.g = p.geometry
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        L0 = ActionLimits()
        self.limits = ActionLimits(v_lat=L0.v_lat * cfg.speed_scale, v_z=L0.v_z * cfg.speed_scale, w=L0.w * cfg.speed_scale)
        ex = np.load(export_npz)
        g = self.g
        obj, tcp = ex["T_base_obj"], ex["T_base_tcp"]
        bottom = obj[:, 6] - g.ring_height / 2
        near = np.where((np.hypot(obj[:, 4] - g.rod_xy[0], obj[:, 5] - g.rod_xy[1]) < 5e-4) & (bottom < g.rod_tip_z + cfg.entry_gap + 2e-3))[0]
        k = int(near[0])
        T_tcp_obj = grasp_T_tcp_obj(M(ex["T_tcp_obj"]), p.geometry, p.in_hand)
        # entry: nominal grasp, ring axis on the rod axis, bottom entry_gap above the tip
        T_ring = M(obj[k])
        T_ring[:3, 3] = [g.rod_xy[0], g.rod_xy[1], g.rod_tip_z + cfg.entry_gap + g.ring_height / 2]
        self.T_entry = T_ring @ np.linalg.inv(T_tcp_obj)
        ik = NumpyIK()
        q_entry, err = ik.ik(self.T_entry, ex["q"][k])
        assert err < 1e-4, err
        self.q_entry = q_entry
        T_link_tcp = ik.u.get_transform("tcp", "robotiq_85_base_link")
        # per-env in-hand error
        T_lr = []
        self.inhand = []
        finray = p.in_hand.grasp_model == "finray"
        # closing direction (gripper link x) in the ring frame: a pinch centres the ring along it
        c = (T_link_tcp @ T_tcp_obj)[:3, :3].T @ np.array([1.0, 0.0, 0.0])
        c = np.array([c[0], c[1], 0.0]) / max(np.hypot(c[0], c[1]), 1e-9)
        across = np.cross([0.0, 0.0, 1.0], c)
        for _ in range(n):
            E = np.eye(4)
            if finray:
                # realistic pinch errors: offset across the fingers (and a little along the closing
                # axis from finger asymmetry), rocking about the line between the two contacts
                off_y = cfg.inhand_offset_max * (2 * self.rng.uniform() - 1)
                off_x = cfg.inhand_closing_offset_max * (2 * self.rng.uniform() - 1)
                rock = np.radians(cfg.inhand_tilt_max) * (2 * self.rng.uniform() - 1)
                pitch = np.radians(cfg.inhand_tilt_max) * 0.25 * (2 * self.rng.uniform() - 1)
                E[:3, :3] = Rotation.from_rotvec(rock * c + pitch * across).as_matrix()
                E[:3, 3] = off_x * c + off_y * across
                r, t = float(np.hypot(off_x, off_y)), float(np.hypot(rock, pitch))
            else:
                r, a = cfg.inhand_offset_max * np.sqrt(self.rng.uniform()), self.rng.uniform(0, 2 * np.pi)
                t, b = np.radians(cfg.inhand_tilt_max) * self.rng.uniform(), self.rng.uniform(0, 2 * np.pi)
                E[:3, :3] = Rotation.from_rotvec(t * np.array([np.cos(b), np.sin(b), 0])).as_matrix()
                E[:3, 3] = [r * np.cos(a), r * np.sin(a), 0]
            T_lr.append(T_link_tcp @ T_tcp_obj @ E)
            self.inhand.append((r, np.degrees(t)))
        ring0 = np.stack([P(ik.fk(q_entry, "robotiq_85_base_link") @ T) for T in T_lr])
        self.scene = RingRodScene(sim.stage, p, n, np.tile(q_entry, (n, 1)), 0.55, np.stack([P(T) for T in T_lr]), ring0,
                                  device=device)
        self.T_lr = torch.tensor(np.stack(T_lr), dtype=torch.float32, device=device)
        self.T_tool0_ring = ik.u.get_transform("tcp", "tool0") @ T_tcp_obj
        self.T_tcp_obj_nom = torch.tensor(T_tcp_obj, dtype=torch.float32, device=device)
        assert cfg.vision_source in ("oracle", "kinematic"), cfg.vision_source
        self.urdf = urdf

    # ------------------------------------------------------------------ setup after sim.reset()
    def post_reset(self):
        s, dev, n = self.scene, self.dev, self.n
        s.post_reset()
        self.robot, self.ring = s.robot, s.ring
        self.dt = self.p.sim.dt
        self.fk = TorchFK(self.urdf, device=dev)
        self.sensor = WrenchSensor(self.p.sensor, n, self.dt, dev, seed=int(self.rng.integers(1 << 30)))
        self.sensor.held_mass = self.p.in_hand.mass
        self.sensor.held_cog_tool0 = torch.tensor(self.T_tool0_ring[:3, 3], dtype=torch.float32, device=dev)
        self.q_cmd = torch.tensor(np.tile(self.q_entry, (n, 1)), dtype=torch.float32, device=dev)
        self.T_entry_t = torch.tensor(self.T_entry, dtype=torch.float32, device=dev)
        self.ctrl_every = max(1, round(1 / (self.p.controller.rate_hz * self.dt)))
        self.ctrl_per_policy = max(1, round(self.p.controller.rate_hz / self.cfg.policy_hz))
        self.lim = self.limits.scale(dev)
        self.wrench_cost_fn = None  # optional: cost of the TCP wrench, averaged over physics substeps

    def _physics(self, k: int = 1):
        for _ in range(k):
            self.robot.write_data_to_sim()
            self.ring.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(self.dt)
            self.ring.update(self.dt)

    def _tool_load(self) -> torch.Tensor:
        raw = self.scene.wrench_incoming()  # gripper-link frame, arm on gripper (validated in the probe)
        lk = self.scene.link_pose("robotiq_85_base_link")
        R = quat_to_mat(lk[:, :4])
        w = -torch.cat([(R @ raw[:, :3, None])[..., 0], (R @ raw[:, 3:, None])[..., 0]], -1)
        return shift_wrench(w, lk[:, 4:], self.scene.link_pose("tool0")[:, 4:])

    def _set_targets(self):
        qt = self.robot.data.default_joint_pos.clone()
        qt[:, self.scene.arm_ids] = self.q_cmd
        self.robot.set_joint_position_target(qt)

    def _ik_to(self, R_cmd, p_cmd, iters: int = 2):
        for _ in range(iters):
            T, J = self.fk.jacobian(self.q_cmd)
            self.q_cmd += dls_ik_step(J, p_cmd - T[:, :3, 3], skew_log(R_cmd @ T[:, :3, :3].transpose(1, 2)), damping=1e-3)

    def _sensor_step(self):
        w = self._tool_load()
        ch = self.sensor.step(w, self.scene.link_pose("tool0"), self.scene.link_pose("tcp")[:, 4:])
        return w, ch

    # ------------------------------------------------------------------ episodes
    def reset(self):
        c, n, dev = self.cfg, self.n, self.dev
        r = c.approach_max * torch.sqrt(torch.rand(n, generator=self.gen, device=dev))
        a = torch.rand(n, generator=self.gen, device=dev) * 2 * torch.pi
        self.approach = torch.stack([r * torch.cos(a), r * torch.sin(a)], -1)
        self.R_cmd = self.T_entry_t[:3, :3].expand(n, 3, 3).clone()
        self.p_cmd = self.T_entry_t[:3, 3].expand(n, 3).clone()
        self.p_cmd[:, :2] += self.approach
        self.p_start = self.p_cmd.clone()
        self.q_cmd[:] = torch.tensor(self.q_entry, dtype=torch.float32, device=dev)
        self._ik_to(self.R_cmd, self.p_cmd, iters=40)
        qt = self.robot.data.default_joint_pos.clone()
        qt[:, self.scene.arm_ids] = self.q_cmd
        self.robot.write_joint_state_to_sim(qt, torch.zeros_like(qt))
        self.robot.set_joint_position_target(qt)
        self._physics(2)
        Tr = _mat(self.scene.link_pose("robotiq_85_base_link")) @ self.T_lr
        from isaaclab.utils.math import quat_from_matrix
        self.scene.set_ring_pose(torch.cat([quat_from_matrix(Tr[:, :3, :3]), Tr[:, :3, 3]], -1))
        self.sensor.reset()
        for _ in range(int(0.25 / self.dt)):
            self._sensor_step()
            self._physics()
        self.sensor.tare()
        self.hist = torch.zeros(n, c.history, 6, device=dev)
        self.prev_a = torch.zeros(n, 5, device=dev)
        self.vbias = torch.randn(n, 2, generator=self.gen, device=dev) * (
            c.registration_bias if c.vision_source == "kinematic" else c.vision_bias)
        self.done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.outcome = torch.zeros(n, dtype=torch.long, device=dev)  # 0 running/timeout, 1 seated, 2 aborted, 3 invalid
        self.peak_f = torch.zeros(n, device=dev)
        self.t = 0.0
        self.t_done = torch.full((n,), float("nan"), device=dev)
        _, ch = self._sensor_step()
        self.hist[:] = ch["filtered"][:, None, :]
        return self.observe()

    def privileged(self) -> dict:
        rp = self.scene.ring_pose()
        m = ring_metrics(rp, self.g)
        R = quat_to_mat(rp[:, :4])
        axis = R[:, :, 2] * torch.sign(R[:, 2:3, 2] + 1e-9)
        bottom = rp[:, 4:] - axis * self.g.ring_height / 2
        rod = torch.tensor(self.g.rod_xy, device=self.dev)
        return dict(e_xy=rod - bottom[:, :2], depth=m["depth"], axis=axis, in_hole=m["in_hole"], penetration=m["penetration"])

    def observe(self) -> torch.Tensor:
        c = self.cfg
        tcp = self.scene.link_pose("tcp")
        pr = self.privileged()
        noise = torch.randn(self.n, 2, generator=self.gen, device=self.dev) * c.vision_noise
        if c.vision_source == "kinematic":
            Tn = _mat(tcp) @ self.T_tcp_obj_nom
            axis = Tn[:, :3, 2] * torch.sign(Tn[:, 2:3, 2] + 1e-9)
            bottom = Tn[:, :3, 3] - axis * self.g.ring_height / 2
            est = torch.tensor(self.g.rod_xy, device=self.dev) - bottom[:, :2] + self.vbias + noise
        else:
            est = pr["e_xy"] + self.vbias + noise
        obs = torch.cat([self.p_cmd - self.p_start, tcp[:, 4:] - self.p_start, self.hist.reshape(self.n, -1), self.prev_a,
                         est, torch.ones(self.n, 1, device=self.dev)], -1)
        assert obs.shape[1] == OBS_DIM
        return obs

    def step(self, a: torch.Tensor):
        """a (N, 5) in [-1, 1]; one policy tick. Returns obs, info."""
        c = self.cfg
        a = a.clamp(-1, 1)
        a = torch.where(self.done[:, None], torch.zeros_like(a), a)
        v = a * self.lim
        dtc = self.dt * self.ctrl_every
        cost = torch.zeros(self.n, device=self.dev)
        n_sub = 0
        for _ in range(self.ctrl_per_policy):
            self.p_cmd = self.p_cmd + v[:, :3] * dtc
            w = torch.cat([v[:, 3:], torch.zeros(self.n, 1, device=self.dev)], -1)
            self.R_cmd = exp_so3(w * dtc) @ self.R_cmd
            self._ik_to(self.R_cmd, self.p_cmd)
            self._set_targets()
            for _ in range(self.ctrl_every):
                w_true, ch = self._sensor_step()
                self._physics()
                f = w_true[:, :3].norm(dim=-1)
                if self.wrench_cost_fn is not None:
                    w_tcp = shift_wrench(w_true, self.scene.link_pose("tool0")[:, 4:], self.scene.link_pose("tcp")[:, 4:])
                    cost += self.wrench_cost_fn(w_tcp)
                    n_sub += 1
                self.peak_f = torch.maximum(self.peak_f, torch.where(self.done, torch.zeros_like(f), f))
                ab = (~self.done) & (f > c.abort_force)
                self.outcome = torch.where(ab, torch.full_like(self.outcome, 2), self.outcome)
                self.t_done = torch.where(ab, torch.full_like(self.t_done, self.t), self.t_done)
                self.done |= ab
        self.t += 1 / c.policy_hz
        self.hist = torch.cat([self.hist[:, 1:], ch["filtered"][:, None, :]], 1)
        self.prev_a = a
        pr = self.privileged()
        invalid = (~self.done) & (pr["penetration"] > 0.0005)  # invalid physics never counts as success
        self.outcome = torch.where(invalid, torch.full_like(self.outcome, 3), self.outcome)
        self.t_done = torch.where(invalid, torch.full_like(self.t_done, self.t), self.t_done)
        self.done |= invalid
        seated = (~self.done) & pr["in_hole"] & (pr["depth"] >= c.target_depth - 0.001)
        self.outcome = torch.where(seated, torch.ones_like(self.outcome), self.outcome)
        self.t_done = torch.where(seated, torch.full_like(self.t_done, self.t), self.t_done)
        self.done |= seated
        return self.observe(), dict(priv=pr, fz=ch["filtered"][:, 2], w_true=w_true, c_wrench=cost / max(n_sub, 1))
