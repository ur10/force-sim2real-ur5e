"""Simulated UR5e wrist force/torque channel (batched over envs, torch).

Input each physics step: the true wrench the tool receives beyond the flange (contact + held-object
loads, without the gripper's own weight because the simulated robot has gravity off), as force and
torque about the tool0 origin in the base frame, plus the tool0 and TCP poses.

Chain, in order (each stage has a measured counterpart in plan step 1):
  1. raw sensor load: + gripper weight at its centre of gravity (the sensor carries it)
  2. + per-episode bias (sensor frame) and its random-walk drift, + white noise
  3. payload compensation with the *configured* payload (mass / CoG errors leave a residual that
     changes with orientation, as on the robot)
  4. shift to the reporting point and express in the reporting frame:
        F_P = F_S,  tau_P = tau_S + (p_S - p_P) x F_S
  5. sample-and-hold at the sensor rate, then a fixed latency
  6. causal first-order low-pass (policy channel); the unfiltered channel is kept for monitoring
Only past samples are ever used.
"""

from __future__ import annotations

import math

import torch

from params import Sensor

G = 9.81


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) wxyz -> (N, 3, 3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def shift_wrench(w: torch.Tensor, p_from: torch.Tensor, p_to: torch.Tensor) -> torch.Tensor:
    """Wrench (N, 6) [F, tau about p_from] -> about p_to; all in one frame."""
    f, tau = w[:, :3], w[:, 3:]
    return torch.cat([f, tau + torch.cross(p_from - p_to, f, dim=-1)], -1)


class WrenchSensor:
    def __init__(self, cfg: Sensor, num_envs: int, sim_dt: float, device="cpu", seed: int | None = None):
        self.cfg, self.n, self.dt, self.device = cfg, num_envs, sim_dt, device
        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)
        self.hold_every = max(1, round((1 / cfg.rate_hz) / sim_dt)) if cfg.rate_hz < 1 / sim_dt else 1
        self.delay = max(0, round(cfg.latency_s / sim_dt))
        sample_dt = max(sim_dt, 1 / cfg.rate_hz)
        self.alpha = 1 - math.exp(-2 * math.pi * cfg.lowpass_hz * sample_dt) if cfg.lowpass_hz > 0 else 1.0
        # held object, part of the configured payload on the robot (its weight is in the true load)
        self.held_mass = 0.0
        self.held_cog_tool0 = torch.zeros(3, device=device)
        self.tare_offset = torch.zeros(num_envs, 6, device=device)
        self.tare_window = max(1, round(0.2 / sim_dt))  # tare averages the last 0.2 s
        self.reset()

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def reset(self, env_ids=None):
        c = self.cfg
        ids = slice(None) if env_ids is None else env_ids
        if env_ids is None:
            self.bias = torch.zeros(self.n, 6, device=self.device)
            self.held = torch.zeros(self.n, 6, device=self.device)
            self.lp = torch.zeros(self.n, 6, device=self.device)
            self.buf = torch.zeros(self.delay + 1, self.n, 6, device=self.device)
            self.step_count = 0
        k = self.bias[ids].shape[0]
        scale = torch.tensor([c.force_bias_n] * 3 + [c.torque_bias_nm] * 3, device=self.device)
        self.bias[ids] = self._randn(k, 6) * scale
        self.held[ids] = 0
        self.lp[ids] = 0
        self.tare_offset[ids] = 0
        self.recent = []
        self.buf[:, ids] = 0
        self.initialised = torch.zeros(self.n, dtype=torch.bool, device=self.device) if env_ids is None else self.initialised
        self.initialised[ids] = False

    def payload_wrench(self, mass: float, cog_tool0: torch.Tensor, R_tool0: torch.Tensor, p_tool0: torch.Tensor) -> torch.Tensor:
        """Weight of a payload as a wrench on the tool about tool0, base frame."""
        cog = p_tool0 + (R_tool0 @ cog_tool0.unsqueeze(-1)).squeeze(-1)
        f = torch.zeros_like(p_tool0)
        f[:, 2] = -mass * G
        return torch.cat([f, torch.cross(cog - p_tool0, f, dim=-1)], -1)

    def step(self, w_tool0_base: torch.Tensor, tool0_pose: torch.Tensor, tcp_pos: torch.Tensor) -> dict:
        """w_tool0_base (N, 6); tool0_pose (N, 7) wxyz_xyz; tcp_pos (N, 3). Returns the channels."""
        c = self.cfg
        R = quat_to_mat(tool0_pose[:, :4])
        p0 = tool0_pose[:, 4:]
        cog = torch.tensor(c.gripper_cog_tool0, device=self.device).expand(self.n, 3)
        # 1. what the strain gauges carry: the tool's own weight plus the external load
        raw = w_tool0_base + self.payload_wrench(c.gripper_mass, cog, R, p0)
        # 2. bias (fixed in the sensor frame, here tool0) with drift, and white noise
        self.bias += self._randn(self.n, 6) * torch.tensor([c.drift_n_per_s] * 3 + [c.drift_nm_per_s] * 3,
                                                           device=self.device) * math.sqrt(self.dt)
        bias_base = torch.cat([(R @ self.bias[:, :3, None])[..., 0], (R @ self.bias[:, 3:, None])[..., 0]], -1)
        noise = self._randn(self.n, 6) * torch.tensor([c.force_noise_n] * 3 + [c.torque_noise_nm] * 3, device=self.device)
        raw_uncomp = raw + bias_base + noise
        # 3. compensation with the configured payload (true + error)
        cog_cfg = cog + torch.tensor(c.payload_cog_error, device=self.device)
        comp = raw_uncomp - self.payload_wrench(c.gripper_mass + c.payload_mass_error, cog_cfg, R, p0)
        if self.held_mass > 0:
            comp = comp - self.payload_wrench(self.held_mass, self.held_cog_tool0.expand(self.n, 3), R, p0)
        # 4. reporting point and frame
        out = {}
        for name, w in (("raw", raw_uncomp), ("compensated", comp)):
            if c.reference == "tcp":
                w = shift_wrench(w, p0, tcp_pos)
            if c.frame == "tool":
                Rt = R.transpose(1, 2)
                w = torch.cat([(Rt @ w[:, :3, None])[..., 0], (Rt @ w[:, 3:, None])[..., 0]], -1)
            out[name] = c.sign * w
        out["compensated"] = out["compensated"] - self.tare_offset
        # 5. sample-and-hold, then latency (ring buffer), applied to the compensated channel
        if self.step_count % self.hold_every == 0:
            self.held = out["compensated"].clone()
        self.step_count += 1
        self.buf = torch.roll(self.buf, 1, 0)
        self.buf[0] = self.held
        delayed = self.buf[-1]
        self.recent.append(delayed.clone())
        self.recent = self.recent[-self.tare_window:]
        # 6. causal low-pass on the delayed signal
        new = ~self.initialised
        self.lp[new] = delayed[new]
        self.initialised |= True
        self.lp = self.lp + self.alpha * (delayed - self.lp)
        self.last = dict(raw_uncompensated=out["raw"], compensated_now=out["compensated"], compensated=delayed,
                         filtered=self.lp.clone())
        return self.last

    def tare(self, env_ids=None):
        """zero_ftsensor(): subtract the mean reading of the last 0.2 s from now on (robot still,
        no contact). It does not remove later drift or orientation-dependent payload errors."""
        ids = slice(None) if env_ids is None else env_ids
        off = torch.stack(self.recent).mean(0)[ids]
        self.tare_offset[ids] = self.tare_offset[ids] + off
        self.lp[ids] = self.lp[ids] - off
        self.buf[:, ids] = self.buf[:, ids] - off
        self.held[ids] = self.held[ids] - off
        self.recent = []
