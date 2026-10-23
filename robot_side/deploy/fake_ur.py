"""A stand-in UR5e for --dry-run (no network, no robot): the flange follows servo targets with a
short lag, and a crude contact model gives the wrench (UR convention: actual_TCP_force, base frame,
torque about the flange). It is a rehearsal of the flow and the safety stops, not a simulator.
"""

from __future__ import annotations

import time

import numpy as np

from deploy_core import Grasp, Rod, SimConstants, _tcp_from_flange


class FakeUR:
    def __init__(self, T_flange0, rod: Rod, grasp: Grasp, k: SimConstants, clearance_m=0.0027, chamfer_m=0.001,
                 k_contact=1500.0, noise_n=0.15, sign=1.0, lag_s=0.01, realtime=False, seed=0):
        self.T = np.array(T_flange0, float)
        self.target = self.T.copy()
        self.rod, self.grasp, self.k = rod, grasp, k
        self.c, self.ch, self.kc, self.noise, self.sign, self.lag = clearance_m, chamfer_m, k_contact, noise_n, sign, lag_s
        self.realtime = realtime
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.calls = 0

    # -- robot interface used by deploy_core.run_episode / deploy_policy.py
    def flange(self):
        return self.T.copy()

    def wrench_raw(self):
        T_tcp = _tcp_from_flange(self.T, self.k)
        b = self.grasp.bottom(T_tcp)
        pen = self.rod.tip_z - b[2]
        F = np.zeros(3)
        p_contact = b.copy()
        if pen > 0:
            d_vec = b[:2] - self.rod.xy
            d = float(np.linalg.norm(d_vec))
            if d > self.c + self.ch:          # landed on the rod top
                F[2] = self.kc * pen
            elif d > self.c:                  # on the chamfer: pushed up and towards the axis
                F[2] = self.kc * pen * 0.5
                F[:2] = -d_vec / max(d, 1e-9) * self.kc * pen * 0.5
        F = F + self.rng.normal(0, self.noise, 3)
        tau = np.cross(p_contact - self.T[:3, 3], F)
        return self.sign * np.concatenate([F, tau])

    def servo(self, T_target, dt):
        self.target = np.array(T_target, float)
        a = 1.0 - np.exp(-dt / self.lag) if self.lag > 0 else 1.0
        self.T[:3, 3] += a * (self.target[:3, 3] - self.T[:3, 3])
        self.T[:3, :3] = self.target[:3, :3]
        self.t += dt
        self.calls += 1

    def move_to(self, T_target, speed):
        self.T = np.array(T_target, float)
        self.target = self.T.copy()

    def period_start(self):
        return time.perf_counter()

    def period_wait(self, t0):
        if self.realtime:
            dt = 1.0 / self.k.ctrl_hz - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)

    def stop(self):
        pass

    def ok(self):
        return True, ""

    def close(self):
        pass
