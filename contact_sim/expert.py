"""Reactive insertion expert that uses only what the policy will observe: the sensed (filtered,
delayed) wrench and the commanded TCP. No ground truth, so a student can learn why it moved.

State machine per env (batched, torch), all motions as offsets in the rod (task) frame:
  descend   damping-controlled descent at speed v toward the target depth
  blocked   axial force above f_block while depth progress stalls -> estimate where the contact is
            from the moment about the TCP and step toward it (a ring resting on the rod tip with an
            axis offset touches it on the rod's side: the contact centroid lies toward the rod, so
            tau = r x F with F ~ (0, 0, Fz) gives r_xy = (-tau_y, tau_x) / Fz); after max_tries / 2 tries without
            progress, fall back to an outward spiral
  backoff   lift until the axial force has released (the grip spring is compressed) plus a margin
  move      the lateral correction at v_lateral while clear of the rod, then descend again
            (a step taken while still pressed on the tip drives the bore wall into the rod edge)
  (axial force above f_max also triggers a back-off)
  done      reference reached target depth (the env decides success from geometry)
Tilt correction (rotating against the moment) is in the code but off by default until tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DESCEND, BLOCKED, BACKOFF, DONE, MOVE = 0, 1, 2, 3, 4


@dataclass
class ExpertCfg:
    v_descend: float = 0.005      # m/s
    f_block: float = 1.5          # N, axial force that means "resting on something"
    f_max: float = 12.0           # N, back off above this
    stall_window_s: float = 0.25  # no depth progress over this long + f_block -> blocked
    stall_progress: float = 3e-4  # m
    step: float = 0.0008          # m, lateral step toward the estimated contact
    max_step: float = 0.004       # m, total lateral correction allowed per attempt sequence
    rot_gain: float = 0.0         # rad per N m of moment, tilt correction per blocked event (off: untested)
    max_rot: float = 0.12         # rad
    backoff: float = 0.003        # m
    clear_lift: float = 0.0008    # m, lift beyond the point where the contact force released
    f_release: float = 0.6        # N
    v_lateral: float = 0.005      # m/s
    spiral_pitch: float = 0.0008  # m per turn
    max_tries: int = 12


class InsertionExpert:
    def __init__(self, cfg: ExpertCfg, num_envs: int, dt: float, device="cpu"):
        self.cfg, self.n, self.dt, self.device = cfg, num_envs, dt, device
        self.reset()

    def reset(self, env_ids=None):
        n, d = self.n, self.device
        if env_ids is None:
            self.mode = torch.zeros(n, dtype=torch.long, device=d)
            self.z = torch.zeros(n, device=d)          # commanded depth below the start (m, >= 0)
            self.xy = torch.zeros(n, 2, device=d)      # lateral correction (task frame)
            self.rxy = torch.zeros(n, 2, device=d)     # tilt correction
            self.tries = torch.zeros(n, dtype=torch.long, device=d)
            self.hist = []
            self.backoff_left = torch.zeros(n, device=d)
            self.spiral_phase = torch.zeros(n, device=d)
            self.w_avg = torch.zeros(n, 6, device=d)
            self.xy_target = torch.zeros(n, 2, device=d)
        else:
            for t in (self.mode, self.z, self.xy, self.rxy, self.tries, self.backoff_left, self.spiral_phase, self.w_avg, self.xy_target):
                t[env_ids] = 0

    def act(self, w: torch.Tensor, depth_cmd_target: float, tcp_depth: torch.Tensor) -> dict:
        """w: (N, 6) sensed wrench on the tool, task frame (z up along the rod), about the TCP.
        tcp_depth: (N,) how far the *measured* TCP has gone below the start (m): proprioception.
        Returns the commanded offsets for this tick."""
        c, dt = self.cfg, self.dt
        # average over the stall window: the contact moment (~1 cm x F) is small next to force noise
        # times the sensor-to-TCP lever, so a single filtered sample cannot say where the rod is
        self.w_avg = self.w_avg + (w - self.w_avg) * min(1.0, dt / c.stall_window_s)
        fz = w[:, 2]
        self.hist.append(tcp_depth.clone())
        win = max(1, int(c.stall_window_s / dt))
        self.hist = self.hist[-win - 1:]
        progress = tcp_depth - self.hist[0]
        stalled = (len(self.hist) > win) & (progress < c.stall_progress)
        desc = self.mode == DESCEND
        # descend: advance the reference, never more than 2 mm ahead of the measured TCP
        adv = desc & (self.z < depth_cmd_target)
        self.z = torch.where(adv, torch.minimum(self.z + c.v_descend * dt, tcp_depth + 0.002), self.z)
        self.mode = torch.where(desc & (self.z >= depth_cmd_target) & (tcp_depth >= depth_cmd_target - 0.002), DONE, self.mode)
        # blocked: resting on something while not progressing, or pressing past f_max while descending
        # (the grip spring can load up before the stall window fills); both get a lateral correction
        hit = desc & (fz > c.f_block) & (stalled | (fz > c.f_max))
        over = (self.mode != DONE) & (self.mode != DESCEND) & (fz > c.f_max)
        self.mode = torch.where(over, BACKOFF, self.mode)
        self.backoff_left = torch.where(over, torch.full_like(self.backoff_left, c.backoff), self.backoff_left)
        # blocked: one corrective step per event, then descend again
        if hit.any():
            wa = self.w_avg
            f = wa[:, 2].clamp_min(0.2)
            r = torch.stack([-wa[:, 4], wa[:, 3]], -1) / f[:, None]  # contact point from tau = r x F (task frame, m)
            nr = r.norm(dim=-1, keepdim=True)
            dirn = torch.where(nr > 1e-4, r / nr.clamp_min(1e-9), torch.zeros_like(r))
            use_spiral = (self.tries >= c.max_tries // 2) | (nr[:, 0] < 1e-4)
            self.spiral_phase = self.spiral_phase + use_spiral.float() * 0.9
            rad = c.spiral_pitch * self.spiral_phase / (2 * torch.pi)
            spiral = torch.stack([torch.cos(self.spiral_phase), torch.sin(self.spiral_phase)], -1) * rad[:, None]
            step = torch.where(use_spiral[:, None], spiral - self.xy, dirn * c.step)
            new_xy = (self.xy + step)
            n_xy = new_xy.norm(dim=-1, keepdim=True)
            new_xy = torch.where(n_xy > c.max_step, new_xy / n_xy * c.max_step, new_xy)
            self.xy_target = torch.where(hit[:, None], new_xy, self.xy_target)  # applied after lifting clear
            # tilt: rotate against the moment component not explained by the lateral step
            drot = torch.stack([-w[:, 3], -w[:, 4]], -1) * c.rot_gain * 0.1
            self.rxy = torch.where(hit[:, None], (self.rxy + drot).clamp(-c.max_rot, c.max_rot), self.rxy)
            self.tries = self.tries + hit.long()
            # lift until the contact releases, then clear_lift more, so the step happens off the rod
            self.backoff_left = torch.where(hit, torch.full_like(self.backoff_left, c.clear_lift), self.backoff_left)
            self.mode = torch.where(hit, BACKOFF, self.mode)
            self.hist = [tcp_depth.clone()]
        bo = self.mode == BACKOFF
        released = fz < c.f_release
        lift = torch.where(released, torch.clamp(self.backoff_left, max=c.v_descend * 4 * dt), torch.full_like(fz, c.v_descend * 4 * dt))
        self.z = torch.where(bo, (self.z - lift).clamp_min(0), self.z)
        self.backoff_left = torch.where(bo & released, self.backoff_left - lift, self.backoff_left)
        self.mode = torch.where(bo & released & (self.backoff_left <= 1e-6), MOVE, self.mode)
        mv = self.mode == MOVE
        d_xy = self.xy_target - self.xy
        dist = d_xy.norm(dim=-1, keepdim=True)
        stepv = torch.where(dist > c.v_lateral * dt, d_xy / dist.clamp_min(1e-9) * c.v_lateral * dt, d_xy)
        self.xy = torch.where(mv[:, None], self.xy + stepv, self.xy)
        arrived = mv & (dist[:, 0] <= c.v_lateral * dt)
        self.mode = torch.where(arrived, DESCEND, self.mode)
        if arrived.any():
            self.hist = [tcp_depth.clone()]
        give_up = self.tries > c.max_tries
        return dict(z=self.z, xy=self.xy, rxy=self.rxy, mode=self.mode, give_up=give_up)
