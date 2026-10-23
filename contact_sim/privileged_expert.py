"""Privileged insertion expert for DAgger: it reads the true ring pose relative to the rod (which a
deployed policy cannot) and outputs TCP velocity commands in the task frame (base axes, z up the
rod). The student never sees these inputs; DAgger relabels the states the student visits with
this expert's actions, so the student learns what it can infer from its own observations.

Two styles:
  hover    centre the ring over the rod while holding just above the tip, enter only once the true
           offset is inside the gap. Not imitable: the enter decision depends on the hidden offset,
           and while hovering nothing touches, so the student's observations cannot tell "wait"
           from "enter" (a DAgger student learned to hover until timeout).
  contact  (default) descend until contact, then hold a light contact force (f_contact) while
           sliding toward the true axis; the ring drops in when aligned. Every decision (descend,
           hold force, stop) follows observable force; only the correction direction is privileged,
           and that is what the student must infer from wrench and vision.
Both keep centring inside the bore, straighten tilt and lift off above f_lift.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

ACT_DIM = 5  # vx, vy, vz, wx, wy (task frame); a = v / limit in [-1, 1]


@dataclass
class ActionLimits:
    v_lat: float = 0.006   # m/s
    v_z: float = 0.008     # m/s
    w: float = 0.10        # rad/s

    def scale(self, device) -> torch.Tensor:
        return torch.tensor([self.v_lat, self.v_lat, self.v_z, self.w, self.w], device=device)


@dataclass
class PrivilegedCfg:
    style: str = "contact"
    f_contact: float = 2.5       # N held on the tip while sliding (contact style)
    k_force: float = 0.004       # m/s per N of force error (contact style)
    k_xy: float = 5.0            # 1/s
    k_tilt: float = 3.0          # 1/s
    hover: float = 0.0015        # m above the tip while misaligned
    enter_frac: float = 0.5      # enter when |offset| < enter_frac * effective gap
    f_lift: float = 6.0          # N sensed axial force -> lift
    target_depth: float = 0.015  # m


class PrivilegedExpert:
    def __init__(self, cfg: PrivilegedCfg, limits: ActionLimits, gap: float, ring_height: float, device="cpu"):
        self.cfg, self.lim, self.gap, self.h, self.device = cfg, limits, gap, ring_height, device

    def act(self, priv: dict, fz_sensed: torch.Tensor) -> torch.Tensor:
        """priv: e_xy (N, 2) ring axis (bottom face) -> rod axis, depth (N,) ring bottom below the tip
        (< 0 above), axis (N, 3) ring axis (unit, up). Returns a (N, 5) in [-1, 1]."""
        c, L = self.cfg, self.lim
        e, depth, axis = priv["e_xy"], priv["depth"], priv["axis"]
        sin_t = axis[:, :2].norm(dim=-1)
        gap_eff = (self.gap - self.h / 2 * sin_t).clamp_min(1e-4)  # tilt eats into the gap
        aligned = e.norm(dim=-1) < c.enter_frac * gap_eff
        inside = depth > 0
        v_xy = (c.k_xy * e).clamp(-L.v_lat, L.v_lat)
        if c.style == "hover":
            # descend freely when aligned or already inside, otherwise hold just above the tip
            hold_z = (-c.hover - depth) * 4.0  # > 0 means go down (depth grows downwards)
            down = torch.where(aligned | inside, torch.full_like(depth, L.v_z), hold_z.clamp(-L.v_z, L.v_z))
        else:
            # force-regulated: full speed in free space, settle at f_contact on the tip, slide in
            f_reg = (c.k_force * (c.f_contact - fz_sensed)).clamp(-L.v_z, L.v_z)
            down = torch.where(inside & (depth > 0.0005), torch.full_like(depth, L.v_z), f_reg)
        down = torch.where(depth >= c.target_depth, torch.zeros_like(down), down)
        down = torch.where(fz_sensed > c.f_lift, torch.full_like(down, -0.25 * L.v_z), down)
        v_z = -down  # task z is up
        w = c.k_tilt * torch.stack([axis[:, 1], -axis[:, 0]], -1)  # rotate the axis back toward +z
        v = torch.cat([v_xy, v_z[:, None], w.clamp(-L.w, L.w)], -1)
        return (v / L.scale(self.device)).clamp(-1, 1)
