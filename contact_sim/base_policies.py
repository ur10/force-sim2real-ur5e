"""Frozen base policies for the residual (normalised TCP-velocity actions, task frame)."""

import torch


def replay_base(fz_sensed: torch.Tensor, v_z_limit: float, speed: float = 0.005, f_stop: float = 3.0) -> torch.Tensor:
    """The generated demonstration's insertion executed blindly: straight down the nominal axis at
    `speed`, no longer pushing once the sensed axial force exceeds f_stop (force-limited)."""
    a = torch.zeros(fz_sensed.shape[0], 5, device=fz_sensed.device)
    a[:, 2] = torch.where(fz_sensed > f_stop, torch.zeros_like(fz_sensed), torch.full_like(fz_sensed, -speed / v_z_limit))
    return a


def priv_vector(priv: dict) -> torch.Tensor:
    """Privileged critic inputs: offset (mm), depth (mm), axis tilt (x10), rod in the bore."""
    return torch.cat([priv["e_xy"] * 1000, priv["depth"][:, None] * 1000, priv["axis"][:, :2] * 10,
                      priv["in_hole"].float()[:, None]], -1)
