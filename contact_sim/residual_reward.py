"""Reward for the force-aware residual (Version 2 of the plan), per policy tick (batched, torch).

    reward = 10 * first_verified_success - 10 * hard_failure + r_progress
             - dt * (0.1 + 2.0 * c_wrench + lambda_residual * c_residual + 0.02 * c_slew)

  Phi(s)      = -w_align * align_err(s) + w_depth * gate(s) * depth_frac(s)    (bounded, ~[-2, 1])
  r_progress  = gamma * Phi(s') - Phi(s), with Phi = 0 at absorbing terminals; time-limit ends are
                truncations (the learner bootstraps), not terminals
  c_wrench    = sum_i max(0, (|w_i| - L_i) / (H_i - L_i))^2, task frame about the TCP, averaged
                over the physics substeps of the tick (the env computes it)
  c_residual  = |residual / residual_scale|^2  (actions are already in residual-scale units)
  c_slew      = |residual - previous residual|^2

Success: rod in the bore, ring bottom within 1 mm of the target depth (geometry, never force).
Hard failure: force above the abort limit, or penetration beyond 0.5 mm (invalid physics never
counts as success). All weights and thresholds are starting choices, not validated settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RewardCfg:
    gamma: float = 0.99
    dt: float = 1 / 30
    w_align: float = 1.0
    w_depth: float = 1.0
    align_scale: float = 0.002     # m: offset that counts as align_err = 1
    gate_width: float = 0.0002     # m: softness of "inside the gap" for the depth term
    target_depth: float = 0.015    # m
    success_bonus: float = 10.0
    failure_cost: float = 10.0
    time_cost: float = 0.1
    w_wrench: float = 2.0
    lambda_residual: float = 0.1
    w_slew: float = 0.02
    # soft thresholds L and limits H per component: Fx Fy Fz (N), tx ty tz (N m); placeholders
    soft: tuple = (5.0, 5.0, 8.0, 0.3, 0.3, 0.3)
    hard: tuple = (30.0, 30.0, 40.0, 2.0, 2.0, 2.0)
    clip_align: float = 2.0
    extra: dict = field(default_factory=dict)


def potential(cfg: RewardCfg, e_xy: torch.Tensor, depth: torch.Tensor, gap: float) -> torch.Tensor:
    err = (e_xy.norm(dim=-1) / cfg.align_scale).clamp(max=cfg.clip_align)
    gate = torch.sigmoid((gap - e_xy.norm(dim=-1)) / cfg.gate_width)
    frac = (depth / cfg.target_depth).clamp(0.0, 1.0)
    return -cfg.w_align * err + cfg.w_depth * gate * frac


def wrench_cost(cfg: RewardCfg, w_tcp: torch.Tensor) -> torch.Tensor:
    L = torch.tensor(cfg.soft, device=w_tcp.device)
    H = torch.tensor(cfg.hard, device=w_tcp.device)
    assert bool((H > L).all())
    return (((w_tcp.abs() - L) / (H - L)).clamp_min(0) ** 2).sum(-1)


def step_reward(cfg: RewardCfg, phi_prev: torch.Tensor, phi_next: torch.Tensor, success: torch.Tensor,
                failure: torch.Tensor, c_wrench: torch.Tensor, residual: torch.Tensor, residual_prev: torch.Tensor) -> dict:
    """Per-tick reward for envs that were running at the start of the tick. success / failure:
    became terminal at this tick (Phi of the absorbing state is 0)."""
    terminal = success | failure
    phi_n = torch.where(terminal, torch.zeros_like(phi_next), phi_next)
    r_prog = cfg.gamma * phi_n - phi_prev
    c_res = (residual ** 2).sum(-1)
    c_slew = ((residual - residual_prev) ** 2).sum(-1)
    costs = cfg.dt * (cfg.time_cost + cfg.w_wrench * c_wrench + cfg.lambda_residual * c_res + cfg.w_slew * c_slew)
    r = cfg.success_bonus * success.float() - cfg.failure_cost * failure.float() + r_prog - costs
    return dict(reward=r, progress=r_prog, costs=costs, c_wrench=c_wrench, c_residual=c_res, c_slew=c_slew)
