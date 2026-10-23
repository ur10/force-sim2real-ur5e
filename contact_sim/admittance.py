"""Task-frame admittance (batched, torch): a virtual mass-damper-spring between the reference pose
and the commanded pose, driven by the measured wrench minus a desired contact wrench:

    M e'' + D e' + K e = W_meas - W_des        (per axis, task frame: x y z rx ry rz)

With K = 0 on the insertion axis this is damping control: free space descends at F_des / D and
contact settles where the rod pushes back with F_des. The command is reference (+) e, then
differential IK (damped least squares) to joint targets for the stiff, position-controlled arm.
"""

from __future__ import annotations

import torch

from params import Controller


def skew_log(R: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) -> rotation vector (N, 3), small-angle safe."""
    cos = ((R.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    ang = torch.arccos(cos)
    v = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    s = torch.where(ang[:, None] < 1e-4, torch.full_like(v, 0.5), ang[:, None] / (2 * torch.sin(ang)[:, None]))
    return v * s


def exp_so3(w: torch.Tensor) -> torch.Tensor:
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    K = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K = K - K.transpose(1, 2)
    th = th[..., None]
    return torch.eye(3, device=w.device, dtype=w.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


class Admittance:
    def __init__(self, cfg: Controller, num_envs: int, device="cpu"):
        self.cfg, self.n, self.device = cfg, num_envs, device
        t = lambda v: torch.tensor(v, dtype=torch.float32, device=device).expand(num_envs, 6).clone()  # noqa: E731
        self.M, self.D, self.K, self.lim = t(cfg.mass), t(cfg.damping), t(cfg.stiffness), t(cfg.max_offset)
        self.db = t(cfg.deadband)
        self.reset()

    def reset(self, env_ids=None):
        if env_ids is None:
            self.e = torch.zeros(self.n, 6, device=self.device)
            self.ed = torch.zeros(self.n, 6, device=self.device)
        else:
            self.e[env_ids] = 0
            self.ed[env_ids] = 0

    def step(self, w_task: torch.Tensor, w_des: torch.Tensor, dt: float) -> torch.Tensor:
        """Backward Euler on (e, e'), stable for any M, D, K and step (an explicit update diverges
        once D dt / M > 2, e.g. M = 1, D = 400 at 120 Hz); returns the offset e (N, 6), clamped."""
        f = w_task - w_des
        f = torch.sign(f) * (f.abs() - self.db).clamp_min(0)  # deadband against noise and drift
        self.ed = (self.M * self.ed + dt * (f - self.K * self.e)) / (self.M + dt * self.D + dt * dt * self.K)
        e = self.e + self.ed * dt
        clamped = e.clamp(-self.lim, self.lim)
        self.ed = torch.where(clamped != e, torch.zeros_like(self.ed), self.ed)
        self.e = clamped
        return self.e

    @staticmethod
    def compose(ref_R: torch.Tensor, ref_p: torch.Tensor, task_R: torch.Tensor, e: torch.Tensor):
        """Commanded TCP: translate by e[:3] (task axes) and rotate by e[3:] about the TCP point
        (task axes), so a rotation does not swing the tool about some other origin."""
        p = ref_p + (task_R @ e[:, :3, None])[..., 0]
        w_base = (task_R @ e[:, 3:, None])[..., 0]
        R = exp_so3(w_base) @ ref_R
        return R, p


def dls_ik_step(J: torch.Tensor, pos_err: torch.Tensor, rot_err: torch.Tensor, damping: float = 0.01) -> torch.Tensor:
    """Joint step (N, n_dof) from a (N, 6, n_dof) Jacobian [linear; angular] and pose error."""
    err = torch.cat([pos_err, rot_err], -1)[..., None]
    Jt = J.transpose(1, 2)
    lam = torch.eye(6, device=J.device) * damping ** 2
    return (Jt @ torch.linalg.solve(J @ Jt + lam, err))[..., 0]
