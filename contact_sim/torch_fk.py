"""Batched UR5e forward kinematics and geometric Jacobian in torch, from the URDF joint chain
(ur_base -> tcp by default). Used for differential IK on the *commanded* configuration, the way a
position-controlled UR is driven through servoJ, so the arm's own tracking lag is not fed back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yourdfpy

ARM = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


class TorchFK:
    def __init__(self, urdf: Path, base: str = "ur_base", tip: str = "tcp", device="cpu"):
        u = yourdfpy.URDF.load(urdf, load_meshes=False, build_collision_scene_graph=False)
        by_child = {j.child: j for j in u.robot.joints}
        chain, link = [], tip
        while link != base:
            j = by_child[link]
            chain.append(j)
            link = j.parent
        chain.reverse()
        self.device = device
        self.origins = torch.tensor(np.stack([j.origin if j.origin is not None else np.eye(4) for j in chain]),
                                    dtype=torch.float32, device=device)
        self.axes = torch.tensor(np.stack([np.asarray(j.axis if j.axis is not None else [0, 0, 1], float) for j in chain]),
                                 dtype=torch.float32, device=device)
        self.qidx = [ARM.index(j.name) if j.type in ("revolute", "continuous") else -1 for j in chain]
        assert sorted(i for i in self.qidx if i >= 0) == list(range(6)), [j.name for j in chain]

    @staticmethod
    def _rot(axis: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
        n = ang.shape[0]
        k = axis.expand(n, 3)
        K = torch.zeros(n, 3, 3, device=ang.device)
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
        K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
        s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
        T = torch.eye(4, device=ang.device).repeat(n, 1, 1)
        T[:, :3, :3] = torch.eye(3, device=ang.device) + s * K + (1 - c) * K @ K
        return T

    def forward(self, q: torch.Tensor):
        """q (N, 6) -> tip pose (N, 4, 4) and per revolute joint (world axis, world position)."""
        n = q.shape[0]
        T = torch.eye(4, device=q.device).repeat(n, 1, 1)
        axes, pos = [None] * 6, [None] * 6
        for k, qi in enumerate(self.qidx):
            T = T @ self.origins[k]
            if qi >= 0:
                axes[qi] = (T[:, :3, :3] @ self.axes[k][:, None])[..., 0]
                pos[qi] = T[:, :3, 3]
                T = T @ self._rot(self.axes[k], q[:, qi])
        return T, torch.stack(axes, 1), torch.stack(pos, 1)

    def jacobian(self, q: torch.Tensor):
        """(N, 4, 4) tip pose and (N, 6, 6) [linear; angular] geometric Jacobian."""
        T, z, p = self.forward(q)
        tip = T[:, :3, 3][:, None, :]
        Jl = torch.cross(z, tip - p, dim=-1)
        return T, torch.cat([Jl, z], -1).transpose(1, 2)
