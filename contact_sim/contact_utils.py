"""Helpers shared by the contact scripts (no Isaac imports): pose conversions, numpy IK on the
planner URDF, and ring-vs-rod geometry."""

from pathlib import Path

import numpy as np
import torch
import yourdfpy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from wrench_sensor import quat_to_mat

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "data/ur5e_robotiq/ur5e_robotiq85_ik.urdf"
ARM = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def M(p):  # wxyz_xyz -> 4x4
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(p)[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = p[4:]
    return T


def P(T):  # 4x4 -> wxyz_xyz
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.r_[q[[3, 0, 1, 2]], T[:3, 3]]


class NumpyIK:
    def __init__(self):
        self.u = yourdfpy.URDF.load(URDF, load_meshes=False, build_collision_scene_graph=False)
        self.order = [ARM.index(n) for n in self.u.actuated_joint_names]

    def fk(self, q, frame="tcp", base="ur_base"):
        self.u.update_cfg(np.asarray(q)[self.order])
        return self.u.get_transform(frame, base)

    def ik(self, T_goal, seed):
        def res(q):
            T = self.fk(q)
            dp = T[:3, 3] - T_goal[:3, 3]
            dr = Rotation.from_matrix(T_goal[:3, :3].T @ T[:3, :3]).as_rotvec()
            return np.r_[dp * 10, dr]
        out = least_squares(res, seed, xtol=1e-12, ftol=1e-12)
        T = self.fk(out.x)
        err = np.linalg.norm(T[:3, 3] - T_goal[:3, 3])
        return out.x, err


_ROD_PTS = {}


def rod_points(g, device) -> torch.Tensor:
    """Points on the rod's side (from the tip down 25 mm) and top disc, env frame."""
    key = (g.rod_r, g.rod_tip_z, tuple(g.rod_xy), str(device))
    if key not in _ROD_PTS:
        a = torch.linspace(0, 2 * np.pi, 49, device=device)[:-1]
        side = [torch.stack([g.rod_r * torch.cos(a), g.rod_r * torch.sin(a), torch.full_like(a, z)], -1)
                for z in np.linspace(g.rod_tip_z, max(g.rod_z0, g.rod_tip_z - 0.025), 26)]
        top = [torch.stack([f * g.rod_r * torch.cos(a), f * g.rod_r * torch.sin(a), torch.full_like(a, g.rod_tip_z)], -1)
               for f in (0.0, 0.33, 0.66)]
        pts = torch.cat(side + top)
        pts[:, 0] += g.rod_xy[0]
        pts[:, 1] += g.rod_xy[1]
        _ROD_PTS[key] = pts
    return _ROD_PTS[key]


def ring_metrics(ring_pose: torch.Tensor, g) -> dict:
    """Ring vs rod: offset of the bottom-face centre from the rod axis, depth over the tip (bottom
    face centre), tilt, whether the rod is in the bore, and penetration: how deep any sampled rod
    surface point sits inside the ring's material (0 when none is), valid for any tilt."""
    R = quat_to_mat(ring_pose[:, :4])
    axis = R[:, :, 2] * torch.sign(R[:, 2:3, 2] + 1e-9)
    c = ring_pose[:, 4:]
    rod = torch.tensor([*g.rod_xy], device=c.device)
    bottom = c - axis * g.ring_height / 2
    depth = g.rod_tip_z - bottom[:, 2]
    off_b = (bottom[:, :2] - rod).norm(dim=-1)
    in_hole = (depth > 0) & (off_b < g.ring_bore_r)
    pts = rod_points(g, c.device)  # (K, 3)
    loc = torch.einsum("nji,nkj->nki", R, pts[None] - c[:, None])  # R^T (p - c)
    rad = loc[..., :2].norm(dim=-1)
    zz = loc[..., 2].abs()
    inside = (rad > g.ring_bore_r) & (rad < g.ring_outer_r) & (zz < g.ring_height / 2)
    d_in = torch.minimum(torch.minimum(rad - g.ring_bore_r, g.ring_outer_r - rad), g.ring_height / 2 - zz)
    pen = torch.where(inside, d_in, torch.zeros_like(d_in)).amax(-1)
    tilt = torch.rad2deg(torch.arccos(axis[:, 2].clamp(-1, 1)))
    return dict(offset=off_b, depth=depth, tilt=tilt, in_hole=in_hole, penetration=pen)


def retime(tcp: np.ndarray, obj: np.ndarray, base_dt: float, scale: float, g, entry_gap: float, v_ins: float,
           target_depth: float, ctrl_dt: float):
    """Reference TCP (T, 4, 4) at the control rate: the carry at base_dt * scale, then from
    `entry_gap` above the tip a straight descent along the rod axis at v_ins to the target depth.
    Returns (refs, index of the entry sample)."""
    T = np.stack([M(x) for x in tcp])
    bottom = obj[:, 6] - g.ring_height / 2
    near = np.where((np.hypot(obj[:, 4] - g.rod_xy[0], obj[:, 5] - g.rod_xy[1]) < 5e-4) & (bottom < g.rod_tip_z + entry_gap))[0]
    k_entry = int(near[0]) if len(near) else len(T) - 1
    t = np.arange(k_entry + 1) * base_dt * scale
    sl = Slerp(t, Rotation.from_matrix(T[:k_entry + 1, :3, :3]))
    # smoothstep time-warp: start and arrive at rest (a generated carry starts mid-motion), never
    # faster than the uniform timing (peak of s'(u) / 1.5 is 1), 1.5x longer overall
    dur = 1.5 * t[-1]
    u = np.arange(0, dur, ctrl_dt) / dur
    tt = t[-1] * (3 * u ** 2 - 2 * u ** 3)
    pos = np.stack([np.interp(tt, t, T[:k_entry + 1, :3, 3][:, i]) for i in range(3)], -1)
    rot = sl(tt).as_matrix()
    refs = np.tile(np.eye(4), (len(tt), 1, 1))
    refs[:, :3, :3], refs[:, :3, 3] = rot, pos
    # straight descent from the entry pose (keep its orientation) to the target depth
    E = T[k_entry].copy()
    drop = (bottom[k_entry] - (g.rod_tip_z - target_depth))
    ramp = 0.2  # s, velocity ramp into the descent
    n_d = int((max(drop, 0) / v_ins + ramp / 2) / ctrl_dt)
    td = np.arange(n_d) * ctrl_dt
    sd = np.minimum(v_ins * np.where(td < ramp, td ** 2 / (2 * ramp), td - ramp / 2), max(drop, 0))
    desc = np.tile(E, (n_d, 1, 1))
    desc[:, 2, 3] = E[2, 3] - sd
    return np.concatenate([refs, desc]), len(refs)


# ----------------------------------------------------------------------------- fin-ray grasp
_MIMIC = {"finger_joint": 1.0, "left_inner_knuckle_joint": 1.0, "left_inner_finger_joint": -1.0,
          "right_inner_knuckle_joint": -1.0, "right_inner_finger_joint": 1.0, "right_outer_knuckle_joint": -1.0}
FULL_URDF = REPO / "data/ur5e_robotiq/ur5e_robotiq85.urdf"


def _gripper_cfg(u, theta):
    cfg = {n: 0.0 for n in u.actuated_joint_names}
    for j, m in _MIMIC.items():
        if j in cfg:
            cfg[j] = theta * m
    return cfg


def _stock_inner_face(u, theta, link="left_inner_finger"):
    """Distance of the stock inner finger's inner face from the gripper axis at a closing angle."""
    import os
    import glob
    import trimesh
    if not hasattr(_stock_inner_face, "pts"):
        out = []
        for v in u.link_map[link].visuals:
            fn = v.geometry.mesh.filename.replace("package://", "")
            cands = [os.path.join(os.path.dirname(FULL_URDF), fn), fn] + glob.glob(
                os.path.join(os.path.dirname(FULL_URDF), "**", os.path.basename(fn)), recursive=True)
            m = trimesh.load(next(c for c in cands if os.path.exists(c)), force="mesh")
            T0 = v.origin if v.origin is not None else np.eye(4)
            out.append((T0[:3, :3] @ m.vertices.T).T + T0[:3, 3])
        _stock_inner_face.pts = np.concatenate(out)
    u.update_cfg(_gripper_cfg(u, theta))
    T = u.get_transform(link, "tcp")
    q = (T[:3, :3] @ _stock_inner_face.pts.T).T + T[:3, 3]
    return float(np.abs(q[:, 0]).min())


def finray_grasp(g, ih) -> dict:
    """Closing angle at which the fingers touch the ring's outside, and the fin-ray blades as
    8-vertex tapered prisms in each inner finger's link frame (visual only): inner face flat on the
    ring's outer radius, from finray_top_above_tcp above the TCP to finray_tip_below_tcp below it."""
    u = yourdfpy.URDF.load(FULL_URDF, load_meshes=False, build_collision_scene_graph=False)
    lo, hi = 0.0, 0.725
    for _ in range(40):  # inner face distance falls as the gripper closes
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _stock_inner_face(u, mid) > g.ring_outer_r else (lo, mid)
    theta = lo
    u.update_cfg(_gripper_cfg(u, theta))
    z0, z1 = -ih.finray_top_above_tcp, ih.finray_tip_below_tcp  # TCP z points toward the fingertips
    (t0, t1), (w0, w1) = ih.finray_thickness, ih.finray_width
    a = g.ring_outer_r
    blades = {}
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        v = []
        for z, t, w in ((z0, t0, w0), (z1, t1, w1)):
            for x in (a, a + t):
                for y in (-w / 2, w / 2):
                    v.append([sgn * x, y, z])
        v = np.array(v)  # order: (top, bottom) x (inner, outer) x (-y, +y)
        T = u.get_transform(f"{side}_inner_finger", "tcp")
        v_link = (T[:3, :3].T @ (v - T[:3, 3]).T).T
        faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
        blades[side] = dict(verts=v_link, faces=faces)
    return dict(theta=theta, blades=blades)


def pivot_frame(T_link_ring: np.ndarray, g, ih) -> np.ndarray:
    """Spring frame in the ring frame: on the ring axis at the middle of the gripped band, x along
    the fingers' closing direction (the gripper link's x), z along the ring axis."""
    T = np.eye(4)
    if ih.grasp_model != "finray":
        return T
    z = np.array([0.0, 0.0, 1.0])
    x = T_link_ring[:3, :3].T @ np.array([1.0, 0.0, 0.0])
    x = x - x.dot(z) * z
    x /= np.linalg.norm(x)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    band = min(max(ih.tip_below_ring_top, 0.0), g.ring_height)
    T[2, 3] = g.ring_height / 2 - band / 2
    return T


FLANGE_TO_TCP = 0.20  # the simulation's TCP (between the pads) below tool0, from the URDF


def grasp_T_tcp_obj(T_tcp_obj: np.ndarray, g, ih) -> np.ndarray:
    """The generator's grasp, with the ring moved along the tool axis to the measured grip when
    in_hand.ring_bottom_below_flange is set (TCP z points toward the fingertips)."""
    T = np.array(T_tcp_obj, float)
    if ih.ring_bottom_below_flange is not None:
        T[2, 3] = ih.ring_bottom_below_flange - FLANGE_TO_TCP - g.ring_height / 2
    return T

