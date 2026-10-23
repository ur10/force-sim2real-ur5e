"""Physics scene for the nut-on-rod task in Isaac Lab 2.0.2 (import after the app is launched).

  robot   UR5e + 2F-85 articulation (our URDF-converted USD), stiff joint position control, gravity
          off on the robot links (the wrist sensor model adds the gripper's weight analytically)
  rod     static: spacer + rod as convex prisms at the measured axis, optional tip chamfer
  ring    dynamic rigid body made of convex annular sectors (the bore stays open), gravity on
  hand    a 6-DoF spring-damper joint from the gripper base link to the ring (in-hand compliance);
          ring-gripper collisions are filtered so only the spring carries the ring
  wrench  PhysX incoming-joint force of the gripper base link: everything beyond the flange

Polygon sizing keeps the clearance conservative: the bore's flat faces sit at the bore radius and
the rod's vertices at rod_r / cos(pi / sides) >= rod_r, so the true clearance is never larger
than the nominal one (smaller by at most rod_r * (1 / cos(pi / sides) - 1), 0.009 mm at 64 sides).
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg

from params import Params

ENV_FMT = "/World/envs/env_{}"
ROBOT, RING, ROD, JOINT = "Robot", "Ring", "Rod", "RingJoint"


def physx_cfg(p: Params) -> sim_utils.SimulationCfg:
    s = p.sim
    return sim_utils.SimulationCfg(dt=s.dt, gravity=(0.0, 0.0, -9.81), physx=sim_utils.PhysxCfg(
        solver_type=1, max_position_iteration_count=s.position_iterations, max_velocity_iteration_count=s.velocity_iterations,
        bounce_threshold_velocity=s.bounce_threshold_velocity, friction_offset_threshold=s.friction_offset_threshold,
        friction_correlation_distance=s.friction_correlation_distance, gpu_max_num_partitions=s.gpu_max_num_partitions))


def _prism(stage, path: str, ring_xy: np.ndarray, z0: float, z1: float, p: Params, approx="convexHull", color=(0.25, 0.68, 0.35)):
    """Convex prism mesh over a closed 2D polygon (counter-clockwise), z0..z1, with a collider."""
    n = len(ring_xy)
    pts = [Gf.Vec3f(float(x), float(y), z0) for x, y in ring_xy] + [Gf.Vec3f(float(x), float(y), z1) for x, y in ring_xy]
    faces = [list(range(n))[::-1], list(range(n, 2 * n))]
    faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approx)
    c = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    c.CreateContactOffsetAttr(p.contact.contact_offset)
    c.CreateRestOffsetAttr(p.contact.rest_offset)
    return prim


def _frustum(stage, path, r0, r1, z0, z1, sides, p, color=(0.62, 0.64, 0.67)):
    """Convex frustum (radius r0 at z0 to r1 at z1), vertices at r / cos(pi / sides)."""
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    k = 1 / math.cos(math.pi / sides)
    pts = [Gf.Vec3f(float(r0 * k * math.cos(t)), float(r0 * k * math.sin(t)), z0) for t in a]
    pts += [Gf.Vec3f(float(r1 * k * math.cos(t)), float(r1 * k * math.sin(t)), z1) for t in a]
    n = sides
    faces = [list(range(n))[::-1], list(range(n, 2 * n))] + [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    c = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    c.CreateContactOffsetAttr(p.contact.contact_offset)
    c.CreateRestOffsetAttr(p.contact.rest_offset)
    return prim


def _material(path: str, p: Params) -> str:
    cfg = sim_utils.RigidBodyMaterialCfg(static_friction=p.contact.static_friction, dynamic_friction=p.contact.dynamic_friction,
                                         restitution=p.contact.restitution)
    cfg.func(path, cfg)
    return path


def build_rod(stage, p: Params, root: str, mat: str):
    g = p.geometry
    path = f"{root}/{ROD}"
    UsdGeom.Xform.Define(stage, path).AddTranslateOp().Set(Gf.Vec3d(g.rod_xy[0], g.rod_xy[1], 0.0))
    _frustum(stage, f"{path}/spacer", g.spacer_r, g.spacer_r, g.spacer_z0, g.rod_z0, 64, p)
    c = min(g.rod_chamfer, g.rod_r * 0.9)
    _frustum(stage, f"{path}/rod", g.rod_r, g.rod_r, g.rod_z0, g.rod_tip_z - c, g.rod_sides, p)
    if c > 0:
        _frustum(stage, f"{path}/tip", g.rod_r, g.rod_r - c, g.rod_tip_z - c, g.rod_tip_z, g.rod_sides, p)
    sim_utils.bind_physics_material(path, mat)


def build_ring(stage, p: Params, root: str, pose_wxyz_xyz: np.ndarray, mat: str):
    """Ring body at the given pose in the env frame; local frame: z = axis, origin at the centre."""
    g, h = p.geometry, p.in_hand
    path = f"{root}/{RING}"
    x = UsdGeom.Xform.Define(stage, path)
    q = pose_wxyz_xyz
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, q[4:])))
    x.AddOrientOp().Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    prim = x.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(h.mass)
    R, r, w = g.ring_outer_r, g.ring_bore_r, g.ring_height
    ixx = h.mass * (3 * (R * R + r * r) + w * w) / 12
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(ixx, ixx, h.mass * (R * R + r * r) / 2))
    m.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateSolverPositionIterationCountAttr(p.sim.position_iterations)
    rb.CreateSolverVelocityIterationCountAttr(p.sim.velocity_iterations)
    rb.CreateMaxDepenetrationVelocityAttr(0.2)  # choice: a solver correction must not launch the ring
    rb.CreateLinearDampingAttr(0.0)
    rb.CreateAngularDampingAttr(0.0)
    n = g.ring_segments
    k = 1 / math.cos(math.pi / n)  # bore faces at r, outer faces at R
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        poly = np.array([[r * k * math.cos(a0), r * k * math.sin(a0)], [R * k * math.cos(a0), R * k * math.sin(a0)],
                         [R * k * math.cos(a1), R * k * math.sin(a1)], [r * k * math.cos(a1), r * k * math.sin(a1)]])
        _prism(stage, f"{path}/seg_{i:02d}", poly, -w / 2, w / 2, p)
    sim_utils.bind_physics_material(path, mat)
    return prim


def build_hand_joint(stage, p: Params, root: str, gripper_link: str, T_link_ring: np.ndarray):
    """6-DoF spring-damper from the gripper link to the ring at rest pose T_link_ring (wxyz_xyz).
    finray: the joint frame sits at the middle of the gripped band (contact_utils.pivot_frame),
    axes along the closing direction, across the fingers and the ring axis, each with its own
    gains; center: isotropic about the ring centre. USD angular drive gains are per degree."""
    from contact_utils import M, P, pivot_frame
    h = p.in_hand
    T_lr = M(np.asarray(T_link_ring, float))
    T_rp = pivot_frame(T_lr, p.geometry, h)
    j = UsdPhysics.Joint.Define(stage, f"{root}/{JOINT}")
    j.CreateBody0Rel().SetTargets([Sdf.Path(gripper_link)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(f"{root}/{RING}")])
    for (pos_attr, rot_attr), T in (((j.CreateLocalPos0Attr, j.CreateLocalRot0Attr), T_lr @ T_rp),
                                    ((j.CreateLocalPos1Attr, j.CreateLocalRot1Attr), T_rp)):
        q = P(T)
        pos_attr(Gf.Vec3f(*map(float, q[4:])))
        rot_attr(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    j.CreateExcludeFromArticulationAttr(True)
    j.CreateCollisionEnabledAttr(False)
    deg = math.pi / 180
    if h.grasp_model == "finray":
        kl, ka, dl, da = h.k_lin_xyz, h.k_ang_xyz, h.d_lin_xyz, h.d_ang_xyz
    else:
        kl, ka, dl, da = (h.k_lin,) * 3, (h.k_ang,) * 3, (h.d_lin,) * 3, (h.d_ang,) * 3
    gains = [(f"trans{ax}", kl[i], dl[i]) for i, ax in enumerate("XYZ")]
    gains += [(f"rot{ax}", ka[i] * deg, da[i] * deg) for i, ax in enumerate("XYZ")]
    for axis, k, d in gains:
        drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), axis)
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(float(k))
        drv.CreateDampingAttr(float(d))
        drv.CreateTargetPositionAttr(0.0)
        drv.CreateTargetVelocityAttr(0.0)
    return j


def add_finray_blades(stage, robot_root: str, blades: dict):
    """Visual-only fin-ray blades on the inner fingers (no collider, no mass)."""
    for side, b in blades.items():
        link = find_prim(stage, robot_root, f"{side}_inner_finger")
        mesh = UsdGeom.Mesh.Define(stage, f"{link}/finray_blade")
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in b["verts"]])
        mesh.CreateFaceVertexCountsAttr([len(f) for f in b["faces"]])
        mesh.CreateFaceVertexIndicesAttr([i for f in b["faces"] for i in f])
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.55, 0.85)])
        mesh.CreateDoubleSidedAttr(True)


def filter_ring_vs(stage, root: str, prims: list[str]):
    f = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f"{root}/{RING}"))
    f.CreateFilteredPairsRel().SetTargets([Sdf.Path(x) for x in prims])


def robot_cfg(q0: list[float], gripper: float, p: Params):
    from real2render2real.isaaclab_viser.configs.articulation_configs.ur5e_robotiq_cfg import (
        ROBOTIQ_MIMIC, UR5E_ARM_JOINTS, UR5E_ROBOTIQ_CFG,
    )
    cfg = copy.deepcopy(UR5E_ROBOTIQ_CFG).replace(prim_path=ENV_FMT.format(".*") + f"/{ROBOT}")
    cfg.spawn.articulation_props.solver_position_iteration_count = p.sim.position_iterations
    cfg.spawn.articulation_props.solver_velocity_iteration_count = p.sim.velocity_iterations
    cfg.init_state.joint_pos = {**dict(zip(UR5E_ARM_JOINTS, map(float, q0))), **{j: float(gripper * m) for j, m in ROBOTIQ_MIMIC.items()}}
    return cfg


def find_prim(stage, root: str, name: str) -> str:
    for prim in stage.Traverse():
        s = str(prim.GetPath())
        if s.startswith(root) and prim.GetName() == name:
            return s
    raise KeyError(f"{name} under {root}")


def gripper_prims(stage, robot_root: str) -> list[str]:
    names = ("robotiq_85_base_link", "left_outer_knuckle", "left_outer_finger", "left_inner_knuckle", "left_inner_finger",
             "right_outer_knuckle", "right_outer_finger", "right_inner_knuckle", "right_inner_finger",
             "left_inner_finger_pad", "right_inner_finger_pad")
    out = []
    for prim in stage.Traverse():
        s = str(prim.GetPath())
        if s.startswith(robot_root + "/") and prim.GetName() in names and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            out.append(s)
    return out


class RingRodScene:
    """N envs, each: robot + rod + ring + spring joint (spawned before sim.reset()). Env i sits at
    x = i * spacing; every pose this class returns is in its env's frame (= that robot's base).
    T_link_ring / ring_pose0: (N, 7) per env, so each env can carry its own in-hand error."""

    def __init__(self, stage, p: Params, n: int, q0, gripper: float, T_link_ring: np.ndarray, ring_pose0: np.ndarray,
                 spacing: float = 2.5, device="cuda:0"):
        self.p, self.n = p, n
        self.finray = None
        if p.in_hand.grasp_model == "finray":
            from contact_utils import finray_grasp
            self.finray = finray_grasp(p.geometry, p.in_hand)
            gripper = self.finray["theta"]  # the fingers just touch the ring's outside
        self.gripper = gripper
        self.origins = torch.zeros(n, 3, device=device)
        self.origins[:, 0] = torch.arange(n, device=device) * spacing
        for i in range(n):
            UsdGeom.Xform.Define(stage, ENV_FMT.format(i)).AddTranslateOp().Set(Gf.Vec3d(i * spacing, 0.0, 0.0))
        self.robot = Articulation(robot_cfg([float(x) for x in np.asarray(q0)[0]], gripper, p))
        rod_mat, ring_mat = _material("/World/Materials/rod", p), _material("/World/Materials/ring", p)
        T_link_ring, ring_pose0 = np.atleast_2d(T_link_ring), np.atleast_2d(ring_pose0)
        for i in range(n):
            root = ENV_FMT.format(i)
            build_rod(stage, p, root, rod_mat)
            build_ring(stage, p, root, ring_pose0[i % len(ring_pose0)], ring_mat)
            link = find_prim(stage, f"{root}/{ROBOT}", "robotiq_85_base_link")
            build_hand_joint(stage, p, root, link, T_link_ring[i % len(T_link_ring)])
            filter_ring_vs(stage, root, gripper_prims(stage, f"{root}/{ROBOT}"))
            if self.finray is not None:
                add_finray_blades(stage, f"{root}/{ROBOT}", self.finray["blades"])
        self.ring = RigidObject(RigidObjectCfg(prim_path=ENV_FMT.format(".*") + f"/{RING}", spawn=None))

    def post_reset(self):
        r = self.robot
        self.arm_ids = r.find_joints(["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                                      "wrist_2_joint", "wrist_3_joint"], preserve_order=True)[0]
        self.link_ids = {n: r.find_bodies(n)[0][0] for n in ("tool0", "tcp", "robotiq_85_base_link")}
        # massless URDF frames come out of the converter as 1 kg bodies; "tcp" sits beyond the
        # wrist sensor 0.16 m below the gripper base, so it would add a phantom 1 kg at the fingertips
        view = r.root_physx_view
        masses, inertias = view.get_masses(), view.get_inertias()
        self.fixed_frames = {}
        for name in ("tcp", "flange", "tool0", "ft_frame"):
            if name in r.body_names:
                j = r.body_names.index(name)
                self.fixed_frames[name] = float(masses[0, j])
                inertias[:, j] *= (1e-3 / masses[:, j])[:, None]
                masses[:, j] = 1e-3
        ids = torch.arange(r.num_instances, dtype=torch.int32)
        view.set_masses(masses, ids)
        view.set_inertias(inertias, ids)

    def wrench_incoming(self) -> torch.Tensor:
        """(N, 6) PhysX incoming-joint spatial force of the gripper base link (raw convention)."""
        return self.robot.root_physx_view.get_link_incoming_joint_force()[:, self.link_ids["robotiq_85_base_link"]]

    def link_pose(self, name: str) -> torch.Tensor:
        """(N, 7) wxyz_xyz in the env frame."""
        s = self.robot.data.body_state_w[:, self.link_ids[name], :7]
        return torch.cat([s[:, 3:7], s[:, :3] - self.origins], -1)

    def ring_pose(self) -> torch.Tensor:
        s = self.ring.data.root_state_w[:, :7]
        return torch.cat([s[:, 3:7], s[:, :3] - self.origins], -1)

    def set_ring_pose(self, pose: torch.Tensor, env_ids=None):
        """pose (N, 7) wxyz_xyz env frame; zero velocity."""
        st = torch.zeros(pose.shape[0], 13, device=pose.device)
        ids = torch.arange(self.n, device=pose.device) if env_ids is None else env_ids
        st[:, :3] = pose[:, 4:] + self.origins[ids]
        st[:, 3:7] = pose[:, :4]
        self.ring.write_root_state_to_sim(st, env_ids=ids)
