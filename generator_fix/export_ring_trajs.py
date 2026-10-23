"""Export generated nut-on-rod demonstrations for physical execution: step 1 of Version 1 in the
force-aware sim2real plan. One .npz per Isaac rollout plus a manifest.json holding the conventions,
cameras, robot frames, scene, object, grasp and provenance. A physics environment or the real
UR5e can consume them without reading the generator.

Sources are the kinematic Isaac rollouts (scripts/run_ur5e_tape.py --task nut): `replay` is the
cleaned human track replayed as-is (the nominal trajectory), `r2r2r` the generator's variants.

Per trajectory (T steps, one per rendered frame of the rollout):
  t               (T,)    s, nominal timing at dt (see manifest "timing")
  phase           (T,)    approach / close / carry / release / retreat
  q               (T, 6)  arm joint targets, rad, UR order shoulder_pan .. wrist_3; unwrapped in time
                          (the IK returns angles in [-pi, pi], so the simulator, which sets joints
                          directly, can jump 2 pi in one step; unwrapping keeps every pose and stays
                          inside the URDF limits, which is checked)
  qd              (T, 6)  finite-difference joint velocity at dt, rad/s
  finger          (T,)    Robotiq finger_joint target, rad
  gripper_closed  (T,)    bool, the rollout's binary gripper command
  held            (T,)    bool, the object follows the gripper (carry)
  T_base_tool0    (T, 7)  FK of q, UR flange frame (the RTDE TCP-zero frame)
  T_base_tcp      (T, 7)  FK of q, the planner's TCP (mid-finger, tool0 + 0.20 m along z)
  T_base_ft       (T, 7)  FK of q, the URDF's ft_frame
  T_base_obj      (T, 7)  intended object pose (kinematic attachment while held)
  radial_offset   (T,)    m, nut-bottom-face centre to the rod axis, horizontal
  bottom_z        (T,)    m, nut bottom-face centre height in the base frame
  depth_over_tip  (T,)    m, rod tip z - bottom_z (> 0: the rod is inside the nut's hole)
  tilt_deg        (T,)    nut axis from vertical
  rod_in_hole     (T,)    bool, the rod tip is inside the bore: depth_over_tip > 0 and the rod axis
                          within the bore radius (beyond the radial clearance that would be contact)
  T_tcp_obj       (7,)    grasp: object pose in the TCP frame (the carry step nearest the median)
Poses are wxyz_xyz (quaternion w first, then position), metres, robot base frame.

    .venv/bin/python scripts/ur_tape/export_ring_trajs.py --output outputs/ur_nut/physical_export
"""

import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import tyro
import yourdfpy
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "data/ur5e_robotiq/ur5e_robotiq85_ik.urdf"
ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
PHASES = dict(approach=30, close=8, release=8, retreat=14)  # tape_plan.TapePhases / ur5e_cube_pnp.PnPPhases
PHASE_NAMES = ("approach", "close", "carry", "release", "retreat")


def mat_to_pose(T: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(T[..., :3, :3]).as_quat()  # xyzw
    return np.concatenate([q[..., [3, 0, 1, 2]], T[..., :3, 3]], -1)


def pose_to_mat(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, np.float64)
    T = np.zeros(p.shape[:-1] + (4, 4))
    T[..., :3, :3] = Rotation.from_quat(p[..., [1, 2, 3, 0]]).as_matrix()
    T[..., :3, 3] = p[..., 4:]
    T[..., 3, 3] = 1
    return T


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def git_rev(path: Path) -> str | None:
    try:
        rev = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--", "scripts", "real2render2real"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("+uncommitted" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class FK:
    """Forward kinematics of the planner's IK URDF (fingers fixed) from the robot base ("ur_base")."""

    def __init__(self, urdf: Path = URDF):
        self.urdf = yourdfpy.URDF.load(urdf, load_meshes=False, build_collision_scene_graph=False)
        names = list(self.urdf.actuated_joint_names)
        assert set(names) == set(ARM_JOINTS), names
        self.idx = [ARM_JOINTS.index(n) for n in names]
        lim = {j.name: j.limit for j in self.urdf.robot.joints if j.name in ARM_JOINTS}
        self.lower = np.array([lim[j].lower for j in ARM_JOINTS])
        self.upper = np.array([lim[j].upper for j in ARM_JOINTS])
        self.velocity = np.array([lim[j].velocity for j in ARM_JOINTS])  # rated, 180 deg/s on the UR5e

    def __call__(self, q: np.ndarray, frames=("tool0", "tcp", "ft_frame")) -> dict:
        out = {f: [] for f in frames}
        for qt in q:
            self.urdf.update_cfg(np.asarray(qt)[self.idx])
            for f in frames:
                out[f].append(self.urdf.get_transform(f, "ur_base"))
        return {f: np.stack(v) for f, v in out.items()}

    def fixed(self, frame: str, parent: str = "tool0") -> np.ndarray:
        return mat_to_pose(self.urdf.get_transform(frame, parent))


def plan_record(source: Path, env: int, start_xy: np.ndarray) -> dict:
    rows = [json.loads(line) for line in (source / "plans.jsonl").read_text().splitlines()]
    best = min((r for r in rows if r["env"] == env), key=lambda r: np.hypot(*(np.asarray(r["start_xy"]) - start_xy)))
    gap = float(np.hypot(*(np.asarray(best["start_xy"]) - start_xy)))
    assert gap < 2e-4, f"{source}: no plan row matches env {env} start {start_xy} (nearest {gap * 1000:.2f} mm)"
    return best


def rod_geometry(obj: np.ndarray, rod_xy, tip_z: float, w: float, bore_r: float) -> dict:
    T = pose_to_mat(obj)
    axis = T[:, :3, 2]
    axis = axis * np.sign(axis[:, 2:3] + 1e-12)  # point up
    bottom = T[:, :3, 3] - axis * w / 2
    radial = np.hypot(*(bottom[:, :2] - np.asarray(rod_xy)).T)
    depth = tip_z - bottom[:, 2]
    return dict(radial_offset=radial, bottom_z=bottom[:, 2], depth_over_tip=depth,
                tilt_deg=np.degrees(np.arccos(np.clip(axis[:, 2], -1, 1))), rod_in_hole=(depth > 0) & (radial < bore_r))


def main(sources: tuple[Path, ...] = (REPO / "output_data/nut/replay", REPO / "output_data/nut/r2r2r"),
         output: Path = REPO / "outputs/ur_nut/physical_export", dt: float = 1 / 30,
         joint_speed_limit: float | None = None,
         scene_file: Path = REPO / "outputs/ur_nut/scene.json", dims_file: Path = REPO / "outputs/ur_nut/nut_dims.json",
         traj_file: Path = REPO / "outputs/ur_nut/traj_ep4_place_clean.json",
         camera_poses: Path = REPO / "data/camera_poses/ur_realsense_front_right.h5",
         intrinsics: Path = REPO / "data/camera_poses/ur_realsense_intrinsics.json") -> None:
    """dt: seconds per step. The carry comes from the human video at 30 Hz; the other phases are
    fixed step counts at the same rate. joint_speed_limit (rad/s, default the URDF's rated speed;
    the lab's safety limits are lower) only flags steps and sets the suggested time scale."""
    scene, dims = json.loads(scene_file.read_text()), json.loads(dims_file.read_text())
    stand = scene["bolt_stand"]
    rod_xy, rod_r, tip_z = stand["xy"], stand["rod"]["radius"], stand["rod"]["z1"]
    fk = FK()
    vlim = fk.velocity if joint_speed_limit is None else np.full(6, joint_speed_limit)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for source in sources:
        kind = "nominal" if source.name == "replay" else "generated"
        for k, roll in enumerate(sorted((source / "successes").iterdir(), key=lambda p: p.name)):
            with h5py.File(roll / "robot_data/robot_data.h5", "r") as f:
                joints = f["joint_angles"][:].astype(np.float64)
                ee = f["ee_poses"][:].astype(np.float64)  # logged as xyz + wxyz
                closed = f["gripper_binary_cmd"][:].ravel().astype(bool)
            names = (roll / "robot_data/joint_names.txt").read_text().split()
            assert tuple(names[:6]) == ARM_JOINTS and names[6] == "finger_joint", names
            obj = np.load(roll / "robot_data/object_poses.npy").astype(np.float64)
            n = len(joints)
            carry = n - sum(PHASES.values())
            assert carry > 0 and len(obj) == n, (roll, n, len(obj))
            counts = [PHASES["approach"], PHASES["close"], carry, PHASES["release"], PHASES["retreat"]]
            phase = np.repeat(np.array(PHASE_NAMES), counts)
            held = phase == "carry"
            q_raw = joints[:, :6]
            q = np.unwrap(q_raw, axis=0)
            unwrapped = [ARM_JOINTS[j] for j in range(6) if np.abs(q[:, j] - q_raw[:, j]).max() > 1e-6]
            in_limits = bool(((q >= fk.lower) & (q <= fk.upper)).all())
            frames = fk(q)
            tool0, tcp, ft = (mat_to_pose(frames[f]) for f in ("tool0", "tcp", "ft_frame"))
            # Frame check: FK of the logged joints against the flange pose the simulator logged.
            logged = np.concatenate([ee[:, 3:7], ee[:, :3]], 1)
            fk_pos_err = float(np.linalg.norm(tool0[:, 4:] - logged[:, 4:], axis=1).max())
            fk_rot_err = float(np.degrees((Rotation.from_quat(tool0[:, [1, 2, 3, 0]]).inv()
                                           * Rotation.from_quat(logged[:, [1, 2, 3, 0]])).magnitude().max()))
            # Grasp: object in the TCP frame over the carry, and how constant it stays.
            rel = np.linalg.inv(frames["tcp"][held]) @ pose_to_mat(obj[held])
            grasp = mat_to_pose(rel)
            grasp_mean = grasp[np.argmin(np.linalg.norm(grasp[:, 4:] - np.median(grasp[:, 4:], 0), axis=1))]
            grasp_dev_mm = float(np.linalg.norm(grasp[:, 4:] - grasp_mean[4:], axis=1).max() * 1000)
            qd = np.gradient(q, dt, axis=0)
            ratio = np.abs(qd) / vlim
            geo = rod_geometry(obj, rod_xy, tip_z, dims["w"], dims["r_in"])
            plan = plan_record(source, int(roll.name.split("_")[1]), obj[0, 4:6])
            name = f"nut_{kind}_{k:03d}"
            over = held & geo["rod_in_hole"]
            vz = np.gradient(obj[:, 6], dt)
            np.savez_compressed(
                output / f"{name}.npz", t=np.arange(n) * dt, phase=phase, q=q, qd=qd, finger=joints[:, 6],
                gripper_closed=closed, held=held, T_base_tool0=tool0, T_base_tcp=tcp, T_base_ft=ft, T_base_obj=obj,
                T_tcp_obj=grasp_mean, **geo)
            entries.append(dict(
                name=name, kind=kind, steps=n, duration_s=round(n * dt, 3),
                phase_steps=dict(zip(PHASE_NAMES, map(int, counts))),
                source=str(roll.relative_to(REPO)),
                plan=dict(episode=plan["episode"], env=plan["env"], status=plan["status"], grasp_azimuth_deg=round(plan["azimuth_deg"], 1),
                          fixture_clearance_m=plan["tree_clearance"], table_clearance_m=plan["table_clearance"],
                          fingertip_clearance_m=plan["fingertip_clearance"]),
                start_xy=obj[0, 4:6].round(4).tolist(),
                checks=dict(fk_vs_logged_flange_mm=round(fk_pos_err * 1000, 3), fk_vs_logged_flange_deg=round(fk_rot_err, 3),
                            grasp_drift_over_carry_mm=round(grasp_dev_mm, 3),
                            unwrapped_joints=unwrapped, within_joint_limits=in_limits,
                            peak_joint_speed_rad_s=round(float(np.abs(qd).max()), 3),
                            peak_joint_speed_by_phase=({p: round(float(np.abs(qd[phase == p]).max()), 3) for p in PHASE_NAMES}),
                            steps_over_joint_speed_limit=int((ratio > 1).any(1).sum()),
                            time_scale_for_limit=round(max(1.0, float(ratio.max())), 3)),
                insertion=dict(first_step_rod_in_hole=int(np.argmax(over)) if over.any() else None,
                               steps_rod_in_hole_while_held=int(over.sum()),
                               max_descent_speed_in_hole_m_s=round(float(-vz[over].min()), 4) if over.any() else None,
                               depth_at_release_m=round(float(geo["depth_over_tip"][held][-1]), 4),
                               radial_offset_at_release_mm=round(float(geo["radial_offset"][held][-1] * 1000), 2),
                               max_radial_offset_in_hole_mm=round(float(geo["radial_offset"][over].max() * 1000), 2) if over.any() else None,
                               max_tilt_in_hole_deg=round(float(geo["tilt_deg"][over].max()), 2) if over.any() else None)))
            c = entries[-1]["checks"]
            print(f"{name}: {n} steps, FK vs log {c['fk_vs_logged_flange_mm']} mm, grasp drift {c['grasp_drift_over_carry_mm']} mm, "
                  f"peak qd {c['peak_joint_speed_rad_s']} rad/s (x{c['time_scale_for_limit']} to the limit), unwrapped {unwrapped or '-'}, "
                  f"depth at release {entries[-1]['insertion']['depth_at_release_m'] * 1000:.1f} mm")

    with h5py.File(camera_poses, "r") as f:
        cam_poses = f["poses"][:]
    intr = json.loads(intrinsics.read_text())
    img = next((sources[0] / "successes").iterdir()) / "camera_0/rgb/0000.jpg"
    from PIL import Image
    scale = Image.open(img).width / intr["width"]
    Kr = np.asarray(intr["shared_K"], np.float64).reshape(3, 3) * np.array([[scale], [scale], [1]])
    cameras = {}
    for i, cam in enumerate(intr["camera_order"]):
        p = cam_poses[i]
        c = intr[cam]
        cameras[cam] = dict(
            T_base_cam=[*p[3:].tolist(), *p[:3].tolist()],
            real=dict(K=[[c["fx"], 0, c["ppx"]], [0, c["fy"], c["ppy"]], [0, 0, 1]], width=c["width"], height=c["height"],
                      distortion=c["coeffs"]),
            rendered=dict(K=Kr.round(4).tolist(), width=round(intr["width"] * scale), height=round(intr["height"] * scale),
                          distortion=None))
    clearance = dims["r_in"] - rod_r
    manifest = dict(
        task="nut onto the bolt-stand rod (insertion over a rod, not threaded fastening)",
        status="generated kinematic demonstrations; not yet executed with physics or on hardware",
        conventions=dict(pose="wxyz_xyz: unit quaternion (w, x, y, z) then position (x, y, z)", units="m, rad, s",
                         frame="robot base (URDF 'ur_base', the frame RTDE reports poses in and T_cam2base uses)",
                         camera_frame="OpenCV/ROS optical: z forward, x right, y down; T_base_cam is camera-to-base"),
        timing=dict(dt_s=dt, rate_hz=round(1 / dt, 3),
                    note="kinematic replay has no time; carry steps are consecutive human-video frames at 30 Hz, the other "
                         "phases are the generator's fixed step counts played at the same rate. Scale time before execution "
                         "if checks.time_scale_for_limit > 1 or the lab's limits are lower.",
                    joint_speed_limit_rad_s=vlim.round(4).tolist()),
        robot=dict(arm="UR5e", gripper="Robotiq 2F-85", urdf=str(URDF.relative_to(REPO)), arm_joints=list(ARM_JOINTS),
                   gripper_joint="finger_joint (rad, 0 = open)",
                   joint_limits_rad=dict(lower=fk.lower.round(4).tolist(), upper=fk.upper.round(4).tolist()),
                   rated_joint_speed_rad_s=fk.velocity.round(4).tolist(),
                   T_tool0_tcp=fk.fixed("tcp").round(6).tolist(), T_tool0_ft=fk.fixed("ft_frame").round(6).tolist(),
                   tcp_note="planner TCP between the finger pads; the real robot's TCP setting must match it, or convert",
                   ft_note="URDF ft_frame (from ur_description); confirm it matches the RTDE wrench frame and reference "
                           "point on the installed software before use"),
        object=dict(name="nut", shape="ring", outer_radius_m=dims["R"], bore_radius_m=dims["r_in"], height_m=dims["w"],
                    source="two-view RGB silhouette fit (fit_ring_dims.py)", caliper_measured=None),
        fixture=dict(rod_axis_xy=rod_xy, rod_radius_m=rod_r, rod_tip_z_m=tip_z, rod_base_z_m=stand["rod"]["z0"],
                     spacers=stand["spacers"], block=stand.get("block"), base=stand.get("base"), table_z_m=scene["table_z"],
                     source="measure_bolt_stand.py (RGB)", caliper_measured=None, threaded=None, chamfer=None),
        clearance=dict(radial_m=round(clearance, 5), diametral_m=round(2 * clearance, 5),
                       note="from the RGB fits; replace with caliper measurements before any contact experiment"),
        cameras=cameras,
        sources={str(p.relative_to(REPO)): sha256(p) for p in (scene_file, dims_file, traj_file, camera_poses, intrinsics, URDF)},
        code_revision=git_rev(REPO),
        trajectories=entries)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1))
    ok = [e for e in entries if e["checks"]["fk_vs_logged_flange_mm"] < 1.0 and e["checks"]["within_joint_limits"]]
    scales = [e["checks"]["time_scale_for_limit"] for e in entries]
    print(f"{len(entries)} trajectories -> {output} ({len(ok)} pass the frame and joint-limit checks; "
          f"{sum(bool(e['checks']['unwrapped_joints']) for e in entries)} unwrapped); time scale to the speed limit "
          f"median x{np.median(scales):.2f}, max x{max(scales):.2f}; radial clearance {clearance * 1000:.1f} mm (RGB fit)")


if __name__ == "__main__":
    tyro.cli(main)
