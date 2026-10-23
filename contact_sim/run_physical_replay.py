"""Version 1, physical replay: execute exported nut trajectories (outputs/ur_nut/physical_export)
with contact physics, a compliant controller and the simulated wrist sensor, under sampled errors.

Each env takes one exported trajectory and an in-hand error (the ring sits offset / tilted in the
gripper; the controller assumes the nominal grasp). The carry is time-scaled to the joint-speed
limit, and the part near the rod is re-timed to a slow insertion speed. Then:
  --mode replay   follow the reference; compliance only along the rod (damping control)
  --mode expert   at the entry point hand over to the reactive expert (expert.py), which sees only
                  the sensed wrench and the measured TCP
Outcomes from geometry (not from the wrench): seated (rod in the bore at the target depth), jammed,
aborted (force above the limit); plus peak loads, penetration and in-hand deflection.

    scripts/ur_cube/isaac_run.sh outputs/ur_nut/contact_sim/replay.log "REPLAY DONE" \
        scripts/ur_contact/run_physical_replay.py --mode replay --max-offset-mm 4 --output outputs/ur_nut/contact_sim/replay
"""

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("outputs/ur_nut/contact_sim/replay"))
parser.add_argument("--mode", choices=["replay", "expert"], default="replay")
parser.add_argument("--num-envs", type=int, default=48)
parser.add_argument("--max-offset-mm", type=float, default=4.0, help="in-hand lateral error, uniform on a disc")
parser.add_argument("--max-tilt-deg", type=float, default=3.0, help="in-hand tilt error, uniform, random axis")
parser.add_argument("--insert-speed", type=float, default=5.0, help="mm/s near the rod")
parser.add_argument("--entry-gap-mm", type=float, default=6.0, help="re-time from this height above the tip")
parser.add_argument("--target-depth-mm", type=float, default=15.0)
parser.add_argument("--extra-time-scale", type=float, default=1.2, help="on top of the joint-speed scale")
parser.add_argument("--max-tcp-speed", type=float, default=0.25, help="m/s cap on the carry (choice: set from lab limits)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--params", type=Path, default=None)
parser.add_argument("--set", type=str, default=None)
parser.add_argument("--export-dir", type=Path, default=Path("outputs/ur_nut/physical_export"))
parser.add_argument("--video", type=int, default=0, help="film the first N envs (close-up each + a wide shot of env 0)")
parser.add_argument("--video-out", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from admittance import Admittance, dls_ik_step, exp_so3, skew_log  # noqa: E402
from expert import DONE, ExpertCfg, InsertionExpert  # noqa: E402
from params import Controller, load, to_dict, unmeasured  # noqa: E402
from ring_rod_sim import RingRodScene, physx_cfg  # noqa: E402
from contact_utils import M, P, NumpyIK, grasp_T_tcp_obj, retime, ring_metrics  # noqa: E402
from torch_fk import TorchFK  # noqa: E402
from wrench_sensor import WrenchSensor, quat_to_mat, shift_wrench  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "data/ur5e_robotiq/ur5e_robotiq85_ik.urdf"


def write_video(frames, rows, args, out):
    """Wide shot of env 0 next to a grid of close-ups, labelled; H.264 via ffmpeg."""
    import subprocess
    import tempfile

    import cv2
    names = {"replay": {0: "carry", 1: "insert"}, "expert": {0: "descend", 1: "blocked", 2: "back off", 3: "done", 4: "move sideways"}}[args.mode]
    v = len(rows)
    cols = 2 if v > 1 else 1
    gr = (v + cols - 1) // cols
    tmp = Path(tempfile.mkdtemp(dir=out))
    for k, f in enumerate(frames):
        tiles = []
        for i in range(v):
            im = cv2.cvtColor(np.ascontiguousarray(f["close"][i][..., :3]), cv2.COLOR_RGB2BGR)
            r = rows[i]
            state = "ABORT (force limit)" if f["aborted"][i] else names.get(int(f["mode"][i]), "?")
            band = im.copy()
            cv2.rectangle(band, (0, 0), (im.shape[1], 50), (25, 25, 25), -1)
            im = cv2.addWeighted(band, 0.6, im, 0.4, 0)
            for j, line in enumerate((f"env {i}: in-hand error {r['err_offset_mm']:.1f} mm / {r['err_tilt_deg']:.1f} deg",
                                      f"{state:20s} Fz {f['fz'][i]:6.2f} N  depth {f['depth'][i] * 1000:6.1f} mm")):
                cv2.putText(im, line, (8, 20 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3, cv2.LINE_AA)
                cv2.putText(im, line, (8, 20 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 250), 1, cv2.LINE_AA)
            if k == len(frames) - 1 or k > len(frames) - 30:
                col = (60, 200, 60) if r["outcome"] == "seated" else (40, 40, 230)
                cv2.putText(im, r["outcome"].upper(), (8, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
            tiles.append(im)
        while len(tiles) < gr * cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[r0 * cols:(r0 + 1) * cols]) for r0 in range(gr)])
        wide = cv2.cvtColor(np.ascontiguousarray(f["wide"][..., :3]), cv2.COLOR_RGB2BGR)
        wide = cv2.resize(wide, (int(wide.shape[1] * grid.shape[0] / wide.shape[0]), grid.shape[0]))
        band = wide.copy()
        cv2.rectangle(band, (0, 0), (wide.shape[1], 36), (25, 25, 25), -1)
        wide = cv2.addWeighted(band, 0.6, wide, 0.4, 0)
        title = {"replay": "blind replay (compliant along the rod only)", "expert": "reactive expert (sensed wrench only)"}[args.mode]
        cv2.putText(wide, f"{title}   t = {f['t']:5.2f} s   simulation, uncalibrated", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (250, 250, 250), 1, cv2.LINE_AA)
        img = np.hstack([wide, grid])
        img = img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2]
        cv2.imwrite(str(tmp / f"f{k:05d}.png"), img)
    dst = args.video_out or out / f"{args.mode}.mp4"
    subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", str(tmp / "f%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "22", str(dst)], check=True)
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()
    print("[VIDEO]", dst, len(frames), "frames", flush=True)


def main():
    p = load(args.params, json.loads(args.set) if args.set else None)
    g = p.geometry
    rng = np.random.default_rng(args.seed)
    man = json.loads((REPO / args.export_dir / "manifest.json").read_text())
    names = [e["name"] for e in man["trajectories"]]
    n = args.num_envs
    ik = NumpyIK()
    T_link_tcp = ik.u.get_transform("tcp", "robotiq_85_base_link")
    ctrl_dt = 1 / p.controller.rate_hz
    envs = []
    for i in range(n):
        name = names[(i + 1) % len(names)] if n < len(names) else names[i % len(names)]
        d = np.load(REPO / args.export_dir / f"{name}.npz")
        e = next(x for x in man["trajectories"] if x["name"] == name)
        carry = np.where(d["held"])[0]
        lift = carry[np.argmax(d["T_base_obj"][carry, 6] > g.table_z + g.ring_height / 2 + 0.03)]  # start once lifted
        sl = slice(lift, carry[-1] + 1)
        v_peak = float(np.linalg.norm(np.diff(d["T_base_tcp"][sl, 4:], axis=0), axis=1).max() * 30)
        scale = max(max(1.0, e["checks"]["time_scale_for_limit"]) * args.extra_time_scale, v_peak / args.max_tcp_speed)
        refs, k_entry = retime(d["T_base_tcp"][sl], d["T_base_obj"][sl], 1 / 30, scale, g, args.entry_gap_mm / 1000,
                               args.insert_speed / 1000, args.target_depth_mm / 1000, ctrl_dt)
        # in-hand error: offset on a disc in the ring's plane and a tilt about a random in-plane axis
        r, a = args.max_offset_mm / 1000 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        tilt, b = np.radians(args.max_tilt_deg) * rng.uniform(), rng.uniform(0, 2 * np.pi)
        E = np.eye(4)
        E[:3, :3] = Rotation.from_rotvec(tilt * np.array([np.cos(b), np.sin(b), 0])).as_matrix()
        E[:3, 3] = [r * np.cos(a), r * np.sin(a), 0]
        T_tcp_obj = grasp_T_tcp_obj(M(d["T_tcp_obj"]), g, p.in_hand)
        envs.append(dict(name=name, q0=d["q"][lift], refs=refs, k_entry=k_entry, scale=scale, T_tcp_obj=T_tcp_obj,
                         T_link_ring=T_link_tcp @ T_tcp_obj @ E, err_offset_mm=r * 1000, err_tilt_deg=float(np.degrees(tilt))))
    q0 = np.stack([e["q0"] for e in envs])
    T_lr = np.stack([P(e["T_link_ring"]) for e in envs])
    ring0 = np.stack([P(ik.fk(e["q0"], "robotiq_85_base_link") @ e["T_link_ring"]) for e in envs])
    sim = sim_utils.SimulationContext(physx_cfg(p))
    scene = RingRodScene(sim.stage, p, n, q0, 0.55, T_lr, ring0, device=args.device)
    cams = None
    if args.video:
        from isaaclab.sensors import Camera, CameraCfg
        sim_utils.DomeLightCfg(intensity=450.0).func("/World/DomeLight", sim_utils.DomeLightCfg(intensity=450.0, color=(0.95, 0.95, 1.0)))
        sim_utils.DistantLightCfg(intensity=900.0).func("/World/Sun", sim_utils.DistantLightCfg(intensity=900.0, angle=1.0), orientation=(0.85, 0.3, 0.3, 0.3))
        table = sim_utils.CuboidCfg(size=(1.0, 1.4, 0.04), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.66, 0.5)))
        for i in range(n):
            table.func(f"/World/envs/env_{i}/Table", table, translation=(0.05, -0.55, g.table_z - 0.02))
        cam_spawn = sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0))
        cams = dict(close=Camera(CameraCfg(prim_path="/World/envs/env_.*/CloseCam", height=360, width=480, data_types=["rgb"], spawn=cam_spawn)),
                    wide=Camera(CameraCfg(prim_path="/World/envs/env_.*/WideCam", height=360, width=480, data_types=["rgb"], spawn=cam_spawn)))
    sim.reset()
    scene.post_reset()
    dev, dt = sim.device, p.sim.dt
    if cams is not None:
        tip = torch.tensor([g.rod_xy[0], g.rod_xy[1], g.rod_tip_z], device=dev)
        cams["close"].set_world_poses_from_view(scene.origins + tip + torch.tensor([0.11, -0.10, 0.05], device=dev),
                                                scene.origins + tip + torch.tensor([0.0, 0.0, 0.004], device=dev))
        cams["wide"].set_world_poses_from_view(scene.origins + torch.tensor([0.95, -1.45, 0.62], device=dev),
                                               scene.origins + torch.tensor([0.02, -0.5, 0.18], device=dev))
        m = scene.robot.root_physx_view.get_masses()[0]
        print("[MASS] gripper subtree:", {nm: round(float(m[j]), 4) for j, nm in enumerate(scene.robot.body_names)
                                           if any(k in nm for k in ("robotiq", "finger", "knuckle", "tcp", "tool0", "flange", "ft_frame", "wrist_3"))}, flush=True)
    frames = []
    robot, ring = scene.robot, scene.ring
    tfk = TorchFK(URDF, device=dev)
    q_cmd = torch.tensor(q0, dtype=torch.float32, device=dev)
    T_lr_t = torch.tensor(np.stack([e["T_link_ring"] for e in envs]), dtype=torch.float32, device=dev)

    def physics_step():
        robot.write_data_to_sim()
        ring.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        ring.update(dt)

    def mat(pose):
        T = torch.eye(4, device=dev).repeat(pose.shape[0], 1, 1)
        T[:, :3, :3] = quat_to_mat(pose[:, :4])
        T[:, :3, 3] = pose[:, 4:]
        return T

    qt = robot.data.default_joint_pos.clone()
    qt[:, scene.arm_ids] = q_cmd
    robot.write_joint_state_to_sim(qt, torch.zeros_like(qt))
    robot.set_joint_position_target(qt)
    physics_step()
    physics_step()
    Tr = mat(scene.link_pose("robotiq_85_base_link")) @ T_lr_t
    scene.set_ring_pose(torch.cat([quat_from_matrix(Tr[:, :3, :3]), Tr[:, :3, 3]], -1))
    for _ in range(int(0.3 / dt)):
        physics_step()

    rot = lambda R, w: torch.cat([(R @ w[:, :3, None])[..., 0], (R @ w[:, 3:, None])[..., 0]], -1)  # noqa: E731

    def tool_load(ref_point: str = "tool0") -> torch.Tensor:
        raw = scene.wrench_incoming()  # child-link frame, force of the arm on the gripper (checked in the probe)
        lk = scene.link_pose("robotiq_85_base_link")
        return shift_wrench(-rot(quat_to_mat(lk[:, :4]), raw), lk[:, 4:], scene.link_pose(ref_point)[:, 4:])

    sensor = WrenchSensor(p.sensor, n, dt, dev, seed=args.seed)
    T_tool0_ring = ik.u.get_transform("tcp", "tool0") @ envs[0]["T_tcp_obj"]  # configured payload: the nominal grasp
    sensor.held_mass = p.in_hand.mass
    sensor.held_cog_tool0 = torch.tensor(T_tool0_ring[:3, 3], dtype=torch.float32, device=dev)
    for _ in range(int(0.2 / dt)):
        sensor.step(tool_load(), scene.link_pose("tool0"), scene.link_pose("tcp")[:, 4:])
        physics_step()
    sensor.tare()

    insert = Controller(**{**to_dict(p)["controller"], "max_offset": (0.0, 0.0, 0.03, 0.0, 0.0, 0.0),
                           "stiffness": (0.0,) * 6, "mass": (1.0,) * 6})
    adm = Admittance(insert, n, dev)
    expert = InsertionExpert(ExpertCfg(v_descend=args.insert_speed / 1000), n, ctrl_dt, dev)
    L = max(len(e["refs"]) for e in envs)
    refs = torch.tensor(np.stack([np.concatenate([e["refs"], np.repeat(e["refs"][-1:], L - len(e["refs"]), 0)]) for e in envs]),
                        dtype=torch.float32, device=dev)  # (N, L, 4, 4), padded with the last pose
    k_entry = torch.tensor([e["k_entry"] for e in envs], device=dev)
    entry_T = refs[torch.arange(n, device=dev), k_entry]
    hold = int(1.0 / ctrl_dt)
    every = max(1, round(ctrl_dt / dt))
    keys = ("w_true", "w_filtered", "depth", "offset", "tilt", "penetration", "mode", "ins", "ring", "tcp", "q")
    log = {k: [] for k in keys}
    aborted = torch.zeros(n, dtype=torch.bool, device=dev)
    peak_f = torch.zeros(n, device=dev)
    abort_force = 40.0
    abort_pos = torch.zeros(n, 3, device=dev)
    t0 = time.time()
    tcp_entry_z = None
    for c in range(L + hold + (int(12.0 / ctrl_dt) if args.mode == "expert" else 0)):
        idx = torch.clamp(torch.full((n,), c, device=dev), max=L - 1)
        ref = refs[torch.arange(n, device=dev), idx]
        in_insert = idx >= k_entry
        for _ in range(every):
            w = tool_load()
            tcp = scene.link_pose("tcp")
            ch = sensor.step(w, scene.link_pose("tool0"), tcp[:, 4:])
            physics_step()
        w_tcp = shift_wrench(ch["filtered"], scene.link_pose("tool0")[:, 4:], tcp[:, 4:]) if p.sensor.reference == "tool0" else ch["filtered"]
        if args.mode == "replay":
            e = adm.step(torch.where(in_insert[:, None], ch["filtered"], torch.zeros_like(ch["filtered"])), torch.zeros(n, 6, device=dev), ctrl_dt)
            R_cmd, p_cmd = Admittance.compose(ref[:, :3, :3], ref[:, :3, 3], torch.eye(3, device=dev).expand(n, 3, 3), e)
        else:
            if tcp_entry_z is None:
                tcp_entry_z = torch.full((n,), float("nan"), device=dev)
            at_entry = in_insert & torch.isnan(tcp_entry_z)
            tcp_entry_z = torch.where(at_entry, tcp[:, 6], tcp_entry_z)
            depth_meas = torch.nan_to_num(tcp_entry_z - tcp[:, 6], nan=0.0)
            target = args.entry_gap_mm / 1000 + args.target_depth_mm / 1000
            act = expert.act(torch.where(in_insert[:, None], w_tcp, torch.zeros_like(w_tcp)), target, depth_meas)
            off = torch.stack([act["xy"][:, 0], act["xy"][:, 1], -act["z"]], -1)
            R_cmd = torch.where(in_insert[:, None, None],
                                exp_so3(torch.cat([act["rxy"], torch.zeros(n, 1, device=dev)], -1)) @ entry_T[:, :3, :3], ref[:, :3, :3])
            p_cmd = torch.where(in_insert[:, None], entry_T[:, :3, 3] + off, ref[:, :3, 3])
            expert.mode = torch.where(~in_insert, torch.zeros_like(expert.mode), expert.mode)
            expert.z = torch.where(~in_insert, torch.zeros_like(expert.z), expert.z)
        p_cmd = torch.where(aborted[:, None], abort_pos + torch.tensor([0, 0, 0.005], device=dev), p_cmd)
        for _ in range(2):
            T, J = tfk.jacobian(q_cmd)
            q_cmd += dls_ik_step(J, p_cmd - T[:, :3, 3], skew_log(R_cmd @ T[:, :3, :3].transpose(1, 2)), damping=1e-3)
        qt = robot.data.default_joint_pos.clone()
        qt[:, scene.arm_ids] = q_cmd
        robot.set_joint_position_target(qt)
        m = ring_metrics(scene.ring_pose(), g)
        f = w[:, :3].norm(dim=-1)
        peak_f = torch.maximum(peak_f, torch.where(in_insert, f, torch.zeros_like(f)))
        newly = (~aborted) & (f > abort_force)
        abort_pos = torch.where(newly[:, None], scene.link_pose("tcp")[:, 4:], abort_pos)
        aborted |= newly
        for key, val in (("w_true", w), ("w_filtered", ch["filtered"]), ("depth", m["depth"]), ("offset", m["offset"]),
                         ("tilt", m["tilt"]), ("penetration", m["penetration"]),
                         ("mode", expert.mode if args.mode == "expert" else in_insert.long()), ("ins", in_insert),
                         ("ring", scene.ring_pose()), ("tcp", scene.link_pose("tcp")), ("q", robot.data.joint_pos[:, scene.arm_ids])):
            log[key].append(val.detach().clone())
        if cams is not None and c % 4 == 0:  # 30 fps at the 120 Hz control rate
            sim.render()
            for cam in cams.values():
                cam.update(dt, force_recompute=True)
            frames.append(dict(close=cams["close"].data.output["rgb"][:args.video].cpu().numpy(),
                               wide=cams["wide"].data.output["rgb"][0].cpu().numpy(), t=c * ctrl_dt,
                               fz=w[:args.video, 2].cpu().numpy(), mode=(expert.mode if args.mode == "expert" else in_insert.long())[:args.video].cpu().numpy(),
                               depth=m["depth"][:args.video].cpu().numpy(), aborted=aborted[:args.video].cpu().numpy()))
        if args.mode == "expert" and c > L and bool(((expert.mode == DONE) | aborted).all()):
            break
    wall = time.time() - t0
    Lg = {k: torch.stack(v, 1).cpu().numpy() for k, v in log.items()}
    rows = []
    for i, e in enumerate(envs):
        depth = Lg["depth"][i]
        ins = Lg["ins"][i].astype(bool)
        seated = bool(depth[-1] > (args.target_depth_mm - 2.0) / 1000 and Lg["offset"][i][-1] < g.ring_bore_r)
        wz = Lg["w_true"][i]
        contact = (wz[:, 2] > 0.8) & ins
        # not seated without ever touching: the compliant axis yielded to sensor drift, not a jam
        outcome = "aborted" if bool(aborted[i]) else ("seated" if seated else ("jammed" if contact.any() else "short_no_contact"))
        rows.append(dict(env=i, traj=e["name"], err_offset_mm=round(e["err_offset_mm"], 2), err_tilt_deg=round(e["err_tilt_deg"], 2),
                         time_scale=round(e["scale"], 2), outcome=outcome, final_depth_mm=round(float(depth[-1] * 1000), 2),
                         peak_force_insert_n=round(float(peak_f[i]), 2),
                         peak_lateral_insert_n=round(float(np.linalg.norm(wz[ins, :2], axis=1).max()) if ins.any() else 0.0, 2),
                         peak_torque_insert_nm=round(float(np.linalg.norm(wz[ins, 3:], axis=1).max()) if ins.any() else 0.0, 3),
                         contact_steps=int(contact.sum()), max_penetration_mm=round(float(Lg["penetration"][i].max() * 1000), 3)))
    out = REPO / args.output
    out.mkdir(parents=True, exist_ok=True)
    clr = g.radial_clearance * 1000
    bins = [(0, clr), (clr, 99)]
    by = {}
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r["err_offset_mm"] < hi]
        by[f"offset {lo:.1f}-{hi:.1f} mm"] = dict(n=len(sel), seated=sum(r["outcome"] == "seated" for r in sel),
                                                  jammed=sum(r["outcome"] == "jammed" for r in sel),
                                                  short_no_contact=sum(r["outcome"] == "short_no_contact" for r in sel),
                                                  aborted=sum(r["outcome"] == "aborted" for r in sel),
                                                  with_contact=sum(r["contact_steps"] > 0 for r in sel))
    res = dict(note="simulation, placeholder hardware parameters (params.py), not calibrated", mode=args.mode,
               radial_clearance_mm=round(clr, 3), errors=dict(max_offset_mm=args.max_offset_mm, max_tilt_deg=args.max_tilt_deg),
               insert_speed_mm_s=args.insert_speed, seated=sum(r["outcome"] == "seated" for r in rows), n=n, by_offset=by,
               wall_s=round(wall, 1), unmeasured=unmeasured(p), envs=rows)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / "traces.npz", **{k: v[:, ::2] for k, v in Lg.items()})
    print("[RESULT]", json.dumps({k: res[k] for k in ("mode", "radial_clearance_mm", "seated", "n", "by_offset", "wall_s")}), flush=True)
    if frames:
        write_video(frames, rows[:args.video], args, out)
    print("REPLAY DONE", out, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("REPLAY DONE (error)", flush=True)
    app.close()
