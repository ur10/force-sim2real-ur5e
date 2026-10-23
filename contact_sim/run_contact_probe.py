"""Simulated contact probe: the ring, held by the UR5e through the in-hand spring, is pushed onto
the measured rod at known offsets and tilts (probe_protocol.py); logs the true and sensed wrench,
ring-rod geometry, in-hand deflection and an outcome per condition. First it checks the wrench
readout against a known load applied to the ring.

    scripts/ur_cube/isaac_run.sh outputs/ur_nut/contact_sim/probe.log "PROBE DONE" \
        scripts/ur_contact/run_contact_probe.py --output outputs/ur_nut/contact_sim/probe

Uncalibrated: every hardware number comes from params.py (placeholders until measured).
"""

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("outputs/ur_nut/contact_sim/probe"))
parser.add_argument("--params", type=Path, default=None, help="JSON with measured values (params.py nesting)")
parser.add_argument("--set", type=str, default=None, help='JSON overrides, e.g. \'{"geometry": {"ring_bore_r": 0.0083}}\'')
parser.add_argument("--offsets", type=float, nargs="*", default=None)
parser.add_argument("--tilts", type=float, nargs="*", default=None)
parser.add_argument("--speed", type=float, default=5.0, help="mm/s")
parser.add_argument("--seed", type=int, default=0, help="sensor noise seed")
parser.add_argument("--controller", choices=["admittance", "stiff"], default="admittance",
                    help="stiff: position-controlled descent that stops and holds once the sensed |F| passes --f-stop (GELLO-like)")
parser.add_argument("--f-stop", type=float, default=8.0, help="N, stiff controller")
parser.add_argument("--export", type=Path, default=Path("outputs/ur_nut/physical_export/nut_nominal_000.npz"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from admittance import Admittance, dls_ik_step, skew_log  # noqa: E402
from params import Controller, load, to_dict, unmeasured  # noqa: E402
from probe_protocol import ProbeProtocol  # noqa: E402
from ring_rod_sim import RingRodScene, physx_cfg  # noqa: E402
from torch_fk import TorchFK  # noqa: E402
from wrench_sensor import WrenchSensor, quat_to_mat, shift_wrench  # noqa: E402
from contact_utils import ARM, M, P, URDF, NumpyIK, grasp_T_tcp_obj, ring_metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main():
    p = load(args.params, json.loads(args.set) if args.set else None)
    g = p.geometry
    proto = ProbeProtocol(offsets_mm=tuple(args.offsets) if args.offsets else ProbeProtocol().offsets_mm,
                          tilts_deg=tuple(args.tilts) if args.tilts else ProbeProtocol().tilts_deg, speed_mm_s=args.speed)
    conds = proto.expand()
    ex = np.load(REPO / args.export)
    T_tcp_obj = grasp_T_tcp_obj(M(ex["T_tcp_obj"]), g, p.in_hand)
    entry = int(np.argmax(ex["rod_in_hole"] & ex["held"]))
    q_seed = ex["q"][max(entry - 4, 0)]
    R_ring_nom = M(ex["T_base_obj"][entry])[:3, :3]
    ik = NumpyIK()
    T_link_tcp = ik.u.get_transform("tcp", "robotiq_85_base_link")
    T_link_ring = T_link_tcp @ T_tcp_obj

    def start_pose(c):
        Rt = Rotation.from_euler("y", np.radians(c.tilt_deg)).as_matrix()
        T_ring = np.eye(4)
        T_ring[:3, :3] = Rt @ R_ring_nom
        axis = T_ring[:3, 2] * np.sign(T_ring[2, 2])
        bottom = np.r_[g.rod_xy[0] + c.offset_mm / 1000, g.rod_xy[1], g.rod_tip_z + c.start_gap_mm / 1000]
        T_ring[:3, 3] = bottom + axis * g.ring_height / 2
        T_tcp = T_ring @ np.linalg.inv(T_tcp_obj)
        q, err = ik.ik(T_tcp, q_seed)
        assert err < 1e-4, f"IK failed for {c}: {err * 1000:.2f} mm"
        return q, T_tcp, T_ring

    starts = [start_pose(c) for c in conds]
    n = len(conds)
    q0 = np.stack([q for q, _, _ in starts])
    ring0 = np.stack([P(T) for _, _, T in starts])
    gripper = 0.55  # the generator's closing for the nut (the fingers are filtered from the ring)
    sim = sim_utils.SimulationContext(physx_cfg(p))
    scene = RingRodScene(sim.stage, p, n, q0, gripper, P(T_link_ring), ring0, device=args.device)
    sim.reset()
    scene.post_reset()
    dev = sim.device
    dt = p.sim.dt
    robot, ring = scene.robot, scene.ring
    tfk = TorchFK(URDF, device=dev)
    q_cmd = torch.tensor(q0, dtype=torch.float32, device=dev)
    T_lr = torch.tensor(T_link_ring, dtype=torch.float32, device=dev)

    def physics_step(k=1):
        for _ in range(k):
            robot.write_data_to_sim()
            ring.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            ring.update(dt)

    def mat(pose):  # (N, 7) -> (N, 4, 4)
        T = torch.eye(4, device=dev).repeat(pose.shape[0], 1, 1)
        T[:, :3, :3] = quat_to_mat(pose[:, :4])
        T[:, :3, 3] = pose[:, 4:]
        return T

    def pose(T):  # (N, 4, 4) -> (N, 7)
        from isaaclab.utils.math import quat_from_matrix
        return torch.cat([quat_from_matrix(T[:, :3, :3]), T[:, :3, 3]], -1)

    def teleport(q):
        q_cmd[:] = torch.as_tensor(q, dtype=torch.float32, device=dev)
        qt = robot.data.default_joint_pos.clone()
        qt[:, scene.arm_ids] = q_cmd
        robot.write_joint_state_to_sim(qt, torch.zeros_like(qt))
        robot.set_joint_position_target(qt)
        physics_step(2)
        scene.set_ring_pose(pose(mat(scene.link_pose("robotiq_85_base_link")) @ T_lr))
        physics_step(1)

    # ---- wrench readout check: known force on every ring, robots still ----
    teleport(q0)
    physics_step(int(0.5 / dt))
    f_app_w = torch.tensor([3.0, -2.0, 4.0], device=dev).expand(n, 3)
    rp = scene.ring_pose()
    f_body = (quat_to_mat(rp[:, :4]).transpose(1, 2) @ f_app_w[..., None])[..., 0]
    ring.set_external_force_and_torque(f_body[:, None, :], torch.zeros(n, 1, 3, device=dev))
    physics_step(int(1.0 / dt))
    raw = scene.wrench_incoming()
    link, rp = scene.link_pose("robotiq_85_base_link"), scene.ring_pose()
    F = f_app_w + torch.tensor([0, 0, -p.in_hand.mass * 9.81], device=dev)
    expected_w = torch.cat([F, torch.cross(rp[:, 4:] - link[:, 4:], F, dim=-1)], -1)
    Rl = quat_to_mat(link[:, :4])
    rot = lambda R, w: torch.cat([(R @ w[:, :3, None])[..., 0], (R @ w[:, 3:, None])[..., 0]], -1)  # noqa: E731
    cands = {"world,+1": raw, "world,-1": -raw, "link,+1": rot(Rl, raw), "link,-1": -rot(Rl, raw)}
    errs = {k: float((v - expected_w).norm(dim=-1).max()) for k, v in cands.items()}
    conv = min(errs, key=errs.get)
    frame, sign = conv.split(",")[0], float(conv.split(",")[1])
    check = dict(applied_world_n=[3.0, -2.0, 4.0], expected_load_on_tool_env0=expected_w[0].tolist(),
                 raw_incoming_env0=raw[0].tolist(), max_residual_over_envs_by_convention=errs, convention=conv)
    print("[CHECK] wrench readout:", json.dumps(check), flush=True)
    ring.set_external_force_and_torque(torch.zeros(n, 1, 3, device=dev), torch.zeros(n, 1, 3, device=dev))

    def tool_load_tool0() -> torch.Tensor:
        """Load the environment applies on the tool, about tool0, base frame."""
        raw = scene.wrench_incoming()
        lk = scene.link_pose("robotiq_85_base_link")
        if frame == "link":
            raw = rot(quat_to_mat(lk[:, :4]), raw)
        return shift_wrench(sign * raw, lk[:, 4:], scene.link_pose("tool0")[:, 4:])

    # probe: compliant only along z (damping control), the other axes held
    ctrl = Controller(**{**to_dict(p)["controller"], "max_offset": (0.0, 0.0, 0.0 if args.controller == "stiff" else 0.03, 0.0, 0.0, 0.0),
                         "stiffness": (0.0,) * 6, "mass": (1.0,) * 6})
    adm = Admittance(ctrl, n, dev)
    sensor = WrenchSensor(p.sensor, n, dt, dev, seed=args.seed)
    T_tool0_ring = ik.u.get_transform("tcp", "tool0") @ T_tcp_obj  # the ring is part of the configured payload
    sensor.held_mass = p.in_hand.mass
    sensor.held_cog_tool0 = torch.tensor(T_tool0_ring[:3, 3], dtype=torch.float32, device=dev)
    ctrl_every = max(1, round(1 / (p.controller.rate_hz * dt)))
    out = REPO / args.output
    out.mkdir(parents=True, exist_ok=True)

    teleport(q0)
    for _ in range(int(0.3 / dt)):  # settle with the sensor running, then tare (zero_ftsensor)
        sensor.step(tool_load_tool0(), scene.link_pose("tool0"), scene.link_pose("tcp")[:, 4:])
        physics_step()
    sensor.tare()
    adm.reset()
    T_tcp0 = torch.tensor(np.stack([T for _, T, _ in starts]), dtype=torch.float32, device=dev)
    v = torch.tensor([c.speed_mm_s / 1000 for c in conds], device=dev)
    travel = torch.tensor([(c.start_gap_mm + c.target_depth_mm) / 1000 for c in conds], device=dev)
    abort_f = torch.tensor([c.abort_force_n for c in conds], device=dev)
    n_move = int((float((travel / v).max()) + 0.1) / dt)
    n_hold = int(max(c.hold_s for c in conds) / dt)
    keys = ("w_true", "w_sensor", "w_filtered", "tcp", "ring", "offset", "depth", "tilt", "in_hole", "penetration",
            "hand_mm", "hand_deg", "e_z")
    log = {k: [] for k in keys}
    aborted = torch.zeros(n, dtype=torch.bool, device=dev)
    stopped = torch.zeros(n, dtype=torch.bool, device=dev)  # stiff controller: force limit reached, holding
    s_frozen = torch.zeros(n, device=dev)
    t0 = time.time()
    for k in range(n_move + n_hold):
        tk, ramp = k * dt, 0.2  # 0.2 s velocity ramp, as a real arm accelerates
        s = torch.minimum(v * (tk * tk / (2 * ramp) if tk < ramp else tk - ramp / 2), travel)
        s = torch.where(aborted | stopped, s_frozen, s)  # aborted or force-stopped envs hold where they stopped
        p_ref = T_tcp0[:, :3, 3] - torch.stack([torch.zeros_like(s), torch.zeros_like(s), s], -1)
        w = tool_load_tool0()
        tool0, tcp = scene.link_pose("tool0"), scene.link_pose("tcp")
        ch = sensor.step(w, tool0, tcp[:, 4:])
        if k % ctrl_every == 0:
            e = adm.step(ch["filtered"], torch.zeros(n, 6, device=dev), dt * ctrl_every)
            R_cmd, p_cmd = Admittance.compose(T_tcp0[:, :3, :3], p_ref, torch.eye(3, device=dev).expand(n, 3, 3), e)
            for _ in range(2):  # differential IK on the commanded configuration (servoJ-style)
                T, J = tfk.jacobian(q_cmd)
                q_cmd += dls_ik_step(J, p_cmd - T[:, :3, 3], skew_log(R_cmd @ T[:, :3, :3].transpose(1, 2)), damping=1e-3)
            qt = robot.data.joint_pos.clone()
            qt[:, scene.arm_ids] = q_cmd
            robot.set_joint_position_target(qt)
        physics_step()
        rp = scene.ring_pose()
        m = ring_metrics(rp, g)
        dT = torch.linalg.inv(mat(scene.link_pose("robotiq_85_base_link")) @ T_lr) @ mat(rp)
        hand_deg = torch.rad2deg(torch.arccos(((dT[:, 0, 0] + dT[:, 1, 1] + dT[:, 2, 2] - 1) / 2).clamp(-1, 1)))
        for key, val in (("w_true", w), ("w_sensor", ch["compensated"]), ("w_filtered", ch["filtered"]), ("tcp", tcp), ("ring", rp),
                         ("offset", m["offset"]), ("depth", m["depth"]), ("tilt", m["tilt"]), ("in_hole", m["in_hole"]),
                         ("penetration", m["penetration"]), ("hand_mm", dT[:, :3, 3].norm(dim=-1) * 1000), ("hand_deg", hand_deg),
                         ("e_z", adm.e[:, 2])):
            log[key].append(val.detach().clone())
        newly = (~aborted) & (w[:, :3].norm(dim=-1) > abort_f)
        s_frozen = torch.where(newly & ~stopped, s, s_frozen)
        aborted |= newly
        if args.controller == "stiff":
            hit = (~stopped) & (~aborted) & (ch["filtered"][:, :3].norm(dim=-1) > args.f_stop)
            s_frozen = torch.where(hit, s, s_frozen)
            stopped |= hit
    wall = time.time() - t0
    L = {k: torch.stack(v, 1).cpu().numpy() for k, v in log.items()}  # (N, T, ...)
    t_axis = np.arange(n_move + n_hold) * dt
    traces, summary = {}, []
    for ci, c in enumerate(conds):
        Li = {k: v[ci] for k, v in L.items()}
        Li["t"] = t_axis
        depth_end = float(Li["depth"][-1])
        reached = depth_end > (c.target_depth_mm - 3.0) / 1000
        f = np.linalg.norm(Li["w_true"][:, :3], axis=1)
        contact = Li["w_true"][:, 2] > 0.8  # the rod pushing up on the ring (motion signal stays below)
        outcome = "aborted" if bool(aborted[ci]) else ("seated" if reached else "jammed")
        row = dict(i=ci, offset_mm=c.offset_mm, tilt_deg=c.tilt_deg, speed_mm_s=c.speed_mm_s, outcome=outcome,
                   final_depth_mm=round(depth_end * 1000, 2), final_offset_mm=round(float(Li["offset"][-1] * 1000), 2),
                   peak_force_n=round(float(f.max()), 2), peak_axial_n=round(float(Li["w_true"][:, 2].max()), 2),
                   peak_lateral_n=round(float(np.linalg.norm(Li["w_true"][:, :2], axis=1).max()), 2),
                   peak_torque_nm=round(float(np.linalg.norm(Li["w_true"][:, 3:], axis=1).max()), 3),
                   first_contact_depth_mm=round(float(Li["depth"][contact][0] * 1000), 2) if contact.any() else None,
                   max_penetration_mm=round(float(Li["penetration"].max() * 1000), 3),
                   max_hand_deflection_mm=round(float(Li["hand_mm"].max()), 3), max_hand_deflection_deg=round(float(Li["hand_deg"].max()), 3))
        summary.append(row)
        traces[f"c{ci:02d}"] = Li
        print("[PROBE]", json.dumps(row), flush=True)
    print(f"[INFO] {n} conditions in parallel, {n_move + n_hold} steps, {wall:.0f} s wall", flush=True)
    np.savez_compressed(out / "traces.npz", **{f"{c}/{k}": v for c, L in traces.items() for k, v in L.items()})
    (out / "summary.json").write_text(json.dumps(dict(
        note="simulation with placeholder hardware parameters (params.py); not calibrated", wrench_check=check,
        radial_clearance_mm=round(g.radial_clearance * 1000, 3), unmeasured=unmeasured(p), params=to_dict(p),
        conditions=summary), indent=1))
    print("PROBE DONE", out, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("PROBE DONE (error)", flush=True)
    app.close()
