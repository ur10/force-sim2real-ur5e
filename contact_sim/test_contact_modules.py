"""Offline checks of the sensor chain and the admittance controller (no Isaac):
    .venv/bin/python scripts/ur_contact/test_contact_modules.py"""

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from admittance import Admittance, dls_ik_step, exp_so3, skew_log  # noqa: E402
from params import Controller, Params, Sensor, load, unmeasured  # noqa: E402
from wrench_sensor import WrenchSensor, quat_to_mat, shift_wrench  # noqa: E402

DT = 1 / 240
ID_POSE = torch.tensor([[1.0, 0, 0, 0, 0.1, -0.5, 0.4]])
TCP = ID_POSE[:, 4:] + torch.tensor([[0, 0, 0.2]])


def quiet(**kw) -> Sensor:
    base = dict(force_noise_n=0.0, torque_noise_nm=0.0, force_bias_n=0.0, torque_bias_nm=0.0, drift_n_per_s=0.0,
                drift_nm_per_s=0.0, latency_s=0.0, lowpass_hz=0.0, rate_hz=1e9, reference="tool0")
    base.update(kw)
    return replace(Sensor(), **base)


def run(sensor, w, steps, pose=ID_POSE, tcp=TCP):
    return [sensor.step(w, pose, tcp) for _ in range(steps)]


def test_shift():
    # 10 N along -z applied 0.2 m below tool0 along +x offset: torque about tool0 = r x F
    f = torch.tensor([[0.0, 0, -10, 0, 0, 0]])
    w_tcp = f.clone()
    w0 = shift_wrench(w_tcp, torch.tensor([[0.05, 0, 0.0]]), torch.zeros(1, 3))  # load applied at (0.05, 0, 0)
    assert torch.allclose(w0[0, 3:], torch.tensor([0.0, 0.5, 0.0])), w0  # (0.05,0,0) x (0,0,-10) = (0, 0.5, 0)
    back = shift_wrench(w0, torch.zeros(1, 3), torch.tensor([[0.05, 0, 0.0]]))
    assert torch.allclose(back, w_tcp, atol=1e-6)


def test_compensation_exact_and_error():
    s = WrenchSensor(quiet(), 1, DT)
    out = run(s, torch.zeros(1, 6), 3)[-1]
    assert out["compensated"].abs().max() < 1e-5, out  # perfect payload model: no contact -> zero
    assert abs(out["raw_uncompensated"][0, 2] + Sensor().gripper_mass * 9.81) < 1e-4  # raw sees the gripper weight
    s = WrenchSensor(quiet(payload_mass_error=0.05), 1, DT)
    up = run(s, torch.zeros(1, 6), 2)[-1]["compensated"][0, 2]
    assert abs(up - 0.05 * 9.81) < 1e-4, up  # over-configured payload reads as +0.49 N
    # a CoG error leaves a torque that depends on orientation: tool pointing down vs sideways
    s = WrenchSensor(quiet(payload_cog_error=(0.0, 0.0, 0.02)), 1, DT)  # CoG error along the tool axis
    t_down = run(s, torch.zeros(1, 6), 2)[-1]["compensated"][0, 3:].norm()
    side = torch.tensor([[math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0, 0.1, -0.5, 0.4]])
    s.reset()
    t_side = run(s, torch.zeros(1, 6), 2, pose=side)[-1]["compensated"][0, 3:].norm()
    assert t_down < 1e-5 and abs(t_side - Sensor().gripper_mass * 9.81 * 0.02) < 1e-4, (t_down, t_side)


def test_latency_and_hold():
    s = WrenchSensor(quiet(latency_s=0.0125, rate_hz=1e9), 1, DT)  # 3 steps at 240 Hz
    w = torch.tensor([[0.0, 0, 5, 0, 0, 0]])
    outs = [s.step(torch.zeros(1, 6), ID_POSE, TCP)] + run(s, w, 6)
    z = [o["compensated"][0, 2].item() for o in outs]
    assert z[:4] == [0, 0, 0, 0] and abs(z[4] - 5) < 1e-5, z  # step at index 1 appears 3 steps later
    s = WrenchSensor(quiet(rate_hz=80.0), 1, DT)  # hold every 3 physics steps
    vals = [s.step(torch.tensor([[0.0, 0, float(k), 0, 0, 0]]), ID_POSE, TCP)["compensated"][0, 2].item() for k in range(7)]
    assert vals == [0, 0, 0, 3, 3, 3, 6], vals


def test_tare_averages():
    s = WrenchSensor(replace(quiet(force_noise_n=0.5, force_bias_n=1.0), lowpass_hz=0.0), 512, DT, seed=1)
    for _ in range(100):
        s.step(torch.zeros(512, 6), ID_POSE.expand(512, 7), TCP.expand(512, 3))
    s.tare()
    x = torch.stack([s.step(torch.zeros(512, 6), ID_POSE.expand(512, 7), TCP.expand(512, 3))["compensated"][:, 2] for _ in range(200)])
    resid = x.mean(0)  # per-env offset left after taring
    assert resid.abs().std() < 0.12, resid.std()  # ~0.5 / sqrt(48) per env, not the 0.5 N of one sample


def test_lowpass():
    fc = 10.0
    s = WrenchSensor(quiet(lowpass_hz=fc), 1, DT)
    s.step(torch.zeros(1, 6), ID_POSE, TCP)
    tau = 1 / (2 * math.pi * fc)
    n = round(tau / DT)
    outs = run(s, torch.tensor([[0.0, 0, 1, 0, 0, 0]]), n)
    v = outs[-1]["filtered"][0, 2].item()
    assert 0.55 < v < 0.70, v  # ~63% after one time constant


def test_noise_statistics():
    s = WrenchSensor(replace(quiet(force_noise_n=0.5, force_bias_n=1.0), lowpass_hz=0.0), 256, DT, seed=0)
    outs = [s.step(torch.zeros(256, 6), ID_POSE.expand(256, 7), TCP.expand(256, 3))["compensated"] for _ in range(400)]
    x = torch.stack(outs)  # (T, N, 6)
    within = x[..., 0].std(0).mean().item()
    across = x[..., 0].mean(0).std().item()
    assert 0.45 < within < 0.55 and 0.8 < across < 1.2, (within, across)


def test_admittance():
    cfg = replace(Controller(), deadband=(0.0,) * 6)
    a = Admittance(cfg, 1)
    w = torch.tensor([[2.0, 0, 0, 0, 0, 0]])
    for _ in range(4000):
        e = a.step(w, torch.zeros(1, 6), DT)
    assert abs(e[0, 0] - 2.0 / cfg.stiffness[0]) < 1e-4, e  # spring axis: e = F / K
    a.reset()
    w_des = torch.tensor([[0.0, 0, 5.0, 0, 0, 0]])  # push with 5 N: free space -> descend at F/D
    for _ in range(480):
        e = a.step(torch.zeros(1, 6), w_des, DT)
    v = a.ed[0, 2].item()
    assert abs(v + 5.0 / cfg.damping[2]) < 1e-3, v
    a.reset()
    for _ in range(100000):  # contact pushing back with exactly F_des: the offset stops moving
        e = a.step(w_des, w_des, DT)
    assert a.ed.abs().max() < 1e-6


def test_admittance_stiff_gains_stable():
    # the probe's gains at the controller rate: explicit Euler diverges here (1 - D dt / M = -2.3)
    cfg = replace(Controller(), mass=(1.0,) * 6, damping=(400.0,) * 6, stiffness=(0.0,) * 6, max_offset=(1.0,) * 6, deadband=(0.0,) * 6)
    a = Admittance(cfg, 1)
    w = torch.zeros(1, 6)
    for k in range(600):
        w[0, 2] = 2.0 * (-1) ** k  # worst case: force flipping every tick
        a.step(w, torch.zeros(1, 6), 1 / 120)
    assert a.ed.abs().max() < 2.0 / 400 + 1e-6 and a.e.abs().max() < 1e-3, (a.ed, a.e)
    a.reset()
    for _ in range(1200):
        a.step(torch.tensor([[0.0, 0, 2.0, 0, 0, 0]]), torch.zeros(1, 6), 1 / 120)
    assert abs(a.ed[0, 2] - 2.0 / 400) < 1e-5  # damping control: v = F / D


def test_expert_overforce_also_corrects():
    from expert import BACKOFF, ExpertCfg, InsertionExpert
    ex = InsertionExpert(ExpertCfg(), 1, 1 / 120)
    r = torch.tensor([[0.0, -0.010, 0.0]])
    F = torch.tensor([[0.0, 0.0, 20.0]])  # above f_max before any stall window has filled
    w = torch.cat([F, torch.cross(r, F, dim=-1)], -1)
    for _ in range(10):  # a few ticks of light contact, fewer than the stall window
        out = ex.act(w * 0.1, 0.02, torch.zeros(1))
    out = ex.act(w, 0.02, torch.zeros(1))
    assert int(out["mode"][0]) == BACKOFF and ex.xy_target[0, 1] < 0, (out["mode"], ex.xy_target)


def test_privileged_expert():
    from privileged_expert import ActionLimits, PrivilegedCfg, PrivilegedExpert
    ex = PrivilegedExpert(PrivilegedCfg(style="hover"), ActionLimits(), gap=0.0005, ring_height=0.0157)
    up = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, 3)
    priv = dict(e_xy=torch.tensor([[0.002, 0.0], [0.0001, 0.0], [0.0001, 0.0]]), depth=torch.tensor([-0.001, -0.004, 0.005]), axis=up)
    a = ex.act(priv, torch.zeros(3))
    assert a[0, 0] > 0.9 and a[0, 2] > -0.05  # far off: move toward the rod, do not go down
    assert a[1, 2] < -0.9 and a[2, 2] < -0.9  # aligned or inside: descend
    a = ex.act(priv, torch.tensor([10.0, 10.0, 10.0]))
    assert (a[:, 2] > 0).all()  # pressing: lift
    tilted = dict(e_xy=torch.zeros(1, 2), depth=torch.tensor([-0.004]), axis=torch.tensor([[0.05, 0.0, 0.9987]]))
    assert ex.act(tilted, torch.zeros(1))[0, 4] < 0  # axis leaning +x: rotate about -y back to vertical


def test_privileged_expert_contact_style():
    from privileged_expert import ActionLimits, PrivilegedCfg, PrivilegedExpert
    ex = PrivilegedExpert(PrivilegedCfg(), ActionLimits(), gap=0.0005, ring_height=0.0157)
    up = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, 3)
    priv = dict(e_xy=torch.tensor([[0.002, 0.0], [0.002, 0.0], [0.0001, 0.0]]), depth=torch.tensor([-0.003, 0.0, 0.004]), axis=up)
    a = ex.act(priv, torch.tensor([0.0, 2.5, 2.5]))
    assert a[0, 2] < -0.9 and a[0, 0] > 0.9       # no contact yet: go down while correcting
    assert abs(a[1, 2]) < 0.05 and a[1, 0] > 0.9  # on the tip at the contact force: hold, slide toward the rod
    assert a[2, 2] < -0.9                          # inside the bore: descend


def test_student_learns_and_masks():
    import numpy as np
    import student as stu
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(6000, stu.OBS_DIM)).astype(np.float32)
    act = np.tanh(obs[:, [59, 60, 8, 9, 10]] * 0.8).astype(np.float32)  # depends on vision and wrench features
    m, info = stu.train(obs, act, np.ones(len(obs)), epochs=25)
    assert info["history"][-1]["val"] < 0.25 * info["history"][0]["val"], info["history"][-1]
    mv, iv = stu.train(obs, act, np.ones(len(obs)), epochs=25, mask_vision=True)
    assert iv["history"][-1]["val"] > 2 * info["history"][-1]["val"]  # the masked input really is gone


def _episode_return(e_mm, depth_mm, wz=None, success_at=None, fail_at=None, gap=0.0005):
    """Discounted return of a scripted state sequence under the residual reward."""
    from residual_reward import RewardCfg, potential, step_reward, wrench_cost
    cfg = RewardCfg()
    e = torch.tensor(e_mm, dtype=torch.float32)[:, None] * 1e-3 * torch.tensor([[1.0, 0.0]])
    d = torch.tensor(depth_mm, dtype=torch.float32) * 1e-3
    T = len(depth_mm)
    w = torch.zeros(T, 6)
    if wz is not None:
        w[:, 2] = torch.tensor(wz, dtype=torch.float32)
    phi = potential(cfg, e, d, gap)
    G, disc = 0.0, 1.0
    for t in range(T - 1):
        succ = torch.tensor([success_at == t + 1])
        fail = torch.tensor([fail_at == t + 1])
        r = step_reward(cfg, phi[t:t + 1], phi[t + 1:t + 2], succ, fail, wrench_cost(cfg, w[t + 1:t + 2]),
                        torch.zeros(1, 5), torch.zeros(1, 5))["reward"]
        G += disc * float(r)
        disc *= cfg.gamma
        if bool(succ) or bool(fail):
            break
    return G


def test_reward_checks():
    import numpy as np
    clean = _episode_return([0] * 61, list(np.linspace(-6, 15, 61)), success_at=60)
    slow = _episode_return([0] * 121, list(np.linspace(-6, 15, 121)), success_at=120)
    hover = _episode_return([1.0] * 240, [-1.0] * 240)
    touch = _episode_return([1.0] * 240, [(-2.0 if (t // 10) % 2 else 0.0) for t in range(240)])
    push = _episode_return([1.0] * 240, [0.0] * 240, wz=[20.0] * 240)
    fail_early = _episode_return([1.0] * 10, [0.0] * 10, fail_at=9)
    assert clean > slow > hover, (clean, slow, hover)       # clean insertion highest; faster beats slower
    assert abs(hover) < 3.0                                  # hovering cannot farm progress
    assert touch <= hover + 0.2, (touch, hover)              # touch-and-retreat earns no recurring bonus
    assert push < hover - 0.5, (push, hover)                 # pushing a blocked rim accumulates cost
    assert fail_early < hover and fail_early < push          # failing early is not preferable to trying


def test_ppo_toy():
    """PPO on a batched 1-D reach task: the return must rise clearly within a few iterations."""
    from ppo import PPO, ActorCritic, PPOCfg
    torch.manual_seed(0)
    N, T = 64, 32
    m = ActorCritic(2, 1, 1, init_log_std=-0.5)
    algo = PPO(m, PPOCfg(lr=1e-3))
    def run(iters):
        out = []
        for _ in range(iters):
            x = torch.rand(N) * 2 - 1
            obs_l, priv_l, a_l, lp_l, r_l, v_l = [], [], [], [], [], []
            for t in range(T):
                obs = torch.stack([x, torch.full_like(x, t / T)], -1)
                m.obs_norm.update(obs)
                m.crit_norm.update(torch.cat([obs, x[:, None]], -1))
                a, lp = m.act(obs)
                v = m.value(obs, x[:, None]).detach()
                x = x + 0.1 * a[:, 0].clamp(-1, 1)
                obs_l.append(obs); priv_l.append(x[:, None]); a_l.append(a); lp_l.append(lp); v_l.append(v)
                r_l.append(-x.abs())
            R = torch.stack(r_l)
            done = torch.zeros(T, N, dtype=torch.bool)
            valid = torch.ones(T, N, dtype=torch.bool)
            obs = torch.stack([x, torch.ones_like(x)], -1)
            adv, ret = algo.gae(R, torch.stack(v_l), None, done, valid, m.value(obs, x[:, None]).detach())
            algo.update(torch.stack(obs_l), torch.stack(priv_l), torch.stack(a_l), torch.stack(lp_l), adv, ret, valid)
            out.append(float(R.mean()))
        return out
    curve = run(25)
    assert np.mean(curve[-5:]) > np.mean(curve[:5]) + 0.05, curve


def test_finray_grasp_and_pivot():
    from dataclasses import replace as rp
    import yourdfpy
    from contact_utils import FULL_URDF, M, _gripper_cfg, finray_grasp, pivot_frame
    p = Params()
    ih = rp(p.in_hand, grasp_model="finray")
    fg = finray_grasp(p.geometry, ih)
    assert 0.40 < fg["theta"] < 0.50, fg["theta"]  # touches a 32 mm ring (stock 0.55 would be 6 mm inside it)
    u = yourdfpy.URDF.load(FULL_URDF, load_meshes=False, build_collision_scene_graph=False)
    u.update_cfg(_gripper_cfg(u, fg["theta"]))
    for side in ("left", "right"):
        T = u.get_transform(f"{side}_inner_finger", "tcp")
        v = (T[:3, :3] @ fg["blades"][side]["verts"].T).T + T[:3, 3]
        assert np.allclose(np.abs(v[[0, 1, 4, 5], 0]), p.geometry.ring_outer_r, atol=1e-6)  # inner faces on the ring
        assert abs(v[:, 2].max() - ih.finray_tip_below_tcp) < 1e-6                        # tips 30 mm below the TCP
    T_link_ring = np.eye(4)
    T_link_ring[:3, :3] = np.diag([1.0, -1.0, -1.0])  # ring axis opposite the TCP z, as in the nut grasp
    Tp = pivot_frame(T_link_ring, p.geometry, ih)
    assert np.allclose(Tp[:3, :3].T @ Tp[:3, :3], np.eye(3)) and np.allclose(Tp[:3, 2], [0, 0, 1])
    assert abs(Tp[2, 3] - (p.geometry.ring_height / 2 - ih.tip_below_ring_top / 2)) < 1e-9
    assert np.allclose(pivot_frame(T_link_ring, p.geometry, rp(ih, grasp_model="center")), np.eye(4))  # earlier model


def test_rotation_helpers():
    w = torch.tensor([[0.1, -0.2, 0.3]])
    assert torch.allclose(skew_log(exp_so3(w)), w, atol=1e-5)
    R = quat_to_mat(torch.tensor([[math.cos(0.25), 0, 0, math.sin(0.25)]]))
    assert torch.allclose(skew_log(R), torch.tensor([[0, 0, 0.5]]), atol=1e-5)
    J = torch.eye(6)[None]
    dq = dls_ik_step(J, torch.tensor([[0.01, 0, 0]]), torch.zeros(1, 3), damping=1e-4)
    assert torch.allclose(dq[0, :3], torch.tensor([0.01, 0, 0]), atol=1e-6)


def test_params():
    p = load(overrides={"sensor": {"force_noise_n": 0.3}, "geometry": {"ring_bore_r": 0.0083}})
    assert p.sensor.force_noise_n == 0.3 and abs(p.geometry.radial_clearance - 0.0005) < 1e-9
    todo = unmeasured(p)
    assert "in_hand.k_lin" in todo and "sensor.latency_s" in todo
    try:
        load(overrides={"sensor": {"nope": 1}})
        raise AssertionError("unknown key accepted")
    except KeyError:
        pass
    assert isinstance(Params().controller.mass, tuple)


def test_torch_fk():
    import numpy as np
    import yourdfpy
    from torch_fk import ARM, TorchFK
    urdf = Path(__file__).resolve().parents[2] / "data/ur5e_robotiq/ur5e_robotiq85_ik.urdf"
    fk = TorchFK(urdf)
    u = yourdfpy.URDF.load(urdf, load_meshes=False, build_collision_scene_graph=False)
    order = [ARM.index(n) for n in u.actuated_joint_names]
    rng = np.random.default_rng(0)
    q = rng.uniform(-2, 2, (5, 6))
    T, J = fk.jacobian(torch.tensor(q, dtype=torch.float32))
    for i in range(5):
        u.update_cfg(q[i][order])
        assert np.allclose(T[i].numpy(), u.get_transform("tcp", "ur_base"), atol=1e-5)
    # Jacobian against finite differences of the tip position and orientation
    eps = 1e-3
    for j in range(6):
        dq = torch.zeros(5, 6)
        dq[:, j] = eps
        T2, _ = fk.jacobian(torch.tensor(q, dtype=torch.float32) + dq)
        dp = (T2[:, :3, 3] - T[:, :3, 3]) / eps
        dr = skew_log(T2[:, :3, :3] @ T[:, :3, :3].transpose(1, 2)) / eps
        assert torch.allclose(dp, J[:, :3, j], atol=2e-3) and torch.allclose(dr, J[:, 3:, j], atol=2e-3), j


def test_expert_steps_toward_contact():
    from expert import BACKOFF, DESCEND, ExpertCfg, InsertionExpert
    cfg = ExpertCfg()
    ex = InsertionExpert(cfg, 4, 1 / 120)
    # contact points 10 mm from the TCP axis in four directions, 4 N pushing up
    dirs = torch.tensor([[1.0, 0], [0, 1.0], [-1.0, 0], [0.7071, -0.7071]])
    r = torch.cat([dirs * 0.010, torch.zeros(4, 1)], -1)
    F = torch.tensor([[0.0, 0, 4.0]]).expand(4, 3)
    w = torch.cat([F, torch.cross(r, F, dim=-1)], -1)
    depth = torch.zeros(4)
    for _ in range(int(cfg.stall_window_s * 120) + 3):  # stalled on the rod tip
        out = ex.act(w, 0.02, depth)
    assert (out["mode"] == BACKOFF).all(), out["mode"]
    assert out["xy"].abs().max() == 0  # no lateral move while still pressed on the tip
    step = ex.xy_target / ex.xy_target.norm(dim=-1, keepdim=True)
    assert torch.allclose(step, dirs, atol=1e-3), step  # aims toward where the rod touches
    for _ in range(30):  # still loaded: keeps lifting, does not move sideways
        out = ex.act(w, 0.02, depth)
    assert (out["mode"] == BACKOFF).all() and out["xy"].abs().max() == 0
    free = torch.zeros(4, 6)
    for _ in range(400):  # released: lift the margin, move sideways at v_lateral, descend again
        out = ex.act(free, 0.02, depth)
    assert torch.allclose(out["xy"], ex.xy_target, atol=1e-6), (out["xy"], ex.xy_target)
    assert (out["mode"] == DESCEND).all(), out["mode"]


def test_penetration_metric():
    from contact_utils import ring_metrics
    g = Params().geometry

    def pose(dx_mm, tilt_deg, below_tip_mm):
        h = math.radians(tilt_deg) / 2
        c = [g.rod_xy[0] + dx_mm / 1000, g.rod_xy[1], g.rod_tip_z - below_tip_mm / 1000 + g.ring_height / 2]
        return torch.tensor([[math.cos(h), 0, math.sin(h), 0, *c]])
    pen = lambda *a: float(ring_metrics(pose(*a), g)["penetration"][0]) * 1000  # noqa: E731
    assert pen(0, 0, 10) == 0 and pen(2.5, 0, 10) == 0  # inside the clearance
    assert abs(pen(3.5, 0, 10) - (3.5 - g.radial_clearance * 1000)) < 0.02  # pushed past it
    assert pen(4, 0, -0.01) == 0 and pen(6, 6, 0.0) == 0  # resting on the tip, upright or tilted


def test_retime_starts_and_arrives_at_rest():
    import numpy as np
    from contact_utils import retime
    g = Params().geometry
    # a carry that starts mid-motion (0.3 m/s) and ends above the rod axis
    n = 40
    xs = np.linspace(-0.12, 0.0, n) + g.rod_xy[0]
    zs = np.linspace(0.05, g.rod_tip_z + 0.004 + g.ring_height / 2, n)
    obj = np.zeros((n, 7))
    obj[:, 0] = 1
    obj[:, 4], obj[:, 5], obj[:, 6] = xs, g.rod_xy[1], zs
    tcp = obj.copy()
    tcp[:, 6] += 0.0249
    refs, k = retime(tcp, obj, 1 / 30, 1.0, g, 0.006, 0.005, 0.015, 1 / 120)
    v = np.linalg.norm(np.diff(refs[:, :3, 3], axis=0), axis=1) * 120
    uniform = np.linalg.norm(np.diff(tcp[:, 4:], axis=0), axis=1).max() * 30
    assert v[0] < 0.05 * uniform and v[k - 2] < 0.05 * uniform, (v[0], v[k - 2], uniform)
    assert v[:k].max() <= uniform * 1.01
    assert abs(np.median(v[k + 30:-1]) - 0.005) < 2e-4  # the descent at the insertion speed


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
