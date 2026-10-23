"""Offline tests of the deployment (numpy only; no robot, no network).

1-5 check that deploy_core rebuilds the simulator's policy inputs exactly, from a reference log of
a simulated run (log_deploy_reference.py): flange -> tcp, the wrench filter, the 62-d observation,
the action integration, and the numpy student against the logged actions. 6-8 run whole episodes
against the stand-in robot (fake_ur.py): an aligned insertion seats, a misaligned blind replay
stops pushing and times out, and an excessive contact force aborts.

    python deploy/test_deploy.py                      (on the robot machine, from ~/ring_rod_probe)
    DEPLOY_DATA=<dir with reference.npz and the bundle> python test_deploy.py
"""

import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deploy_core import (Grasp, Limits, ObsBuilder, ReplayPolicy, Rod, SimConstants, StudentNP, WrenchChannel,  # noqa: E402
                         _tcp_from_flange, entry_pose, integrate, kinematic_estimate, load_bundle, run_episode)
from fake_ur import FakeUR  # noqa: E402

DATA = Path(os.environ.get("DEPLOY_DATA", HERE.parent / "sim_library" / "deploy"))
REF = np.load(DATA / "reference.npz")
BUNDLE, META = load_bundle(DATA / "bundle_dagger_kinematic.npz")
K = SimConstants()


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def T_of(pose7):
    T = np.eye(4)
    T[:3, :3] = qmat(pose7[:4])
    T[:3, 3] = pose7[4:]
    return T


def grasp_ref():
    return Grasp(BUNDLE["R_tcp_obj"], BUNDLE["xy_tcp_obj"], float(REF["ring_height"]), float(REF["ring_bottom_below_flange"]))


def rod_ref():
    return Rod(REF["rod_xy"], float(REF["rod_tip_z"]), "sim")


def running(i):
    """Ticks k whose observation the env produced while env i was still running."""
    done = REF["tick_done"][:, i]
    last = int(np.argmax(done)) if done.any() else len(done) - 1
    return range(0, last + 1)


def test_0_rotations():
    from deploy_core import exp_so3, mat_to_rotvec, pose6_to_T, T_to_pose6
    rng = np.random.default_rng(1)
    axes = [v / np.linalg.norm(v) for v in rng.normal(size=(20, 3))]
    for ax in axes:
        for th in (0.0, 1e-8, 0.3, 2.0, math.pi - 1e-3, math.pi - 1e-7, math.pi):
            R = exp_so3(ax * th)
            assert np.abs(exp_so3(mat_to_rotvec(R)) - R).max() < 1e-6, (ax, th)
    T = np.eye(4); T[:3, :3] = BUNDLE["R_entry"]; T[:3, 3] = [0.1, -0.6, 0.2]
    assert np.abs(pose6_to_T(T_to_pose6(T)) - T).max() < 1e-6  # exactly a half turn: ~2e-8 from the axis extraction


def test_1_tcp_from_flange():
    err = 0.0
    for s in range(0, len(REF["s_tool0"]), 7):
        for i in range(REF["s_tool0"].shape[1]):
            T = _tcp_from_flange(T_of(REF["s_tool0"][s, i]), K)
            err = max(err, float(np.abs(T[:3, 3] - REF["s_tcp"][s, i, 4:]).max()))
    assert err < 2e-6, err
    T_ref = grasp_ref().T_tcp_obj()
    assert np.abs(T_ref - REF["T_tcp_obj"]).max() < 1e-6, (T_ref, REF["T_tcp_obj"])


def test_2_filter():
    ticks = REF["s_tick"]
    i0 = int(np.where(ticks == -1)[0][-1])  # last sensor step of the reset (the hist initialisation)
    dt = float(REF["sensor_dt"])
    for i in range(REF["s_comp"].shape[1]):
        ch = WrenchChannel(lowpass_hz=K.lowpass_hz)
        ch.lp = REF["s_filt"][i0, i].astype(float)
        err = 0.0
        for s in range(i0 + 1, len(ticks)):
            out = ch.step(REF["s_comp"][s, i], dt)
            err = max(err, float(np.abs(out - REF["s_filt"][s, i]).max()))
        assert err < 1e-4, (i, err)
    a = 1 - math.exp(-2 * math.pi * K.lowpass_hz * dt)
    assert abs(a - float(REF["alpha"])) < 1e-9


def test_3_observation():
    ticks = REF["s_tick"]
    i0 = int(np.where(ticks == -1)[0][-1])
    g, rod = grasp_ref(), rod_ref()
    for i in range(REF["tick_obs"].shape[1]):
        ob = ObsBuilder(K)
        ob.reset(REF["p_start"][i], REF["s_filt"][i0, i])
        worst = 0.0
        for k in running(i):
            if k > 0:
                last = int(np.where(ticks == k - 1)[0][-1])
                ob.push(REF["s_filt"][last, i], REF["tick_action"][k - 1, i])
            T_tcp = T_of(REF["tick_tcp"][k, i])
            est = kinematic_estimate(rod, g, T_tcp)
            o = ob.observe(REF["tick_p_cmd"][k, i], T_tcp[:3, 3], est)
            d = np.abs(o - REF["tick_obs"][k, i])
            scale = np.r_[np.full(6, 1e-6), np.full(48, 1e-4), np.full(5, 1e-6), np.full(2, 2e-6), [1e-9]]
            assert (d <= scale).all(), (i, k, np.where(d > scale)[0], d[d > scale])
            worst = max(worst, float(d[:6].max()))
    assert worst < 1e-6


def test_4_integration():
    for i in range(REF["tick_obs"].shape[1]):
        for k in list(running(i))[:-1]:
            p, R = REF["tick_p_cmd"][k, i].astype(float), REF["tick_R_cmd"][k, i].astype(float)
            for _ in range(K.sub_steps):
                p, R = integrate(REF["tick_action"][k, i], p, R, K)
            assert np.abs(p - REF["tick_p_cmd"][k + 1, i]).max() < 5e-7, (i, k)  # float32 positions ~0.6 m from the base
            assert np.abs(R - REF["tick_R_cmd"][k + 1, i]).max() < 1e-5, (i, k)


def test_5_student_actions():
    pol = StudentNP.from_bundle(BUNDLE)
    err = max(float(np.abs(pol(o) - a).max()) for o, a in zip(BUNDLE["ref_obs"], BUNDLE["ref_act"]))
    assert err < 1e-5, err
    worst = 0.0
    for i in range(REF["tick_obs"].shape[1]):
        for k in list(running(i))[:-1]:
            worst = max(worst, float(np.abs(pol(REF["tick_obs"][k, i]) - REF["tick_action"][k, i]).max()))
    assert worst < 1e-4, worst


def test_9_residual_numpy():
    """Numpy residual (actor + normaliser on the student base) against torch, if a residual bundle is present."""
    from deploy_core import ResidualNP
    f = DATA / "residual_equivalence_fixture.npz"
    if not f.exists():
        print("  (no residual bundle: skipped)")
        return
    z, _ = load_bundle(f)
    pol = ResidualNP.from_bundle(z, K)
    err = max(float(np.abs(pol(o) - a).max()) for o, a in zip(z["ref_obs"], z["ref_act_res"]))
    assert err < 1e-5, err
    r = _fake_episode(pol)  # the whole loop with a residual policy (stand-in robot, aligned)
    assert r.outcome in ("seated", "timeout"), (r.outcome, r.reason)


def _fake_episode(policy_name, rod_offset_m=(0.0, 0.0), k_contact=1500.0, lim=None):
    g = Grasp(BUNDLE["R_tcp_obj"], BUNDLE["xy_tcp_obj"], float(BUNDLE["ring_height"]), 0.225)
    rod = Rod(np.array([-0.085, -0.810]), 0.180, "test")
    R = BUNDLE["R_entry"]
    T_tcp = entry_pose(rod, g, R, K)
    T_fl = T_tcp.copy()
    T_fl[:3, 3] = T_tcp[:3, 3] - R @ np.array([0, 0, K.flange_to_tcp])
    world_rod = Rod(rod.xy + np.asarray(rod_offset_m), rod.tip_z, "world")
    robot = FakeUR(T_fl, world_rod, g, K, k_contact=k_contact, noise_n=0.0)
    pol = policy_name if callable(policy_name) else StudentNP.from_bundle(BUNDLE) if policy_name == "student" else ReplayPolicy(K)
    return run_episode(robot, pol, K, g, rod, R, lim or Limits(), WrenchChannel(), verbose=False)


def test_6_fake_aligned_seats():
    for name in ("replay", "student"):
        r = _fake_episode(name)
        assert r.outcome == "seated", (name, r.outcome, r.reason)
        assert r.log["obs"].shape[1] == 62


def test_7_fake_blind_replay_on_the_top_times_out():
    r = _fake_episode("replay", rod_offset_m=(0.006, 0.0))
    assert r.outcome == "timeout", (r.outcome, r.reason)
    assert 2.5 < r.peak_force_n < 6.0, r.peak_force_n  # stops pushing just above 3 N


def test_8_fake_force_abort():
    r = _fake_episode("replay", rod_offset_m=(0.006, 0.0), k_contact=20000.0, lim=Limits(force_abort_n=4.0))
    assert r.outcome == "aborted" and r.reason.startswith("force"), (r.outcome, r.reason)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
