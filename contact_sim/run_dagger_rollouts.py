"""One DAgger collection (or evaluation) process: N envs x R lockstep episodes of the contact
phase. Per episode each env is driven by the privileged expert with probability beta, otherwise
by the student checkpoint; every visited state is labelled with the expert's action.

    scripts/ur_cube/isaac_run.sh outputs/ur_nut/dagger/it0.log "ROLLOUTS DONE" scripts/ur_contact/run_dagger_rollouts.py \
        --beta 1.0 --rounds 4 --output outputs/ur_nut/dagger/shard_it0.npz
    ... --beta 0 --student outputs/ur_nut/dagger/student_it3.pt --eval      # student alone

Writes the dataset shard (obs, expert_action, executed action, valid, episode/env ids, outcome per
episode) and a JSON summary with success counts (Wilson 95% intervals).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--student", type=Path, default=None)
parser.add_argument("--residual", type=Path, default=None, help="residual.pt from run_residual_rl.py")
parser.add_argument("--beta", type=float, default=1.0, help="probability an episode is driven by the expert")
parser.add_argument("--policy", choices=["mix", "expert", "wrench_expert", "replay", "residual"], default="mix",
                    help="mix = expert/student by beta; wrench_expert / replay = baselines for evaluation")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--expert-style", choices=["contact", "hover"], default="contact")
parser.add_argument("--rounds", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval", action="store_true", help="evaluation run: do not keep the dataset shard")
parser.add_argument("--params", type=Path, default=None)
parser.add_argument("--set", type=str, default=None)
parser.add_argument("--env", type=str, default=None, help="JSON overrides for EnvCfg")
parser.add_argument("--export", type=Path, default=Path("outputs/ur_nut/physical_export/nut_nominal_000.npz"))
parser.add_argument("--video-dir", type=Path, default=None, help="film every env close-up (use few envs); writes <policy>_envN.mp4")
parser.add_argument("--video-name", type=str, default=None, help="name used in the video files and labels")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.video_dir:
    args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from contact_env import ContactEnv, EnvCfg  # noqa: E402
from contact_utils import URDF  # noqa: E402
from expert import DONE, ExpertCfg, InsertionExpert  # noqa: E402
from params import load, to_dict  # noqa: E402
from privileged_expert import PrivilegedCfg, PrivilegedExpert  # noqa: E402
from ring_rod_sim import physx_cfg  # noqa: E402
import student as st  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def main():
    p = load(args.params, json.loads(args.set) if args.set else None)
    cfg = EnvCfg(**(json.loads(args.env) if args.env else {}))
    sim = sim_utils.SimulationContext(physx_cfg(p))
    env = ContactEnv(sim, p, args.num_envs, REPO / args.export, URDF, cfg, seed=args.seed, device=args.device)
    cam = None
    if args.video_dir:
        import render_utils as ru
        ru.add_visuals(args.num_envs, p.geometry.table_z)
        cam = ru.make_cameras()
    sim.reset()
    env.post_reset()
    if cam is not None:
        ru.aim(cam, env.scene.origins, p.geometry.rod_xy, p.geometry.rod_tip_z)
        vname = args.video_name or args.policy
        tiles = [[] for _ in range(args.num_envs)]
    dev, n = args.device, args.num_envs
    gap = p.geometry.radial_clearance
    expert = PrivilegedExpert(PrivilegedCfg(style=args.expert_style, target_depth=cfg.target_depth), env.limits, gap,
                              p.geometry.ring_height, dev)
    student = st.load(REPO / args.student, dev) if args.student else None
    resid = None
    if args.policy == "residual":
        from base_policies import replay_base
        from ppo import ActorCritic
        ck = torch.load(REPO / args.residual, map_location=dev)
        resid = ActorCritic(ck["obs_dim"], ck["priv_dim"], 5).to(dev)
        resid.load_state_dict(ck["model"])
        resid.eval()
        res_scale = torch.tensor(ck["res_scale"], device=dev)
        res_base = st.load(REPO / ck["student"], dev) if ck["base"] == "student" else None
    if args.policy == "mix" and args.beta < 1 and student is None:
        raise SystemExit("--beta < 1 needs --student")
    steps = int(cfg.episode_s * cfg.policy_hz)
    rec = {k: [] for k in ("obs", "expert_action", "action", "valid", "env", "episode", "t")}
    episodes = []
    rng = np.random.default_rng(args.seed + 17)
    t0 = time.time()
    for rnd in range(args.rounds):
        obs = env.reset()
        use_expert = torch.tensor(rng.uniform(size=n) < args.beta, device=dev)
        wexp = InsertionExpert(ExpertCfg(v_descend=0.005), n, 1 / cfg.policy_hz, dev) if args.policy == "wrench_expert" else None
        z_start = env.scene.link_pose("tcp")[:, 6].clone()
        info = None
        for k in range(steps):
            priv = env.privileged() if info is None else info["priv"]
            fz = env.hist[:, -1, 2]
            a_exp = expert.act(priv, fz)
            if args.policy == "expert":
                a = a_exp
            elif args.policy == "residual":
                with torch.no_grad():
                    base = res_base(obs) if res_base is not None else replay_base(fz, env.limits.v_z)
                    r_mean = resid.dist(torch.cat([obs, base], -1)).mean
                a = (base + res_scale * r_mean.clamp(-1, 1)).clamp(-1, 1)
            elif args.policy == "replay":  # blind: straight down at 5 mm/s, stop pushing above 3 N (force-limited)
                a = torch.zeros(n, 5, device=dev)
                a[:, 2] = torch.where(fz > 3.0, torch.zeros_like(fz), torch.full_like(fz, -0.005 / env.limits.v_z))
            elif args.policy == "wrench_expert":
                depth_meas = (z_start - env.scene.link_pose("tcp")[:, 6]).clamp_min(0)
                out = wexp.act(env.hist[:, -1], cfg.entry_gap + cfg.target_depth, depth_meas)
                # its offsets -> velocities toward them
                tgt = torch.stack([out["xy"][:, 0], out["xy"][:, 1], -out["z"]], -1)
                cur = env.p_cmd - env.p_start
                v = ((tgt - cur) * 10.0)
                a = torch.cat([v[:, :2] / env.limits.v_lat, v[:, 2:3] / env.limits.v_z, torch.zeros(n, 2, device=dev)], -1)
            else:
                with torch.no_grad():
                    a_st = student(obs) if student is not None else a_exp
                a = torch.where(use_expert[:, None], a_exp, a_st)
            if not args.eval:
                rec["obs"].append(obs.cpu())
                rec["expert_action"].append(a_exp.cpu())
                rec["action"].append(a.cpu())
                rec["valid"].append((~env.done).cpu())
                rec["env"].append(torch.arange(n))
                rec["episode"].append(torch.full((n,), rnd))
                rec["t"].append(torch.full((n,), k / cfg.policy_hz))
            obs, info = env.step(a)
            if cam is not None:
                rgb = ru.grab(sim, cam, env.dt)
                pr_now = info["priv"]
                for i in range(n):
                    state = ["", "SEATED", "ABORTED (force limit)", "INVALID"][int(env.outcome[i])]
                    lines = [f"{vname}   env {i}   t {k / cfg.policy_hz:4.2f} s",
                             f"in-hand {env.inhand[i][0] * 1000:.1f} mm / {env.inhand[i][1]:.1f} deg, approach {float(env.approach[i].norm()) * 1000:.1f} mm",
                             f"Fz {float(env.hist[i, -1, 2]):5.2f} N  depth {float(pr_now['depth'][i]) * 1000:5.1f} mm  offset {float(pr_now['e_xy'][i].norm()) * 1000:4.2f} mm"]
                    tiles[i].append(ru.label(rgb[i], lines, state or None, True if state == "SEATED" else False if state else None))
            elif bool(env.done.all()):
                break
        pr = env.privileged()
        for i in range(n):
            episodes.append(dict(round=rnd, env=i, driver="expert" if (args.policy == "mix" and bool(use_expert[i])) else args.policy if args.policy != "mix" else "student",
                                 outcome=["timeout", "seated", "aborted", "invalid"][int(env.outcome[i])],
                                 time_s=None if math.isnan(float(env.t_done[i])) else round(float(env.t_done[i]), 3),
                                 peak_force_n=round(float(env.peak_f[i]), 2), final_depth_mm=round(float(pr["depth"][i]) * 1000, 2),
                                 inhand_offset_mm=round(env.inhand[i][0] * 1000, 2), inhand_tilt_deg=round(env.inhand[i][1], 2),
                                 approach_mm=round(float(env.approach[i].norm()) * 1000, 2)))
        print(f"[ROUND] {rnd}: seated {int((env.outcome == 1).sum())}/{n}, aborted {int((env.outcome == 2).sum())}, "
              f"{time.time() - t0:.0f} s", flush=True)
    out = REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    if cam is not None:
        for i in range(n):
            ru.write_mp4(tiles[i], REPO / args.video_dir / f"{vname}_env{i}.mp4")
        print("[VIDEO]", REPO / args.video_dir, n, "tiles", flush=True)
    if not args.eval:
        np.savez_compressed(out, **{k: torch.stack(v, 1).reshape(-1, *v[0].shape[1:]).numpy() for k, v in rec.items()})
    by = {}
    for drv in sorted({e["driver"] for e in episodes}):
        es = [e for e in episodes if e["driver"] == drv]
        k = sum(e["outcome"] == "seated" for e in es)
        ts = [e["time_s"] for e in es if e["outcome"] == "seated"]
        by[drv] = dict(n=len(es), seated=k, seated_ci95=wilson(k, len(es)), aborted=sum(e["outcome"] == "aborted" for e in es),
                       median_time_s=float(np.median(ts)) if ts else None,
                       median_peak_force_n=float(np.median([e["peak_force_n"] for e in es])))
    summary = dict(note="simulation, placeholder hardware parameters, not calibrated", policy=args.policy, beta=args.beta,
                   expert_style=args.expert_style,
                   student=str(args.student) if args.student else None, residual=str(args.residual) if args.residual else None,
                   radial_clearance_mm=round(gap * 1000, 3),
                   env_cfg=cfg.__dict__, by_driver=by, wall_s=round(time.time() - t0, 1), params=to_dict(p), episodes=episodes)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print("[RESULT]", json.dumps(dict(policy=args.policy, beta=args.beta, clearance_mm=round(gap * 1000, 2), by_driver=by)), flush=True)
    print("ROLLOUTS DONE", out, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("ROLLOUTS DONE (error)", flush=True)
    app.close()
