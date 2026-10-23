"""Version 2: force-aware residual RL on the nut contact phase (simulation, uncalibrated).

A frozen base (the generated demo's insertion executed blindly, or a DAgger student) plus a
bounded residual trained with PPO. The actor sees only deployable inputs (the student's
observations + the base action); the critic also sees the true ring-vs-rod offset, depth and tilt.
Reward: residual_reward.py. Episodes run in lockstep; time-limit ends are bootstrapped.

    scripts/ur_cube/isaac_run.sh outputs/ur_nut/residual/train.log "RL DONE" scripts/ur_contact/run_residual_rl.py \
        --minutes 45 --output outputs/ur_nut/residual/replay_base
"""

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--minutes", type=float, default=45.0, help="wall-clock training budget")
parser.add_argument("--rollout", type=int, default=48, help="policy ticks per PPO update (divides the episode)")
parser.add_argument("--base", choices=["replay", "student"], default="replay")
parser.add_argument("--student", type=Path, default=None)
parser.add_argument("--res-scale", type=float, nargs=5, default=(0.6, 0.6, 0.6, 0.5, 0.5))
parser.add_argument("--lambda-residual", type=float, default=0.1)
parser.add_argument("--checkpoint-every-min", type=float, default=15.0)
parser.add_argument("--lr-anneal", action="store_true", help="decay the learning rate linearly to 10%% over the budget")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--params", type=Path, default=None)
parser.add_argument("--set", type=str, default='{"geometry": {"ring_bore_r": 0.0083}}')
parser.add_argument("--env", type=str, default=None)
parser.add_argument("--export", type=Path, default=Path("outputs/ur_nut/physical_export/nut_nominal_000.npz"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from base_policies import priv_vector, replay_base  # noqa: E402
from contact_env import ContactEnv, EnvCfg  # noqa: E402
from contact_utils import URDF  # noqa: E402
from params import load, to_dict  # noqa: E402
from ppo import PPO, ActorCritic, PPOCfg  # noqa: E402
from residual_reward import RewardCfg, potential, step_reward, wrench_cost  # noqa: E402
from ring_rod_sim import physx_cfg  # noqa: E402
import student as st  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main():
    torch.manual_seed(args.seed)
    p = load(args.params, json.loads(args.set) if args.set else None)
    ecfg = EnvCfg(**(json.loads(args.env) if args.env else {}))
    sim = sim_utils.SimulationContext(physx_cfg(p))
    env = ContactEnv(sim, p, args.num_envs, REPO / args.export, URDF, ecfg, seed=args.seed, device=args.device)
    sim.reset()
    env.post_reset()
    dev, n = args.device, args.num_envs
    gap = p.geometry.radial_clearance
    rcfg = RewardCfg(dt=1 / ecfg.policy_hz, target_depth=ecfg.target_depth, lambda_residual=args.lambda_residual)
    env.wrench_cost_fn = lambda w: wrench_cost(rcfg, w)
    base_student = st.load(REPO / args.student, dev) if args.base == "student" else None
    res_scale = torch.tensor(args.res_scale, device=dev)
    obs_dim, priv_dim = st.OBS_DIM + 5, 6
    model = ActorCritic(obs_dim, priv_dim, 5).to(dev)
    algo = PPO(model, PPOCfg(gamma=rcfg.gamma))
    ep_ticks = int(ecfg.episode_s * ecfg.policy_hz)
    assert ep_ticks % args.rollout == 0, "rollout must divide the episode length"
    out = REPO / args.output
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.jsonl", "w")

    def base_action(obs):
        if base_student is not None:
            with torch.no_grad():
                return base_student(obs)
        return replay_base(env.hist[:, -1, 2], env.limits.v_z)

    t0, it, samples, tick = time.time(), 0, 0, 0
    next_ckpt = args.checkpoint_every_min
    lr0 = algo.cfg.lr
    finished = []
    obs = None
    while (time.time() - t0) / 60 < args.minutes:
        if tick == 0:
            obs = env.reset()
            priv = env.privileged()
            phi = potential(rcfg, priv["e_xy"], priv["depth"], gap)
            res_prev = torch.zeros(n, 5, device=dev)
        buf = {k: [] for k in ("x", "pv", "res", "logp", "rew", "val", "done", "valid")}
        terms = {k: 0.0 for k in ("progress", "costs", "c_wrench", "c_residual")}
        for _ in range(args.rollout):
            base = base_action(obs)
            x = torch.cat([obs, base], -1)
            pv = priv_vector(priv)
            model.obs_norm.update(x)
            model.crit_norm.update(torch.cat([x, pv], -1))
            res, logp = model.act(x)
            val = model.value(x, pv).detach()
            valid = ~env.done
            a = (base + res_scale * res.clamp(-1, 1)).clamp(-1, 1)
            obs, info = env.step(a)
            priv = info["priv"]
            phi_next = potential(rcfg, priv["e_xy"], priv["depth"], gap)
            newly = valid & env.done
            success = newly & (env.outcome == 1)
            failure = newly & ((env.outcome == 2) | (env.outcome == 3))
            r = step_reward(rcfg, phi, phi_next, success, failure, info["c_wrench"], res.clamp(-1, 1), res_prev)
            for k in terms:
                terms[k] += float(torch.where(valid, r[k], torch.zeros_like(r[k])).sum())
            for k, v in (("x", x), ("pv", pv), ("res", res), ("logp", logp), ("rew", torch.where(valid, r["reward"], torch.zeros_like(r["reward"]))),
                         ("val", val), ("done", success | failure), ("valid", valid)):
                buf[k].append(v)
            phi, res_prev = phi_next, res.clamp(-1, 1)
            tick += 1
            samples += int(valid.sum())
        with torch.no_grad():
            x_next = torch.cat([obs, base_action(obs)], -1)
            trunc_val = model.value(x_next, priv_vector(priv))
        B = {k: torch.stack(v) for k, v in buf.items()}
        adv, ret = algo.gae(B["rew"], B["val"], None, B["done"], B["valid"], trunc_val)
        if args.lr_anneal:
            frac = min((time.time() - t0) / 60 / args.minutes, 1.0)
            for gp in algo.opt.param_groups:
                gp["lr"] = lr0 * (1 - 0.9 * frac)
        stats = algo.update(B["x"], B["pv"], B["res"], B["logp"], adv, ret, B["valid"])
        stats["lr"] = algo.opt.param_groups[0]["lr"]
        if (time.time() - t0) / 60 >= next_ckpt:
            torch.save(dict(model=model.state_dict(), res_scale=list(args.res_scale), base=args.base,
                            student=str(args.student) if args.student else None, obs_dim=obs_dim, priv_dim=priv_dim,
                            minutes=next_ckpt, samples=samples), out / f"residual_{int(next_ckpt)}m.pt")
            print(f"[CKPT] {int(next_ckpt)} min, {samples} samples", flush=True)
            next_ckpt += args.checkpoint_every_min
        if tick == ep_ticks or bool(env.done.all()):  # all episodes finished early: reset now
            finished.append(dict(seated=int((env.outcome == 1).sum()), aborted=int((env.outcome == 2).sum()),
                                 invalid=int((env.outcome == 3).sum()), n=n))
            tick = 0
        nv = max(int(B["valid"].sum()), 1)
        last = finished[-1] if finished else None
        row = dict(iter=it, samples=samples, wall_min=round((time.time() - t0) / 60, 2), reward_per_step=round(float(B["rew"].sum()) / nv, 4),
                   progress=round(terms["progress"] / nv, 4), costs=round(terms["costs"] / nv, 5), c_wrench=round(terms["c_wrench"] / nv, 4),
                   residual_sq=round(terms["c_residual"] / nv, 4), episodes=len(finished),
                   last_episode_success=None if last is None else round(last["seated"] / last["n"], 3),
                   last_episode_aborted=None if last is None else last["aborted"], **{k: (round(v, 6) if k == "lr" else round(v, 4)) for k, v in stats.items() if k != "n"})
        log.write(json.dumps(row) + "\n")
        log.flush()
        if it % 5 == 0 or tick == 0:
            print("[ITER]", json.dumps(row), flush=True)
        if it % 25 == 0:
            torch.save(dict(model=model.state_dict(), res_scale=list(args.res_scale), base=args.base,
                            student=str(args.student) if args.student else None, obs_dim=obs_dim, priv_dim=priv_dim), out / "residual.pt")
        it += 1
    torch.save(dict(model=model.state_dict(), res_scale=list(args.res_scale), base=args.base,
                    student=str(args.student) if args.student else None, obs_dim=obs_dim, priv_dim=priv_dim), out / "residual.pt")
    (out / "config.json").write_text(json.dumps(dict(note="simulation, placeholder hardware parameters, not calibrated",
                                                     args={k: str(v) for k, v in vars(args).items()}, reward=rcfg.__dict__,
                                                     env=ecfg.__dict__, params=to_dict(p), episodes=finished), indent=1))
    print(f"RL DONE {out}: {it} iterations, {samples} samples, {len(finished)} episode rounds", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("RL DONE (error)", flush=True)
    app.close()
