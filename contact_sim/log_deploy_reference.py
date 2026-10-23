"""Reference log for the deployment equivalence test (simulation, Isaac).

Runs the kinematic-estimate DAgger student for one round in ContactEnv with the estimate's noise and
registration bias off (so it is deterministic) and records, per sensor step, the delayed tared
wrench channel and its low-pass output with the tool0 / tcp poses, and per policy tick the
observation, action and commanded pose. robot_side/deploy/test_deploy.py rebuilds the observations
from these with deploy_core and checks they match.

    scripts/ur_cube/isaac_run.sh outputs/ur_nut/deploy/reference.log "REFERENCE DONE" scripts/ur_contact/log_deploy_reference.py \
        --student outputs/ur_nut/measured/dagger_kinematic/student_it4.pt -o outputs/ur_nut/deploy/reference.npz
"""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--student", type=Path, required=True)
parser.add_argument("--params", type=Path, default=Path("outputs/ur_nut/contact/measured.json"))
parser.add_argument("--set", type=str, default='{"geometry": {"ring_bore_r": 0.0085}}')
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("-o", "--output", type=Path, required=True)
parser.add_argument("--export", type=Path, default=Path("outputs/ur_nut/physical_export/nut_nominal_000.npz"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from contact_env import ContactEnv, EnvCfg  # noqa: E402
from contact_utils import URDF  # noqa: E402
from params import load  # noqa: E402
from ring_rod_sim import physx_cfg  # noqa: E402
import student as st  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main():
    p = load(REPO / args.params, json.loads(args.set))
    cfg = EnvCfg(vision_source="kinematic", vision_noise=0.0, registration_bias=0.0)
    sim = sim_utils.SimulationContext(physx_cfg(p))
    env = ContactEnv(sim, p, args.num_envs, REPO / args.export, URDF, cfg, seed=args.seed, device=args.device)
    sim.reset()
    env.post_reset()
    pol = st.load(REPO / args.student, args.device)
    rec = {k: [] for k in ("s_comp", "s_filt", "s_tool0", "s_tcp", "s_tick")}
    tick = {"k": -1}
    orig = env._sensor_step

    def logged():
        w, ch = orig()
        rec["s_comp"].append(ch["compensated"].cpu().numpy())
        rec["s_filt"].append(ch["filtered"].cpu().numpy())
        rec["s_tool0"].append(env.scene.link_pose("tool0").cpu().numpy())
        rec["s_tcp"].append(env.scene.link_pose("tcp").cpu().numpy())
        rec["s_tick"].append(tick["k"])
        return w, ch

    env._sensor_step = logged
    obs = env.reset()
    T = {k: [] for k in ("obs", "action", "p_cmd", "R_cmd", "tcp", "tool0", "done", "outcome")}

    def snap(o, a):
        T["obs"].append(o.cpu().numpy())
        T["action"].append(a.cpu().numpy())
        T["p_cmd"].append(env.p_cmd.cpu().numpy())
        T["R_cmd"].append(env.R_cmd.cpu().numpy())
        T["tcp"].append(env.scene.link_pose("tcp").cpu().numpy())
        T["tool0"].append(env.scene.link_pose("tool0").cpu().numpy())
        T["done"].append(env.done.cpu().numpy())
        T["outcome"].append(env.outcome.cpu().numpy())

    p_start = env.p_start.cpu().numpy()
    steps = int(cfg.episode_s * cfg.policy_hz)
    for k in range(steps):
        tick["k"] = k
        with torch.no_grad():
            a = pol(obs)
        snap(obs, a)
        obs, _ = env.step(a)
        if bool(env.done.all()):
            break
    tick["k"] = k + 1
    snap(obs, torch.zeros_like(a))
    out = {f"tick_{k}": np.stack(v) for k, v in T.items()}
    out.update({k: (np.stack(v) if k != "s_tick" else np.array(v)) for k, v in rec.items()})
    out.update(p_start=p_start, rod_xy=np.array(p.geometry.rod_xy), rod_tip_z=np.array(p.geometry.rod_tip_z),
               ring_height=np.array(p.geometry.ring_height), ring_bottom_below_flange=np.array(p.in_hand.ring_bottom_below_flange),
               T_tcp_obj=env.T_tcp_obj_nom.cpu().numpy(), sensor_dt=np.array(env.dt), alpha=np.array(env.sensor.alpha),
               outcome=env.outcome.cpu().numpy())
    (REPO / args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REPO / args.output, **out)
    print(f"REFERENCE DONE {args.output}: {k + 1} ticks, outcomes {env.outcome.tolist()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("REFERENCE DONE (error)", flush=True)
    app.close()
