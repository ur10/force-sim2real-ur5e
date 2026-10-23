"""Export a trained policy for the robot machine: a numpy-only bundle (no torch there).

Contents: the student MLP (weights, normalisation, masks), the trained constants (rates, limits,
filter, entry gap, target depth), the nominal grasp (nut frame in the tcp frame, from the generated
demo's export) and the simulated entry orientation, plus reference input/output pairs so the robot
side can check its numpy forward pass against torch.

    .venv/bin/python scripts/ur_contact/export_deploy_bundle.py \
        --student outputs/ur_nut/measured/dagger_kinematic/student_it4.pt --vision kinematic \
        --params outputs/ur_nut/contact/measured.json -o outputs/ur_nut/deploy/bundle_dagger_kinematic.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "robot_side" / "deploy"))
import student as st  # noqa: E402
from contact_utils import M, grasp_T_tcp_obj  # noqa: E402
from deploy_core import SimConstants  # noqa: E402
from params import load, to_dict  # noqa: E402
from privileged_expert import ActionLimits  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def sim_entry(p, export, entry_gap):
    """contact_env.ContactEnv.__init__: nominal grasp and entry tcp pose."""
    g = p.geometry
    ex = np.load(export)
    obj = ex["T_base_obj"]
    bottom = obj[:, 6] - g.ring_height / 2
    near = np.where((np.hypot(obj[:, 4] - g.rod_xy[0], obj[:, 5] - g.rod_xy[1]) < 5e-4) & (bottom < g.rod_tip_z + entry_gap + 2e-3))[0]
    k = int(near[0])
    Tto = grasp_T_tcp_obj(M(ex["T_tcp_obj"]), g, p.in_hand)
    T_ring = M(obj[k])
    T_ring[:3, 3] = [g.rod_xy[0], g.rod_xy[1], g.rod_tip_z + entry_gap + g.ring_height / 2]
    return Tto, T_ring @ np.linalg.inv(Tto)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", type=Path, default=None, help="DAgger student (or the residual's base student)")
    ap.add_argument("--residual", type=Path, default=None, help="residual.pt from run_residual_rl.py: exports the residual on its base")
    ap.add_argument("--vision", choices=["kinematic", "oracle"], required=True, help="what the student was trained with")
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--export", type=Path, default=Path("outputs/ur_nut/physical_export/nut_nominal_000.npz"))
    ap.add_argument("--shard", type=Path, default=None, help="DAgger shard for the reference inputs")
    ap.add_argument("--results", type=str, default="", help="simulation results to record in the bundle (text)")
    ap.add_argument("-o", "--output", type=Path, required=True)
    a = ap.parse_args()
    ck_res = None
    if a.residual is not None:
        ck_res = torch.load(REPO / a.residual, map_location="cpu", weights_only=False)
        if ck_res["base"] == "student":
            a.student = Path(ck_res["student"])
        cfg_json = (REPO / a.residual).parent / "config.json"
        if cfg_json.exists():
            vs = json.loads(cfg_json.read_text()).get("env", {}).get("vision_source", "oracle")
            if vs != a.vision:
                print(f"NOTE: the residual was trained with vision_source={vs}; recording that instead of --vision {a.vision}")
                a.vision = vs
    if a.student is None:
        raise SystemExit("--student or --residual needed")
    if a.vision != "kinematic":
        print("WARNING: an oracle-vision student needs an in-hand tracker on the robot; the deployment only builds the kinematic estimate")
    p = load(REPO / a.params)
    k = SimConstants()
    L = ActionLimits()
    assert (k.v_lat, k.v_z, k.w) == (L.v_lat, L.v_z, L.w), "deploy_core.SimConstants out of date with ActionLimits"
    assert abs(k.lowpass_hz - p.sensor.lowpass_hz) < 1e-9 and abs(k.ctrl_hz - p.controller.rate_hz) < 1e-9
    Tto, T_entry = sim_entry(p, REPO / a.export, k.entry_gap)
    ck = torch.load(REPO / a.student, map_location="cpu", weights_only=False)
    sd = ck["state"]
    lin = sorted({kk.split(".")[1] for kk in sd if kk.startswith("net.") and kk.endswith(".weight")}, key=int)
    out = {f"W{i}": sd[f"net.{j}.weight"].numpy().astype(np.float64) for i, j in enumerate(lin)}
    out.update({f"b{i}": sd[f"net.{j}.bias"].numpy().astype(np.float64) for i, j in enumerate(lin)})
    out.update(n_layers=np.array(len(lin)), mu=sd["mu"].numpy().astype(np.float64), sd=sd["sd"].numpy().astype(np.float64),
               mask_wrench=np.array(bool(ck["mask_wrench"])), mask_vision=np.array(bool(ck["mask_vision"])),
               R_entry=T_entry[:3, :3], R_tcp_obj=Tto[:3, :3], xy_tcp_obj=Tto[:2, 3], ring_height=np.array(p.geometry.ring_height),
               sim_rod_xy=np.array(p.geometry.rod_xy), sim_rod_tip_z=np.array(p.geometry.rod_tip_z), sim_T_entry=T_entry)
    # reference pairs for the numpy forward pass
    model = st.load(REPO / a.student, "cpu")
    if a.shard is not None:
        obs = np.load(REPO / a.shard)["obs"]
        ref = obs[np.random.default_rng(0).choice(len(obs), 64, replace=False)]
    else:
        ref = np.random.default_rng(0).normal(size=(64, st.OBS_DIM)) * sd["sd"].numpy() + sd["mu"].numpy()
    with torch.no_grad():
        out["ref_obs"] = ref.astype(np.float64)
        out["ref_act"] = model(torch.tensor(ref, dtype=torch.float32)).numpy().astype(np.float64)
    meta = dict(note="Policy bundle for the UR5e deployment. Trained in SIMULATION (Isaac Lab) with provisional parameters.",
                student=str(a.student), vision=a.vision, params=str(a.params), export=str(a.export), sim_results=a.results,
                constants=k.__dict__, ring_bottom_below_flange_trained=p.in_hand.ring_bottom_below_flange,
                geometry=to_dict(p)["geometry"])
    if ck_res is not None:
        from ppo import ActorCritic
        ms = ck_res["model"]
        alin = sorted({kk.split(".")[1] for kk in ms if kk.startswith("actor.") and kk.endswith(".weight")}, key=int)
        out.update({f"rW{i}": ms[f"actor.{j}.weight"].numpy().astype(np.float64) for i, j in enumerate(alin)})
        out.update({f"rb{i}": ms[f"actor.{j}.bias"].numpy().astype(np.float64) for i, j in enumerate(alin)})
        ac = ActorCritic(ck_res["obs_dim"], ck_res["priv_dim"], 5)
        ac.load_state_dict(ms)
        ac.eval()
        out.update(n_res_layers=np.array(len(alin)), r_mean=ms["obs_norm.mean"].numpy().astype(np.float64),
                   r_var=ms["obs_norm.var"].numpy().astype(np.float64), r_clip=np.array(float(ac.obs_norm.clip)),
                   res_scale=np.array(ck_res["res_scale"], float), res_base=np.array(ck_res["base"]))
        with torch.no_grad():
            o = torch.tensor(out["ref_obs"], dtype=torch.float32)
            if ck_res["base"] == "student":
                base = model(o)
            else:
                from base_policies import replay_base
                base = replay_base(o[:, 6 + 6 * 7 + 2], ActionLimits().v_z)
            r = ac.dist(torch.cat([o, base], -1)).mean
            out["ref_act_res"] = (base + torch.tensor(ck_res["res_scale"]) * r.clamp(-1, 1)).clamp(-1, 1).numpy().astype(np.float64)
        meta.update(residual=str(a.residual), residual_base=ck_res["base"])
    out["meta_json"] = np.array(json.dumps(meta))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, **out)
    print(f"wrote {a.output} ({a.output.stat().st_size / 1e3:.0f} kB): {len(lin)} layers, masks wrench={ck['mask_wrench']} vision={ck['mask_vision']}")
    print("entry tcp rotation\n", np.round(T_entry[:3, :3], 5), "\nnut in tcp\n", np.round(Tto, 5))


if __name__ == "__main__":
    main()
