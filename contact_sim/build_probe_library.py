"""Turn the simulated probe sweep into the feature library the offline check compares against.

Each sweep run (run_contact_probe.py --controller stiff, one clearance x grip stiffness x friction)
becomes an entry: per-attempt features from robot_side/probe_features.py, computed from the same
signals the real export carries (TCP pose, compensated wrench in the base frame with torque about
the TCP, the ring bottom a fixed distance below the TCP), their bin summary and the capture range.

    .venv/bin/python scripts/ur_contact/build_probe_library.py outputs/ur_nut/probe_library \
        -o outputs/ur_nut/probe_library/library.json

Also: sim_run_to_export() writes a sim run in the real export format (flange TCP, torque about the
flange) so probe_check.py can be tested end to end on held-out simulated data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "robot_side"))
from probe_features import attempt_features, clearance_estimate, quat_to_mat, summarise_bins  # noqa: E402

FLANGE_TO_TCP = 0.20  # contact_utils.FLANGE_TO_TCP: the sim's tcp link is 200 mm below the flange
NAME = re.compile(r"c(?P<c>[\d.]+)_k(?P<k>[\d.]+)_mu(?P<mu>[\d.]+)")


def run_features(run: Path) -> tuple[dict, list[dict]]:
    s = json.loads((run / "summary.json").read_text())
    g, ih = s["params"]["geometry"], s["params"]["in_hand"]
    below = ih["ring_bottom_below_flange"] - FLANGE_TO_TCP
    d = np.load(run / "traces.npz")
    rows = []
    for c in s["conditions"]:
        k = f"c{c['i']:02d}"
        f = attempt_features(d[f"{k}/t"], d[f"{k}/tcp"], d[f"{k}/w_sensor"], below, g["rod_tip_z"], g["rod_xy"])
        f.update(outcome=c["outcome"], cmd_offset_mm=c["offset_mm"], cmd_tilt_deg=c["tilt_deg"])
        rows.append(f)
    return s, rows


def keyed(bins: dict) -> dict:
    return {f"{o:g}|{t:g}": v for (o, t), v in sorted(bins.items())}


def build(root: Path) -> dict:
    entries = []
    for run in sorted(p for p in root.iterdir() if (p / "summary.json").exists()):
        m = NAME.fullmatch(run.name)
        if not m:
            continue
        s, rows = run_features(run)
        entries.append(dict(name=run.name, clearance_mm=float(m["c"]), k_scale=float(m["k"]), mu=float(m["mu"]),
                            bins=keyed(summarise_bins(rows)), capture=clearance_estimate(rows),
                            attempts=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]))
    p0 = json.loads((root / entries[0]["name"] / "summary.json").read_text())["params"]
    return dict(note="SIMULATION library (Isaac Lab, fin-ray grasp model, measured parameters); not real data",
                protocol=dict(controller="stiff position control, stop at 8 N filtered force", speed_mm_s=5.0,
                              offsets_mm=sorted({a["cmd_offset_mm"] for a in entries[0]["attempts"]}),
                              tilts_deg=sorted({a["cmd_tilt_deg"] for a in entries[0]["attempts"]})),
                grid=dict(clearance_mm=sorted({e["clearance_mm"] for e in entries}), k_scale=sorted({e["k_scale"] for e in entries}),
                          mu=sorted({e["mu"] for e in entries})),
                k_nominal=dict(k_lin_xyz=[6000, 1500, 2500], k_ang_xyz=[0.3, 1.0, 0.3]),
                sim_sensor=p0["sensor"], geometry=p0["geometry"], in_hand=p0["in_hand"], entries=entries)


def sim_run_to_export(run: Path, out: Path, registration_error_m=(0.0, 0.0)):
    """Write a simulated run as a real export (summary.json + traces.npz, flange TCP)."""
    s = json.loads((run / "summary.json").read_text())
    g = s["params"]["geometry"]
    d = np.load(run / "traces.npz")
    out.mkdir(parents=True, exist_ok=True)
    tr, conds = {}, []
    for c in s["conditions"]:
        k = f"c{c['i']:02d}"
        tcp, w = d[f"{k}/tcp"].astype(float), d[f"{k}/w_sensor"].astype(float)
        z = quat_to_mat(tcp[:, :4])[:, :, 2]
        p_fl = tcp[:, 4:] - FLANGE_TO_TCP * z
        tau_fl = w[:, 3:] + np.cross(tcp[:, 4:] - p_fl, w[:, :3])
        tr[f"{k}/t"] = d[f"{k}/t"] - 0.5
        tr[f"{k}/tcp"] = np.concatenate([tcp[:, :4], p_fl], 1)
        tr[f"{k}/w_sensor"] = np.concatenate([w[:, :3], tau_fl], 1)
        conds.append(dict(i=c["i"], offset_mm=c["offset_mm"], tilt_deg=c["tilt_deg"], speed_mm_s=c["speed_mm_s"],
                          outcome=c["outcome"], final_depth_mm=c["final_depth_mm"], flags=[]))
    xy = [g["rod_xy"][0] + registration_error_m[0], g["rod_xy"][1] + registration_error_m[1]]
    (out / "summary.json").write_text(json.dumps(dict(note=f"SIMULATED export from {run}", rod_axis=dict(xy=xy, tip_z=g["rod_tip_z"]),
                                                      conditions=conds), indent=1))
    np.savez_compressed(out / "traces.npz", **tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    a = ap.parse_args()
    lib = build(a.root)
    a.output.write_text(json.dumps(lib, indent=1))
    for e in lib["entries"]:
        cap = e["capture"]
        print(f"{e['name']:18s} seated {sum(x['outcome'] == 'seated' for x in e['attempts']):2d}/{len(e['attempts'])}  "
              f"capture {cap.get('clearance_mm')} (seated <= {cap.get('largest_seated_mm')}, jammed >= {cap.get('smallest_jammed_mm')})")
    print(f"wrote {a.output} ({a.output.stat().st_size / 1e3:.0f} kB, {len(lib['entries'])} entries)")


if __name__ == "__main__":
    main()
