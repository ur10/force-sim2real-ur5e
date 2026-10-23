"""Compare contact-probe runs (plan step 5) and rank simulated parameter sets against a target run.

A probe run directory holds summary.json (conditions) and traces.npz with, per condition cNN:
  t (T,), tcp (T, 7) wxyz_xyz, w_sensor (T, 6) the reported wrench [F, tau] in the base frame
The real run must be written in the same layout (a logger can record RTDE actual_TCP_pose and
actual_TCP_force at the same rate; only the robot's own signals are used, never the ring pose).

Features per condition, all computable on the real robot:
  onset_mm      TCP travel from the start until |F| first exceeds the rest noise by 5 sigma
  jam_n         median axial force over the last 30% of the descent (0 when the ring went in)
  rise_n_per_mm slope of axial force vs TCP travel during the first 1.5 mm after onset: the
                series stiffness of in-hand grip, rod and arm (identifies in_hand.k_lin)
  moment_dir    angle of (-tau_y, tau_x) while jammed: where the contact is (the expert's signal)
  rest_std_n    axial-force std over the first 0.2 s (sensor noise)
    .venv/bin/python scripts/ur_contact/compare_probe.py --target outputs/ur_nut/contact_sim/probe \
        --candidates outputs/ur_nut/contact_sim/fit_*
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def features(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text())
    d = np.load(run / "traces.npz")
    out = {}
    for c in s["conditions"]:
        k = f"c{c['i']:02d}"
        t, tcp, w = d[f"{k}/t"], d[f"{k}/tcp"], d[f"{k}/w_sensor"]
        travel = (tcp[0, 6] - tcp[:, 6]) * 1000
        rest = t < 0.2
        sig = max(float(w[rest, 2].std()), 0.05)
        f = np.linalg.norm(w[:, :3], axis=1)
        hit = np.where((f > 5 * sig + abs(float(w[rest, 2].mean()))) & (t > 0.2))[0]
        onset = float(travel[hit[0]]) if len(hit) else None
        moving = np.where(np.diff(travel, prepend=travel[0]) > 0)[0]
        last = moving[int(len(moving) * 0.7):] if len(moving) else np.arange(len(t))
        jam = float(np.median(w[last, 2]))
        rise = None
        if onset is not None:
            win = (travel >= onset) & (travel <= onset + 1.5)
            if win.sum() > 5 and np.ptp(travel[win]) > 0.2:
                rise = float(np.polyfit(travel[win], w[win, 2], 1)[0])
        mdir = None
        if jam > 1.0:
            m = np.median(w[last, 3:5], axis=0)
            mdir = float(np.degrees(np.arctan2(m[0], -m[1])))
        out[(c["offset_mm"], c["tilt_deg"])] = dict(onset_mm=onset, jam_n=jam, rise_n_per_mm=rise, moment_dir_deg=mdir,
                                                     rest_std_n=sig, outcome=c["outcome"])
    return out


def distance(a: dict, b: dict) -> tuple[float, dict]:
    """Mismatch between two feature sets over the conditions they share (lower is better)."""
    terms = {"onset": [], "jam": [], "rise": [], "outcome": [], "dir": []}
    for key in set(a) & set(b):
        x, y = a[key], b[key]
        terms["outcome"].append(float(x["outcome"] != y["outcome"]))
        terms["jam"].append(((x["jam_n"] - y["jam_n"]) / 1.0) ** 2)
        if x["onset_mm"] is not None and y["onset_mm"] is not None:
            terms["onset"].append(((x["onset_mm"] - y["onset_mm"]) / 0.5) ** 2)
        elif (x["onset_mm"] is None) != (y["onset_mm"] is None):
            terms["onset"].append(4.0)
        if x["rise_n_per_mm"] is not None and y["rise_n_per_mm"] is not None:
            terms["rise"].append(np.log(max(x["rise_n_per_mm"], 1e-3) / max(y["rise_n_per_mm"], 1e-3)) ** 2)
        if x["moment_dir_deg"] is not None and y["moment_dir_deg"] is not None:
            dd = (x["moment_dir_deg"] - y["moment_dir_deg"] + 180) % 360 - 180
            terms["dir"].append((dd / 30) ** 2)
    parts = {k: float(np.mean(v)) if v else 0.0 for k, v in terms.items()}
    return sum(parts.values()), parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    ft = features(a.target)
    rows = []
    for c in a.candidates:
        if not (c / "traces.npz").exists():
            continue
        score, parts = distance(ft, features(c))
        params = json.loads((c / "summary.json").read_text()).get("params", {})
        rows.append(dict(run=str(c), score=round(score, 3), parts={k: round(v, 3) for k, v in parts.items()},
                         k_lin=params.get("in_hand", {}).get("k_lin"), friction=params.get("contact", {}).get("static_friction")))
    rows.sort(key=lambda r: r["score"])
    for r in rows:
        print(json.dumps(r))
    print(json.dumps({"target_features": {f"{k[0]}mm/{k[1]}deg": v for k, v in sorted(ft.items())}}, indent=0)[:3000])


if __name__ == "__main__":
    main()
