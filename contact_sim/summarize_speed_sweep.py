"""Summarise the speed x impedance probe sweep (run_contact_probe.py, admittance controller) and the
privileged-expert speed sweep (run_dagger_rollouts.py --env '{"speed_scale": s}'). Simulation.

    .venv/bin/python scripts/ur_contact/summarize_speed_sweep.py outputs/ur_nut/measured -o outputs/ur_nut/measured/speed_summary.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

SEAT_MM = 12.0  # run_contact_probe.py: seated = final depth > target (15 mm) - 3 mm


def probe_run(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text())
    d = np.load(run / "traces.npz")
    rows = []
    for c in s["conditions"]:
        k = f"c{c['i']:02d}"
        t, dep = d[f"{k}/t"], d[f"{k}/depth"]
        hit = np.where(dep * 1e3 >= SEAT_MM)[0]
        rows.append(dict(offset_mm=c["offset_mm"], tilt_deg=c["tilt_deg"], outcome=c["outcome"], peak_n=c["peak_force_n"],
                         peak_axial_n=c["peak_axial_n"], t_seat_s=float(t[hit[0]]) if (c["outcome"] == "seated" and len(hit)) else None))
    seated = [r for r in rows if r["outcome"] == "seated"]
    ts = [r["t_seat_s"] for r in seated if r["t_seat_s"] is not None]
    return dict(n=len(rows), seated=len(seated), aborted=sum(r["outcome"] == "aborted" for r in rows),
                seated_offsets_mm=sorted({r["offset_mm"] for r in seated}),
                peak_n_median=float(np.median([r["peak_n"] for r in rows])), peak_n_max=float(max(r["peak_n"] for r in rows)),
                peak_axial_n_median=float(np.median([r["peak_axial_n"] for r in rows])),
                t_seat_s_median=float(np.median(ts)) if ts else None, rows=rows)


def result_line(log: Path):
    for line in log.read_text().splitlines():
        if line.startswith("[RESULT]"):
            return json.loads(line[len("[RESULT] "):])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    a = ap.parse_args()
    out = dict(note="SIMULATION, measured parameters, fin-ray grasp, 0.5 mm radial clearance")
    probe = {}
    for run in sorted((a.root / "speed_impedance").iterdir()):
        m = re.fullmatch(r"v([\d.]+)_d([\d.]+)", run.name)
        if m and (run / "summary.json").exists():
            r = probe_run(run)
            probe[run.name] = dict(speed_mm_s=float(m[1]), damping_scale=float(m[2]), **{k: v for k, v in r.items() if k != "rows"})
            print(f"probe v {m[1]:>4} mm/s  damping x{m[2]:<3}  seated {r['seated']:2d}/{r['n']}  offsets {r['seated_offsets_mm']}  "
                  f"peak median {r['peak_n_median']:.2f} N max {r['peak_n_max']:.2f} N  t_seat {r['t_seat_s_median']}")
    out["probe"] = probe
    exp = {}
    xd = a.root / "expert_speed"
    if xd.exists():
        for log in sorted(xd.glob("s*.log")):
            r = result_line(log)
            if r:
                b = next(iter(r["by_driver"].values()))
                exp[log.stem] = dict(speed_scale=float(log.stem[1:]), **b)
                print(f"expert x{log.stem[1:]}: {b}")
    out["expert"] = exp
    a.output.write_text(json.dumps(out, indent=1))
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
