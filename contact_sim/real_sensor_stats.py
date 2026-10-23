"""Sensor numbers from a real ring_rod_probe session, next to the simulated sensor model (numpy only).

Reads session.npz + events.jsonl (still_* / free_* marks from collect_contact.py) and reports per
segment: tool tilt, mean force, per-axis force/torque std, drift (linear slope), the rest-noise
statistic probe_check.py uses (std of |F - F_rest| over 0.2 s windows), and how far the TCP moved.

    .venv/bin/python scripts/ur_contact/real_sensor_stats.py <session dir> --params outputs/ur_nut/contact/measured.json \
        -o outputs/ur_nut/contact/sensor_identified.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from params import load, to_dict  # noqa: E402


def rotvec_mat(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def segments(events):
    out, st = [], None
    for e in events:
        if e["type"] in ("still_start", "free_start"):
            st = e
        elif e["type"] in ("still_end", "free_end") and st is not None:
            out.append((st["type"].split("_")[0], st["idx"], e["idx"]))
            st = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--params", type=Path, default=None)
    ap.add_argument("--channel", default="actual_TCP_force")
    ap.add_argument("-o", "--output", type=Path, required=True)
    a = ap.parse_args()
    d = np.load(a.session / "session.npz")
    ev = [json.loads(line) for line in (a.session / "events.jsonl").read_text().splitlines() if line.strip()]
    W, ts, pose = d[a.channel], d["timestamp"], d["actual_TCP_pose"]
    rate = float(1 / np.median(np.diff(ts)))
    n02 = int(round(0.2 * rate))
    rows = []
    for kind, i0, i1 in segments(ev):
        w, t = W[i0:i1], ts[i0:i1] - ts[i0]
        R = rotvec_mat(pose[i0, 3:])
        rest = [float(np.linalg.norm(w[j:j + n02, :3] - w[j:j + n02, :3].mean(0), axis=1).std()) for j in range(0, len(w) - n02, n02)]
        rows.append(dict(kind=kind, duration_s=round(float(t[-1]), 2), tilt_deg=round(float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1)))), 2),
                         mean_force_n=np.round(w[:, :3].mean(0), 3).tolist(), force_std_n=np.round(w[:, :3].std(0), 4).tolist(),
                         torque_std_nm=np.round(w[:, 3:].std(0), 5).tolist(),
                         drift_n_per_s=np.round([np.polyfit(t, w[:, k], 1)[0] for k in range(3)], 5).tolist(),
                         rest_stat_n=round(float(np.median(rest)), 4) if rest else None,
                         moved_mm=round(float(np.linalg.norm(pose[i0:i1, :3] - pose[i0, :3], axis=1).max() * 1e3), 2)))
    still = [r for r in rows if r["kind"] == "still"]
    free = [r for r in rows if r["kind"] == "free"]
    summ = dict(force_std_n_per_axis_median=np.round(np.median([r["force_std_n"] for r in still], 0), 4).tolist() if still else None,
                torque_std_nm_per_axis_median=np.round(np.median([r["torque_std_nm"] for r in still], 0), 5).tolist() if still else None,
                drift_n_per_s_max_abs=float(np.max(np.abs([r["drift_n_per_s"] for r in still]))) if still else None,
                rest_stat_n_median=float(np.median([r["rest_stat_n"] for r in still])) if still else None,
                bias_spread_over_orientations_n=np.round(np.ptp([r["mean_force_n"] for r in still], 0), 3).tolist() if len(still) > 1 else None,
                free_motion_force_std_n_per_axis_median=np.round(np.median([r["force_std_n"] for r in free], 0), 4).tolist() if free else None)
    sim = to_dict(load(a.params))["sensor"] if a.params else None
    out = dict(note="REAL data (UR5e built-in wrench, RTDE); 'sim_model' is the simulated sensor model for comparison",
               session=str(a.session), channel=a.channel, rate_hz=round(rate, 2), payload_kg=float(d["payload"][0]),
               payload_cog_m=d["payload_cog"][0].tolist(), tcp_offset=d["tcp_offset"][0].tolist(), segments=rows, summary=summ, sim_model=sim)
    a.output.write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))
    if sim:
        print(f"sim model: force noise {sim['force_noise_n']} N/axis, torque noise {sim['torque_noise_nm']} N m, drift {sim['drift_n_per_s']} N/s, "
              f"rate {sim['rate_hz']} Hz, latency {sim['latency_s']} s")


if __name__ == "__main__":
    main()
