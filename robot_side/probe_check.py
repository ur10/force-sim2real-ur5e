"""Offline check: compare a real probe export with the simulated probe library (numpy only).

Reads the export (summary.json + traces.npz from scripts/export_probe.py), the measured grasp
(config/measured.json: grasp.ring_bottom_below_tcp_mm, relative to the TCP the export logs) and the
library (sim_library/probe_library.json, built on the simulation machine). No robot connection.

    python scripts/probe_check.py outputs/export --measured config/measured.json \
        --library sim_library/probe_library.json -o outputs/probe_check

Steps
  1. Contact features per attempt (probe_features.py, same code as the library).
  2. Height check: attempts that land on the rod top (jammed, >= 3.5 mm off) should start touching
     where the simulation does in the same bin. The median difference is the combined error of
     grasp.ring_bottom_below_tcp_mm and the registered tip z; it is reported and removed before the
     comparison, by default as a grasp-length error (--height-correction tip|none).
  3. Registration check from outcomes (level attempts in several directions): the rod-axis shift
     that best separates seated from jammed attempts by distance, and that distance (the capture
     radius: clearance + chamfer + what the grip compliance absorbs). Reported, not applied unless
     --apply-registration-fit.
  4. Per bin (offset x tilt) seat fraction, onset height, force rise per mm, moment direction;
     mismatch score against every library entry (clearance x grip stiffness x friction); ranking
     and per-parameter marginals.
Writes report.json and report.md. Everything compared against is simulation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_features import attempt_features, bin_key, clearance_estimate, score, summarise_bins  # noqa: E402

LEVEL_DEG = 1.5


def load_export(d: Path):
    s = json.loads((d / "summary.json").read_text())
    tr = np.load(d / "traces.npz")
    return s, tr


def real_features(s: dict, tr, below: float) -> list[dict]:
    ax = s["rod_axis"]
    width = max(2, len(str(max(len(s["conditions"]) - 1, 0))))  # as utils/export.py names the traces
    rows = []
    for c in s["conditions"]:
        k = f"c{int(c['i']):0{width}d}"
        if f"{k}/t" not in tr:
            continue
        f = attempt_features(tr[f"{k}/t"], tr[f"{k}/tcp"], tr[f"{k}/w_sensor"], below, ax["tip_z"], ax["xy"])
        f.update(i=c["i"], outcome=c.get("outcome"), flags=c.get("flags", []))
        rows.append(f)
    return rows


def top_landings(rows: list[dict], min_offset_mm: float) -> list[dict]:
    """Jammed attempts far enough off the axis that the ring lands on the rod top (any tilt)."""
    return [r for r in rows if r["contact"] and r["offset_mm"] >= min_offset_mm and r["onset_height_mm"] is not None
            and r.get("outcome") == "jammed"]


def height_bias(rows: list[dict], lib: dict, min_offset_mm: float) -> dict:
    """Median onset height of the top landings minus the library's onset in the same (offset, tilt)
    bin (median over entries): removes the tilt geometry (a tilted ring touches with its low rim)."""
    ref = {}
    for e in lib["entries"]:
        for a in top_landings(e["attempts"], min_offset_mm):
            ref.setdefault(bin_key(a["offset_mm"], a["tilt_deg"]), []).append(a["onset_height_mm"])
    ref = {k: float(np.median(v)) for k, v in ref.items()}
    d = [(r["onset_height_mm"] - ref[bin_key(r["offset_mm"], r["tilt_deg"])], r["onset_height_mm"]) for r in top_landings(rows, min_offset_mm)
         if bin_key(r["offset_mm"], r["tilt_deg"]) in ref]
    if not d:
        return dict(n=0, bias_mm=None, note=f"needs jammed attempts at >= {min_offset_mm} mm offset (rod-top landings)")
    x = np.array([u for u, _ in d])
    out = dict(n=len(d), bias_mm=float(np.median(x)), spread_mm=float(np.percentile(x, 75) - np.percentile(x, 25)),
               real_onset_median_mm=float(np.median([v for _, v in d])))
    if len(d) < 3:
        out["note"] = "fewer than 3 rod-top landings: weak"
    return out


def registration_fit(rows: list[dict], span_mm: float = 6.0, step_mm: float = 0.25) -> dict:
    lv = [r for r in rows if r["tilt_deg"] < LEVEL_DEG and r.get("outcome") in ("seated", "jammed")]
    if len(lv) < 8:
        return dict(n=len(lv), note="needs >= 8 level attempts with outcomes")
    v = np.array([[r["offset_x_mm"], r["offset_y_mm"]] for r in lv])
    y = np.array([r["outcome"] == "seated" for r in lv])
    if y.all() or not y.any():
        return dict(n=len(lv), note="all level attempts have the same outcome")
    ang = np.degrees(np.arctan2(v[~y, 1], v[~y, 0]))
    if np.ptp(np.unwrap(np.radians(np.sort(ang)))) < np.radians(150):
        dirs_ok = False
    else:
        dirs_ok = True
    best = None
    g = np.arange(-span_mm, span_mm + 1e-9, step_mm)
    for sx in g:
        for sy in g:
            d = np.hypot(v[:, 0] - sx, v[:, 1] - sy)
            o = np.argsort(d)
            ds, ys = d[o], y[o]
            # threshold between consecutive sorted distances: errors = jammed inside + seated outside
            jam_in = np.concatenate([[0], np.cumsum(~ys)])
            seat_out = np.concatenate([np.cumsum(ys[::-1])[::-1], [0]])
            err = jam_in + seat_out
            kk = int(np.argmin(err))
            lo = ds[kk - 1] if kk > 0 else 0.0
            hi = ds[kk] if kk < len(ds) else ds[-1] + 1.0
            margin = hi - lo
            key = (int(err[kk]), -margin, np.hypot(sx, sy))
            if best is None or key < best[0]:
                best = (key, sx, sy, (lo + hi) / 2, margin)
    (errs, negm, _), sx, sy, rad, margin = best
    return dict(n=len(lv), shift_x_mm=float(sx), shift_y_mm=float(sy), capture_radius_mm=float(rad), misclassified=errs,
                margin_mm=float(margin), directions_ok=dirs_ok,
                note="" if dirs_ok else "jammed attempts cover < 150 deg of directions: the shift is poorly determined")


def apply_corrections(rows: list[dict], dz_mm: float, shift=(0.0, 0.0)) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        if r["onset_height_mm"] is not None:
            r["onset_height_mm"] -= dz_mm
        if shift != (0.0, 0.0):
            x, y = r["offset_x_mm"] - shift[0], r["offset_y_mm"] - shift[1]
            r.update(offset_x_mm=x, offset_y_mm=y, offset_mm=float(np.hypot(x, y)))
        out.append(r)
    return out


def unkey(bins: dict) -> dict:
    return {tuple(float(x) for x in k.split("|")): v for k, v in bins.items()}


def check(export: Path, measured: Path, library: Path, height_correction="grasp", apply_reg=False, top_offset_mm=3.5) -> dict:
    s, tr = load_export(export)
    m = json.loads(measured.read_text())
    below_mm = m["grasp"]["ring_bottom_below_tcp_mm"]
    lib = json.loads(library.read_text())
    rows = real_features(s, tr, below_mm / 1e3)
    out = dict(note="real export vs SIMULATED library; the library is Isaac Lab simulation, not real data", export=str(export),
               n_attempts=len(rows), outcomes={o: sum(r.get("outcome") == o for r in rows) for o in ("seated", "jammed", "aborted", None)},
               flags={}, ring_bottom_below_tcp_mm=below_mm, rod_axis=s["rod_axis"])
    for r in rows:
        for f in r.get("flags") or []:
            out["flags"][f] = out["flags"].get(f, 0) + 1
    rs = [r["rest_std_n"] for r in rows]
    out["sensor"] = dict(rest_force_std_n_median=float(np.median(rs)) if rs else None,
                         sim_force_noise_n=lib["sim_sensor"]["force_noise_n"],
                         sim_rest_std_n_median=float(np.median([a["rest_std_n"] for e in lib["entries"] for a in e["attempts"]])))
    # 2. height
    hb = height_bias(rows, lib, top_offset_mm)
    dz = hb["bias_mm"] if (hb["bias_mm"] is not None and height_correction != "none") else 0.0
    hb.update(applied_mm=dz, applied_as=height_correction if dz else "none")
    if hb["bias_mm"] is not None:
        hb["meaning"] = (f"contact starts with the computed ring bottom {hb['bias_mm']:+.1f} mm from where the simulation has it: "
                         f"ring_bottom_below_tcp_mm {below_mm} -> about {below_mm + hb['bias_mm']:.1f}, or the registered tip z is off by "
                         f"{-hb['bias_mm']:+.1f} mm (the two cannot be told apart here)")
    out["height"] = hb
    onset_shift = 0.0
    if dz and height_correction == "grasp":   # recompute everything with the corrected grasp length (lever arm, tilt xy too)
        rows = real_features(s, tr, (below_mm + dz) / 1e3)
        hb["ring_bottom_below_tcp_mm_used"] = below_mm + dz
    elif dz:                                    # tip z: only the height reference moves
        onset_shift = dz
    # 3. registration from outcomes
    reg = registration_fit(rows)
    out["registration_fit"] = reg
    shift = (reg["shift_x_mm"], reg["shift_y_mm"]) if (apply_reg and "shift_x_mm" in reg) else (0.0, 0.0)
    out["registration_fit"]["applied"] = shift != (0.0, 0.0)
    rows_c = apply_corrections(rows, onset_shift, shift)
    out["capture_real"] = clearance_estimate(rows_c)
    # 4. bins and scores
    real_bins = summarise_bins(rows_c)
    ranked = []
    for e in lib["entries"]:
        sc, parts = score(real_bins, unkey(e["bins"]))
        ranked.append(dict(name=e["name"], clearance_mm=e["clearance_mm"], k_scale=e["k_scale"], mu=e["mu"], score=round(sc, 4),
                           parts={k: round(v, 4) if isinstance(v, float) else v for k, v in parts.items()},
                           capture_sim_mm=e["capture"].get("clearance_mm")))
    ranked.sort(key=lambda r: r["score"])
    out["ranking"] = ranked
    marg = {}
    for p in ("clearance_mm", "k_scale", "mu"):
        vals = sorted({r[p] for r in ranked})
        marg[p] = {f"{v:g}": min(r["score"] for r in ranked if r[p] == v) for v in vals}
    out["marginals"] = marg
    best = ranked[0]
    close = [r["name"] for r in ranked if r["score"] <= best["score"] * 1.1 + 1e-9]
    out["best"] = dict(best, indistinguishable_within_10pct=close)
    be = unkey(next(e for e in lib["entries"] if e["name"] == best["name"])["bins"])
    out["bins"] = {f"{o:g}|{t:g}": dict(real=real_bins[(o, t)], best_sim=be.get((o, t))) for (o, t) in sorted(real_bins)}
    out["attempts"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows_c]
    return out


def fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def markdown(r: dict) -> str:
    L = ["# Probe check: real export vs simulated library", "",
         f"Export `{r['export']}`: {r['n_attempts']} attempts, outcomes {r['outcomes']}, flags {r['flags'] or 'none'}.",
         "The library is **simulation** (Isaac Lab, fin-ray grasp model, measured parameters).", "",
         "## Sensor", "",
         f"Force noise at rest (median std of |F - F_rest|): real {fmt(r['sensor']['rest_force_std_n_median'])} N, "
         f"simulated {fmt(r['sensor']['sim_rest_std_n_median'])} N (sim noise parameter {r['sensor']['sim_force_noise_n']} N per axis).", "",
         "## Height (grasp length / tip z)", ""]
    h = r["height"]
    L.append(f"{h['n']} rod-top landings; median onset height relative to the simulation in the same bins {fmt(h.get('bias_mm'))} mm "
             f"(spread {fmt(h.get('spread_mm'))} mm). " + (h.get("meaning", "") + ". " if h.get("bias_mm") is not None else "") + h.get("note", ""))
    L += ["", "## Registration from outcomes", ""]
    g = r["registration_fit"]
    if "shift_x_mm" in g:
        L.append(f"Best rod-axis shift ({g['shift_x_mm']:+.2f}, {g['shift_y_mm']:+.2f}) mm, capture radius {g['capture_radius_mm']:.2f} mm, "
                 f"{g['misclassified']} of {g['n']} level attempts misclassified, margin {g['margin_mm']:.2f} mm"
                 + (f". {g['note']}" if g["note"] else "") + (" (applied)" if g["applied"] else " (not applied)") + ".")
    else:
        L.append(g["note"] + ".")
    c = r["capture_real"]
    L += [f"Capture range from level attempts (after corrections): {fmt(c.get('clearance_mm'))} mm "
          f"(largest seated {fmt(c.get('largest_seated_mm'))}, smallest jammed {fmt(c.get('smallest_jammed_mm'))}). {c.get('note', '')}", "",
          "## Best matching simulation parameters", "",
          f"**{r['best']['name']}**: radial clearance {r['best']['clearance_mm']} mm, grip stiffness x{r['best']['k_scale']:g}, "
          f"friction {r['best']['mu']} (score {r['best']['score']}). Within 10%: {', '.join(r['best']['indistinguishable_within_10pct'])}.", "",
          "| rank | entry | score | seat | onset | rise | dir | sim capture (mm) |", "|---|---|---|---|---|---|---|---|"]
    for i, e in enumerate(r["ranking"][:8]):
        p = e["parts"]
        L.append(f"| {i + 1} | {e['name']} | {e['score']} | {p['seat']} | {p['onset']} | {p['rise']} | {p['dir']} | {fmt(e['capture_sim_mm'])} |")
    L += ["", "Best score per parameter value (lower is better):", ""]
    for p, v in r["marginals"].items():
        L.append(f"- {p}: " + ", ".join(f"{k}: {s:.3f}" for k, s in v.items()))
    L += ["", "## Per bin (real vs best simulation entry)", "",
          "| offset (mm) | tilt (deg) | n | seat real | seat sim | onset real | onset sim | rise real | rise sim | dir real | dir sim |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in r["bins"].items():
        o, t = k.split("|")
        a, b = v["real"], v["best_sim"] or {}
        L.append(f"| {o} | {t} | {a['n']} | {fmt(a['seat_frac'])} | {fmt(b.get('seat_frac'))} | {fmt(a['onset_height_mm'])} | "
                 f"{fmt(b.get('onset_height_mm'))} | {fmt(a['rise_n_per_mm'])} | {fmt(b.get('rise_n_per_mm'))} | {fmt(a['dir_err_deg'], 0)} | "
                 f"{fmt(b.get('dir_err_deg'), 0)} |")
    L += ["", "Scores: seat 4 x (fraction difference)^2, onset ((mm difference) / 1.0)^2, rise (log(ratio) / 0.4)^2, each capped at 4 per bin "
          "and averaged over the bins both sides have; the moment direction (dir) is shown but not scored (too noisy between simulated "
          "seeds: median 26 deg). The grid is coarse (clearance 0.5 / 1.0 / 2.7 mm, stiffness x0.5 / 1 / 2, "
          "friction 0.3 / 0.6): read the result as the nearest grid point, and bins with n = 1 as weak evidence."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", type=Path, help="directory with summary.json and traces.npz")
    ap.add_argument("--measured", type=Path, default=Path("config/measured.json"))
    ap.add_argument("--library", type=Path, default=Path("sim_library/probe_library.json"))
    ap.add_argument("-o", "--output", type=Path, default=Path("outputs/probe_check"))
    ap.add_argument("--height-correction", choices=["grasp", "tip", "none"], default="grasp",
                    help="remove the height bias of the top landings as a grasp-length error (default; recomputes all features), "
                         "a tip-z error (shifts onsets only) or not at all")
    ap.add_argument("--apply-registration-fit", action="store_true")
    ap.add_argument("--top-offset-mm", type=float, default=3.5, help="jams at least this far off count as rod-top landings")
    a = ap.parse_args()
    r = check(a.export, a.measured, a.library, a.height_correction, a.apply_registration_fit, a.top_offset_mm)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "report.json").write_text(json.dumps(r, indent=1))
    md = markdown(r)
    (a.output / "report.md").write_text(md)
    print(md)
    print(f"wrote {a.output / 'report.md'} and report.json")


if __name__ == "__main__":
    main()
