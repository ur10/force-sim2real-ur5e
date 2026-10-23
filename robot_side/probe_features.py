"""Contact features of one probe attempt, computed the same way for real and simulated data
(numpy only; runs on the robot machine and here).

Inputs per attempt: t (T,) s, tcp (T, 7) [qw qx qy qz x y z] of the logged TCP, w (T, 6) [F, tau]
in the base frame with tau about that TCP, the ring bottom's distance below the TCP along the tool
z axis, and the registered rod axis xy and tip z. All lengths in metres.

Features (geometry, not who drove the arm):
  rest_std_n         force noise in the first 0.2 s (still)
  onset_height_mm    ring bottom above the tip at first contact (< 0: already below the tip), from the
                     force rise back-extrapolated to the rest level
  offset_mm, tilt_deg  actual ring-axis offset from the rod axis and tool tilt at the onset (without
                     contact: where the ring bottom passes the tip height); offset_x/y_mm the vector
  rise_n_per_mm      force rise per mm of further descent after the onset (series stiffness of
                     grip + rod + arm), over the first 1 mm
  dir_err_deg        angle between the contact point estimated from the moment, r = (-tau_y, tau_x)
                     / Fz with tau about the ring bottom, and the true direction from the ring axis
                     to the rod axis (0 = the moment points at the rod)
  peak_n, final_depth_mm
"""

from __future__ import annotations

import numpy as np

OFFSET_BINS_MM = (0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0)
TILT_BINS_DEG = (0.0, 3.0, 6.0)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average over n samples, edges replicated."""
    if n <= 1:
        return x
    pad = np.concatenate([np.full(n // 2, x[0]), x, np.full(n - 1 - n // 2, x[-1])])
    c = np.cumsum(np.concatenate([[0.0], pad]))
    return (c[n:] - c[:-n]) / n


def attempt_features(t, tcp, w, bottom_below_tcp: float, tip_z: float, rod_xy, rest_s: float = 0.2,
                     smooth_s: float = 0.02, k_sigma: float = 5.0, min_dev_n: float = 0.5, persist_s: float = 0.02,
                     max_backtrack_mm: float = 3.0) -> dict:
    t, tcp, w = np.asarray(t, float), np.asarray(tcp, float), np.asarray(w, float)
    R = quat_to_mat(tcp[:, :4])
    ztool = R[:, :, 2]
    p = tcp[:, 4:]
    bottom = p + bottom_below_tcp * ztool
    F, tau = w[:, :3], w[:, 3:]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.002
    rest = t <= t[0] + rest_s
    n_rest = int(rest.sum())
    F_rest = F[rest].mean(0)
    dev = np.linalg.norm(F - F_rest, axis=1)
    std = float(dev[rest].std()) if n_rest > 2 else 0.0
    # contact: the smoothed force deviation leaves its rest band and stays out (time-based windows, so
    # 240 Hz simulation and 500 Hz RTDE are treated alike)
    ns = max(1, int(round(smooth_s / dt)))
    dev_s = _smooth(dev, ns)
    core = dev_s[ns:n_rest] if n_rest > ns + 2 else dev_s[:n_rest]
    mu_r, sd_r = float(core.mean()), float(core.std())
    thr = mu_r + max(k_sigma * sd_r, min_dev_n)
    k = max(1, int(round(persist_s / dt)))
    run = np.convolve((dev_s > thr).astype(int), np.ones(k, int), mode="valid") == k
    cand = np.where(run & (np.arange(len(run)) > n_rest))[0]
    j = int(cand[0]) if len(cand) else None
    rod = np.asarray(rod_xy, float)
    # where the offset is read: at the contact onset, else where the ring bottom passes the tip height
    below_tip = np.where(bottom[n_rest:, 2] <= tip_z + 5e-4)[0]
    jj = j if j is not None else (n_rest + int(below_tip[0]) if len(below_tip) else n_rest)
    off_vec = bottom[jj, :2] - rod
    tilt = float(np.degrees(np.arccos(np.clip(-ztool[jj, 2], -1, 1))))
    out = dict(rest_std_n=std, onset_height_mm=None, offset_mm=float(np.linalg.norm(off_vec) * 1e3), tilt_deg=tilt,
               offset_x_mm=float(off_vec[0] * 1e3), offset_y_mm=float(off_vec[1] * 1e3),
               rise_n_per_mm=None, dir_err_deg=None, peak_n=float(dev_s.max()),
               final_depth_mm=float((tip_z - bottom[-1, 2]) * 1e3), contact=j is not None)
    if j is None:
        return out
    h_cross = float((bottom[j, 2] - tip_z) * 1e3)
    out["onset_height_mm"] = h_cross
    # rise: smoothed force deviation against further descent of the ring bottom, over the 1 mm after the
    # crossing; the line back-extrapolated to the rest level gives the onset (the crossing lags it by
    # threshold / rise, about 1 mm on a compliant grip)
    down = (bottom[j, 2] - bottom[j:, 2]) * 1e3
    win = (down >= 0) & (down <= 1.0)
    idx = np.arange(j, len(t))[win]
    if len(idx) > 5 and np.ptp(down[win]) > 0.2:
        b, a = np.polyfit(down[win], dev_s[idx], 1)
        out["rise_n_per_mm"] = float(b)
        if b > 0:
            back = float(np.clip((a - mu_r) / b, 0.0, max_backtrack_mm))
            out["onset_height_mm"] = h_cross + back
    # moment direction about the ring bottom while pressing (axial force change above 1 N; either sign,
    # and r below is unchanged if the whole wrench is negated, so the sensor sign convention drops out)
    tau_b = tau + np.cross(p - bottom, F)
    press = (np.arange(len(t)) >= j) & (np.abs(F[:, 2] - F_rest[2]) > 1.0)
    if press.sum() > 5:
        fz = (F[press, 2] - F_rest[2]).mean()
        m = (tau_b[press] - tau_b[rest].mean(0)).mean(0)
        r = np.array([-m[1], m[0]]) / fz
        to_rod = -off_vec
        if np.linalg.norm(r) > 1e-5 and np.linalg.norm(to_rod) > 2e-4:
            c = np.dot(r, to_rod) / (np.linalg.norm(r) * np.linalg.norm(to_rod))
            out["dir_err_deg"] = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return out


def bin_key(offset_mm: float, tilt_deg: float) -> tuple[float, float]:
    ob = min(OFFSET_BINS_MM, key=lambda b: abs(b - offset_mm))
    tb = min(TILT_BINS_DEG, key=lambda b: abs(b - tilt_deg))
    return ob, tb


def summarise_bins(rows: list[dict]) -> dict:
    """rows: attempt features + 'outcome'. -> {(offset bin, tilt bin): medians and seat fraction}."""
    groups = {}
    for r in rows:
        groups.setdefault(bin_key(r["offset_mm"], r["tilt_deg"]), []).append(r)
    out = {}
    for key, g in groups.items():
        med = lambda k: (float(np.median([x[k] for x in g if x[k] is not None])) if any(x[k] is not None for x in g) else None)  # noqa: E731
        seat = [x["outcome"] == "seated" for x in g if x.get("outcome") in ("seated", "jammed")]
        out[key] = dict(n=len(g), seat_frac=float(np.mean(seat)) if seat else None, onset_height_mm=med("onset_height_mm"),
                        rise_n_per_mm=med("rise_n_per_mm"), dir_err_deg=med("dir_err_deg"), rest_std_n=med("rest_std_n"))
    return out


def clearance_estimate(rows: list[dict]) -> dict:
    """Effective radial clearance from where outcomes switch from seated to jammed (level attempts,
    tilt < 1.5 deg): the midpoint between the largest seated and the smallest jammed offset."""
    lv = [r for r in rows if r["tilt_deg"] < 1.5 and r.get("outcome") in ("seated", "jammed")]
    seated = [r["offset_mm"] for r in lv if r["outcome"] == "seated"]
    jammed = [r["offset_mm"] for r in lv if r["outcome"] == "jammed"]
    if not seated or not jammed:
        return dict(clearance_mm=None, note="needs level attempts that seated and that jammed")
    lo, hi = max(seated), min(jammed)
    return dict(clearance_mm=float((lo + hi) / 2), largest_seated_mm=float(lo), smallest_jammed_mm=float(hi),
                consistent=bool(lo < hi), note="" if lo < hi else "seated and jammed offsets overlap (tilt, direction or noise)")


def score(real: dict, sim: dict, seat_w: float = 4.0, onset_mm: float = 1.0, rise_log: float = 0.4, dir_deg: float = 60.0,
          dir_w: float = 0.0, cap: float = 4.0) -> tuple[float, dict]:
    """Mismatch between two bin summaries over shared bins (lower is better).

    Scales from the spread between two simulated runs of the same parameters with different sensor
    noise (2026-11-22): onset median |diff| 0.15 mm (rms 1.3 mm, outliers to 5 mm), rise rms log-ratio
    0.39, one outcome flip in 24 bins, moment direction median |diff| 26 deg (rms 56 deg). The moment
    direction is therefore reported but not scored by default (dir_w = 0). Each per-bin term is capped
    at `cap` so one outlying bin cannot decide the ranking.
    """
    parts = {"seat": [], "onset": [], "rise": [], "dir": []}
    for key in set(real) & set(sim):
        a, b = real[key], sim[key]
        if a["seat_frac"] is not None and b["seat_frac"] is not None:
            parts["seat"].append(min(seat_w * (a["seat_frac"] - b["seat_frac"]) ** 2, cap))
        if a["onset_height_mm"] is not None and b["onset_height_mm"] is not None:
            parts["onset"].append(min(((a["onset_height_mm"] - b["onset_height_mm"]) / onset_mm) ** 2, cap))
        if a["rise_n_per_mm"] and b["rise_n_per_mm"] and a["rise_n_per_mm"] > 0 and b["rise_n_per_mm"] > 0:
            parts["rise"].append(min((np.log(a["rise_n_per_mm"] / b["rise_n_per_mm"]) / rise_log) ** 2, cap))
        if a["dir_err_deg"] is not None and b["dir_err_deg"] is not None:
            parts["dir"].append(dir_w * min(((a["dir_err_deg"] - b["dir_err_deg"]) / dir_deg) ** 2, cap))
    means = {k: float(np.mean(v)) if v else 0.0 for k, v in parts.items()}
    return sum(means.values()), dict(means, shared_bins=len(set(real) & set(sim)))
