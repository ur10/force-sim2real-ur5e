"""Close-up animation of simulated probe conditions: the vertical section through the rod axis
(rod, ring cut into its two annulus walls, TCP), and below it the axial force, true (what the
physics transmits) and sensed (after the simulated F/T chain), with a moving cursor.

    .venv/bin/python scripts/ur_contact/animate_probe.py outputs/ur_nut/contact_sim/probe --conds 0 11 18 \
        --output outputs/ur_nut/contact_sim/media/probe_cases.mp4
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ROD, RING, RING_EDGE, TCP = "#9aa0a6", "#3fae5a", "#1f6e35", "#2b2b2b"
TRUE_C, SENSED_C = "#1f4e9c", "#e08a1e"


def ring_walls(pose, R, r, w, rod_y):
    """The two wall sections of the ring in the x-z plane (polygons, metres)."""
    Rm = Rotation.from_quat(np.asarray(pose)[[1, 2, 3, 0]]).as_matrix()
    c = np.asarray(pose[4:])
    out = []
    for x0, x1 in ((r, R), (-R, -r)):
        loc = np.array([[x0, 0, -w / 2], [x1, 0, -w / 2], [x1, 0, w / 2], [x0, 0, w / 2]])
        out.append((loc @ Rm.T + c)[:, [0, 2]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--conds", type=int, nargs="+", default=[0, 11, 18])
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--layout", choices=["wide", "tall"], default="wide", help="tall: one row per condition (phones)")
    ap.add_argument("--title", type=str, default=None, help="figure title (default: clearance + 'uncalibrated placeholder parameters')")
    a = ap.parse_args()
    s = json.loads((a.run / "summary.json").read_text())
    d = np.load(a.run / "traces.npz")
    g = s["params"]["geometry"]
    conds = [next(c for c in s["conditions"] if c["i"] == i) for i in a.conds]
    t = d[f"c{a.conds[0]:02d}/t"]
    step = max(1, int(round(a.speed / a.fps / (t[1] - t[0]))))
    frames = range(0, len(t), step)
    rx, tip = g["rod_xy"][0], g["rod_tip_z"]
    tall = a.layout == "tall"
    if tall:
        fig = plt.figure(figsize=(7.2, 3.3 * len(conds) + 0.6), dpi=110)
        gs = fig.add_gridspec(len(conds), 2, width_ratios=[1.0, 1.15], hspace=0.5, wspace=0.3)
    else:
        fig = plt.figure(figsize=(4.4 * len(conds), 7.2), dpi=110)
        gs = fig.add_gridspec(2, len(conds), height_ratios=[2.2, 1.0], hspace=0.28, wspace=0.22)
    art = []
    f_hi = max(4.0, 1.12 * max(float(d[f"c{c['i']:02d}/w_true"][:, 2].max()) for c in conds))
    for j, c in enumerate(conds):
        k = f"c{c['i']:02d}"
        ax = fig.add_subplot(gs[j, 0] if tall else gs[0, j])
        ax.set_aspect("equal")
        ax.set_xlim(-26, 26)
        ax.set_ylim(-22, 30)
        ax.add_patch(Rectangle((-g["rod_r"] * 1000, (g["rod_z0"] - tip) * 1000), 2 * g["rod_r"] * 1000,
                               (tip - g["rod_z0"]) * 1000, color=ROD, zorder=1))
        ax.axhline(0, color="0.75", lw=0.6, ls=":", zorder=0)
        ax.set_title(f"offset {c['offset_mm']:g} mm, tilt {c['tilt_deg']:g} deg -> {c['outcome']}", fontsize=10)
        ax.set_xlabel("x from rod axis (mm)", fontsize=8)
        if j == 0 or tall:
            ax.set_ylabel("height above rod tip (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        walls = [Polygon(np.zeros((4, 2)), closed=True, fc=RING, ec=RING_EDGE, lw=1.0, zorder=3) for _ in range(2)]
        for wp in walls:
            ax.add_patch(wp)
        tcp_dot, = ax.plot([], [], marker="v", color=TCP, ms=7, zorder=4)
        txt = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=8, family="monospace")
        fx = fig.add_subplot(gs[j, 1] if tall else gs[1, j])
        wt, ws = d[f"{k}/w_true"][:, 2], d[f"{k}/w_filtered"][:, 2]
        fx.plot(t, ws, color=SENSED_C, lw=0.8, label="sensed (F/T model)")
        fx.plot(t, wt, color=TRUE_C, lw=1.2, label="true (physics)")
        fx.set_xlim(t[0], t[-1])
        fx.set_ylim(-1.5, f_hi)
        fx.set_xlabel("time (s)", fontsize=8)
        if j == 0 or tall:
            fx.set_ylabel("axial force Fz (N)", fontsize=8)
        if j == 0:
            fx.legend(fontsize=7, frameon=False, loc="upper left")
        fx.tick_params(labelsize=7)
        fx.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            fx.spines[sp].set_visible(False)
        cur = fx.axvline(0, color="0.3", lw=0.8)
        art.append((k, walls, tcp_dot, txt, cur))
    if tall and a.title:  # wrap a long title for the narrow figure
        import textwrap
        a.title = "\n".join(textwrap.wrap(a.title, 60))
    fig.suptitle(a.title or f"Simulated contact probe (radial clearance {s['radial_clearance_mm']:.2f} mm, uncalibrated placeholder parameters)",
                 fontsize=11)
    out = a.output or a.run / "probe_cases.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=a.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "23"])
    with writer.saving(fig, str(out), dpi=110):
        for f in frames:
            for k, walls, tcp_dot, txt, cur in art:
                ring = d[f"{k}/ring"][f]
                for wp, poly in zip(walls, ring_walls(ring, g["ring_outer_r"], g["ring_bore_r"], g["ring_height"], g["rod_xy"][1])):
                    xy = poly.copy()
                    xy[:, 0] = (xy[:, 0] - rx) * 1000
                    xy[:, 1] = (xy[:, 1] - tip) * 1000
                    wp.set_xy(xy)
                tp = d[f"{k}/tcp"][f]
                z_show = min((tp[6] - tip) * 1000, 29)
                tcp_dot.set_data([(tp[4] - rx) * 1000], [z_show])
                fz = d[f"{k}/w_true"][f, 2]
                txt.set_text(f"t {t[f]:4.2f} s\ndepth {d[f'{k}/depth'][f] * 1000:6.2f} mm\nFz {fz:5.2f} N")
                cur.set_xdata([t[f], t[f]])
            writer.grab_frame()
    plt.close(fig)
    print(out, f"{len(frames)} frames")


if __name__ == "__main__":
    main()
