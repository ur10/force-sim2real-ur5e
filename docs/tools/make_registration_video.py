"""Illustration video of how the rod was registered from the robot's own kinematics (REAL data).

Three GELLO-teleoperated insertions on the real UR5e (camera episodes + the robot's flange pose at
10 Hz, collect.py replay buffer). The nut bottom is the flange position plus the grasp length along
the tool axis; where the gripper opens over the rod, the nut sits on the rod. The video plays one
insertion (cropped side camera + wrist camera) next to a top view of the nut path with the release
points of all three insertions and the resulting rod axis, and the nut height over time.

    python docs/tools/make_registration_video.py <replay_buffer.zarr> <videos dir> -o registration_demo.mp4
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9"


def episodes(z, grasp_m):
    ends = z["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    pos, rv, grip, ts = (z["data/robot0_eef_pos"][:], z["data/robot0_eef_quat"][:], z["data/robot0_gripper_qpos"][:, 0],
                         z["data/timestamp"][:])
    out = []
    for s, e in zip(starts, ends):
        R = Rot.from_rotvec(rv[s:e]).as_matrix()
        bottom = pos[s:e] + grasp_m * R[:, :, 2]
        g = grip[s:e]
        t = ts[s:e] - ts[s]
        held = g < 0.5 * g.max()                      # closed on the nut
        opens = np.where(held[:-1] & ~held[1:])[0]     # closed -> open
        rel = int(opens[-1]) + 1                       # the last opening: the release over the rod
        out.append(dict(t=t, bottom=bottom, grip=g, rel=rel, rel_pt=bottom[rel], tilt=float(np.degrees(np.arccos(-R[rel, 2, 2])))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr", type=Path)
    ap.add_argument("videos", type=Path, help="directory with <episode>/<camera>.mp4 (cropped side cameras)")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--side-cam", type=int, default=1)
    ap.add_argument("--wrist-cam", type=int, default=3)
    ap.add_argument("--grasp-mm", type=float, default=225.0, help="nut bottom below the flange")
    ap.add_argument("--axis", type=float, nargs=2, default=(-85.0, -810.0), help="registered rod axis xy (mm)")
    ap.add_argument("--tip", type=float, default=180.0, help="registered tip z (mm)")
    ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("-o", "--output", type=Path, required=True)
    a = ap.parse_args()
    z = zarr.open(str(a.zarr), "r")
    eps = episodes(z, a.grasp_mm / 1000)
    ep = eps[a.episode]
    side = cv2.VideoCapture(str(a.videos / str(a.episode) / f"{a.side_cam}.mp4"))
    wrist = cv2.VideoCapture(str(a.videos / str(a.episode) / f"{a.wrist_cam}.mp4"))
    vfps = side.get(cv2.CAP_PROP_FPS)
    dur = ep["t"][-1]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#fcfcfb")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], height_ratios=[1.2, 1.0, 0.32], hspace=0.38, wspace=0.12,
                          left=0.02, right=0.97, top=0.9, bottom=0.04)
    fig.suptitle("Rod registration from the robot's own kinematics   (REAL: UR5e, GELLO teleoperation)", fontsize=14,
                 color=INK, x=0.02, ha="left", fontweight="bold")
    ax_side = fig.add_subplot(gs[0:2, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_z = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[2, :])
    for x in (ax_side, ax_txt):
        x.axis("off")
    ax_txt.text(0, 0.95, "Nut bottom = flange position (RTDE) + 225 mm along the tool axis.  Where the gripper opens over the rod, the "
                "nut rests on it:\nthe releases of the three insertions give the rod axis "
                f"({a.axis[0]:.0f}, {a.axis[1]:.0f}) mm and the tip at {a.tip:.0f} mm in the robot base frame.",
                fontsize=11.5, color=MUTED, va="top", transform=ax_txt.transAxes)
    img_side = ax_side.imshow(np.zeros((10, 10, 3), np.uint8))
    inset = ax_side.inset_axes([0.64, 0.64, 0.36, 0.36])  # upper right: the rod stand is at the lower left of this view
    inset.axis("off")
    img_wrist = inset.imshow(np.zeros((10, 10, 3), np.uint8))
    ax_side.set_title(f"insertion {a.episode + 1} of 3: side camera (cropped), wrist camera inset", fontsize=10.5, color=MUTED, loc="left")
    # top view
    c = np.array(a.axis)
    for axx in (ax_top, ax_z):
        axx.set_facecolor("#fcfcfb")
        axx.grid(color=GRID, lw=0.8)
        for sp in ("top", "right"):
            axx.spines[sp].set_visible(False)
        axx.tick_params(colors="#898781", labelsize=8.5)
    ax_top.set_aspect("equal")
    ax_top.set_xlim(c[0] - 30, c[0] + 30)
    ax_top.set_ylim(c[1] - 30, c[1] + 30)
    ax_top.set_title("top view near the rod (mm, base frame)", fontsize=10.5, color=MUTED, loc="left")
    ax_top.add_patch(plt.Circle(c, 8.0, color="#9aa0a6", alpha=0.5, lw=0))
    ax_top.plot(*c, "+", color=INK, ms=12, mew=1.5)
    ax_top.annotate(f"rod axis ({c[0]:.0f}, {c[1]:.0f})", c, xytext=(10, -22), textcoords="offset points", fontsize=9, color=INK)
    for i, e in enumerate(eps):
        if i != a.episode:
            p = e["rel_pt"] * 1000
            ax_top.plot(p[0], p[1], "o", mfc="none", mec=ORANGE, ms=8, mew=1.6)
    path_line, = ax_top.plot([], [], color=BLUE, lw=1.6)
    here, = ax_top.plot([], [], "o", color=BLUE, ms=6)
    rel_dot, = ax_top.plot([], [], "o", color=ORANGE, ms=9)
    ax_top.plot([], [], "o", mfc="none", mec=ORANGE, ms=8, mew=1.6, label="releases, other insertions")
    ax_top.plot([], [], "o", color=ORANGE, ms=8, label="release, this insertion")
    ax_top.plot([], [], color=BLUE, lw=1.6, label="nut bottom path")
    ax_top.legend(fontsize=8, frameon=False, loc="upper left")
    # height over time
    t, bz = ep["t"], ep["bottom"][:, 2] * 1000
    ax_z.plot(t, bz, color="#c3c2b7", lw=1.2)
    ax_z.axhline(a.tip, color=INK, lw=1, ls="--")
    ax_z.text(t[0] + 0.3, a.tip + 8, f"rod tip {a.tip:.0f} mm", fontsize=8.5, color=INK)
    open_now = ep["grip"] > 0.5 * ep["grip"].max()
    ax_z.fill_between(t, -40, 360, where=~open_now, color=BLUE, alpha=0.06, lw=0)
    ax_z.text(t[np.argmax(~open_now)] + 0.3, 330, "gripper closed", fontsize=8.5, color=BLUE)
    ax_z.axvline(t[ep["rel"]], color=ORANGE, lw=1.2)
    ax_z.text(t[ep["rel"]] + 0.3, 20, "release", fontsize=8.5, color=ORANGE)
    ax_z.set_xlim(0, dur)
    ax_z.set_ylim(-40, 360)
    ax_z.set_xlabel("time (s)", fontsize=9, color=MUTED)
    ax_z.set_ylabel("nut bottom height (mm)", fontsize=9, color=MUTED)
    z_line, = ax_z.plot([], [], color=BLUE, lw=1.8)
    cursor = ax_z.axvline(0, color=MUTED, lw=0.8)
    writer = FFMpegWriter(fps=a.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart"])
    n_frames = int(dur / a.speed * a.fps)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(a.output), dpi=100):
        for f in range(n_frames):
            tv = f * a.speed / a.fps
            for cap, im in ((side, img_side), (wrist, img_wrist)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(tv * vfps)))
                ok, fr = cap.read()
                if ok:
                    im.set_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                    im.set_extent((0, fr.shape[1], fr.shape[0], 0))
            ax_side.set_xlim(0, img_side.get_extent()[1])
            ax_side.set_ylim(img_side.get_extent()[2], 0)
            inset.set_xlim(0, img_wrist.get_extent()[1])
            inset.set_ylim(img_wrist.get_extent()[2], 0)
            k = int(np.searchsorted(t, tv))
            k = min(max(k, 1), len(t) - 1)
            p = ep["bottom"][:k] * 1000
            path_line.set_data(p[:, 0], p[:, 1])
            here.set_data([p[-1, 0]], [p[-1, 1]])
            if k > ep["rel"]:
                rp = ep["rel_pt"] * 1000
                rel_dot.set_data([rp[0]], [rp[1]])
            z_line.set_data(t[:k], bz[:k])
            cursor.set_xdata([tv, tv])
            writer.grab_frame()
    print(f"wrote {a.output}: {n_frames} frames; releases (mm):",
          [tuple(np.round(e["rel_pt"] * 1000, 1)) for e in eps], "mean xy", np.round(np.mean([e["rel_pt"][:2] for e in eps], 0) * 1000, 1))


if __name__ == "__main__":
    main()
