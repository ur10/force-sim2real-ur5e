"""Force traces of a contact-probe run (simulated now; the real run will use the same layout):
one column per tilt; rows: axial force, lateral force, torque magnitude, each against the ring's
depth over the rod tip. Offset is a sequential (single-hue) colour, light = small.

    .venv/bin/python scripts/ur_contact/plot_probe.py outputs/ur_nut/contact_sim/probe
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(run: Path, channel: str = "w_true"):
    s = json.loads((run / "summary.json").read_text())
    d = np.load(run / "traces.npz")
    conds = s["conditions"]
    tilts = sorted({c["tilt_deg"] for c in conds})
    offs = sorted({c["offset_mm"] for c in conds})
    cmap = plt.get_cmap("Blues")
    col = {o: cmap(0.35 + 0.65 * i / max(len(offs) - 1, 1)) for i, o in enumerate(offs)}
    fig, ax = plt.subplots(3, len(tilts), figsize=(4.2 * len(tilts), 8.4), sharex=True, sharey="row", squeeze=False)
    for c in conds:
        j = tilts.index(c["tilt_deg"])
        k = f"c{c['i']:02d}"
        depth = d[f"{k}/depth"] * 1000
        w = d[f"{k}/{channel}"]
        for r, y in enumerate((w[:, 2], np.linalg.norm(w[:, :2], axis=1), np.linalg.norm(w[:, 3:], axis=1))):
            ax[r, j].plot(depth, y, color=col[c["offset_mm"]], lw=1.4, label=f"{c['offset_mm']:g} mm ({c['outcome']})" if r == 0 else None)
    clr = s["radial_clearance_mm"]
    for j, t in enumerate(tilts):
        ax[0, j].set_title(f"tilt {t:g} deg  (radial clearance {clr:.2f} mm)", fontsize=10)
        ax[2, j].set_xlabel("ring bottom below rod tip (mm)")
        for r in range(3):
            ax[r, j].axvline(0, color="0.6", lw=0.8, ls=":")
            ax[r, j].grid(alpha=0.25, lw=0.5)
            for sp in ("top", "right"):
                ax[r, j].spines[sp].set_visible(False)
        ax[0, j].legend(title="offset", fontsize=7, title_fontsize=8, frameon=False)
    ax[0, 0].set_ylabel("axial force Fz (N)")
    ax[1, 0].set_ylabel("lateral force |Fxy| (N)")
    ax[2, 0].set_ylabel("torque |tau| (N m)")
    fig.suptitle(f"Simulated probe, {channel} (placeholder parameters, uncalibrated)", fontsize=11)
    fig.tight_layout()
    out = run / f"probe_{channel}.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), *(sys.argv[2:3] or []))
