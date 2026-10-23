"""Checks of the held-ring vs bolt-stand clearance and the post-warp lift (no robot, no Isaac):
    JAX_PLATFORMS=cpu .venv/bin/python scripts/ur_tape/test_ring_clearance.py"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tape_plan import lift_ring_over_rod, ring_fixture_clearance  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCENE = json.loads((REPO / "outputs/ur_nut/scene.json").read_text())
DIMS = json.loads((REPO / "outputs/ur_nut/nut_dims.json").read_text())
B = SCENE["bolt_stand"]
TIP, ROD_R = B["rod"]["z1"], B["rod"]["radius"]


def path(xy, bottom_z):
    T = np.zeros((len(xy), 7))
    T[:, 0] = 1
    T[:, 4:6] = xy
    T[:, 6] = np.asarray(bottom_z) + DIMS["w"] / 2
    return T


def test_seated_ring_has_bore_gap():
    T = path(np.tile(B["xy"], (3, 1)), [TIP - 0.02] * 3)
    c = ring_fixture_clearance(T, SCENE, DIMS)
    assert np.allclose(c, DIMS["r_in"] - ROD_R, atol=1e-6), c


def test_side_pass_detected_and_lifted():
    # rises past the rod 23 mm from its axis (outer wall 0.8 mm into the rod) at tip height
    n = 40
    x = np.linspace(B["xy"][0] - 0.10, B["xy"][0] - 0.023, n)
    xy = np.stack([x, np.full(n, B["xy"][1])], 1)
    T = path(xy, np.linspace(TIP - 0.05, TIP - 0.005, n))
    assert ring_fixture_clearance(T, SCENE, DIMS).min() < 0
    L = lift_ring_over_rod(T, SCENE, DIMS)
    assert ring_fixture_clearance(L, SCENE, DIMS).min() > 0.002
    assert np.abs(np.diff(L[:, 6] - T[:, 6])).max() < 0.02  # a ramp, not a jump


def test_descent_down_the_axis_untouched():
    n = 20
    T = path(np.tile(B["xy"], (n, 1)), np.linspace(TIP + 0.006, TIP - 0.02, n))
    L = lift_ring_over_rod(T, SCENE, DIMS)
    assert np.allclose(L, T)
    assert ring_fixture_clearance(T, SCENE, DIMS).min() > 0.0028


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
