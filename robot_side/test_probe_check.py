"""Offline test of the probe check (no robot, numpy only).

Uses a held-out SIMULATED probe run written in the real export format (flange TCP, torque about the
flange) and checks that the check finds its parameters and removes a planted grasp-length error.

On the robot machine (layout ~/ring_rod_probe/{scripts,tests,sim_library}):
    python tests/test_probe_check.py          # or: python -m pytest tests/test_probe_check.py
Elsewhere: PROBE_CHECK_ROOT=<dir with scripts/ and sim_library/> python test_probe_check.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("PROBE_CHECK_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))
from probe_check import check, markdown  # noqa: E402

LIB = ROOT / "sim_library" / "probe_library.json"
HELDOUT = ROOT / "sim_library" / "heldout_c1.0_k1_mu0.3"


def _measured(below_mm: float) -> Path:
    d = Path(tempfile.mkdtemp())
    p = d / "measured.json"
    p.write_text(json.dumps({"grasp": {"ring_bottom_below_tcp_mm": below_mm}}))
    return p


def test_finds_heldout_parameters():
    r = check(HELDOUT, _measured(225.0), LIB)
    assert r["best"]["name"] == "c1.0_k1_mu0.3", r["best"]
    assert abs(r["height"]["bias_mm"]) < 0.3, r["height"]
    assert r["marginals"]["clearance_mm"]["1"] == min(r["marginals"]["clearance_mm"].values())


def test_removes_grasp_length_error():
    for below in (228.0, 213.0):  # +3 mm and -12 mm wrong
        r = check(HELDOUT, _measured(below), LIB)
        assert abs(r["height"]["ring_bottom_below_tcp_mm_used"] - 225.0) < 0.3, r["height"]
        assert r["best"]["name"] == "c1.0_k1_mu0.3", (below, r["best"])


def test_report_renders():
    md = markdown(check(HELDOUT, _measured(225.0), LIB))
    assert "SIMULATED" in md or "simulation" in md
    assert "| rank |" in md


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
