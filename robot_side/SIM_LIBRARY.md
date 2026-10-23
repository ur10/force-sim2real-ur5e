# Simulated probe library (for `scripts/probe_check.py`)

**Everything in this folder is simulation** (Isaac Lab 2.0 / PhysX, on the simulation machine), not
recorded data. Built 2026-11-22 by `scripts/ur_contact/build_probe_library.py` in the
real2render2real repo (branch `ur5e-contact-sim`).

## What was simulated

- The ring held by the fin-ray grasp model (6-DoF spring between gripper and ring, pivot in the
  gripped band), the provisional `config/measured.json` values: rod 16 mm, ring 32 mm outer, 14 mm
  high, 8 g, ring bottom 225 mm below the flange, rod tip chamfer 1 mm.
- The same probe a GELLO operator does, scripted: line up at an offset (0, 1, 2, 2.5, 3, 3.5, 4, 6 mm,
  all in one direction) and tilt (0, 3, 6 deg), descend at 5 mm/s with a stiff arm, stop when the
  filtered force passes 8 N.
- The UR-like sensor chain: 500 Hz, 8 ms latency, 0.5 N force noise per axis, bias and drift,
  payload compensation, tare before each attempt.
- A grid of the unknowns: radial clearance 0.5 / 1.0 / 2.7 mm (bore unmeasured; 2.7 mm is the RGB
  fit), grip stiffness x0.5 / x1 / x2 of the nominal (6000, 1500, 2500) N/m, friction 0.3 / 0.6.
  18 entries x 24 attempts.

`probe_library.json` holds per entry the features of every attempt (`scripts/probe_features.py`,
the same code `probe_check.py` runs on the real export), their per-bin summary and the capture range.

## How well the check works (on simulated data only)

Tested on runs that are not in the library (new sensor noise seed), written in the real export
format (`heldout_c1.0_k1_mu0.3/` is one of them, used by `tests/test_probe_check.py`):

| case | result |
|---|---|
| held-out run, parameters on the grid | correct entry ranked first (score 1.49, next 2.45) |
| same, grasp length stated 3 mm too long / 12 mm too short | error recovered to 0.05 mm; same ranking |
| same, plus 1 mm registration error | true entry among the 3 within 10 %; stiffness right, clearance blurs between 0.5 and 1.0 mm |
| run off the grid (1.5 mm, x0.7, mu 0.45) | nearest grid neighbours (1.0 mm, x0.5); 2.7 mm rejected |

## Known limits

- One attempt per bin and one offset direction in the library: the registration-from-outcomes fit
  in `probe_check.py` cannot be tested on it (it needs real attempts in several directions).
- The real descents are by hand (GELLO), not a steady 5 mm/s with a clean stop; expect more spread.
- The moment-direction feature is reported, not scored: between two simulated seeds it differs
  by 26 deg (median).
- The grid is coarse. Read the result as "nearest grid point"; a finer sweep runs on the simulation
  machine once real data says where to look.
