#!/usr/bin/env bash
# Assemble the robot-machine folder (numpy + ur_rtde only; no Isaac, no torch):
#   <dest>/deploy/                 deploy_policy.py (MOVES THE ROBOT), deploy_core.py, fake_ur.py, test_deploy.py
#   <dest>/sim_library/deploy/     policy bundle, simulation reference log, rod-registration guess
#   <dest>/scripts/, <dest>/tests/ offline probe check (+ the GELLO collection script, which needs the
#                                  lab's GELLO repo and the ring_rod_probe toolkit to run)
#   <dest>/sim_library/            simulated probe library + held-out fixture
#   <dest>/config/                 measured.json (the PROVISIONAL values) and a robot.yaml template,
#                                  unless they already exist: set the robot IP in robot.yaml
#   <dest>/SESSION_GUIDE.md
# Works from this repository or from an installed real2render2real checkout.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:?usage: make_robot_folder.sh <dest>}"
if [ -d "$HERE/../assets" ]; then          # this repository
  A="$(cd "$HERE/../assets" && pwd)"
  LIB="$A/probe_library"; DEP="$A/deploy"; MEAS="$A/contact/measured_robot_format.json"
else                                      # installed: <r2r>/scripts/ur_contact/robot_side
  R="$(cd "$HERE/../../.." && pwd)"
  LIB="$R/outputs/ur_nut/probe_library"; DEP="$R/outputs/ur_nut/deploy"; MEAS="$R/outputs/ur_nut/contact/measured_robot_format.json"
fi
mkdir -p "$DEST/deploy" "$DEST/sim_library/deploy" "$DEST/scripts" "$DEST/tests" "$DEST/config"
cp "$HERE"/deploy/*.py "$DEST/deploy/"
cp "$DEP"/bundle_dagger_kinematic.npz "$DEP"/reference.npz "$DEP"/rod_registration_guess.json "$DEST/sim_library/deploy/"
for f in residual_equivalence_fixture.npz bundle_residual_kinematic.npz; do if [ -f "$DEP/$f" ]; then cp "$DEP/$f" "$DEST/sim_library/deploy/"; fi; done
cp "$HERE"/probe_features.py "$HERE"/probe_check.py "$HERE"/collect_contact.py "$DEST/scripts/"
cp "$HERE"/test_probe_check.py "$HERE"/test_collect_contact.py "$DEST/tests/"
cp "$LIB"/library.json "$DEST/sim_library/probe_library.json"
cp -r "$LIB"/heldout_c1.0_k1_mu0.3 "$DEST/sim_library/"
cp "$HERE"/SIM_LIBRARY.md "$DEST/sim_library/README.md"
cp "$HERE"/SESSION_GUIDE.md "$DEST/"
[ -f "$DEST/config/measured.json" ] || cp "$MEAS" "$DEST/config/measured.json"
[ -f "$DEST/config/robot.yaml" ] || printf 'robot:\n  ip: "0.0.0.0"   # set the UR5e IP here\n' > "$DEST/config/robot.yaml"
echo "assembled $DEST (config/measured.json = the PROVISIONAL values; set the robot IP in config/robot.yaml)"
echo "then: cd $DEST && python deploy/test_deploy.py && python deploy/deploy_policy.py --dry-run -y --policy student"
