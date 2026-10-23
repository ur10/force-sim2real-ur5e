#!/usr/bin/env bash
# Install this project into a real2render2real checkout that already has the real2render2real-ur5e
# overlay installed (that overlay provides the UR5e + Robotiq 2F-85 config, the URDF/USD build, the
# nut task and Isaac Lab setup):  https://github.com/ur10/real2render2real-ur5e
#
#   ./install.sh /path/to/real2render2real
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
R2R="$(cd "${1:?usage: ./install.sh /path/to/real2render2real}" && pwd)"
V="$R2R/real2render2real/isaaclab_viser"
[ -f "$V/configs/articulation_configs/ur5e_robotiq_cfg.py" ] || {
  echo "the real2render2real-ur5e overlay is not installed in $R2R (run its install.sh first)" >&2; exit 1; }
mkdir -p "$R2R/scripts/ur_contact/robot_side/deploy" "$R2R/scripts/ur_tape" \
         "$R2R/outputs/ur_nut/physical_export" "$R2R/outputs/ur_nut/contact" "$R2R/outputs/ur_nut/probe_library" "$R2R/outputs/ur_nut/deploy"
cp "$HERE"/contact_sim/*.py "$HERE"/contact_sim/*.sh "$HERE"/contact_sim/README.md "$R2R/scripts/ur_contact/"
cp "$HERE"/robot_side/*.py "$HERE"/robot_side/*.md "$HERE"/robot_side/*.sh "$R2R/scripts/ur_contact/robot_side/"
cp "$HERE"/robot_side/deploy/*.py "$R2R/scripts/ur_contact/robot_side/deploy/"
# generator fix: the held nut is checked against the full bolt stand, and lifted over the rod tip after the warp
cp "$HERE"/generator_fix/tape_plan.py "$HERE"/generator_fix/export_ring_trajs.py "$HERE"/generator_fix/test_ring_clearance.py "$R2R/scripts/ur_tape/"
cp "$HERE"/generator_fix/ur5e_tape_pnp.py "$V/ur5e_simulators/"
cp "$HERE"/assets/physical_export/* "$R2R/outputs/ur_nut/physical_export/"
cp "$HERE"/assets/contact/*.json "$R2R/outputs/ur_nut/contact/"
cp -r "$HERE"/assets/probe_library/. "$R2R/outputs/ur_nut/probe_library/"
cp "$HERE"/assets/deploy/* "$R2R/outputs/ur_nut/deploy/"
echo "installed into $R2R"
echo "offline tests:   cd $R2R && .venv/bin/python scripts/ur_contact/test_contact_modules.py"
echo "robot folder:    $R2R/scripts/ur_contact/robot_side/make_robot_folder.sh <dest>   (or from this repo: robot_side/make_robot_folder.sh)"
