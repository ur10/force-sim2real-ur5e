#!/usr/bin/env bash
# Simulated probe library for the offline check: GELLO-like stiff descent with a force stop,
# fin-ray grasp, provisional measurements; clearance x grip stiffness x friction.
set -u
cd "${R2R:?set R2R to the real2render2real checkout}"
O=outputs/ur_nut/probe_library
for clr in 2.7 1.0 0.5; do
  bore=$(python3 -c "print(round(0.008 + $clr / 1000, 6))")
  for ks in 0.5 1 2; do
    for mu in 0.3 0.6; do
      name=c${clr}_k${ks}_mu${mu}
      set_json=$(python3 -c "
import json
k=$ks
print(json.dumps({'geometry': {'ring_bore_r': $bore}, 'contact': {'static_friction': $mu, 'dynamic_friction': $mu},
 'in_hand': {'k_lin_xyz': [6000*k, 1500*k, 2500*k], 'k_ang_xyz': [0.3*k, 1.0*k, 0.3*k]}}))")
      scripts/ur_cube/isaac_run.sh $O/$name.log "PROBE DONE" scripts/ur_contact/run_contact_probe.py --controller stiff --f-stop 8 \
        --params outputs/ur_nut/contact/measured.json --set "$set_json" --output $O/$name > /dev/null 2>&1
      echo "== $name: $(grep -c '^\[PROBE\]' $O/$name.log) conditions $(grep -E 'Traceback' $O/$name.log | head -1)"
    done
  done
done
echo SWEEP DONE
