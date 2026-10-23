#!/usr/bin/env bash
# Simulation runs with the provisional measured parameters (outputs/ur_nut/contact/measured.json:
# fin-ray grasp, measured grip ring_bottom_below_flange 0.225, rod 16 mm, ring 8 g). Training at
# 0.5 mm radial clearance (bore unmeasured; the RGB fit says 2.7 mm), evaluation at 0.5 / 1.0 / 2.7.
# Everything here is simulation.
set -u
cd "${R2R:?set R2R to the real2render2real checkout}"
Q=outputs/ur_nut/measured
P=outputs/ur_nut/contact/measured.json
G05='{"geometry": {"ring_bore_r": 0.0085}}'
KIN='{"vision_source": "kinematic", "vision_noise": 0.0001}'
mkdir -p $Q
run() {  # log, done regex, script, args...
  log=$1; shift; re=$1; shift
  scripts/ur_cube/isaac_run.sh $log "$re" "$@" > /dev/null 2>&1
}
res() { grep -E '^\[RESULT\]' $1 | cut -c1-330; grep -E 'Traceback' -A8 $1 | tail -8; }

# 0. smoke test of the kinematic vision source (4 envs, 1 round)
run $Q/smoke_kinematic.log "ROLLOUTS DONE" scripts/ur_contact/run_dagger_rollouts.py --policy replay --num-envs 4 --rounds 1 \
  --seed 5 --eval --params $P --set "$G05" --env "$KIN" --output $Q/smoke_kinematic.npz
echo "== smoke kinematic: $(res $Q/smoke_kinematic.log)"
grep -q '^\[RESULT\]' $Q/smoke_kinematic.log || { echo "== smoke test failed; stopping"; exit 1; }

# 1. held-out probe runs for the offline check (new sensor seed; one on the grid, one off it)
L=outputs/ur_nut/probe_library/heldout; mkdir -p $L
for spec in "c1.0_k1_mu0.3 0.009 1 0.3" "c1.5_k0.7_mu0.45 0.0095 0.7 0.45"; do
  set -- $spec
  sj=$(python3 -c "
import json; k=$3
print(json.dumps({'geometry': {'ring_bore_r': $2}, 'contact': {'static_friction': $4, 'dynamic_friction': $4},
 'in_hand': {'k_lin_xyz': [6000*k, 1500*k, 2500*k], 'k_ang_xyz': [0.3*k, 1.0*k, 0.3*k]}}))")
  run $L/$1.log "PROBE DONE" scripts/ur_contact/run_contact_probe.py --controller stiff --f-stop 8 --seed 7 \
    --params $P --set "$sj" --output $L/$1
  echo "== heldout probe $1: $(grep -c '^\[PROBE\]' $L/$1.log) conditions"
done

# 2. speed x impedance (admittance controller) at 0.5 mm clearance: seat rate, peak force, time
S=$Q/speed_impedance; mkdir -p $S
for v in 2.5 5 10 20; do
  for d in 0.5 1 2; do
    sj=$(python3 -c "
import json; d=$d
print(json.dumps({'geometry': {'ring_bore_r': 0.0085}, 'controller': {'damping': [300*d, 300*d, 400*d, 2*d, 2*d, 2*d]}}))")
    run $S/v${v}_d${d}.log "PROBE DONE" scripts/ur_contact/run_contact_probe.py --speed $v --params $P --set "$sj" --output $S/v${v}_d${d}
    echo "== speed $v mm/s, damping x$d: $(grep -c '^\[PROBE\]' $S/v${v}_d${d}.log) conditions"
  done
done

# 3. DAgger, oracle vision stand-in (as before) and the deployable kinematic estimate
PARAMS=$P EXPERT_STYLE=contact bash scripts/ur_contact/dagger.sh $Q/dagger "$G05" > $Q/dagger_driver.log 2>&1
grep -E "^== eval" $Q/dagger_driver.log | cut -c1-330
PARAMS=$P ENV_JSON="$KIN" EXPERT_STYLE=contact bash scripts/ur_contact/dagger.sh $Q/dagger_kinematic "$G05" > $Q/dagger_kinematic_driver.log 2>&1
grep -E "^== eval" $Q/dagger_kinematic_driver.log | cut -c1-330

# 4. residual RL: blind-replay base (90 min) and DAgger-student base (60 min)
ev() {  # out log stem, args
  o=$1; shift
  run $o.log "ROLLOUTS DONE" scripts/ur_contact/run_dagger_rollouts.py --num-envs 64 --rounds 2 --seed 100 --eval --params $P \
    --output $o.npz "$@"
}
for spec in "replay_base_90m 90 replay" "student_base_60m 60 student"; do
  set -- $spec
  R=$Q/residual_$1; mkdir -p $R
  b=(--base $3); [ "$3" = student ] && b+=(--student $Q/dagger/student_it4.pt)
  run $R/train.log "RL DONE" scripts/ur_contact/run_residual_rl.py --num-envs 128 --minutes $2 --lr-anneal --checkpoint-every-min 15 \
    --params $P --set "$G05" "${b[@]}" --output $R
  echo "== residual $1 train: $(grep -E '^RL DONE' $R/train.log)"
  for ck in $R/residual_*m.pt $R/residual.pt; do
    [ -f "$ck" ] || continue
    n=$(basename $ck .pt)
    ev $R/eval_$n --policy residual --residual $ck --set "$G05"
    echo "== residual $1 eval $n: $(res $R/eval_$n.log)"
  done
done

# 5. the same policies at the other clearances (bore unmeasured: 1.0 mm and the RGB fit's 2.7 mm)
X=$Q/clearance; mkdir -p $X
for c in 1.0 2.7; do
  gj="{\"geometry\": {\"ring_bore_r\": $(python3 -c "print(round(0.008 + $c / 1000, 6))")}}"
  ev $X/c${c}_expert --policy expert --set "$gj"
  ev $X/c${c}_wrench_expert --policy wrench_expert --set "$gj"
  ev $X/c${c}_replay --policy replay --set "$gj"
  ev $X/c${c}_dagger --beta 0 --student $Q/dagger/student_it4.pt --set "$gj"
  ev $X/c${c}_dagger_kinematic --beta 0 --student $Q/dagger_kinematic/student_it4.pt --env "$KIN" --set "$gj"
  ev $X/c${c}_residual_replay --policy residual --residual $Q/residual_replay_base_90m/residual.pt --set "$gj"
  ev $X/c${c}_residual_student --policy residual --residual $Q/residual_student_base_60m/residual.pt --set "$gj"
  for f in $X/c${c}_*.log; do echo "== clearance $c $(basename $f .log): $(res $f)"; done
done
echo QUEUE DONE
