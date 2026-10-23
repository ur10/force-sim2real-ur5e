#!/usr/bin/env bash
# Trimmed replacement for the rest of queue.sh + queue2.sh + queue3.sh (2026-11-22, to finish in
# about 2.25 h): residuals 40 min each (same budget for both bases), final checkpoints only, no
# wrench-only expert at other clearances, no duplicate baselines. Simulation, provisional parameters.
set -u
cd "${R2R:?set R2R to the real2render2real checkout}"
Q=outputs/ur_nut/measured
P=outputs/ur_nut/contact/measured.json
G05='{"geometry": {"ring_bore_r": 0.0085}}'
KIN='{"vision_source": "kinematic", "vision_noise": 0.0001}'
SEN='{"geometry": {"ring_bore_r": 0.0085}, "sensor": {"force_noise_n": 0.15, "torque_noise_nm": 0.004, "drift_n_per_s": 0.02,
 "gripper_mass": 1.35, "gripper_cog_tool0": [-0.004, -0.010, 0.043], "payload_mass_error": -0.35, "payload_cog_error": [0.004, 0.010, -0.043]}}'
WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 5; done; fi
res() { grep -E '^\[RESULT\]' $1 | cut -c1-330; grep -E 'Traceback' -A8 $1 | tail -8; }
run() { log=$1; shift; re=$1; shift; scripts/ur_cube/isaac_run.sh $log "$re" "$@" > /dev/null 2>&1; }
ev() { o=$1; shift; run $o.log "ROLLOUTS DONE" scripts/ur_contact/run_dagger_rollouts.py --num-envs 64 --rounds 2 --seed 100 --eval --params $P \
  --output $o.npz "$@"; echo "== $(basename $(dirname $o))/$(basename $o): $(res $o.log)"; }
D=$Q/dagger/student_it4.pt
DK=$Q/dagger_kinematic/student_it4.pt

# 1. kinematic-vision student, held-out at 0.5 mm
ev $Q/dagger_kinematic/eval_student_it4 --beta 0 --student $DK --env "$KIN" --set "$G05"

# 2. residual RL, 40 min each: DAgger-student base, then blind-replay base
for spec in "student_base_40m student" "replay_base_40m replay"; do
  set -- $spec
  R=$Q/residual_$1; mkdir -p $R
  b=(--base $2); [ "$2" = student ] && b+=(--student $D)
  run $R/train.log "RL DONE" scripts/ur_contact/run_residual_rl.py --num-envs 128 --minutes 40 --lr-anneal --checkpoint-every-min 20 \
    --params $P --set "$G05" "${b[@]}" --output $R
  echo "== residual $1 train: $(grep -E '^RL DONE' $R/train.log)"
  ev $R/eval_residual --policy residual --residual $R/residual.pt --set "$G05"
done
RS=$Q/residual_student_base_40m/residual.pt
RR=$Q/residual_replay_base_40m/residual.pt

# 3. other clearances (bore unmeasured: 1.0 mm and the RGB fit's 2.7 mm)
X=$Q/clearance; mkdir -p $X
for c in 1.0 2.7; do
  gj="{\"geometry\": {\"ring_bore_r\": $(python3 -c "print(round(0.008 + $c / 1000, 6))")}}"
  ev $X/c${c}_expert --policy expert --set "$gj"
  ev $X/c${c}_replay --policy replay --set "$gj"
  ev $X/c${c}_dagger --beta 0 --student $D --set "$gj"
  ev $X/c${c}_dagger_kinematic --beta 0 --student $DK --env "$KIN" --set "$gj"
  ev $X/c${c}_residual_student --policy residual --residual $RS --set "$gj"
  ev $X/c${c}_residual_replay --policy residual --residual $RR --set "$gj"
done

# 4. identified real sensor (0.5 mm)
I=$Q/identified_sensor; mkdir -p $I
ev $I/replay --policy replay --set "$SEN"
ev $I/dagger --beta 0 --student $D --set "$SEN"
ev $I/dagger_kinematic --beta 0 --student $DK --env "$KIN" --set "$SEN"
ev $I/residual_student --policy residual --residual $RS --set "$SEN"
ev $I/residual_replay --policy residual --residual $RR --set "$SEN"

# 5. videos: 4 held-out episodes per policy, 0.5 mm
V=$Q/videos; mkdir -p $V
vid() { name=$1; shift; run $V/$name.log "ROLLOUTS DONE" scripts/ur_contact/run_dagger_rollouts.py --num-envs 4 --rounds 1 --seed 100 --eval \
  --params $P --set "$G05" --video-dir $V --video-name $name --output $V/$name.npz "$@"; echo "== video $name: $(res $V/$name.log | cut -c1-200)"; }
vid replay --policy replay
vid dagger_kinematic --beta 0 --student $DK --env "$KIN"
vid dagger --beta 0 --student $D
vid residual_dagger --policy residual --residual $RS
vid expert --policy expert
cols="replay dagger_kinematic dagger residual_dagger expert"
inputs=""; layout=""; k=0
for i in 0 1 2 3; do for p in $cols; do
  inputs="$inputs -i $V/${p}_env$i.mp4"; layout="$layout${layout:+|}$(( (k % 5) * 320 ))_$(( (k / 5) * 240 ))"; k=$((k+1)); done; done
filt=""; ins=""; for j in $(seq 0 19); do filt="$filt[$j:v]scale=320:240[s$j];"; ins="$ins[s$j]"; done
/usr/bin/ffmpeg -v error -y $inputs -filter_complex "${filt}${ins}xstack=inputs=20:layout=$layout[v]" -map "[v]" -c:v libx264 \
  -pix_fmt yuv420p -crf 24 -movflags +faststart $V/comparison_measured.mp4 && echo "== video comparison: $V/comparison_measured.mp4"

# 6. expert speed sweep (x1 is dagger/eval_expert)
XS=$Q/expert_speed; mkdir -p $XS
for s in 0.5 2 4; do ev $XS/s$s --policy expert --set "$G05" --env "{\"speed_scale\": $s}"; done
echo FAST DONE
