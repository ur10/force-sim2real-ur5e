#!/usr/bin/env bash
# Fin-ray grasp: cross-evaluate the old policies, videos, then retrain DAgger and residual RL.
set -u
cd "${R2R:?set R2R to the real2render2real checkout}"
F=outputs/ur_nut/finray
G='{"geometry": {"ring_bore_r": 0.0083}}'
until grep -q "LONG DONE" outputs/ur_nut/residual/long.log 2>/dev/null; do sleep 15; done
# 1. fin-ray becomes the default for every run from here on
sed -i 's/    grasp_model: str = "center"           # choice: "finray" for new runs (default switches once the 90-min run is evaluated)/    grasp_model: str = "finray"           # choice: "center" reproduces the runs before 2026-11-21 evening/' scripts/ur_contact/params.py
grep -q 'grasp_model: str = "finray"' scripts/ur_contact/params.py && echo "== default grasp model: finray" || { echo "== could not switch the default; stopping"; exit 1; }
ev() {  # name, args
  name=$1; shift
  scripts/ur_cube/isaac_run.sh $F/eval_$name.log "ROLLOUTS DONE" scripts/ur_contact/run_dagger_rollouts.py --num-envs 64 --rounds 2 \
    --seed 100 --eval --set "$G" --output $F/eval_$name.npz "$@" > /dev/null 2>&1
  echo "== eval $name: $(grep -E '^\[RESULT\]' $F/eval_$name.log | cut -c1-330)"
  grep -E "Traceback" -A6 $F/eval_$name.log | tail -6
}
# 2. held-out evaluation under fin-ray (policies trained with the old grasp model, unchanged)
ev expert --policy expert
ev wrench_expert --policy wrench_expert
ev replay --policy replay
ev dagger_center_it4 --beta 0 --student outputs/ur_nut/dagger_contact/student_it4.pt
ev residual_center_45m --policy residual --residual outputs/ur_nut/residual/replay_base/residual.pt
ev residual_center_90m --policy residual --residual outputs/ur_nut/residual/replay_base_90m/residual.pt
# 3. videos, 4 held-out episodes
vid() {
  name=$1; shift
  scripts/ur_cube/isaac_run.sh $F/videos/$name.log "ROLLOUTS DONE" scripts/ur_contact/run_dagger_rollouts.py --num-envs 4 --rounds 1 \
    --seed 100 --eval --set "$G" --video-dir $F/videos --video-name $name --output $F/videos/$name.npz "$@" > /dev/null 2>&1
  echo "== video $name: $(grep -E '^\[RESULT\]' $F/videos/$name.log | cut -c1-200)"
}
vid replay --policy replay
vid dagger_center --beta 0 --student outputs/ur_nut/dagger_contact/student_it4.pt
vid residual_center_90m --policy residual --residual outputs/ur_nut/residual/replay_base_90m/residual.pt
vid expert --policy expert
inputs=""; layout=""; k=0
for i in 0 1 2 3; do for p in replay dagger_center residual_center_90m expert; do
  inputs="$inputs -i $F/videos/${p}_env$i.mp4"; layout="$layout${layout:+|}$(( (k % 4) * 400 ))_$(( (k / 4) * 300 ))"; k=$((k+1)); done; done
filt=""; ins=""; for j in $(seq 0 15); do filt="$filt[$j:v]scale=400:300[s$j];"; ins="$ins[s$j]"; done
/usr/bin/ffmpeg -v error -y $inputs -filter_complex "${filt}${ins}xstack=inputs=16:layout=$layout[v]" -map "[v]" -c:v libx264 \
  -pix_fmt yuv420p -crf 23 $F/videos/comparison_4episodes.mp4 && echo "== video comparison: $F/videos/comparison_4episodes.mp4"
# 4. DAgger under fin-ray (contact teacher)
EXPERT_STYLE=contact bash scripts/ur_contact/dagger.sh $F/dagger > $F/dagger_driver.log 2>&1
grep -E "^== eval" $F/dagger_driver.log | cut -c1-330
# 5. residual RL under fin-ray, 90 min, checkpoint evals
R=$F/residual_90m; mkdir -p $R
scripts/ur_cube/isaac_run.sh $R/train.log "RL DONE" scripts/ur_contact/run_residual_rl.py --num-envs 128 --minutes 90 --lr-anneal \
  --checkpoint-every-min 15 --output $R > /dev/null 2>&1
echo "== residual train: $(grep -E '^RL DONE' $R/train.log)"
for ck in $R/residual_*m.pt $R/residual.pt; do
  [ -f "$ck" ] || continue
  n=$(basename $ck .pt)
  scripts/ur_cube/isaac_run.sh $R/eval_$n.log "ROLLOUTS DONE" scripts/ur_contact/run_dagger_rollouts.py --policy residual --residual $ck \
    --num-envs 64 --rounds 2 --seed 100 --eval --set "$G" --output $R/eval_$n.npz > /dev/null 2>&1
  echo "== residual eval $n: $(grep -E '^\[RESULT\]' $R/eval_$n.log | cut -c1-330)"
done
echo QUEUE DONE
