#!/usr/bin/env bash
# Privileged expert + DAgger on the nut contact phase (simulation, uncalibrated).
# Collections run in Isaac on GPU 0; the student trains on GPU 1.
#   scripts/ur_contact/dagger.sh [out_dir] [geometry JSON override]
set -u
cd "$(dirname "$0")/../.."
out=${1:-outputs/ur_nut/dagger}
geom=${2:-'{"geometry": {"ring_bore_r": 0.0083}}'}   # printed ring, 0.5 mm radial clearance
envs=${ENVS:-64}
style=${EXPERT_STYLE:-contact}   # privileged expert: contact (imitable) or hover (not imitable)
extra=()                          # PARAMS=<measured.json>, ENV_JSON=<EnvCfg overrides> (e.g. kinematic vision)
[ -n "${PARAMS:-}" ] && extra+=(--params "$PARAMS")
[ -n "${ENV_JSON:-}" ] && extra+=(--env "$ENV_JSON")
mkdir -p $out
roll() {  # name, extra args
  name=$1; shift
  scripts/ur_cube/isaac_run.sh $out/$name.log "ROLLOUTS DONE" scripts/ur_contact/run_dagger_rollouts.py \
    --num-envs $envs --expert-style $style --set "$geom" "${extra[@]}" --output $out/$name.npz "$@" > /dev/null 2>&1
  echo "== $name: $(grep -E '^\[RESULT\]' $out/$name.log | cut -c1-500)"
  grep -E 'Traceback' -A8 $out/$name.log | tail -8
}
train() {  # output, extra args
  o=$1; shift
  .venv/bin/python scripts/ur_contact/train_student.py --shards $out/shard_it*.npz --output $out/$o "$@" 2>&1 | tail -1
}
roll shard_it0 --beta 1.0 --rounds 4 --seed 0
train student_it1.pt
roll shard_it1 --beta 0.5 --rounds 4 --seed 1 --student $out/student_it1.pt
train student_it2.pt
roll shard_it2 --beta 0.25 --rounds 4 --seed 2 --student $out/student_it2.pt
train student_it3.pt
roll shard_it3 --beta 0.0 --rounds 4 --seed 3 --student $out/student_it3.pt
train student_it4.pt
train student_it4_nowrench.pt --no-wrench
train student_it4_novision.pt --no-vision
# held-out evaluation (new seed: new in-hand and approach errors)
for s in student_it1 student_it2 student_it3 student_it4 student_it4_nowrench student_it4_novision; do
  roll eval_$s --beta 0 --rounds 2 --seed 100 --eval --student $out/$s.pt
done
for pol in expert wrench_expert replay; do
  roll eval_$pol --policy $pol --rounds 2 --seed 100 --eval
done
echo DAGGER DONE
