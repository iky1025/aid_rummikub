#!/bin/zsh
# R10 DAgger round 1: student plays (visiting its own failure states),
# teacher labels every decision with action + candidate scores.
# Launch detached: nohup zsh run_dagger_gen.sh >> R8/dagger1_datagen.log 2>&1 &
PY=/opt/homebrew/Caskroom/miniconda/base/envs/rummikub/bin/python
OUT=data/dagger1
TARGET=500
cd "$(dirname "$0")"
for attempt in {1..50}; do
  n=$(ls "$OUT" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -ge "$TARGET" ]; then
    echo "[wrapper] COMPLETE: $n/$TARGET pairs ($(date))"
    exit 0
  fi
  echo "[wrapper] attempt $attempt: $n/$TARGET pairs, (re)starting ($(date))"
  caffeinate -is "$PY" selfplay_data.py \
    --teacher rollout --consistent --greedy-margin 1.0 --endgame-search \
    --determinizations 8 --rollout-turns 24 --candidate-cap 4 \
    --actor student --actor-model distill_s1s_t03_v05.pt \
    --pairs $TARGET --seed 220000 --initial-meld-value 30 \
    --out $OUT --workers 8
  sleep 10
done
echo "[wrapper] gave up after 50 attempts ($(date))"
exit 1
