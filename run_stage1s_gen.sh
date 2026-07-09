#!/bin/zsh
# R10: self-healing overnight generation of the scored teacher dataset.
# Launch detached (nohup zsh run_stage1s_gen.sh >> R8/stage1s_datagen.log 2>&1 &)
# so session-side SIGTERMs can't reach it; the loop restarts the resumable
# generator after any crash until all pairs are on disk.
PY=/opt/homebrew/Caskroom/miniconda/base/envs/rummikub/bin/python
OUT=data/stage1s
TARGET=1000
cd "$(dirname "$0")"
for attempt in {1..50}; do
  n=$(ls "$OUT" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -ge "$TARGET" ]; then
    echo "[wrapper] COMPLETE: $n/$TARGET pairs ($(date))"
    exit 0
  fi
  echo "[wrapper] attempt $attempt: $n/$TARGET pairs on disk, (re)starting generator ($(date))"
  caffeinate -is "$PY" selfplay_data.py \
    --teacher rollout --consistent --greedy-margin 1.0 --endgame-search \
    --determinizations 8 --rollout-turns 24 --candidate-cap 4 \
    --pairs $TARGET --seed 210000 --initial-meld-value 30 \
    --out $OUT --workers 8
  sleep 10
done
echo "[wrapper] gave up after 50 attempts ($(date))"
exit 1
