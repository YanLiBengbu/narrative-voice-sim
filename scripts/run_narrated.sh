#!/usr/bin/env bash
# Improvement experiment 2: speak-then-narrate. Each Actor speaks in its own voice and the Director
# narrates the speech in free indirect style (pronouns, tense and at most a reporting phrase change).
# Same excerpts, models and framework settings as the main experiment, seeds 0-2.
# Finished sessions and judgements are skipped.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
RUNS=${RUNS:-runs}
RES=${RES:-results}
W=${WORKERS:-4}

while pgrep -f "^bash scripts/run_voice.sh" > /dev/null; do sleep 60; done   # do not compete with improvement experiment 1
# completion pass for improvement 1 (sessions that failed because of GPU contention) and their judgements
$PY scripts/run_generation.py --models llama3.1:8b qwen2.5:14b gemma3:12b --modes third_person_voice \
    --seeds 0 1 2 3 4 --condition voice --workers "$W" --out "$RUNS"
for J in phi4:14b qwen3.5:9b-q4_K_M; do
  $PY scripts/run_judge.py --runs "$RUNS" --out "$RES" --judge "$J" --task session --conditions voice
  $PY scripts/run_judge.py --runs "$RUNS" --out "$RES" --judge "$J" --task speaker --conditions voice --seeds 0 1 2
done
$PY scripts/run_generation.py --models llama3.1:8b qwen2.5:14b gemma3:12b --modes third_person_narrated \
    --seeds 0 1 2 --condition narrated --workers "$W" --out "$RUNS"
for J in phi4:14b qwen3.5:9b-q4_K_M; do
  $PY scripts/run_judge.py --runs "$RUNS" --out "$RES" --judge "$J" --task session --conditions narrated
  $PY scripts/run_judge.py --runs "$RUNS" --out "$RES" --judge "$J" --task speaker --conditions narrated --seeds 0 1 2
done
$PY scripts/featurize.py --runs "$RUNS" --out "$RES"
$PY scripts/analyze.py --res "$RES"
$PY scripts/make_figures.py --res "$RES"
echo NARRATED_DONE
