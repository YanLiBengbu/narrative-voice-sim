#!/usr/bin/env bash
# Improvement experiment: voice-preserving third-person narration (free indirect style).
# Same excerpts, models, seeds and framework settings as the main experiment; only the
# third-person mode instruction differs. Finished sessions and judgements are skipped.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
RUNS=${RUNS:-runs}
RES=${RES:-results}
W=${WORKERS:-4}

$PY scripts/run_generation.py --models llama3.1:8b qwen2.5:14b gemma3:12b --modes third_person_voice \
    --seeds 0 1 2 3 4 --condition voice --workers "$W" --out "$RUNS"
for J in phi4:14b qwen3.5:9b-q4_K_M; do
  $PY scripts/run_judge.py --runs "$RUNS" --out "$RES" --judge "$J" --task session --conditions voice
  $PY scripts/run_judge.py --runs "$RUNS" --out "$RES" --judge "$J" --task speaker --conditions voice --seeds 0 1 2
done
$PY scripts/featurize.py --runs "$RUNS" --out "$RES"
$PY scripts/analyze.py --res "$RES"
echo VOICE_DONE
