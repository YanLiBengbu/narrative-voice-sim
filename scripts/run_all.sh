#!/usr/bin/env bash
# Full pipeline used for the revised manuscript (resumable: finished sessions are skipped).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
RUNS=${RUNS:-runs}
RES=${RES:-results}
W=${WORKERS:-4}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

# E1: main experiment - 6 texts x 3 models x 3 modes x 5 seeds
$PY scripts/run_generation.py --models llama3.1:8b qwen2.5:14b gemma3:12b \
    --modes first_person third_person neutral --seeds 0 1 2 3 4 --condition full --workers "$W" --out "$RUNS"

# E2: ablations - llama3.1:8b, 6 texts x 2 modes x 3 seeds
for c in no_memory uniform round_robin no_enforce single_prompt; do
  $PY scripts/run_generation.py --models llama3.1:8b --modes first_person third_person --seeds 0 1 2 \
      --condition "$c" --workers "$W" --out "$RUNS"
done

# E3: LLM judges (different model families from the generators where possible)
for J in phi4:14b qwen3.5:9b-q4_K_M; do
  $PY scripts/run_judge.py --runs "$RUNS" --out "$RES" --judge "$J" --task session \
      --conditions full no_memory uniform round_robin no_enforce single_prompt
  $PY scripts/run_judge.py --runs "$RUNS" --out "$RES" --judge "$J" --task speaker --conditions full --seeds 0 1 2
done

# Analysis
$PY scripts/eval_extraction.py --runs "$RUNS" --out "$RES"
$PY scripts/featurize.py --runs "$RUNS" --out "$RES"
$PY scripts/analyze.py --res "$RES"
echo PIPELINE_DONE
