#!/usr/bin/env bash
# Extension experiment (voice-exemplar role cards), a completion pass that re-runs any session or
# judgement that failed (finished ones are skipped), and the complete re-analysis.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
RUNS=${RUNS:-runs}
RES=${RES:-results}
W=${WORKERS:-4}

# completion pass for the main experiment and the ablations
$PY scripts/run_generation.py --models llama3.1:8b qwen2.5:14b gemma3:12b \
    --modes first_person third_person neutral --seeds 0 1 2 3 4 --condition full --workers "$W" --out "$RUNS"
for c in no_memory uniform round_robin no_enforce single_prompt; do
  $PY scripts/run_generation.py --models llama3.1:8b --modes first_person third_person --seeds 0 1 2 \
      --condition "$c" --workers "$W" --out "$RUNS"
done
# extension: voice exemplars
$PY scripts/run_generation.py --models llama3.1:8b qwen2.5:14b gemma3:12b --modes first_person third_person \
    --seeds 0 1 2 --condition exemplars --workers "$W" --out "$RUNS"
# judges (finished judgements are skipped)
for J in phi4:14b qwen3.5:9b-q4_K_M; do
  $PY scripts/run_judge.py --runs "$RUNS" --out "$RES" --judge "$J" --task session \
      --conditions full no_memory uniform round_robin no_enforce single_prompt exemplars
  $PY scripts/run_judge.py --runs "$RUNS" --out "$RES" --judge "$J" --task speaker --conditions full exemplars --seeds 0 1 2
done
$PY scripts/eval_extraction.py --runs "$RUNS" --out "$RES"
$PY scripts/featurize.py --runs "$RUNS" --out "$RES"
$PY scripts/analyze.py --res "$RES"
$PY scripts/make_figures.py --res "$RES"
$PY scripts/export_tables.py --res "$RES" --runs "$RUNS"
$PY scripts/make_facts.py --res "$RES"
$PY scripts/export_supplement.py --res "$RES" --runs "$RUNS" --out "$RES/supplement.tex"
echo EXTENSION_DONE
