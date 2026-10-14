# narrative-voice-sim

Code, prompts, generated dialogues and analysis scripts for the paper

> Lili Zhao and Hui Shi. *An LLM-driven multi-agent simulation framework for quantitative analysis of narrative voice shifts in literary texts.* Manuscript submitted to PeerJ Computer Science (CS-2026:04:138274).

Project page: https://yanlibengbu.github.io/narrative-voice-sim/ (GitHub Pages, source: `docs/index.html`). Repository: https://github.com/YanLiBengbu/narrative-voice-sim

## What the framework does

1. **Director Agent** extracts the direct-speech turns and the characters of a literary excerpt, writes a role card for every character, maps speaker labels to a verified cast list (`texts/excerpts.json`), and rewrites the opening turns into third-person narration when needed.
2. **Actor Agents** (one per character) generate the next turn from their role card, the scene context and the last *k* turns (sliding-window memory, default *k* = 5).
3. A **seeded speaker-selection policy** chooses the next speaker (addressee-aware softmax, Eq. 1 of the paper; ablations: uniform, round robin).
4. The **narrative mode** is controlled by the Director: `first_person` (direct speech), `third_person` (external narrator's report) or `neutral` (no perspective instruction). Every turn is checked; non-compliant turns are regenerated or rewritten, and all of this is logged per turn.
   Two further third-person modes were added to improve the preservation of character voice: `third_person_voice` (voice-preserving narration requested in the Actor prompt) and `third_person_narrated` (speak-then-narrate: the Actor speaks in its own voice and the Director narrates the speech in free indirect style; the original speech is stored with each turn).
5. **Analysis**: pronoun profile (manipulation check); sentiment (DistilBERT, TextBlob, VADER), subjectivity, utterance length, content-word length and Zipf frequency, MATTR/MTLD; turn-taking measures; Jensen–Shannon character consistency against a between-character baseline; paired Wilcoxon tests with bootstrap CIs, effect sizes, Holm correction and mixed models; LLM-judge evaluation; a small blinded expert evaluation (three raters, 12 sessions, 36 masked turns).

All LLM calls use fixed seeds and are cached (`runs/cache_*.jsonl`), so every number in the paper can be regenerated from the released outputs without re-running the models.

## Repository layout

```
narrsim/            framework package
  llm.py            Ollama client (seeded, JSON mode, response cache)
  prompts.py        all prompt templates (also printed in the paper's Appendix)
  agents.py         Director and Actor agents, mode-compliance check, single-prompt baseline
  session.py        speaker selection, memory window, enforcement loop (Algorithm 1)
  jsonfix.py        deterministic JSON repair
  metrics.py        utterance/session measures, rule-based transposition baseline
  stats.py          paired tests, bootstrap CIs, Holm, mixed models, JSD
scripts/
  run_generation.py generate sessions (main experiment and ablations)
  run_judge.py      LLM-judge evaluation (session ratings, masked speaker identification)
  analyze_human_eval.py  expert evaluation -> results/human_eval_summary.json, human_eval_*.csv
  eval_extraction.py Director extraction vs. gold quotation attribution
  featurize.py      compute all measures -> results/utterances.csv, sessions.csv
  analyze.py        statistics -> results/table_*.csv, summary.json
  make_figures.py   figures of the paper
  run_all.sh        the complete pipeline used for the paper
  run_extension.sh  voice-exemplar extension and a completion pass
  run_voice.sh      improvement experiment 1 (voice-preserving third-person instruction)
  run_narrated.sh   improvement experiment 2 (speak-then-narrate)
texts/excerpts.json six excerpts with metadata, verified cast lists and gold quotation attributions
texts/good_earth_release.py, good_earth_manifest.json  restore the copyrighted The Good Earth excerpt (see Data)
runs/               generated sessions, scene initialisations and LLM response caches
results/            tables, judge outputs, expert ratings (anonymized), figures
expert_evaluation_protocol/  blinded rating sheet (Chinese instructions) and condition key of the expert evaluation
docs/index.html     project page
```

## Requirements

* Python >= 3.10 and the packages in `requirements.txt`
* `python -m spacy download en_core_web_sm`; NLTK `vader_lexicon`
* [Ollama](https://ollama.com) >= 0.4 with the models
  `llama3.1:8b`, `qwen2.5:14b`, `gemma3:12b` (generators) and `phi4:14b`, `qwen3.5:9b-q4_K_M` (judges).
  The paper used Ollama 0.40.3 on an RTX 4070 Ti SUPER (16 GB).
* No API keys are needed; everything runs locally.

## Reproducing the paper

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon')"
for m in llama3.1:8b qwen2.5:14b gemma3:12b phi4:14b qwen3.5:9b-q4_K_M; do ollama pull $m; done
PY=python WORKERS=4 bash scripts/run_all.sh      # generation, judges, analysis
PY=python WORKERS=4 bash scripts/run_extension.sh
PY=python WORKERS=4 bash scripts/run_voice.sh
PY=python WORKERS=4 bash scripts/run_narrated.sh
python scripts/make_figures.py --res results     # figures
python scripts/make_facts.py --res results       # every number quoted in the paper -> results/facts.json
```

The expert evaluation used the rating sheet in `expert_evaluation_protocol/` (12 sessions and 36 masked turns, blind to model and condition). The raters' responses were entered into the sheets by one of the authors; the anonymized item-level ratings are in `results/human_eval_session_ratings.csv` and `results/human_eval_speaker_guesses.csv`, and `scripts/analyze_human_eval.py --key HumanEval_Short_Key.xlsx --kits rater*.xlsx --res results` computes the statistics reported in subsection 3.11 of the paper.

To recompute only the statistics from the released outputs:

```bash
python scripts/featurize.py --runs runs --out results
python scripts/analyze.py --res results
python scripts/make_figures.py --res results
```

Single run on your own excerpt (any `.json` in the format of `texts/excerpts.json`):

```bash
python scripts/run_generation.py --excerpts my_texts.json --models llama3.1:8b \
       --modes first_person third_person neutral --seeds 0 1 2 --condition full --out my_runs
```

## Data

`texts/excerpts.json` contains six short excerpts: five public-domain texts from Project Gutenberg (*Pride and Prejudice*, *Great Expectations*, *A Scandal in Bohemia*, *The Importance of Being Earnest*, *The Home and the World*) and an excerpt of chapter 26 of *The Good Earth* (Pearl S. Buck, 1931). For each excerpt the file lists the verified cast (with aliases) and the manually attributed quotations used to evaluate the Director's extraction.

*The Good Earth* is still under copyright, so the released files do not contain the text of its excerpt: every passage of 25 or more characters from it (in `texts/excerpts.json`, the scene initialisations, the session openings, the cached LLM responses and `results/source_reference.csv`) is replaced by a placeholder `[GE-OMITTED:n:enc]`. The dialogues generated by the models are not affected, except for two generated turns in which a model reproduced a sentence of the excerpt verbatim; the published statistics were computed on the complete data and are reproduced exactly after restoration. With a copy of the book, the files can be restored exactly; the script checks the excerpt against SHA-256 checksums before it writes anything:

```bash
python texts/good_earth_release.py restore --excerpt good_earth_ch26.txt --root .
```

`good_earth_ch26.txt` must contain the excerpt of chapter 26 whose first and last words are listed in `texts/good_earth_manifest.json` (901 words; curly or straight quotation marks and line breaks are accepted).

## License

Code: MIT (see `LICENSE`). Generated dialogues and result tables: CC BY 4.0.

## Citation

See `CITATION.cff`.
