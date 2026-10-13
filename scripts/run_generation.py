#!/usr/bin/env python3
"""Generate dialogue sessions for the main experiment and the ablations.

Conditions
  full          Director-enforced mode, addressee-aware speaker selection, memory k=5
  no_memory     memory window k=1 (actor sees only the previous turn)
  uniform       seeded uniform-random speaker selection (as in the original release)
  round_robin   fixed round-robin speaker order
  no_enforce    mode instruction kept, Director compliance check / regeneration disabled
  single_prompt one LLM call writes the whole continuation (no agents)
  exemplars     verbatim source lines of each character added to its role card (extension)
  voice         improvement 1: voice-preserving third-person instruction (mode third_person_voice)
  narrated      improvement 2: speak-then-narrate; the Actor speaks, the Director narrates the speech
                in free indirect style (mode third_person_narrated)
Modes: first_person, third_person, neutral (no perspective instruction), third_person_voice, third_person_narrated.

Example:
  python scripts/run_generation.py --models llama3.1:8b qwen2.5:14b gemma3:12b \
      --modes first_person third_person neutral --seeds 0 1 2 3 4 --condition full
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from narrsim.agents import DirectorAgent, mode_violation, single_prompt_session  # noqa: E402
from narrsim.llm import OllamaClient  # noqa: E402
from narrsim.session import run_session  # noqa: E402

CONDITIONS = {
    "full": dict(memory_k=5, policy="addressee", enforce=True),
    "no_memory": dict(memory_k=1, policy="addressee", enforce=True),
    "uniform": dict(memory_k=5, policy="uniform", enforce=True),
    "round_robin": dict(memory_k=5, policy="round_robin", enforce=True),
    "no_enforce": dict(memory_k=5, policy="addressee", enforce=False),
    "single_prompt": dict(memory_k=None, policy=None, enforce=False),
    "exemplars": dict(memory_k=5, policy="addressee", enforce=True, exemplars=True),
    "voice": dict(memory_k=5, policy="addressee", enforce=True),  # used with mode third_person_voice
    "narrated": dict(memory_k=5, policy="addressee", enforce=True),  # used with mode third_person_narrated
}


def slug(model):
    return model.replace(":", "_").replace("/", "_")


def log(msg, path):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_scene(llm, text, out_dir, logp):
    path = os.path.join(out_dir, "scenes", f"{text['id']}__{slug(llm.model)}.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for seed in range(4):
        d = DirectorAgent(llm, seed=seed)
        try:
            scene = d.initialise(text["text"], cast=text.get("cast"))
        except ValueError as e:
            log(f"scene init failed ({text['id']}, {llm.model}, seed {seed}): {e}", logp)
            continue
        if len(scene["dialogue"]) >= 2 and len(scene["characters"]) >= 2:
            third = d.rewrite_third(scene["dialogue"])
            for i, t in enumerate(third):  # Director re-renders opening items that still violate the mode
                if mode_violation(t["text"], "third_person", t["speaker"]):
                    r = d.render(scene["dialogue"][i]["text"], "third_person", t["speaker"], t.get("speak_to") or [], seed + i)
                    if r:
                        t["text"] = r
            scene["opening_third"] = third
            scene["director_stats"] = d.stats.as_dict()
            scene["init_seed"] = seed
            json.dump(scene, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            return scene
    raise RuntimeError(f"could not initialise scene {text['id']} with {llm.model}")


def one_session(llm, text, scene, mode, seed, cond, args, path):
    opening = scene["opening_third"] if mode.startswith("third_person") else scene["dialogue"]
    t0 = time.time()
    if cond == "single_prompt":
        turns, stats = single_prompt_session(llm, scene, opening[-5:], mode, args.max_turns, seed)
        res = dict(turns=turns, termination="single_call", attempts=1, call_stats=stats)
    else:
        c = CONDITIONS[cond]
        res = run_session(llm, scene, opening, mode, seed, max_turns=args.max_turns,
                          memory_k=c["memory_k"], policy=c["policy"], enforce=c["enforce"],
                          exemplars=c.get("exemplars", False))
    res.update(meta=dict(text_id=text["id"], model=llm.model, mode=mode, seed=seed, condition=cond,
                         max_turns=args.max_turns, temperature=llm.temperature, **{k: v for k, v in CONDITIONS[cond].items()}),
               opening=opening, characters=list(scene["characters"]), seconds=round(time.time() - t0, 1))
    tmp = path + ".tmp"
    json.dump(res, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return len(res["turns"]), res["seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama3.1:8b"])
    ap.add_argument("--texts", nargs="+", default=None)
    ap.add_argument("--modes", nargs="+", default=["first_person", "third_person", "neutral"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--condition", default="full", choices=list(CONDITIONS))
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--excerpts", default=os.path.join(os.path.dirname(__file__), "..", "texts", "excerpts.json"))
    ap.add_argument("--out", default="runs")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    args = ap.parse_args()

    texts = json.load(open(args.excerpts, encoding="utf-8"))
    if args.texts:
        texts = [t for t in texts if t["id"] in args.texts]
    os.makedirs(args.out, exist_ok=True)
    logp = os.path.join(args.out, "progress.log")

    for model in args.models:
        llm = OllamaClient(model, base_url=args.base_url, temperature=args.temperature,
                           cache_path=os.path.join(args.out, f"cache_{slug(model)}.jsonl"))
        jobs = []
        for text in texts:
            scene = get_scene(llm, text, args.out, logp)
            for mode in args.modes:
                for seed in args.seeds:
                    path = os.path.join(args.out, "sessions", args.condition, slug(model), text["id"], f"{mode}_s{seed}.json")
                    if os.path.exists(path):
                        continue
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    jobs.append((text, scene, mode, seed, path))
        log(f"{model} / {args.condition}: {len(jobs)} sessions to run", logp)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(one_session, llm, t, s, m, sd, args.condition, args, p): (t["id"], m, sd) for t, s, m, sd, p in jobs}
            for f in as_completed(futs):
                tid, m, sd = futs[f]
                try:
                    n, secs = f.result()
                    log(f"done {model} {args.condition} {tid} {m} s{sd}: {n} turns in {secs}s", logp)
                except Exception as e:
                    log(f"FAILED {model} {args.condition} {tid} {m} s{sd}: {e!r}", logp)


if __name__ == "__main__":
    main()
