#!/usr/bin/env python3
"""LLM-as-judge evaluation (used as a scalable proxy for, not a replacement of, human judgement).

session  : blind 1-5 ratings of coherence, character consistency, scene fidelity, subjectivity,
           emotional intensity, plus a classification of the realised narrative form.
           Judges see neither the generating model nor the experimental condition.
speaker  : for sampled turns with every character name and gendered pronoun masked, the judge
           identifies the speaker among the scene's characters and rates the turn's subjectivity.
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from narrsim.jsonfix import parse_json  # noqa: E402
from narrsim.llm import OllamaClient  # noqa: E402

SESSION_PROMPT = """You are an expert in literary studies evaluating a computer-generated continuation of a scene from "{title}" by {author}.

Characters:
{cards}

Scene context: {context}

Continuation ({n} turns):
{turns}

Rate the continuation on each criterion from 1 (very poor / very low) to 5 (excellent / very high):
- coherence: the turns follow sensibly from one another and form a meaningful exchange;
- character_consistency: each character's turns fit the character description and stay consistent;
- scene_fidelity: the continuation is plausible for this scene of the work (situation, relationships, period);
- subjectivity: how strongly the turns foreground personal opinions, evaluations and feelings (1 = detached and factual, 5 = highly personal);
- emotional_intensity: how strong the expressed emotions are (1 = flat, 5 = intense).
Also classify the dominant narrative form as "first_person_direct_speech", "third_person_narration" or "mixed".
Return only JSON: {{"coherence": 1-5, "character_consistency": 1-5, "scene_fidelity": 1-5, "subjectivity": 1-5, "emotional_intensity": 1-5, "perspective": "...", "rationale": "one sentence"}}
"""

SPEAKER_PROMPT = """You are an expert in literary studies. Below are the characters of a scene from "{title}" by {author}.

Characters:
{cards}

Scene context: {context}

One turn of a generated dialogue is shown below. All character names were replaced by [NAME] and gendered pronouns by neutral ones, so you must decide from the content and style alone.
Turn: {turn}

1) Which character most probably produced this turn? Answer with one name from this list: {names}.
2) Rate the subjectivity of the turn from 1 (detached, factual) to 5 (highly personal, evaluative, emotional).
Return only JSON: {{"speaker": "...", "subjectivity": 1-5}}
"""

GENDERED = {"he": "they", "she": "they", "him": "them", "his": "their", "her": "their", "hers": "theirs",
            "himself": "themselves", "herself": "themselves"}


def cards_text(scene):
    return "\n".join(f"- {c['name']}: {c.get('personality', '')} (speech style: {c.get('speech_style', '')})"
                     for c in scene["characters"].values())


def mask(text, names):
    toks = sorted({w for n in names for w in re.split(r"[\s\-]+", n) if len(w) > 2 and w.lower() not in {"the"}} | set(names),
                  key=len, reverse=True)
    for w in toks:
        text = re.sub(rf"\b{re.escape(w)}\b", "[NAME]", text, flags=re.I)
    text = re.sub(r"(\[NAME\][\s\-]*)+", "[NAME] ", text)
    return re.sub(r"\b(he|she|him|his|her|hers|himself|herself)\b", lambda m: GENDERED[m.group(0).lower()], text, flags=re.I)


def ints(obj, keys):
    out = {}
    for k in keys:
        try:
            v = int(round(float(obj.get(k))))
            out[k] = v if 1 <= v <= 5 else None
        except Exception:
            out[k] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--judge", default="phi4:14b")
    ap.add_argument("--task", choices=["session", "speaker"], default="session")
    ap.add_argument("--conditions", nargs="+", default=["full"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--per-session", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--excerpts", default=os.path.join(os.path.dirname(__file__), "..", "texts", "excerpts.json"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta = {t["id"]: t for t in json.load(open(args.excerpts, encoding="utf-8"))}
    slug = args.judge.replace(":", "_")
    llm = OllamaClient(args.judge, temperature=0.0, max_tokens=300, cache_path=os.path.join(args.runs, f"cache_judge_{slug}.jsonl"))
    scenes = {}
    for f in glob.glob(os.path.join(args.runs, "scenes", "*.json")):
        tid, ms = os.path.basename(f)[:-5].split("__")
        scenes[(tid, ms)] = json.load(open(f, encoding="utf-8"))
    out_path = os.path.join(args.out, f"judge_{args.task}_{slug}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["key"] for l in open(out_path, encoding="utf-8")}
    fo = open(out_path, "a", encoding="utf-8")
    lock = threading.Lock()

    def write(r):
        with lock:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            fo.flush()

    def judge_session(key, base, p):
        try:
            obj, _ = parse_json(llm.generate(p, seed=0))
        except Exception:
            obj = {}
        write(dict(base, key=key, **ints(obj, ["coherence", "character_consistency", "scene_fidelity", "subjectivity", "emotional_intensity"]),
                   perspective=obj.get("perspective"), rationale=obj.get("rationale")))

    def judge_speaker(key, base, p, i, x, names):
        try:
            obj, _ = parse_json(llm.generate(p, seed=0))
        except Exception:
            obj = {}
        guess = str(obj.get("speaker", ""))
        g = next((n for n in names if n.lower() == guess.lower()), None) or next((n for n in names if n.lower() in guess.lower() or guess.lower() in n.lower()), None)
        write(dict(base, key=key, turn=i, speaker=x["speaker"], guess=g, correct=int(g == x["speaker"]), n_chars=len(names),
                   chance=1 / len(names), text=x["text"], **ints(obj, ["subjectivity"])))

    jobs = []
    files = []
    for c in args.conditions:
        files += sorted(glob.glob(os.path.join(args.runs, "sessions", c, "*", "*", "*.json")))
    for f in files:
        rec = json.load(open(f, encoding="utf-8"))
        m = rec["meta"]
        if args.seeds is not None and m["seed"] not in args.seeds or not rec["turns"]:
            continue
        scene = scenes[(m["text_id"], m["model"].replace(":", "_"))]
        t = meta[m["text_id"]]
        base = dict(text_id=m["text_id"], model=m["model"], mode=m["mode"], seed=m["seed"], condition=m["condition"], judge=args.judge)
        if args.task == "session":
            key = os.path.relpath(f, args.runs)
            if key in done:
                continue
            turns = "\n".join(f"{x['turn'] if 'turn' in x else i + 1}. {x['speaker']}: {x['text']}" for i, x in enumerate(rec["turns"]))
            p = SESSION_PROMPT.format(title=t["title"], author=t["author"], cards=cards_text(scene), context=scene["context"],
                                      n=len(rec["turns"]), turns=turns)
            jobs.append((judge_session, (key, base, p)))
        else:
            names = list(scene["characters"])
            if len(names) < 2:
                continue
            rng = random.Random(f"{f}")  # deterministic sample of turns per session
            idx = sorted(rng.sample(range(len(rec["turns"])), min(args.per_session, len(rec["turns"]))))
            for i in idx:
                key = f"{os.path.relpath(f, args.runs)}#{i}"
                if key in done:
                    continue
                x = rec["turns"][i]
                p = SPEAKER_PROMPT.format(title=t["title"], author=t["author"], cards=cards_text(scene), context=mask(scene["context"], names),
                                          turn=mask(x["text"], names), names=json.dumps(names))
                jobs.append((judge_speaker, (key, base, p, i, x, names)))
    print(f"{args.judge} / {args.task}: {len(jobs)} judgements to run", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in [ex.submit(fn, *a) for fn, a in jobs]:
            fut.result()
    fo.close()
    print("judge finished:", out_path)


if __name__ == "__main__":
    main()
